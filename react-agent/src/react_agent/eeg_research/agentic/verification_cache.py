"""Process-local caches for immutable manifest reads and byte hashes.

Hashes and parsed manifests are keyed by the file's stat identity (path, device,
inode, size, mtime_ns, ctime_ns). Any rewrite, replacement or truncation changes
that identity and forces a fresh read. Files modified within RACY_SECONDS are
never cached, so a same-size rewrite inside one coarse timestamp tick is still
re-hashed. Nothing is persisted across processes. Mutable state/ledger/goal reads
never use the JSON cache, and cached parsed manifests reject mutation.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import json
import threading
import time

_SCOPE = ContextVar("native_verification_scope", default=None)
RACY_SECONDS = 2.0
_JSON_LIMIT = 64
_LOCK = threading.Lock()
_HASHES: dict[str, tuple[tuple, str]] = {}
_JSON: OrderedDict[str, tuple[tuple, object]] = OrderedDict()
_MEMO_LIMIT = 512
_MEMO: OrderedDict[tuple, object] = OrderedDict()
_MANIFESTS = {"method_search_manifest.json", "study_manifest.json", "suite_manifest.json",
              "execution_protocol.json", "protocol.json"}


def _immutable(*_args, **_kwargs):
    raise TypeError("cached_manifest_is_immutable")


class FrozenDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo):
        import copy
        return {copy.deepcopy(key, memo): copy.deepcopy(value, memo) for key, value in self.items()}


class FrozenList(list):
    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = sort = reverse = __iadd__ = __imul__ = _immutable

    def __deepcopy__(self, memo):
        import copy
        return [copy.deepcopy(value, memo) for value in self]


def freeze(value):
    if isinstance(value, dict):
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze(item) for item in value)
    return value


def signature(path):
    path = Path(path)
    stat = path.stat()
    return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


@contextmanager
def verification_scope():
    if _SCOPE.get() is not None:
        yield
        return
    token = _SCOPE.set({"json": {}, "identities": {}})
    try:
        yield
    finally:
        _SCOPE.reset(token)


def _stable_signature(path):
    """Stat identity, or None when the file is too fresh to trust its metadata."""
    try:
        sig = signature(path)
    except OSError:
        return None
    if time.time_ns() - max(sig[4], sig[5]) < RACY_SECONDS * 1e9:
        return None
    return sig


def read_json(path, reader):
    scope = _SCOPE.get()
    if scope is None or Path(path).name not in _MANIFESTS:
        return reader(path)
    sig = _stable_signature(path)
    if sig is None:
        # Too fresh for stat identity: fall back to an exact byte-keyed scope cache.
        data = Path(path).read_bytes()
        key = (str(Path(path).resolve()), hashlib.sha256(data).hexdigest())
        if key not in scope["json"]:
            scope["json"][key] = freeze(json.loads(data.decode("utf-8")))
        return scope["json"][key]
    with _LOCK:
        hit = _JSON.get(sig[0])
        if hit is not None and hit[0] == sig:
            _JSON.move_to_end(sig[0])
            return hit[1]
    value = freeze(json.loads(Path(path).read_bytes().decode("utf-8")))
    if _stable_signature(path) != sig:
        return value
    with _LOCK:
        _JSON[sig[0]] = (sig, value)
        _JSON.move_to_end(sig[0])
        while len(_JSON) > _JSON_LIMIT:
            _JSON.popitem(last=False)
    return value


def file_hash(path, hasher):
    sig = _stable_signature(path)
    if sig is None:
        return hasher(path)
    with _LOCK:
        hit = _HASHES.get(sig[0])
        if hit is not None and hit[0] == sig:
            return hit[1]
    value = hasher(path)
    # Only remember the hash if the file did not change while it was read.
    if _stable_signature(path) == sig:
        with _LOCK:
            _HASHES[sig[0]] = (sig, value)
    return value


def file_memo(name, paths, extra, compute):
    """Reuse a derived value while every file it was derived from is unchanged.

    Callers must return immutable values (or copy them on the way out).
    """
    sigs = tuple(_stable_signature(path) for path in paths)
    if any(sig is None for sig in sigs):
        return compute()
    key = (name, sigs, extra)
    with _LOCK:
        if key in _MEMO:
            _MEMO.move_to_end(key)
            return _MEMO[key]
    value = compute()
    if tuple(_stable_signature(path) for path in paths) != sigs:
        return value
    with _LOCK:
        _MEMO[key] = value
        while len(_MEMO) > _MEMO_LIMIT:
            _MEMO.popitem(last=False)
    return value


def clear_caches() -> None:
    with _LOCK:
        _HASHES.clear()
        _JSON.clear()
        _MEMO.clear()


def object_identity(value, hasher):
    scope = _SCOPE.get()
    if scope is None or not isinstance(value, dict):
        return hasher(value)
    parts = []
    retained = []
    for key, item in sorted(value.items()):
        if isinstance(item, (FrozenDict, FrozenList)):
            parts.append((key, "frozen", id(item)))
            retained.append(item)
        elif item is None or type(item) in {str, int, float, bool}:
            parts.append((key, type(item).__name__, item))
        else:
            return hasher(value)
    key = tuple(parts)
    if key not in scope["identities"]:
        # Keep immutable children alive for the key's lifetime. Objects not
        # owned by the JSON cache can otherwise die and have their IDs reused.
        scope["identities"][key] = (hasher(value), retained)
    return scope["identities"][key][0]
