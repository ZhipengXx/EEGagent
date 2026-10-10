"""Operation-local caches for exact immutable manifest reads and byte hashes.

No persistent verdict cache: dependency bytes are always hashed again. Cached
manifests are keyed by the actual bytes, not filesystem timestamps (some mounted
filesystems expose coarse metadata). Mutable state/ledger/goal reads never use
the JSON cache, and cached parsed manifests reject mutation.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import json

_SCOPE = ContextVar("native_verification_scope", default=None)
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


def read_json(path, reader):
    scope = _SCOPE.get()
    if scope is None or Path(path).name not in _MANIFESTS:
        return reader(path)
    data = Path(path).read_bytes()
    key = (str(Path(path).resolve()), hashlib.sha256(data).hexdigest())
    if key not in scope["json"]:
        value = json.loads(data.decode("utf-8"))
        scope["json"][key] = freeze(value)
    return scope["json"][key]


def file_hash(path, hasher):
    # Re-reading bytes is required even if inode/timestamps/size are unchanged.
    return hasher(path)


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
