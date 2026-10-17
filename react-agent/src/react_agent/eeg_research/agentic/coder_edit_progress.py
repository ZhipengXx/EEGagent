"""Small, receipt-backed Coder progress without replaying successful edits."""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from collections.abc import Iterable
from typing import Any

from .coder import validate_tool_request
from .coder_edit_feedback import valid_feedback


_CANDIDATE_PATH = "extension/eeg_candidate.py"
_RECEIPT_VERSION = "eeg_research.coder_edit_receipt.v1"


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _args_digest(args: Any) -> str | None:
    _tool, valid, error = validate_tool_request({"tool": "edit_candidate_code", "args": args})
    if error is not None or valid.get("path") != _CANDIDATE_PATH or not _sha(valid.get("expected_base_hash")):
        return None
    return hashlib.sha256(json.dumps(valid, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def edit_receipt(args: dict[str, Any], result: dict[str, Any], before_sha256: str | None) -> dict[str, Any] | None:
    """Bind exact arguments to an actual edit, before log display shortening.

    A successful no-op or a replay is not evidence for another successful edit.
    This helper reads no files and does not certify any check result.
    """
    digest = _args_digest(args)
    if (digest is None or result.get("ok") is not True or result.get("already_applied")
            or result.get("changed") is False or not _sha(before_sha256)
            or before_sha256 != args["expected_base_hash"]
            or not _sha(result.get("sha256")) or result["sha256"] == before_sha256
            or type(result.get("edit_count")) is not int or result["edit_count"] != len(args["edits"])
            or result.get("path", args["path"]) != args["path"]):
        return None
    return {"schema_version": _RECEIPT_VERSION, "args_sha256": digest,
            "path": args["path"], "base_sha256": before_sha256,
            "source_sha256": result["sha256"], "edit_count": result["edit_count"], "changed": True}


def _row_receipt(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("tool") != "edit_candidate_code":
        return None
    result = row.get("result")
    if (not isinstance(result, dict) or result.get("ok") is not True
            or result.get("already_applied") or result.get("changed") is False):
        return None
    proof = row.get("edit_receipt")
    if isinstance(proof, dict):
        if (proof.get("schema_version") != _RECEIPT_VERSION or proof.get("changed") is not True
                or proof.get("path") != _CANDIDATE_PATH or not _sha(proof.get("args_sha256"))
                or not _sha(proof.get("base_sha256")) or not _sha(proof.get("source_sha256"))
                or proof["base_sha256"] == proof["source_sha256"]
                or type(proof.get("edit_count")) is not int or not 1 <= proof["edit_count"] <= 20
                or result.get("sha256") != proof["source_sha256"]
                or result.get("path", proof["path"]) != proof["path"]
                or ("edit_count" in result and result["edit_count"] != proof["edit_count"])):
            return None
        digest = _args_digest(row.get("args"))
        if digest is not None and digest != proof["args_sha256"]:
            return None
        return dict(proof)
    # Old logs can prove an edit only when complete arguments and the native
    # successful edit_count/source receipt survived their display shortening.
    args = row.get("args")
    if not isinstance(args, dict):
        return None
    return edit_receipt(args, result, args.get("expected_base_hash"))


class EditingProgress:
    """Index already-loaded rows once, then observe each newly persisted row."""

    def __init__(self, rows: Iterable[dict[str, Any]] = ()) -> None:
        self._by_args: dict[str, dict[str, Any]] = {}
        self._completed: list[dict[str, Any]] = []
        self._last_tool: dict[str, Any] | None = None
        self._last_successful_write: dict[str, Any] | None = None
        self._edit_feedback: dict[str, dict[str, Any]] = {}
        for row in rows:
            self.observe(row)

    def observe(self, row: dict[str, Any]) -> None:
        result = row.get("result")
        if not isinstance(result, dict):
            return
        self._last_tool = {"tool": row.get("tool"), "step": row.get("step"),
                           **{key: result[key] for key in ("ok", "error", "already_applied", "changed") if key in result}}
        if row.get("tool") in {"apply_candidate_patch", "edit_candidate_code"} and result.get("ok") is False:
            # A new failed edit supersedes the previous edit observation, even
            # if this particular native error has no optional source feedback.
            self._edit_feedback = {}
        if (row.get("tool") in {"apply_candidate_patch", "edit_candidate_code"}
                and result.get("ok") is True and result.get("changed") is not False
                and not result.get("already_applied") and _sha(result.get("sha256"))
                and result.get("path") == _CANDIDATE_PATH):
            self._last_successful_write = {"tool": row["tool"], "step": row.get("step"),
                                           "path": result["path"], "source_sha256": result["sha256"],
                                           **({"edit_count": result["edit_count"]}
                                              if type(result.get("edit_count")) is int else {})}
            # A real new write supersedes old edit diagnostics, even when an
            # older producer did not record the optional local source view.
            self._edit_feedback = {}
        for key in ("edit_success_context", "edit_failure_context"):
            context = row.get(key)
            if (row.get("tool") in {"apply_candidate_patch", "edit_candidate_code"}
                    and valid_feedback(context)
                    and ((key == "edit_success_context" and result.get("ok") is True
                          and result.get("changed") is not False and not result.get("already_applied")
                          and result.get("sha256") == context["source_sha256"])
                         or (key == "edit_failure_context" and result.get("ok") is False
                             and result.get("error") == context.get("error")))):
                self._edit_feedback = {key: deepcopy(context)}
        proof = _row_receipt(row)
        if proof is None:
            return
        proof["original_step"] = row.get("step")
        if isinstance(row.get("call_id"), str):
            proof["original_call_id"] = row["call_id"]
        self._by_args[proof["args_sha256"]] = proof
        self._completed.append({key: proof[key] for key in (
            "original_step", "path", "base_sha256", "source_sha256", "edit_count")})
        self._completed = self._completed[-3:]

    def repeated_edit(self, args: dict[str, Any], current_sha256: str | None) -> dict[str, Any] | None:
        """Return a no-write receipt only for the exact proven resulting source."""
        digest = _args_digest(args)
        proof = self._by_args.get(digest) if digest is not None else None
        if proof is None or not _sha(current_sha256) or proof["source_sha256"] != current_sha256:
            return None
        return {"ok": True, "path": proof["path"], "sha256": current_sha256,
                "edit_count": proof["edit_count"], "already_applied": True, "changed": False,
                "original_step": proof["original_step"],
                **({"original_call_id": proof["original_call_id"]} if "original_call_id" in proof else {})}

    def edit_feedback(self, current_sha256: str | None) -> dict[str, dict[str, Any]]:
        """Project the latest recorded local page only onto its actual source.

        Reads do not discard a useful failed-edit diagnostic. A successful write
        or a newer diagnostic replaces it. The full log remains unchanged.
        """
        return {key: deepcopy(value) for key, value in self._edit_feedback.items()
                if _sha(current_sha256) and value["source_sha256"] == current_sha256}

    def snapshot(self, current_sha256: str | None, last_check: dict[str, Any] | None,
                 check_status: dict[str, Any]) -> dict[str, Any]:
        """Add compact facts; the original last_check/repair context stays intact."""
        check = last_check or {}
        fresh = check_status.get("fresh") is True
        passed = fresh and check.get("ok") is True
        next_tool = ("apply_candidate_patch" if not _sha(current_sha256) else
                     "finish_patch" if passed else
                     "edit_candidate_code" if fresh and last_check is not None else
                     "run_candidate_check")
        return {"current_source_sha256": current_sha256,
                "completed_edits": [dict(row) for row in self._completed],
                "last_tool": dict(self._last_tool) if self._last_tool is not None else None,
                "last_successful_write": (dict(self._last_successful_write)
                                          if self._last_successful_write is not None else None),
                "current_check": {"fresh": fresh, "passed": passed,
                                  **{key: check[key] for key in ("error", "source_sha256", "check_scope") if key in check}},
                "next_tool": next_tool}
