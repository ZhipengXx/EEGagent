"""Read-only verification of a proposed historical source checkout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def checkout_preview(camp: Path, checkout: Path, original_project: Path) -> dict:
    manifest = json.loads((camp / "method_search_manifest.json").read_text())
    dependencies = manifest.get("framework_dependencies") or []
    if not dependencies:
        raise ValueError("historical_frozen_framework_dependencies_missing")
    rows = []
    for dependency in dependencies:
        original = Path(dependency["path"])
        if not original.is_absolute():
            raise ValueError("historical_dependency_absolute_path_required")
        try:
            relative = original.relative_to(original_project.resolve())
        except ValueError:
            rows.append({"original_path": str(original), "status": "external_dependency_requires_original_environment"})
            continue
        proposed = (checkout.resolve() / relative).resolve()
        if not proposed.is_relative_to(checkout.resolve()):
            raise ValueError("historical_checkout_dependency_escape")
        actual = hashlib.sha256(proposed.read_bytes()).hexdigest() if proposed.is_file() else None
        rows.append({"original_path": str(original), "checkout_path": str(proposed),
                     "expected_sha256": dependency["sha256"], "actual_sha256": actual,
                     "status": "matches" if actual == dependency["sha256"] else "missing" if actual is None else "mismatch"})
    return {"preview_only": True, "campaign": str(camp.resolve()), "checkout": str(checkout.resolve()),
            "framework_files": rows, "source_checkout_matches": all(row["status"] == "matches" for row in rows),
            "native_resume_authorized": False,
            "required_original_absolute_mount": str(original_project.resolve()),
            "data_environment_and_process_verification": "required_in_original_runtime_before_resume",
            "instruction": "Map matching old source into its original absolute path inside an isolated mount/container; keep original data/campaign paths and native verification. This preview writes nothing and starts no worker."}
