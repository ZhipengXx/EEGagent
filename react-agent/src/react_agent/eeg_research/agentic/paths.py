"""Resolve campaign and job identities. Client paths are never trusted."""

from __future__ import annotations

from pathlib import Path


def safe_name(value: str) -> str | None:
    if not value or value != Path(value).name:
        return None
    if value in {".", ".."} or "/" in value or "\\" in value:
        return None
    return value


def resolve_campaign(root: Path, campaign: str) -> Path | None:
    name = safe_name(campaign)
    if name is None:
        return None
    base = root.resolve()
    camp = (base / name).resolve()
    try:
        camp.relative_to(base)
    except ValueError:
        return None
    if not (camp / "campaign_state.json").is_file():
        return None
    return camp


def resolve_job_dir(camp: Path, job_id: str) -> Path | None:
    name = safe_name(job_id)
    if name is None:
        return None
    jobs = (camp / "jobs").resolve()
    target = (jobs / name).resolve()
    try:
        target.relative_to(jobs)
    except ValueError:
        return None
    if not target.is_dir():
        return None
    return target


def resolve_candidate_dir(camp: Path, candidate_id: str) -> Path | None:
    name = safe_name(candidate_id)
    if name is None:
        return None
    root = (camp / "candidates").resolve()
    target = (root / name).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    if not target.is_dir():
        return None
    return target
