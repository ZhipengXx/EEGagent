"""SQLite episodes, compatibility filter, and verification."""

from __future__ import annotations

from pathlib import Path

from react_agent.fmri.memory.repository import SqliteMemoryRepository
from react_agent.fmri.memory.retrieval import retrieve_bundle
from react_agent.fmri.memory.service import should_curate


def test_episode_write_and_no_duplicate(tmp_path: Path) -> None:
    repo = SqliteMemoryRepository(tmp_path / "mem.sqlite3", namespace="t")
    episode = {
        "run_id": "r1",
        "sample_id": "apple_01b",
        "input_hash": "abc",
        "generation_profile_id": "static_gray4_image1_gray11_tribev2_v1",
        "space_name": "fsaverage5",
        "normalization": "none_raw_signed",
        "t_len": 16,
        "checkpoint_hash": "facebook/tribev2",
        "execution_domain": "real",
        "screening_decision": "pass_configured",
        "verification_status": "unverified",
    }
    first = repo.append_episode(episode)
    second = repo.append_episode(episode)
    assert first == second
    inspect = repo.inspect()
    assert len(inspect["episodes"]) == 1


def test_cross_profile_numeric_not_retrieved(tmp_path: Path) -> None:
    repo = SqliteMemoryRepository(tmp_path / "mem.sqlite3", namespace="t")
    repo.append_episode(
        {
            "run_id": "old",
            "sample_id": "old12",
            "input_hash": "x",
            "generation_profile_id": "static_1s_7s_tribev2",
            "space_name": "fsaverage5",
            "normalization": "none_raw_signed",
            "t_len": 12,
            "checkpoint_hash": "facebook/tribev2",
            "execution_domain": "real",
            "screening_decision": "flagged",
        }
    )
    items = repo.retrieve({"namespace": "t"})
    bundle = retrieve_bundle(
        items,
        query={
            "generation_profile_id": "static_gray4_image1_gray11_tribev2_v1",
            "space_name": "fsaverage5",
            "normalization": "none_raw_signed",
            "t_len": 16,
            "checkpoint_hash": "facebook/tribev2",
            "execution_domain": "real",
        },
        exclude_sample_ids={"apple_01b"},
    )
    assert bundle["hit_count"] == 0
    assert bundle["excluded"][0]["reason"].startswith("incompatible")


def test_unverified_not_confirmed_and_overturned_kept(tmp_path: Path) -> None:
    repo = SqliteMemoryRepository(tmp_path / "mem.sqlite3", namespace="t")
    eid = repo.append_episode(
        {
            "run_id": "r",
            "sample_id": "s",
            "input_hash": "h",
            "generation_profile_id": "static_gray4_image1_gray11_tribev2_v1",
            "t_len": 16,
            "space_name": "fsaverage5",
            "normalization": "none_raw_signed",
            "checkpoint_hash": "facebook/tribev2",
            "screening_decision": "flagged",
        }
    )
    repo.record_verification(
        {
            "episode_id": eid,
            "verification_status": "overturned",
            "verification_scope": "numeric_consistency_only",
            "source": "human",
            "evidence": "known false flag",
        }
    )
    rows = repo.retrieve({"namespace": "t"})
    assert rows[0]["verification_status"] == "overturned"
    assert rows[0]["screening_decision"] == "flagged"
    assert should_curate({"screening_decision": "flagged"}) is True
    assert should_curate({"screening_decision": "pass_configured"}) is False
