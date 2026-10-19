"""The one Reviewer override accepted by review revalidation.

The paid 32k retry proof modeled the removed single-plan v1 Reviewer and is
retired until it is ported to the v2 Reviewer; worker.revalidate_review
rejects the override explicitly.
"""
from __future__ import annotations

OVERRIDE = {"candidate_reviewer": {"max_tokens": 32768}}


def validate_override(value):
    if value and (value != OVERRIDE or type(value["candidate_reviewer"]["max_tokens"]) is not int):
        raise ValueError("review_revalidation_only_allows_candidate_reviewer_max_tokens_32768")
    return value or {}
