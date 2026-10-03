"""Availability and matched-seed arithmetic with explicit incomplete metadata."""
from __future__ import annotations

import math
from typing import Any


def _score(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1:
        return float(value)
    return None


def coverage_summary(candidate_runs: list[dict[str, Any]], matched_runs: list[dict[str, Any]],
                     facts: list[dict[str, Any]], feedback_history: list[dict[str, Any]]) -> dict[str, Any]:
    by_evidence = {row['evidence_id']: row for row in matched_runs if row.get('evidence_id')}
    by_job = {row['job_id']: row for row in facts if row.get('job_id')}
    feedback_by_job = {row['job_id']: row for row in feedback_history if row.get('job_id')}
    rows, paired = [], []
    for candidate in sorted(candidate_runs, key=lambda row: (not isinstance(row.get('seed'), int),
                                                              row.get('seed') if isinstance(row.get('seed'), int) else 0)):
        seed, job = candidate.get('seed'), candidate.get('job_id')
        control = by_evidence.get(candidate.get('control_id'))
        fact, feedback = by_job.get(job), feedback_by_job.get(job) or {}
        groups = (fact or {}).get('development_subject_groups') or []
        count = ((fact or {}).get('scores') or {}).get('query_count')
        subjects = [g.get('subject') for g in groups]
        full_groups = bool(fact and isinstance(count, int) and count > 0 and groups
                           and all(isinstance(g.get('query_count'), int) and g['query_count'] > 0 for g in groups)
                           and all(isinstance(s, str) and s for s in subjects)
                           and len(set(subjects)) == len(subjects)
                           and sum(g['query_count'] for g in groups) == count)
        score = _score(candidate.get('fixed_bank_top1'))
        comparison = candidate.get('comparison') or {}
        comparable = bool(score is not None and isinstance(seed, int) and not isinstance(seed, bool)
                          and control and comparison.get('comparable') is True
                          and control.get('candidate_id') == 'baseline' and control.get('seed') == seed
                          and control.get('fidelity') == candidate.get('fidelity')
                          and candidate.get('execution_fingerprint')
                          and control.get('execution_fingerprint') == candidate['execution_fingerprint'])
        control_score = _score(control.get('fixed_bank_top1')) if comparable else None
        comparable = comparable and control_score is not None
        delta = 100 * (score - control_score) if comparable else None
        if comparable: paired.append((score, control_score))
        parent = feedback.get('parent_comparison') or {}
        rows.append({'job_id': job, 'evidence_id': candidate.get('evidence_id'), 'training_seed': seed,
                     'missing_identity_fields': [key for key in ('job_id', 'seed') if candidate.get(key) is None],
                     'candidate_top1': score, 'matched_baseline_evidence_id': candidate.get('control_id'),
                     'baseline_comparable': comparable, 'matched_baseline_top1': control_score,
                     'delta_vs_baseline_pp': delta, 'full_development_diagnostic_status': feedback.get('status'),
                     'full_query_groups_available': full_groups, 'full_query_subject_count': len(groups) if full_groups else 0,
                     'full_query_margin_mean': (fact or {}).get('mean_margin'),
                     'diagnostic_artifact_id': (fact or {}).get('artifact_id'),
                     'parent_comparison_status': feedback.get('parent_comparison_status'),
                     'parent_comparison_artifact_id': parent.get('artifact_id'),
                     'delta_vs_parent_pp': parent.get('candidate_minus_parent_pp')})
    return {'authority': 'verified_run_arithmetic_and_input_availability', 'scope': 'frozen development only',
            'seed_records': rows, 'paired_baseline_seed_count': len(paired),
            'paired_candidate_mean_top1': sum(a for a, b in paired) / len(paired) if paired else None,
            'paired_baseline_mean_top1': sum(b for a, b in paired) / len(paired) if paired else None,
            'paired_mean_delta_pp': 100 * sum(a - b for a, b in paired) / len(paired) if paired else None,
            'limits': ['These facts are not hypothesis assessments or runtime promotion.',
                       'Missing identity fields do not create an independent seed or matched comparison.',
                       'Different checkpoint epochs under one frozen stopping policy remain allowed.',
                       'Parent absence does not make a verified full-query diagnostic absent.']}
