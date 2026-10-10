"""Read-only measurement facts, distinct from diagnostic samples and approval."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluation_population_context(protocol: dict[str, Any] | None) -> dict[str, Any]:
    protocol = protocol if isinstance(protocol, dict) else {}
    groups = {key: protocol.get(key) for key in ('validation_query_ids', 'gallery_image_ids')}
    for key, values in groups.items():
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != len(values)):
            return {'status': 'unavailable', 'reason': 'missing_or_invalid_' + key,
                    'scope': 'No evaluation count inferred from image count or diagnostics.'}
    queries, images = len(groups['validation_query_ids']), len(groups['gallery_image_ids'])
    result = {'status': 'frozen_protocol_bound', 'execution_fingerprint': protocol.get('fingerprint'),
            'development_query_count': queries, 'gallery_image_count': images,
            'one_query_hit_delta_pp': 100.0 / queries,
            'uniform_gallery_chance_top1': 1.0 / images,
            'count_basis': 'Frozen validation query IDs and gallery image IDs.',
            'scope': 'Development evaluation identities. Diagnostic sample sizes do not change the score denominator; no final-subject data read.'}
    if protocol.get('evaluation_mode') == 'loso_method_search':
        # This mode freezes ten equal-weight 200-query benchmark folds. The
        # protocol's validation IDs describe the nine-source-subject monitor,
        # rather than the primary method benchmark's hit denominator.
        result.update(primary_metric='benchmark.loso_mean_fixed_gallery_top1',
            development_gallery_image_count=images,
            development_one_query_hit_delta_pp=100.0 / queries,
            benchmark_query_count_per_fold=200, benchmark_fold_count=10,
            benchmark_query_count_total=2000, gallery_image_count=200,
            one_query_hit_delta_pp=0.05, per_fold_one_query_hit_delta_pp=0.5,
            uniform_gallery_chance_top1=0.005,
            count_basis='Frozen native method contract: ten equal-weight folds, 200 held-out queries and 200 gallery items per fold. Actual complete-suite counts remain in aggregate.scores; validation_query_ids belong to the separate source-subject development monitor.',
            scope='Method-development benchmark primary metric; one extra hit in one fold changes the ten-fold mean by 0.05 percentage points. Declared contract counts, not a new measurement or independent final test.')
    return result


def parameter_count_comparison(report: dict[str, Any]) -> dict[str, Any]:
    used = report.get('capabilities_used')
    used = used if isinstance(used, dict) else {}
    def count(side, kind):
        section = used.get(side)
        counts = section.get('parameter_counts') if isinstance(section, dict) else None
        value = counts.get(kind) if isinstance(counts, dict) else None
        return value if type(value) is int and value >= 0 else None
    child, parent = count('child', 'encoder'), count('parent', 'encoder')
    return {'status': 'available' if child is not None and parent is not None else 'incomplete',
            'candidate_encoder_scalar_parameters': child, 'parent_encoder_scalar_parameters': parent,
            'candidate_objective_scalar_parameters': count('child', 'objective'),
            'parent_objective_scalar_parameters': count('parent', 'objective'),
            'candidate_to_parent_encoder_ratio': child / parent if child is not None and parent else None,
            'basis': 'Parameter scalar counts saved by the actual training runtime in the registered source-bound comparison.',
            'limits': 'These counts include all encoder parameters, not a trainability classification. Parameter tensor counts in CPU checks and equal embedding widths are not scalar capacity counts. A ratio alone does not prove a predeclared matched-capacity criterion.'}


def verified_parameter_count_comparisons(camp: Path, state: dict[str, Any], *, limit: int = 12,
                                        protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact, file_digest
    from react_agent.eeg_research.agentic.run_context import load_approved_binding
    if protocol is None:
        try:
            protocol = json.loads((camp / 'execution_protocol.json').read_text())
        except (OSError, ValueError):
            protocol = {}
    fingerprint = protocol.get('fingerprint') if isinstance(protocol, dict) else None
    rows = []; seen = set()
    for evidence in reversed(state.get('evidence') or []):
        if evidence.get('kind') != 'controlled_parent_comparison' or evidence.get('status') != 'verified':
            continue
        for ref in evidence.get('artifact_refs') or []:
            if ref in seen:
                continue
            try:
                registered = resolve_verified_artifact(camp, ref)
                report = json.loads(Path(registered['path']).read_text())
                if (not isinstance(report, dict)
                        or report.get('schema_version') != 'eeg_research.controlled_parent_comparison.v1'
                        or report.get('status') != 'verified'
                        or report.get('final_test_accessed') is not False
                        or not fingerprint or report.get('execution_fingerprint') != fingerprint):
                    continue
                for side, key in (('candidate', 'candidate_id'), ('parent', 'parent_candidate_id')):
                    target = str(report[key]); binding = load_approved_binding(camp, target) or {}
                    if (binding.get('source_hash') != report[side + '_source_hash']
                            or binding.get('spec_hash') != report[side + '_spec_hash']
                            or file_digest(camp/'candidates'/target/'extension/eeg_candidate.py') != binding['source_hash']):
                        raise ValueError('comparison_binding_changed')
                    run = next(row for row in state.get('evidence') or []
                               if row.get('evidence_id') == report[side + '_evidence_id'])
                    if (run.get('evaluation_valid') is not True or run.get('candidate_id') != target
                            or run.get('seed') != report.get('seed') or run.get('fidelity') != report.get('fidelity')
                            or run.get('source_hash') != binding['source_hash'] or run.get('spec_hash') != binding['spec_hash']
                            or run.get('execution_fingerprint') != report.get('execution_fingerprint')):
                        raise ValueError('comparison_run_binding_changed')
                seen.add(ref)
                rows.append({key: report.get(key) for key in ('candidate_id', 'parent_candidate_id', 'seed', 'fidelity',
                    'candidate_source_hash', 'parent_source_hash', 'candidate_spec_hash', 'parent_spec_hash')} | {
                    'evidence_id': evidence['evidence_id'], 'artifact_id': ref, 'artifact_sha256': registered['sha256'],
                    'parameter_count_comparison': parameter_count_comparison(report)})
            except (OSError, ValueError, KeyError, TypeError, StopIteration):
                continue
    return {'comparisons': rows[:limit], 'verified_comparison_count': len(rows),
            'omitted_comparison_count': max(0, len(rows) - limit),
            'scope': 'Accepted paired development runs and current source/spec bindings; no new score, approval or capacity-match conclusion.'}
