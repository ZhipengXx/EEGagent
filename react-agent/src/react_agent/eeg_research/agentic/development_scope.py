"""Development identities derived from the frozen protocol, without holdout I/O."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def development_scope(protocol: dict[str, Any]) -> dict[str, Any]:
    from react_agent.eeg_research.agentic.execution_protocol import design_from_protocol
    from react_agent.eeg_training.protocol import split_plan

    queries = protocol.get('validation_query_ids')
    images = protocol.get('validation_image_ids')
    if not isinstance(queries, list) or not queries or not isinstance(images, list) or not images:
        raise ValueError('frozen_development_identities_unavailable')
    if len(set(queries)) != len(queries) or len(set(images)) != len(images):
        raise ValueError('frozen_development_identities_duplicated')
    plan = split_plan(Path(protocol['data_root']), design_from_protocol(protocol))
    files = sorted(str(p.resolve()) for p in plan.val_files)
    forbidden = {str(p.resolve()) for p in plan.forbidden_files}
    if not files or set(files) & forbidden:
        raise ValueError('development_holdout_overlap_or_empty')
    return {'query_count': len(queries), 'gallery_size': len(images), 'input_files': files,
            'subjects': sorted({Path(p).parent.name for p in files})}
