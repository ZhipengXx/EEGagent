"""Select completed ablations whose interpretation changed with a new parent run."""
from __future__ import annotations
from pathlib import Path
from typing import Any

def analysis_targets(camp:Path,state:dict[str,Any],settled:dict[str,Any])->list[dict[str,Any]]:
 from react_agent.eeg_research.agentic.handoffs import analysis_experiment_binding
 targets=[settled]
 if settled.get('evaluation_valid') is not True or settled.get('fidelity')!='full':return targets
 seen={settled.get('evidence_id')};queue=[settled]
 while queue:
  parent=queue.pop(0)
  for row in reversed(state.get('evidence') or []):
   if (row.get('evidence_id') in seen or row.get('evaluation_valid') is not True
       or row.get('fidelity')!=parent.get('fidelity') or row.get('seed')!=parent.get('seed')
       or row.get('execution_fingerprint')!=parent.get('execution_fingerprint')
       or row.get('candidate_id')==parent.get('candidate_id')):continue
   binding=analysis_experiment_binding(camp,row)
   if binding.get('status')!='verified':continue
   experiment=binding.get('experiment') or {}
   declared_parent=experiment.get('ablation_of_candidate_id') or experiment.get('parent_candidate_id')
   if declared_parent!=parent.get('candidate_id'):continue
   # An already settled duplicate is not an independent seed or a new child.
   key=(row.get('candidate_id'),row.get('seed'),row.get('source_hash'),row.get('spec_hash'))
   if any((old.get('candidate_id'),old.get('seed'),old.get('source_hash'),old.get('spec_hash'))==key for old in targets):continue
   seen.add(row.get('evidence_id'));targets.append(row);queue.append(row)
 return targets
