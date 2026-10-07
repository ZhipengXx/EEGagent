"""Worker-owned, source-bound complete development feedback before analysis."""
from __future__ import annotations
import csv,hashlib,json,os,subprocess,sys,time
from react_agent.eeg_research.agentic.measurement_context import parameter_count_comparison
from pathlib import Path
from typing import Any

def _read(p:Path)->dict[str,Any]:
 value=json.loads(p.read_text(encoding='utf-8'))
 if not isinstance(value,dict):raise ValueError('expected_json_object')
 return value

def _sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def _validated_report(camp:Path,latest:dict[str,Any],ref:str)->tuple[dict,dict,dict]:
 from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
 record=resolve_verified_artifact(camp,ref);report=_read(Path(record['path']))
 protocol=_read(camp/'execution_protocol.json');
 from react_agent.eeg_research.agentic.development_scope import development_scope
 scope=development_scope(protocol);job=Path(str(latest.get('job_dir') or ''))
 if not job.resolve().is_relative_to(camp.resolve()):raise ValueError('job_outside_campaign')
 j=_read(job/'job.json');frozen=_read(job/'frozen_run_spec.json');hook=_read(job/'hook_config.json');source=_read(job/'source_binding.json');metrics=_read(job/'metrics.json')
 if j.get('status')!='finished' or metrics.get('test_result') is not None:raise ValueError('run_not_development_finished')
 if not (latest.get('seed')==j.get('seed')==frozen.get('seed')==hook.get('seed')):raise ValueError('seed_identity_mismatch')
 if latest.get('spec_hash')!=frozen.get('spec_hash'):raise ValueError('run_spec_hash_mismatch')
 if latest.get('fidelity')!=j.get('fidelity') or j.get('fidelity')!=frozen.get('fidelity'):raise ValueError('fidelity_identity_mismatch')
 source_path=Path(str(source.get('class_file') or ''));source_path=source_path if source_path.is_absolute() else job/source_path
 if _sha(source_path)!=source.get('file_sha256') or source.get('file_sha256')!=latest.get('source_hash'):raise ValueError('run_source_changed')
 if report.get('final_test_accessed') is not False or report.get('gpu_seconds')!=0:raise ValueError('diagnostic_scope_invalid')
 if report.get('execution_fingerprint')!=protocol['fingerprint'] or latest.get('execution_fingerprint')!=protocol['fingerprint']:raise ValueError('protocol_identity_mismatch')
 if sorted(str(Path(row['path']).resolve()) for row in report.get('input_files') or [])!=scope['input_files']:raise ValueError('development_files_mismatch')
 if _sha(Path(report['predictions_csv_ref']))!=report['predictions_csv_sha256']:raise ValueError('prediction_artifact_changed')
 run=next((row for row in report.get('runs') or [] if row.get('job_id')==j.get('job_id')),None)
 if not run or run.get('candidate_id')!=j.get('candidate_id') or run.get('seed')!=j['seed']:raise ValueError('diagnostic_run_identity_mismatch')
 for name in ('checkpoint','source_binding','frozen_run_spec'):
  path=job/('last.ckpt' if name=='checkpoint' else name+'.json')
  if run.get(name+'_sha256')!=_sha(path):raise ValueError(name+'_changed')
 scores=run.get('scores') or {}
 if scores.get('query_count')!=scope['query_count'] or scores.get('candidate_count')!=scope['gallery_size']:raise ValueError('all_query_scope_mismatch')
 for key in ('fixed_bank_top1','fixed_bank_top5'):
  if abs(float(scores[key])-float(metrics[key]))>1e-12:raise ValueError('checkpoint_score_not_reproduced')
 if abs(float(scores['fixed_bank_top1'])-float(latest['fixed_bank_top1']))>1e-12:raise ValueError('accepted_score_mismatch')
 return record,report,run

def _existing_diagnostic(camp:Path,state:dict,latest:dict)->tuple[dict,dict,dict]|None:
 for evidence in reversed(state.get('evidence') or []):
  if evidence.get('kind')!='diagnostic_bundle':continue
  for ref in evidence.get('artifact_refs') or []:
   try:
    record,report,run=_validated_report(camp,latest,ref)
    return record,report,run
   except (OSError,ValueError,KeyError,TypeError,StopIteration):continue
 return None

def _ensure_report(camp:Path,state:dict,latest:dict)->tuple[dict,dict,dict]:
 found=_existing_diagnostic(camp,state,latest)
 if found:return found
 if latest.get('evaluation_valid') is not True or latest.get('fidelity')!='full':raise ValueError('accepted_full_required')
 job_id=str(latest.get('job_id') or '')
 if not job_id or Path(job_id).name!=job_id:raise ValueError('job_id_invalid')
 source_root=Path(__file__).resolve().parents[3]
 collector=source_root/'react_agent/eeg_training/development_diagnostics.py'
 name='all_queries_'+job_id+'_'+_sha(collector)[:8]+'.json';path=camp/'diagnostics'/name
 from react_agent.eeg_research.agentic.artifacts import register
 if path.exists():
  artifact=register(camp,path,kind='diagnostic_bundle',candidate_id=latest.get('candidate_id'))
 else:
  env={key:value for key,value in os.environ.items() if not any(word in key.upper() for word in ('API_KEY','SECRET','TOKEN','PASSWORD'))}
  env.update({'PYTHONPATH':str(source_root),'CUDA_VISIBLE_DEVICES':'','EEG_TRAIN_DEVICE':'cpu','EEG_FINAL_TEST':'0','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8'})
  from react_agent.eeg_training.protocol import torch_python
  training_python=torch_python()
  if not training_python or not Path(training_python).is_file():raise ValueError('cpu_training_interpreter_missing')
  cmd=[training_python,'-m','react_agent.eeg_training.development_diagnostics','--camp',str(camp),'--jobs',job_id,'--output',name]
  result=subprocess.run(cmd,cwd=str(camp),env=env,text=True,capture_output=True,timeout=120)
  if result.returncode:raise ValueError('cpu_diagnostic_failed:'+result.stderr[-1200:])
  output=json.loads(result.stdout)
  if output.get('diagnostic_registered') is not True:raise ValueError('collector_receipt_missing')
  artifact={'artifact_id':output['artifact_id']}
 record,report,run=_validated_report(camp,latest,artifact['artifact_id'])
 eid='ev_full_development_'+job_id
 if not any(row.get('evidence_id')==eid for row in state.get('evidence') or []):
  state.setdefault('evidence',[]).append({'evidence_id':eid,'kind':'diagnostic_bundle','candidate_id':latest['candidate_id'],'status':'verified',
   'execution_fingerprint':report['execution_fingerprint'],'artifact_refs':[record['artifact_id']],
   'source_run_evidence_ids':[latest['evidence_id']],
   'summary':{'scope':'complete frozen development queries only','runs':[run],'definitions':report['diagnostic_definitions'],
              'interpretation_limits':'Source-subject development diagnostics are descriptive, not final-subject performance or causal proof.'},
   'local_only':True,'recorded_at':time.time()})
 return record,report,run

def _predictions(report:dict,job_id:str,expected_queries:int)->dict[str,dict]:
 rows={}
 with Path(report['predictions_csv_ref']).open(newline='') as f:
  for row in csv.DictReader(f):
   if row['job_id']!=job_id:continue
   query=row['query_id']
   if query in rows:raise ValueError('duplicate_query')
   rows[query]=row
 if len(rows)!=expected_queries:raise ValueError('paired_query_count_mismatch')
 return rows

def _parent_comparison(camp:Path,state:dict,latest:dict,child_report:dict,child_run:dict,child_artifact:dict)->dict|None:
 target=str(latest['candidate_id']);binding=_read(camp/'candidates'/target/'approved_binding.json') if target!='baseline' else {}
 spec=_read(camp/'candidates'/target/'spec.json').get('experiment') or {} if target!='baseline' else {}
 parent=str(spec.get('ablation_of_candidate_id') or spec.get('parent_candidate_id') or '')
 if not parent or parent=='baseline':return None
 if parent!=binding.get('parent_candidate_id'):raise ValueError('declared_parent_identity_mismatch')
 matches=[row for row in state.get('evidence') or [] if row.get('evaluation_valid') is True and row.get('candidate_id')==parent
          and row.get('fidelity')==latest['fidelity'] and row.get('seed')==latest['seed'] and row.get('execution_fingerprint')==latest.get('execution_fingerprint')]
 if not matches:raise ValueError('same_seed_parent_full_missing')
 previous=matches[-1]
 from react_agent.eeg_research.agentic.handoffs import analysis_experiment_binding
 for row in (latest,previous):
  if analysis_experiment_binding(camp,row).get('status')!='verified':raise ValueError('paired_approval_binding_invalid')
 parent_artifact,parent_report,parent_run=_ensure_report(camp,state,previous)
 metadata={'hooks','note','note_zh','description','reuse','change_summary'};parent_binding=_read(camp/'candidates'/parent/'approved_binding.json')
 semantic=lambda value,section:{key:item for key,item in (value.get(section) or {}).items() if key not in metadata}
 changed=[section for section in ('model','objective','transform') if semantic(binding,section)!=semantic(parent_binding,section)]
 if len(changed)!=1:raise ValueError('ablation_not_single_config_section')
 child_recipe=child_run.get('frozen_recipe') or {};parent_recipe=parent_run.get('frozen_recipe') or {}
 if child_recipe!=parent_recipe:raise ValueError('paired_frozen_recipe_mismatch')
 query_count=child_run['scores']['query_count'];gallery_size=child_run['scores']['candidate_count']
 left=_predictions(child_report,latest['job_id'],query_count);right=_predictions(parent_report,previous['job_id'],query_count)
 if set(left)!=set(right):raise ValueError('paired_query_ids_mismatch')
 counts={'both_correct':0,'child_only_correct':0,'parent_only_correct':0,'both_wrong':0}
 for query in sorted(left):
  a,b=left[query],right[query]
  if (a['positive_index'],a['subject'])!=(b['positive_index'],b['subject']):raise ValueError('paired_positive_identity_mismatch')
  ah=a['top1_hit']=='True';bh=b['top1_hit']=='True'
  key='both_correct' if ah and bh else 'child_only_correct' if ah else 'parent_only_correct' if bh else 'both_wrong'
  counts[key]+=1
 difference=(counts['child_only_correct']-counts['parent_only_correct'])/query_count
 if abs(difference-(float(child_run['scores']['fixed_bank_top1'])-float(parent_run['scores']['fixed_bank_top1'])))>1e-12:raise ValueError('paired_tally_mismatch')
 child_job=Path(latest['job_dir']);parent_job=Path(previous['job_dir']);used={side:_read(job/'capabilities_used.json') for side,job in (('child',child_job),('parent',parent_job))}
 retained_stats=None
 if changed==['transform'] and binding.get('model',{}).get('statistics_on',True):
  a=_read(child_job/'train_statistics.json');b=_read(parent_job/'train_statistics.json')
  if a.get('source')!='train_files' or b.get('source')!='train_files' or a.get('values_per_channel')!=b.get('values_per_channel'):raise ValueError('retained_statistics_scope_mismatch')
  deltas={key:max(abs(float(x)-float(y)) for x,y in zip(a[key],b[key])) for key in ('mean','std')}
  if any(len(a[key])!=17 or len(b[key])!=17 for key in ('mean','std')) or max(deltas.values())>1e-10:raise ValueError('retained_statistics_differ')
  retained_stats={'source':'train_files','values_per_channel':a['values_per_channel'],'channel_count':17,'maximum_absolute_differences':deltas}
 report={'schema_version':'eeg_research.controlled_parent_comparison.v1','status':'verified','scope':'same-seed source-subject development; no final holdout',
  'final_test_accessed':False,'execution_fingerprint':latest['execution_fingerprint'],'fidelity':latest['fidelity'],'seed':latest['seed'],
  'candidate_id':target,'parent_candidate_id':parent,'candidate_evidence_id':latest['evidence_id'],'parent_evidence_id':previous['evidence_id'],
  'declared_ablation':spec.get('ablation_of_candidate_id')==parent,'comparison_role':'declared_same_seed_parent',
  'candidate_source_hash':latest['source_hash'],'parent_source_hash':previous['source_hash'],
  'candidate_spec_hash':latest['spec_hash'],'parent_spec_hash':previous['spec_hash'],
  'candidate_diagnostic_ref':child_artifact['artifact_id'],'parent_diagnostic_ref':parent_artifact['artifact_id'],
  'changed_hook_config_sections':changed,'frozen_recipe_identical':True,'training_statistics_retained':retained_stats,
  'capabilities_used':used,'query_count':query_count,'gallery_size':gallery_size,'paired_top1_counts':counts,'candidate_minus_parent_pp':100*difference,
  'candidate_top1':child_run['scores']['fixed_bank_top1'],'parent_top1':parent_run['scores']['fixed_bank_top1'],
  'baseline_comparison':latest.get('comparison'),'limits':['Single-seed paired development observation does not establish cross-seed reliability.','Identical training stopping policies allow different selected checkpoints and realized epochs.','These source subjects contributed training data; this is not unseen-subject accuracy.'],
  'created_at':time.time()}
 path=camp/'comparisons'/('parent_'+latest['job_id']+'.json')
 from react_agent.eeg_research.agentic.artifacts import register,resolve_verified_artifact
 eid='ev_parent_comparison_'+latest['job_id']
 existing=next((row for row in state.get('evidence') or [] if row.get('evidence_id')==eid),None)
 if existing:
  registered=resolve_verified_artifact(camp,existing['artifact_refs'][0]);stored=_read(Path(registered['path']))
  for key in ('candidate_spec_hash','parent_spec_hash','candidate_source_hash','parent_source_hash','candidate_minus_parent_pp'):
   if stored[key]!=report[key]:raise ValueError('existing_parent_comparison_changed')
  return {**existing['summary'],'parameter_count_comparison':parameter_count_comparison(stored)}
 if path.exists():
  stored=_read(path)
  for key in ('candidate_spec_hash','parent_spec_hash','candidate_source_hash','parent_source_hash','candidate_minus_parent_pp'):
   if stored[key]!=report[key]:raise ValueError('unregistered_parent_comparison_changed')
 else:
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 artifact=register(camp,path,kind='controlled_parent_comparison',candidate_id=target)
 summary={key:report[key] for key in ('candidate_id','parent_candidate_id','fidelity','seed','changed_hook_config_sections','training_statistics_retained','query_count','paired_top1_counts','candidate_minus_parent_pp','candidate_top1','parent_top1','limits')}
 summary['artifact_id']=artifact['artifact_id']
 state.setdefault('evidence',[]).append({'evidence_id':eid,'kind':'controlled_parent_comparison','candidate_id':target,'status':'verified','artifact_refs':[artifact['artifact_id']],
  'source_run_evidence_ids':[latest['evidence_id'],previous['evidence_id']],'summary':summary,'local_only':True,'recorded_at':time.time()})
 return {**summary,'parameter_count_comparison':parameter_count_comparison(_read(path))}

def ensure_development_feedback(camp:Path,state:dict,latest:dict)->dict:
 """Full-query diagnostics and parent availability have separate statuses."""
 if latest.get('fidelity')!='full':return {'status':'not_requested','scope':'pilot remains a routing signal'}
 try:
  artifact,report,run=_ensure_report(camp,state,latest)
 except (OSError,ValueError,KeyError,TypeError,StopIteration,subprocess.SubprocessError) as exc:
  detail=str(exc)[:1600];eid='ev_development_feedback_failure_'+str(latest.get('job_id'))
  if not any(row.get('evidence_id')==eid for row in state.get('evidence') or []):
   state.setdefault('evidence',[]).append({'evidence_id':eid,'kind':'diagnostic_failure','candidate_id':latest.get('candidate_id'),'status':'unavailable',
       'summary':{'reason':detail,'run_evidence_id':latest.get('evidence_id'),'scope':'full development diagnostics failed; scientific metrics remain their accepted values'},'local_only':True,'recorded_at':time.time()})
  return {'status':'unavailable','reason':detail,'scope':'diagnostic failure; no causal or final-holdout claim'}
 feedback={'status':'verified','all_query_artifact_id':artifact['artifact_id'],'query_count':run['scores']['query_count'],
           'parent_comparison':None,'parent_comparison_status':'not_applicable',
           'scope':'development diagnostics only; zero GPU seconds; immutable source/checkpoint scores reproduced'}
 try:
  comparison=_parent_comparison(camp,state,latest,report,run,artifact)
  feedback['parent_comparison']=comparison
  if comparison:feedback['parent_comparison_status']='verified'
 except (OSError,ValueError,KeyError,TypeError,StopIteration,subprocess.SubprocessError) as exc:
  detail=str(exc)[:1600]
  feedback['parent_comparison_reason']=detail
  if isinstance(exc,ValueError) and detail=='same_seed_parent_full_missing':
   feedback['parent_comparison_status']='missing_parent_run'
  else:
   feedback['parent_comparison_status']='unavailable'
   eid='ev_parent_feedback_failure_'+str(latest.get('job_id'))
   if not any(row.get('evidence_id')==eid for row in state.get('evidence') or []):
    state.setdefault('evidence',[]).append({'evidence_id':eid,'kind':'parent_comparison_failure','candidate_id':latest.get('candidate_id'),'status':'unavailable',
        'summary':{'reason':detail,'run_evidence_id':latest.get('evidence_id'),'scope':'parent comparison failed; verified full development diagnostics remain available'},'local_only':True,'recorded_at':time.time()})
 return feedback
