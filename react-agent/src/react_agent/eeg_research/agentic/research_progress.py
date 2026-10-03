"""Evidence-based continuation requirements; these do not authorize execution."""
from __future__ import annotations
import hashlib,json,statistics,re
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    try:
        value=json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):
        return {}


def verified_baseline_off_definition(camp: Path, target: str, binding: dict[str, Any]) -> dict[str, Any] | None:
    """A fresh source/config-bound check of the exact frozen off callable."""
    from react_agent.eeg_research.agentic.handoffs import candidate_cpu_check
    from react_agent.eeg_research.agentic.objective_effectiveness import objective_switch
    switch = objective_switch(binding.get('objective') or {})
    if not switch:
        return None
    view = candidate_cpu_check(camp, target, binding, binding.get('source_hash'))
    probe = (view.get('checks') or {}).get('objective_ablation_probe') or {}
    if (view.get('status') != 'verified_source_config_bound_cpu_receipt'
            or view.get('fresh_for_current_runtime') is not True
            or probe.get('switch_key') != switch or probe.get('switch_value') is not False
            or probe.get('exact_baseline_callable') is not True):
        return None
    return {key: view.get(key) for key in ('source_hash', 'spec_hash', 'check_ref', 'check_sha256', 'check_fingerprint')} | {
        'switch': switch, 'definition': 'exact_frozen_baseline_callable',
        'scope': 'Synthetic implementation identity only; no training score or authorization.'}


def _same_baseline_off_definition(parent: dict[str, Any], child: dict[str, Any], switch: str) -> bool:
    """Omitted inactive parameters are not an extra mechanism when both off paths
    are freshly verified as the identical frozen baseline callable. Unknown or
    changed supplied child parameters remain a mismatch.
    """
    for record in (parent, child):
        proof = record.get('baseline_off_definition') or {}
        binding = record['binding']
        if (proof.get('definition') != 'exact_frozen_baseline_callable'
                or proof.get('switch') != switch or proof.get('source_hash') != binding.get('source_hash')
                or proof.get('spec_hash') != binding.get('spec_hash')):
            return False
    metadata = {'hooks','note','note_zh','description','reuse','change_summary'}
    left = {k:v for k,v in (parent['binding'].get('objective') or {}).items() if k not in metadata | {switch}}
    right = {k:v for k,v in (child['binding'].get('objective') or {}).items() if k not in metadata | {switch}}
    return all(key in left and value == left[key] for key, value in right.items())


def declared_measurement_followups(executed: dict[str, Any], full: dict[str, dict[int, float]],
                                  seeds: list[int], *, fulfilled_switches: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expose only explicit signed comparisons after actual approved execution.

    This is a continuation requirement, never approval to design or train. Prose
    without literal seed numbers or an exact switch=false promise stays advisory.
    """
    pending_seeds=[];pending_switches=[]
    for target,record in executed.items():
        spec=record['spec'];binding=record['binding']
        comparisons=[str(item) for item in spec.get('planned_comparisons') or []]
        signed_measurements=comparisons+[str(spec.get('prediction') or '')]
        full_comparisons=[item for item in signed_measurements if 'full' in item.lower() or '全保真' in item]
        named=set()
        for item in full_comparisons:
            for value in re.findall(r'(?:\bseeds?\b|种子)\s*[\[\(（]?\s*(\d+(?:\s*[,/、，]\s*\d+)*)',item,re.I):
                named.update(int(part) for part in re.findall(r'\d+',value))
        if seeds and set(seeds).issubset(named):
            missing=[seed for seed in seeds if seed not in full.get(target,{})]
            if missing:pending_seeds.append({'candidate_id':target,'source_hash':binding['source_hash'],
                'spec_hash':binding['spec_hash'],'observed_training_seeds':sorted(full.get(target,{})),
                'missing_training_seeds':missing,'reason':'Signed planned_comparisons or prediction explicitly promises these paired full seeds in English or Chinese.'})
        from react_agent.eeg_research.agentic.objective_effectiveness import objective_switch
        objective=binding.get('objective') or {};switch=objective_switch(objective)
        if not isinstance(switch,str) or not switch:continue
        promised=any(re.search(re.escape(switch)+r'\s*=\s*false\b',item,re.I) for item in comparisons)
        if not promised or objective.get(switch,objective.get('ablation_default',True)) is not True:continue
        matched=[]
        for child,child_record in executed.items():
            cb=child_record['binding'];cs=child_record['spec'];co=cb.get('objective') or {}
            if (cb.get('parent_candidate_id')!=target or cs.get('ablation_of_candidate_id')!=target
                or co.get(switch,co.get('ablation_default',True)) is not False):continue
            semantic=lambda value:{k:v for k,v in value.items() if k not in {'hooks','note','note_zh','description','reuse','change_summary'}}
            left=semantic(dict(objective));right=semantic(dict(co));left.pop(switch,None);right.pop(switch,None)
            exact_config = left == right
            verified_off = _same_baseline_off_definition(record, child_record, switch)
            if (not (exact_config or verified_off)
                    or any(semantic(cb.get(k) or {})!=semantic(binding.get(k) or {}) for k in ('model','transform'))):continue
            paired=sorted(set(full.get(target,{}))&set(full.get(child,{})))
            if paired:matched.append({'candidate_id':child,'paired_training_seeds':paired,
                'matching_basis':'exact_other_hook_config' if exact_config else 'fresh_source_config_bound_exact_baseline_off_callables',
                'parent_off_definition': record.get('baseline_off_definition') if not exact_config else None,
                'child_off_definition': child_record.get('baseline_off_definition') if not exact_config else None})
        if matched and fulfilled_switches is not None:
            fulfilled_switches.append({'candidate_id':target,'source_hash':binding['source_hash'],
                'spec_hash':binding['spec_hash'],'switch':switch,'required_value':False,'matched_off_controls':matched,
                'scope':'Accepted paired full seeds of reviewed children; missing child seeds remain independent requirements.'})
        if not matched:pending_switches.append({'candidate_id':target,'source_hash':binding['source_hash'],
            'spec_hash':binding['spec_hash'],'switch':switch,'required_value':False,
            'reason':'Signed planned_comparisons promises a same-seed full objective-off run; a different mechanism ablation does not fulfill it.',
            'next_action_hint':'Design a child with this exact switch false and inherited model/transform; runtime approval and source review remain required.'})
    return pending_seeds,pending_switches


def objective_effectiveness_findings(camp: Path, state: dict[str, Any], target: str,
                                    source_hash: str, spec_hash: str) -> list[dict[str, Any]]:
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    findings=[]
    for row in state.get('evidence') or []:
        if row.get('kind')!='objective_validity_diagnostic' or row.get('candidate_id')!=target:continue
        for ref in row.get('artifact_refs') or []:
            try:
                registry=resolve_verified_artifact(camp,ref);report=_read(Path(registry['path']))
            except (OSError,ValueError,KeyError):continue
            if (report.get('schema_version')!='eeg_research.objective_tied_target_equivalence.v1'
                or report.get('source_hash')!=source_hash or report.get('spec_hash')!=spec_hash
                or report.get('final_test_accessed') is not False):continue
            findings.append({k:v for k,v in report.items() if k!='measurements'}|{
                'evidence_id':row['evidence_id'],'artifact_id':ref,'artifact_sha256':registry['sha256'],
                'effectiveness_status':'baseline_equivalent_under_tied_frozen_targets'})
    return findings


def objective_semantic_findings(camp: Path, state: dict[str, Any], target: str,
                                source_hash: str, spec_hash: str) -> list[dict[str, Any]]:
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    findings=[]
    for row in state.get("evidence") or []:
        if row.get("kind")!="objective_semantic_diagnostic" or row.get("candidate_id")!=target:continue
        for ref in row.get("artifact_refs") or []:
            try:
                registry=resolve_verified_artifact(camp,ref);report=_read(Path(registry["path"]))
            except (OSError,ValueError,KeyError):continue
            if (report.get("schema_version")!="eeg_research.objective_semantics.v1"
                or report.get("source_hash")!=source_hash or report.get("spec_hash")!=spec_hash
                or report.get("final_test_accessed") is not False):continue
            findings.append({k:v for k,v in report.items() if k not in {"probe","source_excerpts"}} |
                {"probe_summary":{k:v for k,v in report["probe"].items() if k!="probes"},
                 "evidence_id":row["evidence_id"],"artifact_id":ref,"artifact_sha256":registry["sha256"]})
    return findings


def progress_context(camp: Path, state: dict[str, Any], goal: dict[str, Any],
                     budget: dict[str, Any], actions: list[str]) -> dict[str, Any]:
    """Count applied full-fidelity mechanisms from accepted, source-bound jobs."""
    labels={
        'encoder':'encoder', 'objective':'objective', 'statistics':'train_statistics',
        'augmentation':'train_augmentation', 'ablation':'controlled_ablation',
    }
    required=list(dict.fromkeys(family for text in goal.get('allowed_changes') or []
                    for token,family in labels.items() if token in str(text).lower()))
    coverage={family:[] for family in required}
    pilot_coverage={family:[] for family in required}
    seeds=[int(seed) for seed in goal.get('training_seeds') or state.get('training_seeds') or []]
    full:dict[str,dict[int,float]]={}
    seen=set()
    invalid=[]
    executed={}
    semantic_findings={}
    objective_hook_execution_coverage=[]
    ineffective_objective_jobs=[]
    effectiveness={}
    for row in state.get('evidence') or []:
        if row.get('evaluation_valid') is not True or row.get('fidelity') not in ('pilot','full'):
            continue
        name=str(row.get('job_id') or Path(str(row.get('job_dir') or '')).name)
        if not name or name in seen:continue
        seen.add(name)
        target=str(row.get('candidate_id') or '')
        job=camp/'jobs'/name
        record=_read(job/'job.json');frozen=_read(job/'frozen_run_spec.json')
        binding=_read(job/'source_binding.json');metrics=_read(job/'metrics.json')
        consumed=_read(job/'hook_consumed.json');hook=_read(job/'hook_config.json')
        used=_read(job/'capabilities_used.json')
        seed=record.get('seed')
        source=Path(str(binding.get('class_file') or ''))
        if not source.is_absolute():source=job/source
        valid=(record.get('status')=='finished' and record.get('candidate_id')==target
               and isinstance(seed,int) and seed==frozen.get('seed')==hook.get('seed')
               and metrics.get('test_result') is None and source.is_file())
        if valid:
            valid=hashlib.sha256(source.read_bytes()).hexdigest()==binding.get('file_sha256')
        if not valid:
            invalid.append(name);continue
        if target=='baseline':
            if row['fidelity']=='full' and isinstance(metrics.get('fixed_bank_top1'),(int,float)):
                full.setdefault(target,{})[seed]=float(metrics['fixed_bank_top1'])
            continue
        approved=_read(camp/'candidates'/target/'approved_binding.json')
        spec=(_read(camp/'candidates'/target/'spec.json').get('experiment') or {})
        if (approved.get('spec_hash')!=frozen.get('spec_hash')
                or approved.get('source_hash')!=binding.get('file_sha256')
                or any((approved.get(k) or {})!=(frozen.get(k) or {}) for k in ('model','objective','transform'))
                or (approved.get('approval_record') or {}).get('status')!='approved'):
            invalid.append(name);continue
        executed[target]={'spec':spec,'binding':approved,
            'baseline_off_definition':verified_baseline_off_definition(camp,target,approved)}
        findings=objective_semantic_findings(camp,state,target,approved['source_hash'],approved['spec_hash'])
        if findings:semantic_findings[target]=findings
        if row['fidelity']=='full' and isinstance(metrics.get('fixed_bank_top1'),(int,float)):
            full.setdefault(target,{})[seed]=float(metrics['fixed_bank_top1'])
        capabilities=set(spec.get('required_capability_ids') or [])
        # Legacy signed proposals placed hook names in model/transform. Count
        # their actual executed mechanism, while new approvals reject metadata.
        for section in ('model','transform'):
            capabilities.update(approved.get(section,{}).get('hooks') or [])
        effective=[]
        if 'build_encoder' in capabilities and consumed.get('model',{}).get('applied') is True:
            effective.append('encoder')
        if (approved.get('objective') and consumed.get('objective',{}).get('applied') is True
                and int(used.get('objective_train_calls') or 0)>0):
            objective_hook_execution_coverage.append({'candidate_id':target,'job_id':name,'seed':seed,'fidelity':row['fidelity']})
            findings=objective_effectiveness_findings(camp,state,target,approved['source_hash'],approved['spec_hash'])
            if findings:
                effectiveness[target]=findings
                ineffective_objective_jobs.append({'candidate_id':target,'job_id':name,'reason':'source-bound algebra diagnostic establishes baseline-equivalent training gradients for tied frozen image targets'})
            else:
                effective.append('objective')
        used_hooks=used.get('hooks') or {}
        if ('fit_statistics' in capabilities and used_hooks.get('fit_statistics') is True
                and approved.get('model',{}).get('statistics_on',True) is not False):
            effective.append('train_statistics')
        if (used_hooks.get('build_training_transform') is True and int(used.get('transform_train_calls') or 0)>0
                and int(used.get('transform_eval_calls') or 0)==0):
            effective.append('train_augmentation')
        control=approved.get('control_candidate_id') or 'baseline'
        parent=approved.get('parent_candidate_id') or 'baseline'
        if parent!='baseline' and spec.get('ablation_of_candidate_id')==parent:
            parent_binding=_read(camp/'candidates'/parent/'approved_binding.json')
            metadata={'hooks','note','note_zh','description','reuse','change_summary'}
            sections=('model','objective','transform')
            semantic=lambda b,k:{key:value for key,value in (b.get(k) or {}).items() if key not in metadata}
            changed=[key for key in sections if semantic(approved,key)!=semantic(parent_binding,key)]
            matched_parent=any(e.get('candidate_id')==parent and e.get('evaluation_valid') is True
                         and e.get('fidelity')==row['fidelity'] and e.get('seed')==seed
                         and e.get('execution_fingerprint')==row.get('execution_fingerprint')
                         for e in state.get('evidence') or [])
            if len(changed)==1 and matched_parent:effective.append('controlled_ablation')
        destination=coverage if row['fidelity']=='full' else pilot_coverage
        for family in effective:
            if family in destination:destination[family].append({'candidate_id':target,'job_id':name,'seed':seed})
    missing=[family for family,rows in coverage.items() if not rows]
    means={target:statistics.mean(values.values()) for target,values in full.items() if values}
    best=max(means,key=means.get) if means else None
    pending=[]
    if best and best!='baseline':
        missing_seeds=[seed for seed in seeds if seed not in full[best]]
        if missing_seeds:pending.append({'candidate_id':best,'missing_training_seeds':missing_seeds})
    questions=[{'family':family,'reason':'No accepted full-fidelity execution of this requested mechanism yet.'} for family in missing]
    questions.extend({'family':'paired_confirmation','reason':str(item)} for item in pending)
    fulfilled_switches=[]
    declared_seeds,declared_switches=declared_measurement_followups(executed,full,seeds,fulfilled_switches=fulfilled_switches)
    questions.extend({'family':'declared_full_seed_comparison',**item} for item in declared_seeds)
    questions.extend({'family':'declared_switch_ablation',**item} for item in declared_switches)
    legal=[action for action in actions if action in {'design_experiment','propose_experiment','implement_candidate',
                'repair_candidate','run_pilot','run_full','replicate','retrieve_methods','collect_diagnostics','diagnose_results'}]
    design_room=(int(budget.get('llm_calls_left') or 0)>=13 and
                 int(budget.get('training_jobs_left') or 0)>0 and
                 float(budget.get('gpu_seconds_left') or 0)>0 and
                 sum(row.get('status') not in {'implementation_failed','requires_framework_extension'}
                     for row in state.get('candidates') or [])<int(goal.get('max_candidates') or state.get('max_candidates') or 0))
    training_room=(int(budget.get('llm_calls_left') or 0)>=3 and
                   int(budget.get('training_jobs_left') or 0)>0 and float(budget.get('gpu_seconds_left') or 0)>0)
    can_continue=(design_room and any(action in actions for action in ('design_experiment','propose_experiment','implement_candidate','repair_candidate'))
                  or training_room and any(action in actions for action in ('run_pilot','run_full','replicate')))
    return {'schema_version':'eeg_research.progress.v1','authority':'runtime_source_bound_development_records',
            'required_mechanism_families':required,'full_fidelity_coverage':coverage,'pilot_only_coverage':pilot_coverage,
            'missing_full_fidelity_families':missing,'pending_best_method_confirmation':pending,
            'pending_declared_full_seed_comparisons':declared_seeds,'pending_declared_switch_ablations':declared_switches,
            'fulfilled_declared_switch_ablations':fulfilled_switches,
            'unfinished_requirements':questions,'best_development_method_so_far':best,
            'legal_research_next_actions':legal,'budget_can_continue':bool(can_continue),
            'coverage_rejected_job_ids':invalid,
            'objective_hook_execution_coverage':objective_hook_execution_coverage,
            'objective_semantic_findings':semantic_findings,
            'objective_effectiveness_findings':effectiveness,'ineffective_objective_jobs':ineffective_objective_jobs,
            'limits':['Coverage is execution evidence, not evidence of benefit or causality.',
                      'This is not a final-holdout result or execution permission.',
                      'Missing trainable targets do not disable legal experiment design.']}


def stopping_block_reason(scope: dict[str, Any], reason: Any) -> str | None:
    """Do not accept a scientific stop while requested executable work remains."""
    progress=scope.get('research_progress') or {}
    if not progress.get('budget_can_continue'):return None
    if reason=='budget_exhausted':return 'premature_stop:authorized_budget_remains'
    if progress.get('unfinished_requirements') and reason in {
            'goal_addressed','no_supported_next_experiment','no_progress','blocked'}:
        return 'premature_stop:requested_executable_research_remains'
    return None


def _factual_correction_projection(camp: Path, state: dict[str, Any], report: dict[str, Any]) -> dict[str, Any] | None:
    """Preserve measured correction fields from the installed complete-query audit.

    The original receipt independently checked every CSV row. This projection
    checks current artifact/run/source bindings; it does not repeat inference.
    """
    from react_agent.eeg_research.agentic.artifacts import file_digest
    from react_agent.eeg_research.agentic.handoffs import development_view
    if report.get('schema_version') != 'c7_parent_complete_development_comparison.v1':
        return None
    try:
        protocol = _read(camp / 'execution_protocol.json')
        from react_agent.eeg_research.agentic.development_scope import development_scope
        expected = development_scope(protocol)
        if (report.get('final_test_accessed') is not False or report.get('gpu_seconds') != 0
                or report.get('execution_fingerprint') != protocol.get('fingerprint')
                or report.get('query_count_per_run') != expected['query_count'] or report.get('gallery_size') != expected['gallery_size']):
            raise ValueError('factual_correction_scope_mismatch')
        accepted = {e.get('job_id'): e for e in state.get('evidence') or []
                    if e.get('evaluation_valid') is True and e.get('fidelity') == 'full'
                    and e.get('execution_fingerprint') == protocol.get('fingerprint')}
        def checked_file(path: str, digest: str) -> Path:
            p = Path(path)
            if not p.resolve().is_relative_to(camp.resolve()) or file_digest(p) != digest:
                raise ValueError('factual_correction_artifact_binding_mismatch')
            return p
        runs = {}
        for run in report.get('runs') or []:
            evidence = accepted.get(run.get('job_id'))
            if (not evidence or evidence.get('evidence_id') != run.get('evidence_ref')
                    or evidence.get('candidate_id') != run.get('candidate_id')
                    or evidence.get('seed') != run.get('seed')
                    or run.get('query_count') != expected['query_count'] or run.get('gallery_size') != expected['gallery_size']
                    or abs(float(evidence['fixed_bank_top1']) - float(run['scores']['fixed_bank_top1'])) > 1e-12):
                raise ValueError('factual_correction_run_binding_mismatch')
            source = run['loaded_source_binding']
            checked_file(source['class_file'], source['file_sha256'])
            if source['file_sha256'] != evidence.get('source_hash'):
                raise ValueError('factual_correction_source_mismatch')
            for path, digest in run['saved_artifact_hashes'].items():
                checked_file(path, digest)
            runs[run['job_id']] = run
        if not runs:
            raise ValueError('factual_correction_runs_missing')
        comparisons = report.get('matched_comparisons') or []
        if not comparisons:
            raise ValueError('factual_correction_comparisons_missing')
        for row in comparisons:
            parent, child = runs[row['parent_job']], runs[row['candidate_job']]
            if (parent['seed'] != row['seed'] or child['seed'] != row['seed']
                    or row['matched_query_count'] != expected['query_count'] or row['gallery_size'] != expected['gallery_size']
                    or abs(row['delta_pp'] - 100 * (child['scores']['fixed_bank_top1'] - parent['scores']['fixed_bank_top1'])) > 1e-10):
                raise ValueError('factual_correction_pair_binding_mismatch')
        checked_file(report['csv_ref'], report['csv_sha256'])
        prior = report['c1_encoder_execution']
        for run in prior['runs']:
            evidence = accepted.get(run['job_id'])
            job = camp / 'jobs' / run['job_id']
            if (not evidence or evidence.get('seed') != run['seed']
                    or evidence.get('source_hash') != prior['source_hash']
                    or evidence.get('spec_hash') != prior['spec_hash']
                    or abs(float(evidence['fixed_bank_top1']) - float(run['top1'])) > 1e-12):
                raise ValueError('factual_correction_prior_run_mismatch')
            checked_file(str(job / 'metrics.json'), run['metrics_sha256'])
            checked_file(str(job / 'source_binding.json'), run['source_binding_sha256'])
            source = _read(job / 'source_binding.json')
            checked_file(source['class_file'], prior['source_hash'])
        return development_view({
            'status': 'verified_current_source_run_artifact_bindings',
            'provenance': report.get('provenance'),
            'scope': 'All frozen development queries and gallery per run; named matched seeds and source/spec bindings only. No final subject.',
            'matched_parent_prediction_comparisons': comparisons,
            'prior_completed_encoder_execution': prior,
            'corrections': report.get('corrections'),
            'limitations': report.get('limitations'),
            'authority': 'Measured external factual correction; no hypothesis assessment or training authorization.',
        })
    except (OSError, ValueError, KeyError, TypeError):
        return {'status': 'unavailable', 'missing_inputs': ['current_factual_correction_scope_or_binding_verification_failed'],
                'authority': 'No correction payload delivered; inspect original registered artifact.'}


def verified_diagnostic_facts(camp: Path, state: dict[str, Any], *, job_ids: set[str] | None = None, target_ids: set[str] | None = None, max_facts: int = 6) -> list[dict[str, Any]]:
    """Inject compact verified all-query diagnostic facts before history truncation."""
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    facts=[];seen=set();limit=max(1,min(int(max_facts),12))
    for row in reversed(state.get('evidence') or []):
        if row.get('kind')!='diagnostic_bundle':continue
        for ref in row.get('artifact_refs') or []:
            if ref in seen:continue
            try:
                registry=resolve_verified_artifact(camp,ref)
                report=_read(Path(registry['path']))
            except (OSError,ValueError,KeyError):continue
            if report.get('final_test_accessed') is not False:continue
            seen.add(ref)
            correction = _factual_correction_projection(camp, state, report) if row.get('status') == 'verified' else None
            correction_delivered = False
            for run in report.get('runs') or []:
                job=str(run.get('job_id') or '')
                if job_ids is not None and job not in job_ids:continue
                if target_ids is not None and run.get('candidate_id') not in target_ids:continue
                accepted = next((e for e in state.get('evidence') or [] if e.get('evaluation_valid') is True and e.get('job_id')==job), None)
                if accepted is None:continue
                try:
                    from react_agent.eeg_research.agentic.development_feedback import _validated_report
                    if report.get('schema_version') == 'eeg_inter.development_diagnostics.v2':
                        _validated_report(camp, accepted, ref)
                    elif correction is None or correction.get('status') != 'verified_current_source_run_artifact_bindings':continue
                except (OSError, ValueError, KeyError, TypeError):continue
                facts.append({'evidence_id':row.get('evidence_id'),'artifact_id':ref,'artifact_sha256':registry['sha256'],
                    'job_id':job,'seed':run.get('seed'),'scores':run.get('scores'),'representation':run.get('representation'),
                    'mean_direction_norm':run.get('mean_direction_norm'),'mean_margin':run.get('retrieval',{}).get('margin_mean'),
                    'retrieval':run.get('retrieval'),'development_subject_groups':run.get('development_subject_groups'),
                    'frozen_training_policy':{key:run.get('frozen_recipe',{}).get(key) for key in ('epochs','stop','negative_sampling_policy')},
                    'realized_training':run.get('realized_training'),
                    'scope':'all frozen development queries; no final subject'})
                if correction is not None and not correction_delivered:
                    facts[-1]['factual_correction'] = correction
                    correction_delivered = True
            if len(facts)>=limit:return facts[:limit]
    return facts


def verified_training_diagnostic_facts(camp: Path, state: dict[str, Any], candidate_id: str,
                                      source_hash: str, spec_hash: str) -> list[dict[str, Any]]:
    """Keep sampled training exposure separate from all-query retrieval facts."""
    from react_agent.eeg_research.agentic.artifacts import resolve_verified_artifact
    facts=[]
    for row in state.get('evidence') or []:
        if row.get('kind')!='training_diagnostic' or row.get('candidate_id')!=candidate_id:continue
        for ref in row.get('artifact_refs') or []:
            try:
                registered=resolve_verified_artifact(camp,ref);report=_read(Path(registered['path']))
            except (OSError,ValueError,KeyError):continue
            if (report.get('schema_version')!='eeg_research.local_duplicate_scope.v1'
                or report.get('final_test_accessed') is not False
                or report.get('source_hash')!=source_hash or report.get('spec_hash')!=spec_hash):continue
            if not any(run.get('evaluation_valid') is True and run.get('job_id')==report.get('job_id')
                       and run.get('source_hash')==source_hash and run.get('spec_hash')==spec_hash
                       for run in state.get('evidence') or []):continue
            facts.append({**{k:v for k,v in report.items() if k!='replica_counts'},
                          'evidence_id':row['evidence_id'],'artifact_id':ref,'artifact_sha256':registered['sha256']})
    return facts[-6:]
