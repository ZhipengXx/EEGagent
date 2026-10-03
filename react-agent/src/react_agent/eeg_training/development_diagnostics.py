from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--camp',required=True)
parser.add_argument('--jobs',nargs='+',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
CAMP=Path(args.camp).resolve()
SRC=Path(__file__).resolve().parents[2]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['EEG_TRAIN_DEVICE']='cpu';os.environ['EEG_FINAL_TEST']='0'
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from react_agent.eeg_research.agentic.execution_protocol import design_from_protocol
from react_agent.eeg_training.protocol import split_plan,geometry
from react_agent.eeg_training.data import load_feature_cache,collect_records,RetrievalTrials,collate_retrieval
from react_agent.eeg_training.fixed_bank import frozen_bank,FixedBankTally
from react_agent.eeg_training.train_entry import rebuild_encoder
from react_agent.eeg_research.agentic.artifacts import register

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def rank_summary(x):
    x=x.double();s=torch.linalg.svdvals(x-x.mean(0,keepdim=True))
    probabilities=s/s.sum().clamp_min(1e-12)
    effective=float(torch.exp(-(probabilities*probabilities.clamp_min(1e-12).log()).sum()))
    variance=s.square();variance=variance/variance.sum().clamp_min(1e-12)
    k90=int(torch.searchsorted(variance.cumsum(0),torch.tensor(.9,dtype=variance.dtype)).item()+1)
    centered_energy=float((x-x.mean(0,keepdim=True)).square().sum()/x.square().sum().clamp_min(1e-12))
    return {'centered_singular_entropy_effective_rank':effective,'pcs_for_90_percent_variance':k90,'centered_energy_fraction':centered_energy,'sample_count':int(x.shape[0]),'feature_count':int(x.shape[1])}

if Path(args.output).name!=args.output or not args.output.endswith('.json'):raise RuntimeError('output_must_be_a_json_basename')
if any(Path(name).name!=name for name in args.jobs):raise RuntimeError('job_must_be_an_exact_basename')
torch.set_num_threads(8);started=time.time()
output=CAMP/'diagnostics'/args.output
if output.exists():raise RuntimeError('diagnostic_already_exists_inspect_before_repeat')
output.parent.mkdir(parents=True,exist_ok=True)
p=read(CAMP/'execution_protocol.json');design=design_from_protocol(p);plan=split_plan(Path(p['data_root']),design)
if set(plan.val_files)&set(plan.forbidden_files):raise RuntimeError('development_holdout_overlap')
spec=geometry(p['dataset']);features=load_feature_cache(plan.feature_caches[1])
records,images=collect_records(plan.val_files,features,spec['channels'],None)
assert len(records)==len(p['validation_query_ids']) and set(images)==set(p['validation_image_ids'])
assert set(str(r['query_id']) for r in records)==set(p['validation_query_ids'])
loader=DataLoader(RetrievalTrials(records,spec['timesteps']),batch_size=256,shuffle=False,collate_fn=collate_retrieval)
bank,labels=frozen_bank(records);bank=F.normalize(bank,dim=-1)
reports=[];csv_path=output.with_suffix('.csv')
with csv_path.open('x',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=['job_id','seed','query_id','subject','positive_index','prediction_index','top1_hit','top5_hit','positive_cosine','strongest_negative_cosine','margin'])
    writer.writeheader()
    for name in args.jobs:
        job=CAMP/'jobs'/name;j=read(job/'job.json');metrics=read(job/'metrics.json')
        assert j['status']=='finished' and metrics['test_result'] is None
        binding=read(job/'source_binding.json');bound_source=Path(binding['class_file'])
        if not bound_source.is_absolute():bound_source=job/bound_source
        if sha(bound_source)!=binding['file_sha256']:raise RuntimeError('bound_source_changed:'+name)
        frozen=read(job/'frozen_run_spec.json');hook=read(job/'hook_config.json')
        if not (j['seed']==frozen['seed']==hook['seed']):raise RuntimeError('actual_seed_identity_mismatch:'+name)
        encoder=rebuild_encoder(spec,job/'last.ckpt',job).cpu().eval()
        tally=FixedBankTally(bank,labels);parts=[]
        with torch.inference_mode():
            for batch in loader:
                z=encoder(batch['eeg']);assert z.shape==(len(batch['eeg']),bank.shape[1]) and torch.isfinite(z).all()
                tally.add(z);parts.append(F.normalize(z,dim=-1))
        z=torch.cat(parts);scores=z@bank.T;top=scores.topk(min(5,bank.shape[0]),dim=-1).indices
        hit1=(top[:,0]==labels);hit5=(top==labels[:,None]).any(-1)
        positive=scores.gather(1,labels[:,None]).squeeze(1)
        negative=scores.clone();negative.scatter_(1,labels[:,None],float('-inf'));strongest=negative.max(-1).values;margin=positive-strongest
        measured=tally.result()
        if abs(measured['fixed_bank_top1']-metrics['fixed_bank_top1'])>1e-12 or abs(measured['fixed_bank_top5']-metrics['fixed_bank_top5'])>1e-12:
            raise RuntimeError('checkpoint_score_not_reproduced:'+name)
        subjects=sorted(set(str(r['subject']) for r in records));groups=[]
        for subject in subjects:
            ix=torch.tensor([i for i,r in enumerate(records) if r['subject']==subject])
            groups.append({'subject':subject,'query_count':len(ix),'top1':float(hit1[ix].double().mean()),'top5':float(hit5[ix].double().mean()),'mean_margin':float(margin[ix].double().mean())})
        for i,r in enumerate(records):
            writer.writerow({'job_id':name,'seed':j['seed'],'query_id':r['query_id'],'subject':r['subject'],'positive_index':int(labels[i]),'prediction_index':int(top[i,0]),'top1_hit':bool(hit1[i]),'top5_hit':bool(hit5[i]),'positive_cosine':float(positive[i]),'strongest_negative_cosine':float(strongest[i]),'margin':float(margin[i])})
        reports.append({'job_id':name,'fidelity':j['fidelity'],'frozen_recipe':j.get('effective_config'),'realized_training':{'maximum_epochs':j['epochs'],'completed_epochs':metrics.get('last_epoch'),'selected_checkpoint_epoch':metrics['selected_checkpoint_epoch']},'candidate_id':j['candidate_id'],'evidence_ref':'ev_'+name,'seed':j['seed'],'checkpoint_sha256':sha(job/'last.ckpt'),'source_binding_sha256':sha(job/'source_binding.json'),'frozen_run_spec_sha256':sha(job/'frozen_run_spec.json'),'selected_checkpoint_epoch':metrics['selected_checkpoint_epoch'],'checkpoint_score_reproduced':True,'scores':measured,'representation':rank_summary(z),'mean_direction_norm':float(z.double().mean(0).norm()),'retrieval':{'query_count':len(records),'candidate_count':len(bank),'positive_cosine_mean':float(positive.double().mean()),'strongest_negative_cosine_mean':float(strongest.double().mean()),'margin_mean':float(margin.double().mean()),'margin_median':float(margin.median()),'ties_for_best_score_queries':int(((scores==scores.max(-1,keepdim=True).values).sum(-1)>1).sum()),'unique_top1_gallery_predictions':int(top[:,0].unique().numel())},'development_subject_groups':groups})
report={'collector_source_sha256':sha(Path(__file__)), 'schema_version':'eeg_inter.development_diagnostics.v2','kind':'diagnostic_bundle','scope':'source-subject development only; no final test access, no training, no candidate performance claim','execution_fingerprint':p['fingerprint'],'final_test_accessed':False,'device':'cpu','input_files':[{'path':str(f),'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in plan.val_files],'training_runtime_sha256':sha(SRC/'react_agent/eeg_training/train_entry.py'),'training_policy_note':'Maximum epochs and stopping policy are the frozen protocol values, shared by baseline and candidates. Completed epochs below 100 are not a separate budget intervention. A build_encoder hook cannot change that policy.','diagnostic_definitions':{'rank':'center normalized EEG embeddings over all frozen development queries; exp(entropy of singular values normalized by their sum)); centered_energy_fraction is squared centered Frobenius norm divided by squared uncentered Frobenius norm','margin':'positive cosine minus maximum nonpositive gallery cosine','group_results':'metadata-only evaluation on each declared development subject; not unseen-subject performance','limitations':'Observed development rank/margin do not diagnose a unique causal mechanism. These subjects also contributed training EEG; group dispersion is not a final subject confidence interval.'},'gallery_representation':rank_summary(bank),'runs':reports,'predictions_csv_ref':str(csv_path),'predictions_csv_sha256':sha(csv_path),'created_at':time.time(),'wall_seconds':time.time()-started,'gpu_seconds':0}
with output.open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
artifact=register(CAMP,output,kind='diagnostic_bundle',candidate_id=reports[0]['candidate_id'] if len({r['candidate_id'] for r in reports})==1 else None)
print(json.dumps({'diagnostic_registered':True,'artifact_id':artifact['artifact_id'],'path':str(output),'report_sha256':sha(output),'wall_seconds':report['wall_seconds'],'gpu_seconds':0},ensure_ascii=False))
