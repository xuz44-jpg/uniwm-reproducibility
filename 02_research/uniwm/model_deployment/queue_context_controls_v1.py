"""Append a new immutable task family after preflight; speech guard owns execution."""
import argparse
from datetime import datetime,timezone
import fcntl
from pathlib import Path
from suite_common import read,digest,atomic,frozen
from context_controls_common import root


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--append',action='store_true');a=ap.parse_args();out=root(a.study);monitor=a.root/'monitor'
    assert read(out/'prepare_complete.json')['status']=='passed'
    assert read(out/'training_audit/complete.json')['status']=='passed'
    names=['context_controls_common.py','prepare_context_controls_v1.py','run_context_controls_v1.py',
        'score_context_controls_v1.py','summarize_context_controls_v1.py','audit_context_training_v1.py',
        'queue_context_controls_v1.py','package_context_controls_v1.py','watch_context_controls_v1.py',
        'suite_common.py','suite_backends.py','train_habitat_model.py','confirmation_common.py','uniwm_v3_common.py',
        'habitat_flow_common.py','habitat_study_common.py','habitat_v2_metrics.py','probe_habitat_flow_v1.py',
        'verify_uniwm_embeddings.py']
    freeze=dict(status='frozen',source_hashes={str(a.study/n):digest(a.study/n) for n in names},
        protocol_sha256=digest(out/'protocol.json'),weights_sha256=digest(out/'weights.json'),
        native_manifest_hashes={m:digest(out/f'native_{m}.json') for m in ['nwm','raenwm','uniwm']},
        native_reuse_provenance_frozen=True,world_model_updates=0)
    frozen(out/'execution_freeze.json',freeze)
    python=str(a.root/'venv/bin/python');uni='external/uniwm-action-pilot/venv/bin/python'
    sim=str(a.study/'env/bin/python');jobs=[]
    def job(name,script,args=(),minimum=5000,env=python,cwd=None):
        jobs.append(dict(id='context_v1_'+name,resumable=True,cwd=str(cwd or a.study),min_free_mib=minimum,
            argv=[env,'-u',str(a.study/script),'--study',str(a.study),*args]))
    # Reference rendering does not query any world model and catches pose/replay errors early.
    job('render96','prepare_context_controls_v1.py',['--root',str(a.root),'--phase','render'],1000,sim)
    for model in ['raenwm','nwm','uniwm']:
        args=['--root',str(a.root),'--model',model,'--cohort','native']
        env=uni if model=='uniwm' else python;cwd=a.study if model=='uniwm' else a.root/'repos'/model
        job(model+'_native_smoke','run_context_controls_v1.py',args+['--smoke'],20000 if model=='uniwm' else 10000,env,cwd)
        job(model+'_native40','run_context_controls_v1.py',args,20000 if model=='uniwm' else 10000,env,cwd)
        job(model+'_native_score40','score_context_controls_v1.py',args,5000,python,cwd)
    for model in ['raenwm','nwm','uniwm']:
        args=['--root',str(a.root),'--model',model,'--cohort','habitat'];env=uni if model=='uniwm' else python
        cwd=a.study if model=='uniwm' else a.root/'repos'/model
        job(model+'_context_smoke','run_context_controls_v1.py',args+['--smoke'],20000 if model=='uniwm' else 10000,env,cwd)
    for model,count in [('raenwm',216),('nwm',216),('uniwm',432)]:
        args=['--root',str(a.root),'--model',model,'--cohort','habitat'];env=uni if model=='uniwm' else python
        cwd=a.study if model=='uniwm' else a.root/'repos'/model
        job(model+f'_context{count}','run_context_controls_v1.py',args,20000 if model=='uniwm' else 10000,env,cwd)
        job(model+f'_context_score{count}','score_context_controls_v1.py',args,5000,python,cwd)
    job('summary984','summarize_context_controls_v1.py',minimum=0)
    job('package','package_context_controls_v1.py',minimum=0)
    frozen(out/'queue_append.json',dict(jobs=jobs))
    if not a.append:print('QUEUE_PREPARED',len(jobs));return
    assert not (monitor/'pause').exists(),'An explicit research pause is still active; do not silently clear it'
    status=read(monitor/'status.json')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(status['utc'])).total_seconds()
    assert age<10 and status['phase']=='complete' and status['worker_pid'] is None
    lock=(monitor/'context_queue_update.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    q=read(monitor/'queue.json');states=read(monitor/'jobs_state.json')
    assert all(states[j['id']]['status']=='done' for j in q['jobs'])
    assert not {j['id'] for j in jobs}&{j['id'] for j in q['jobs']}
    frozen(out/'queue_before.json',q);q['jobs']+=jobs;atomic(monitor/'queue.json',q)
    atomic(out/'queue_deployment.json',dict(status='appended',utc=datetime.now(timezone.utc).isoformat(),
        jobs=len(jobs),job_ids=[j['id'] for j in jobs],queue_sha256=digest(monitor/'queue.json'),
        protocol_sha256=digest(out/'protocol.json'),new_prediction_budget=816,world_model_updates=0))
    print('QUEUE_APPENDED',len(jobs),flush=True)


if __name__=='__main__':main()
