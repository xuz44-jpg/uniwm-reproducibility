"""Reuse the verified symmetric images; audit native continuous action encodings."""
import argparse
import ast
from collections import Counter
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
from suite_common import atomic,digest,frozen,read
from recon_symmetric_v3_common import experiment


def function(path,name,namespace,class_name=None):
    tree=ast.parse(Path(path).read_text())
    body=next(x.body for x in tree.body if isinstance(x,ast.ClassDef) and x.name==class_name) if class_name else tree.body
    node=next(x for x in body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    return namespace[name]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    source=a.study/'v2/uniwm_symmetric_aligned_v3';out=experiment(a.study);out.mkdir(exist_ok=True)
    protocol=read(source/'protocol.json');frames={t['trajectory_id']:t['frames'] for t in read(a.study/'collection_v1/dataset.json')['trajectories']}
    frames.update({t['frames'][0]['trajectory_id']:t['frames'] for t in protocol['collection']})
    configs={m:yaml.safe_load((a.root/'repos'/m/'config/data_config.yaml').read_text()) for m in ['nwm','raenwm']};functions={};sources={}
    for m,cfg in configs.items():
        repo=a.root/'repos'/m;ns=dict(np=np)
        for name in ['yaw_rotmat','to_local_coords','angle_difference','normalize_data']:function(repo/'misc.py',name,ns)
        compute=function(repo/'datasets.py','_compute_actions',ns,'BaseDataset')
        obj=SimpleNamespace(dataset_name='recon',len_traj_pred=1,normalize=True,data_config=cfg['recon'])
        functions[m]=(compute,obj,ns['normalize_data'])
        for p in [repo/'misc.py',repo/'datasets.py',repo/'config/data_config.yaml']:sources[str(p)]=digest(p)
    hashes={};reports={}
    for split,count in [('train',336),('validation',42)]:
        rows=read(source/(split+'.json'));assert len(rows)==count;encoded=[];maxerr=np.zeros(3);counts=Counter();seen={}
        for row in rows:
            assert row['split']==split and row['room_id']!=9
            fs=frames[row['trajectory_id']];i=row['current_index'];f,g=fs[i:i+2]
            assert row['target_index']==i+1 and len(row['history_paths'])==4
            assert [row['current_pose'],row['target_pose']]==[f['planar_pose'],g['planar_pose']]
            R,S=np.array(f['camera_rotation_c2w']),np.array(g['camera_rotation_c2w']);local=R.T@(np.array(g['camera_position_world'])-f['camera_position_world']);rel=R.T@S
            independent=np.array([-local[2],-local[0],math.atan2(rel[0,2],rel[0,0])]);act=np.array(row['action']);error=np.abs(independent-act);maxerr=np.maximum(maxerr,error);assert error.max()<1e-6
            traj=dict(position=np.array([x['planar_pose'][:2] for x in fs]),yaw=np.array([x['planar_pose'][2] for x in fs]))
            entry=dict(row);entry['encoded_actions']={}
            for m,cfg in configs.items():
                xy=act[:2]/float(cfg['recon']['metric_waypoint_spacing']);lo=np.array(cfg['action_stats']['min']);hi=np.array(cfg['action_stats']['max'])
                normalized=2*(xy-lo)/(hi-lo)-1;value=np.r_[normalized,act[2]]
                compute,obj,normalize=functions[m];_,native=compute(obj,traj,i,np.array([i+1]));native[:,:2]=normalize(native[:,:2],{k:np.array(v)[None] for k,v in cfg['action_stats'].items()})
                assert np.allclose(value,native[0],atol=1e-10,rtol=0)
                entry['encoded_actions'][m]=value.tolist()
            # Recorded histories are observed frames only, with first-frame padding.
            paths=[fs[max(0,j)].get('image_path',str(a.study/'collection_v1/trajectories'/row['trajectory_id']/f'{max(0,j):03d}.png')) for j in range(i-3,i+1)]
            assert paths==row['history_paths'],row['sample_id']
            assert row['current_image']==paths[-1]
            for p in set(paths+[row['target_image']]):
                assert Path(p).exists();hashes[p]=digest(p)
            for p,fm in [(row['current_image'],f),(row['target_image'],g)]:assert hashes[p]==fm['image_sha256']
            pure=np.linalg.norm(act[:2])<1e-5
            if pure and abs(act[2])>.01:counts['left' if act[2]>0 else 'right']+=1
            if pure and abs(act[2])>.01 and abs((row['current_pose'][2]-row['start_pose'][2]+math.pi)%(2*math.pi)-math.pi)<1e-9:
                group=row.get('source_trajectory_id',row['trajectory_id']);seen.setdefault(group,[]).append(round(math.degrees(act[2])))
            encoded.append(entry)
        assert all(sorted(v)==[-7,-5,-3,3,5,7] for v in seen.values())
        assert len(seen)==(16 if split=='train' else 2)
        atomic(out/(split+'.json'),encoded);reports[split]=dict(count=count,pure_yaw_counts=dict(counts),six_action_start_groups=len(seen),camera_max_abs_error=maxerr.tolist(),native_encoding_mismatches=0)
    frozen(out/'protocol.json',dict(version='recon_symmetric_v3',script_sha256=digest(__file__),source_protocol_sha256=digest(source/'protocol.json'),
        source_rows_sha256={s:digest(source/(s+'.json')) for s in ['train','validation']},native_sources_sha256=sources,
        train_pairs=336,validation_pairs=42,models=['raenwm','nwm'],initialization='fresh released EMA checkpoints; not earlier Habitat weights',
        train_images='same symmetric corpus as completed UniWM; no new rendering or starts',history='four observed frames, early history repeats first; +1 target only in loss',
        validation='all42 rows x 2 fixed noise/time seeds; restore training RNG after evaluation; no test-based selection',
        selection='actual minimum validation mean including step0; min300, significant relative improvement .005, patience3, max2000 budget',
        schedule='lr1e-5, warmup100 then constant',optimizer='AdamW .9/.999 weight_decay0 foreachFalse; clip1; batch1; bf16 autocast',
        evaluation_manifest_sha256=digest(a.study/'v2/eval_manifest.json'),predictions_per_model=72,original_results_preserved=True,
        uniwm_comparator='reuse completed symmetric validation-best900; identical source336/42',limitations=['one house and two test positions','combined data and selection/schedule change; no isolated causal attribution','flow proxy is not angle accuracy','unknown is not failure']))
    atomic(out/'input_image_hashes.json',hashes);atomic(out/'data_complete.json',dict(status='passed',splits=reports,images=len(hashes),optimizer_updates=0,new_renders=0,
        rows_sha256={s:digest(out/(s+'.json')) for s in ['train','validation']},protocol_sha256=digest(out/'protocol.json')))
    flow=out/'flow_probe_v1';flow.mkdir(exist_ok=True);old=a.study/'v2/flow_probe_v1'
    for name in ['references','cache']:
        p=flow/name
        if not p.exists():p.symlink_to(old/name,target_is_directory=True)
        assert p.resolve()==(old/name).resolve()
    print('SYMMETRIC_DATA_READY',reports,flush=True)


if __name__=='__main__':main()
