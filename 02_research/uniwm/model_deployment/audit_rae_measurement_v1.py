"""RAE measurement diagnosis on existing images only; frozen results stay intact."""
import argparse
from collections import Counter,defaultdict
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from suite_common import read,digest,atomic,frozen
from habitat_study_common import scaled_K
from habitat_flow_common import texture_mask,compare_flow,direction_label,direction_counts
from habitat_v2_metrics import pixel_metrics

ANGLES=[0,3,-3,5,-5,7,-7]
SIFT_RULES=dict(ratio=.75,min_matches=12,min_inliers=12,min_inlier_fraction=.5,
    ransac_threshold_px=2.,minimum_grid_cells=4,rotation_projection_median_px=2.)


def flow_from_H(H):
    yy,xx=np.mgrid[:256,:256];p=np.stack([xx,yy,np.ones_like(xx)],-1);q=p@H.T
    z=q[...,2];v=q[...,:2]/z[...,None];flow=v-np.stack([xx,yy],-1)
    valid=(z>0)&(v[...,0]>=1)&(v[...,0]<255)&(v[...,1]>=1)&(v[...,1]<255)
    return flow.astype(np.float32),valid


def H_from_views(camera,first,second):
    K=scaled_K(camera['K'],camera['width'],camera['height']);D=np.diag([1.,-1.,-1.])
    R=D@np.asarray(second['camera_rotation_c2w']).T@np.asarray(first['camera_rotation_c2w'])@D
    H=K@R@np.linalg.inv(K);return H/H[2,2]


def avg(values):
    x=[float(v) for v in values if v is not None];return float(np.mean(x)) if x else None


def med(values):
    x=[float(v) for v in values if v is not None];return float(np.median(x)) if x else None


class Images:
    def __init__(self):self.arrays={};self.hashes={};self.features={};self.sift=cv2.SIFT_create(nfeatures=1500)
    def sha(self,p):
        p=str(p)
        if p not in self.hashes:self.hashes[p]=digest(p)
        return self.hashes[p]
    def image(self,p):
        p=str(p)
        if p not in self.arrays:
            with Image.open(p) as im:self.arrays[p]=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC),copy=True)
        return self.arrays[p]
    def features_of(self,p):
        sha=self.sha(p)
        if sha not in self.features:
            kp,des=self.sift.detectAndCompute(cv2.cvtColor(self.image(p),cv2.COLOR_RGB2GRAY),None)
            points=np.asarray([k.pt for k in kp],np.float32).reshape(-1,2)
            self.features[sha]=(points,des)
        return self.features[sha]


def match(images,first,second,mask,H_expected,K):
    p,d1=images.features_of(first);q,d2=images.features_of(second)
    if d1 is None or d2 is None:return dict(valid=False,reason='no_descriptors',source_keypoints=len(p),target_keypoints=len(q))
    xy=np.round(p).astype(int).clip(0,255);take=mask[xy[:,1],xy[:,0]];p=p[take];d1=d1[take]
    base=dict(valid=False,source_keypoints=len(p),target_keypoints=len(q))
    if len(p)<2 or len(q)<2:return dict(base,reason='insufficient_keypoints')
    matcher=cv2.BFMatcher(cv2.NORM_L2);forward=matcher.knnMatch(d1,d2,k=2);back=matcher.knnMatch(d2,d1,k=2)
    reverse={a.queryIdx:a.trainIdx for row in back if len(row)==2 for a,b in [row] if a.distance<.75*b.distance}
    accepted=[a for row in forward if len(row)==2 for a,b in [row] if a.distance<.75*b.distance and reverse.get(a.trainIdx)==a.queryIdx]
    base['matches']=len(accepted)
    if len(accepted)<12:return dict(base,reason='insufficient_mutual_ratio_matches')
    x=np.asarray([p[a.queryIdx] for a in accepted]);y=np.asarray([q[a.trainIdx] for a in accepted])
    cv2.setRNGSeed(20261027);H,inliers=cv2.findHomography(x,y,cv2.RANSAC,2.,maxIters=5000,confidence=.999)
    if H is None or not np.isfinite(H).all():return dict(base,reason='homography_fit_failed')
    good=inliers.ravel().astype(bool);n=int(good.sum());fraction=float(good.mean())
    cells=len({(min(3,int(a//64)),min(3,int(b//64))) for a,b in x[good]})
    projected=cv2.perspectiveTransform(x[None],H)[0];residual=np.linalg.norm(projected-y,axis=-1)
    expected=cv2.perspectiveTransform(x[None],H_expected)[0];expected_error=np.linalg.norm(expected-y,axis=-1)
    H=H/H[2,2];estimated,visible=flow_from_H(H);truth,truth_visible=flow_from_H(H_expected)
    cmp=compare_flow(estimated,truth,mask&visible&truth_visible)
    M=np.linalg.inv(K)@H@K;det=float(np.linalg.det(M))
    rotation_error=None
    if abs(det)>1e-10:
        M=M/np.cbrt(det);U,_,V=np.linalg.svd(M);R=U@V
        if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@V
        RH=K@R@np.linalg.inv(K);rf,rv=flow_from_H(RH)
        rotation_error=compare_flow(estimated,rf,mask&visible&rv)['median_epe_px']
    valid=n>=12 and fraction>=.5 and cells>=4 and float(np.median(residual[good]))<=2.
    return dict(base,valid=bool(valid),reason='passed' if valid else 'insufficient_geometric_support',inliers=n,
        inlier_fraction=fraction,grid_cells=cells,reprojection_median_px=float(np.median(residual[good])) if n else None,
        expected_inlier_residual_median_px=float(np.median(expected_error[good])) if n else None,
        expected_flow_comparison=cmp,rotation_projection_median_px=rotation_error,
        rotation_compatible=bool(valid and rotation_error is not None and rotation_error<=2.),homography=H.tolist())


def load_cache(images,cache,source,target):
    key=hashlib.sha256((images.sha(source)+images.sha(target)).encode()).hexdigest();path=cache/(key+'.npz')
    if not path.exists():return None
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def make_manifest(study,out,images):
    root=study/'v2/paper_confirmation_v1';refs=read(root/'references_manifest.json');cases=read(root/'confirmation_manifest.json')['cases']
    gen=root/'confirmation/raenwm';done=read(gen/'complete.json');assert done['successful']==252 and done['parse_failed']==0
    for c in cases:
        p=gen/(c['key']+'.json');assert digest(p)==done['record_hashes'][c['key']]
        rec=read(p);assert images.sha(p.with_suffix('.png'))==rec['image_sha256']
    codecs=read(root/'codec_controls/raenwm/complete.json');assert codecs['count']==84
    cm={(c['state_id'],c['angle_deg']):c['output_path'] for c in codecs['rows']}
    for c in codecs['rows']:
        assert images.sha(c['output_path'])==c['image_sha256'] and images.sha(c['source_path'])==c['source_sha256']
    by={(c['state_id'],c['repeat'],c['target_yaw_deg']):str(gen/(c['key']+'.png')) for c in cases}
    pairs=[]
    def add(kind,sid,scene,angle,rep,first,second,H,known=False,gpu=True):
        pairs.append(dict(key=f'{kind}/{sid}/r{rep}/yaw_{angle:+d}',kind=kind,state_id=sid,scene_id=scene,
            angle=angle,repeat=rep,source=str(first),target=str(second),H=H.tolist(),known_synthetic=known,gpu=gpu))
    for sid,entry in refs['references'].items():
        scene=entry['scene_id'];views=entry['reference_views'];raw0=views['0']['path'];codec0=cm[(sid,0)]
        for angle in ANGLES:
            raw=views[str(angle)]['path'];codec=cm[(sid,angle)];H=H_from_views(refs['camera'],views['0'],views[str(angle)])
            add('raw_reference',sid,scene,angle,0,raw0,raw,H)
            add('mixed_codec',sid,scene,angle,0,raw0,codec,H)
            add('dual_codec',sid,scene,angle,0,codec0,codec,H)
            add('target_codec',sid,scene,angle,0,raw,codec,np.eye(3),gpu=False)
            for rep in range(3):
                pred=by[(sid,rep,angle)];zero=by[(sid,rep,0)]
                add('raw_generated',sid,scene,angle,rep,raw0,pred,H)
                add('codec_generated',sid,scene,angle,rep,codec0,pred,H)
                add('target_generated',sid,scene,angle,rep,raw,pred,np.eye(3),gpu=False)
                if angle:add('generated_action',sid,scene,angle,rep,zero,pred,H)
            add('same_action_seed',sid,scene,angle,0,by[(sid,0,angle)],by[(sid,1,angle)],np.eye(3))
        for domain,source in [('raw',raw0),('codec',codec0),('generated',by[(sid,0,0)])]:
            controls=[('zero',np.eye(3))]+[(f'shift{v:+d}',np.array([[1.,0.,v],[0.,1.,0.],[0.,0.,1.]])) for v in [-8,8]]
            controls += [(f'yaw{v:+d}',H_from_views(refs['camera'],views['0'],views[str(v)])) for v in [3,-3,5,-5,7,-7]]
            for ci,(name,H) in enumerate(controls):
                target=out/'synthetic'/domain/sid/(name+'.png');target.parent.mkdir(parents=True,exist_ok=True)
                array=cv2.warpPerspective(images.image(source),H,(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                if target.exists():assert np.array_equal(images.image(target),array)
                else:Image.fromarray(array).save(target)
                add('synthetic_'+domain,sid,scene,ci,0,source,target,H,known=True)
                pairs[-1]['control']=name
    assert len(pairs)==1716 and sum(p['gpu'] for p in pairs)==1380
    assert len({p['key'] for p in pairs})==len(pairs)
    for p in pairs:
        images.sha(p['source']);images.sha(p['target'])
    result=dict(pairs=pairs,references_sha256=digest(root/'references_manifest.json'),generation_complete_sha256=digest(gen/'complete.json'),
        codec_complete_sha256=digest(root/'codec_controls/raenwm/complete.json'),source_hashes=dict(images.hashes),new_world_model_predictions=0,
        new_training_updates=0,new_evaluation_positions=0,derived_synthetic_images=324)
    frozen(out/'manifest.json',result);return result


def cached(study,out):
    images=Images();root=study/'v2/paper_confirmation_v1'
    protocol=dict(script_sha256=digest(__file__),main_protocol_sha256=digest(root/'protocol.json'),
        data='existing12 positions,84 true references,84 frozen RAE reconstructions,252 generated frames',
        source_results_unchanged=True,original_unknown=195,original_nonzero=216,
        primary='supplementary diagnosis only; never replace frozen direction results',
        new_world_model_predictions=0,new_training_updates=0,new_positions=0,
        raft='unchanged original RAFT Large,12updates,FP32,original source-only masks and thresholds',
        sift=SIFT_RULES,known_controls='36 fixed source images:12raw0,12codec0,12generated-zero repeat0; each0/±8px shifts/±3/±5/±7deg homographies',
        interpretation='synthetic control truth is known; expected camera geometry on generated pairs is a hypothesis, not generated physical truth')
    frozen(out/'protocol.json',protocol);manifest=make_manifest(study,out,images);refs=read(root/'references_manifest.json')
    K=scaled_K(refs['camera']['K'],refs['camera']['width'],refs['camera']['height']);masks={}
    for sid in refs['references']:
        with np.load(root/'flow/references'/(sid+'.npz'),allow_pickle=False) as f:masks[sid]=f['common_mask'].copy()
    rows=[]
    for pair in manifest['pairs']:
        path=out/'cpu_rows'/(pair['key']+'.json')
        if path.exists():rows.append(read(path));continue
        H=np.asarray(pair['H']);expected,visible=flow_from_H(H)
        mask=(texture_mask(images.image(pair['source']))&visible) if pair['known_synthetic'] else masks[pair['state_id']]
        sift=match(images,pair['source'],pair['target'],mask,H,K)
        if pair['known_synthetic']:
            cmp=sift.get('expected_flow_comparison',{})
            sift['known_control_passed']=bool(sift['valid'] and cmp.get('median_epe_px') is not None and
                cmp['median_epe_px']<=1. and cmp['p90_epe_px']<=2.)
        x=images.image(pair['source']);y=images.image(pair['target'])
        gray=cv2.cvtColor(y,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        row=dict(pair=pair,sift=sift,pixel=pixel_metrics(x,y),target_laplacian_variance=float(cv2.Laplacian(gray,cv2.CV_32F).var()),
            target_gradient_mean=float((np.abs(cv2.Sobel(gray,cv2.CV_32F,1,0))+np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1))).mean()))
        flow=load_cache(images,root/'flow/cache',pair['source'],pair['target'])
        if flow is not None:row['cached_flow']=dict(coverage=float((flow['valid']&mask).sum()/max(1,mask.sum())),comparison=compare_flow(flow['flow'],expected,mask))
        atomic(path,row);rows.append(row)
        atomic(out/'cpu_progress.json',dict(scored=len(rows),total=len(manifest['pairs']),last=pair['key']))
    atomic(out/'cpu_rows.json',rows)
    for sid,entry in refs['references'].items():
        sheet=Image.new('RGB',(1792,834),'white');draw=ImageDraw.Draw(sheet)
        by={(p['kind'],p['state_id'],p['angle'],p['repeat']):p for p in manifest['pairs'] if not p['known_synthetic']}
        for i,angle in enumerate(ANGLES):
            for j,kind in enumerate(['raw_reference','mixed_codec','raw_generated']):
                p=by[(kind,sid,angle,0)];sheet.paste(Image.fromarray(images.image(p['target'])),(i*256,j*278+22))
                draw.text((i*256+3,j*278+3),f'{kind} {angle:+d} deg',fill='black')
        folder=out/'contacts';folder.mkdir(exist_ok=True);sheet.save(folder/(sid+'_repeat0.png'))
    atomic(out/'cpu_complete.json',dict(status='passed',records=len(rows),rows_sha256=digest(out/'cpu_rows.json'),
        manifest_sha256=digest(out/'manifest.json'),new_world_model_predictions=0,new_training_updates=0))


def gpu(study,out):
    root=study/'v2/paper_confirmation_v1';assert read(out/'cpu_complete.json')['status']=='passed'
    assert read(out/'protocol.json')['script_sha256']==digest(__file__)
    assert read(root/'flow/references/summary.json')['all_reference_gates_passed']
    import torch
    from probe_habitat_flow_v1 import FlowStore
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    folder=out/'raft';folder.mkdir(exist_ok=True);cache=folder/'cache'
    if not cache.exists():cache.symlink_to(root/'flow/cache',target_is_directory=True)
    assert cache.resolve()==(root/'flow/cache').resolve()
    engine=FlowStore(folder);manifest=read(out/'manifest.json');refs=read(root/'references_manifest.json');masks={}
    for p,sha in manifest['source_hashes'].items():assert digest(p)==sha
    for sid in refs['references']:
        with np.load(root/'flow/references'/(sid+'.npz'),allow_pickle=False) as f:masks[sid]=f['common_mask'].copy()
    pairs=[p for p in manifest['pairs'] if p['gpu']];rows=[]
    for pair in pairs:
        path=out/'flow_rows'/(pair['key']+'.json')
        if path.exists():rows.append(read(path));continue
        H=np.asarray(pair['H']);expected,visible=flow_from_H(H)
        mask=(texture_mask(engine.image(pair['source']))&visible) if pair['known_synthetic'] else masks[pair['state_id']]
        flow=engine.get(pair['source'],pair['target']);coverage=float((flow['valid']&mask).sum()/max(1,mask.sum()))
        cmp=compare_flow(flow['flow'],expected,mask)
        gate=bool(int(mask.sum())>=400 and coverage>=.5 and cmp['median_epe_px']<=1. and cmp['p90_epe_px']<=2.)
        if pair['known_synthetic'] and pair.get('control','').startswith('shift'):gate=gate and cmp['median_epe_px']<=.8
        row=dict(pair=pair,coverage=coverage,reference_mask_pixels=int(mask.sum()),comparison=cmp,
            known_control_passed=gate if pair['known_synthetic'] else None)
        atomic(path,row);rows.append(row);atomic(out/'gpu_progress.json',dict(scored=len(rows),total=len(pairs),last=pair['key']))
        print('RAE_AUDIT_FLOW',len(rows),len(pairs),pair['key'],flush=True)
    # Compare measurement anchors without changing the original primary score.
    by={(p['kind'],p['state_id'],p['angle'],p['repeat']):p for p in pairs if not p['known_synthetic']}
    actions=[]
    for sid in refs['references']:
        with np.load(root/'flow/references'/(sid+'.npz'),allow_pickle=False) as f:truth={k:f[k].copy() for k in f.files}
        mask=truth['common_mask']
        for kind,repeats in [('mixed_codec',range(1)),('dual_codec',range(1)),('raw_generated',range(3)),('codec_generated',range(3))]:
            for rep in repeats:
                z=by[(kind,sid,0,rep)];f0=engine.get(z['source'],z['target'])
                for angle in ANGLES[1:]:
                    p=by[(kind,sid,angle,rep)];fa=engine.get(p['source'],p['target']);delta=fa['flow']-f0['flow'];reference=truth['flow_'+str(angle)]-truth['flow_0']
                    ca=float((fa['valid']&mask).sum()/mask.sum());cz=float((f0['valid']&mask).sum()/mask.sum());intersection=float((fa['valid']&f0['valid']&mask).sum()/mask.sum())
                    actions.append(dict(kind=kind,state_id=sid,scene_id=p['scene_id'],angle=angle,repeat=rep,
                        action_coverage=ca,zero_coverage=cz,intersection_coverage=intersection,
                        both_individual_coverages_pass_but_intersection_fails=bool(ca>=.2 and cz>=.2 and intersection<.2),
                        direction=direction_label(delta,reference,mask,fa['valid']&f0['valid'],True),comparison=compare_flow(delta,reference,mask)))
        for rep in range(3):
            for angle in ANGLES[1:]:
                p=by[('generated_action',sid,angle,rep)];f=engine.get(p['source'],p['target']);reference=truth['flow_'+str(angle)]-truth['flow_0']
                actions.append(dict(kind='generated_action',state_id=sid,scene_id=p['scene_id'],angle=angle,repeat=rep,
                    direction=direction_label(f['flow'],reference,mask,f['valid'],True),comparison=compare_flow(f['flow'],reference,mask)))
    atomic(out/'flow_rows.json',rows);atomic(out/'anchor_actions.json',actions)
    atomic(out/'gpu_complete.json',dict(status='passed',records=len(rows),rows_sha256=digest(out/'flow_rows.json'),
        anchor_actions_sha256=digest(out/'anchor_actions.json'),new_world_model_predictions=0,new_training_updates=0))


def report(study,out):
    assert read(out/'cpu_complete.json')['status']=='passed' and read(out/'gpu_complete.json')['status']=='passed'
    cpu=read(out/'cpu_rows.json');rows=read(out/'flow_rows.json');actions=read(out/'anchor_actions.json');result=dict(by_kind={},anchor_directions={},by_scene={},independent_sift_gates={})
    for kind in sorted({r['pair']['kind'] for r in cpu}):
        cs=[r for r in cpu if r['pair']['kind']==kind];fs=[r for r in rows if r['pair']['kind']==kind]
        result['by_kind'][kind]=dict(total=len(cs),sift_valid=sum(r['sift']['valid'] for r in cs),
            sift_rotation_compatible=sum(r['sift'].get('rotation_compatible',False) for r in cs),
            sift_known_control_passed=sum(r['sift'].get('known_control_passed',False) for r in cs),
            sift_median_expected_residual_px=med([r['sift'].get('expected_inlier_residual_median_px') for r in cs]),
            flow_scored=len(fs),flow_coverage_median=float(np.median([r['coverage'] for r in fs])) if fs else None,
            flow_known_control_passed=sum(r.get('known_control_passed',False) is True for r in fs),
            laplacian_variance_mean=avg([r['target_laplacian_variance'] for r in cs]))
    for kind in sorted({a['kind'] for a in actions}):
        xs=[a for a in actions if a['kind']==kind]
        result['anchor_directions'][kind]=dict(direction=direction_counts([a['direction'] for a in xs]),
            intersection_only_failures=sum(a.get('both_individual_coverages_pass_but_intersection_fails',False) for a in xs),
            median_action_coverage=med([a.get('action_coverage') for a in xs]),median_zero_coverage=med([a.get('zero_coverage') for a in xs]),
            median_intersection_coverage=med([a.get('intersection_coverage') for a in xs]))
    for sid in sorted({r['pair']['state_id'] for r in cpu}):
        cs=[r for r in cpu if r['pair']['state_id']==sid and r['pair']['kind']=='raw_reference']
        passed=all(r['sift']['valid'] and r['sift']['expected_flow_comparison']['median_epe_px']<=1. and r['sift']['expected_flow_comparison']['p90_epe_px']<=2. for r in cs)
        result['independent_sift_gates'][sid]=dict(passed=passed,total=len(cs),fit_valid=sum(r['sift']['valid'] for r in cs))
    for scene in sorted({a['scene_id'] for a in actions}):
        result['by_scene'][scene]={kind:direction_counts([a['direction'] for a in actions if a['scene_id']==scene and a['kind']==kind]) for kind in sorted({a['kind'] for a in actions})}
    result['limitations']=['supplementary exploratory diagnosis; main frozen scores unchanged','unknown is not model failure',
        'known2D homographies validate measurement only, not the physical correctness of generated actions',
        'SIFT cannot independently confirm a state if its real-reference gate fails','copying or sharper imagery is not evidence of correct rotation']
    atomic(out/'summary.json',result)
    lines=['# RAE 图像与光流测量排查','', '全部检查使用已有12位置、84真实参考、84冻结codec重建和252预测。无新增世界模型生成或训练。',
        '', '| 对照 | 同向/反向/近零/未知 |','|---|---|']
    for kind,b in result['anchor_directions'].items():lines.append('| '+kind+' | '+'/'.join(str(b['direction'][k]) for k in ['correct','reverse','near_zero','unknown'])+' |')
    lines+=['','## 已知变换验收','', '| 图像域 | RAFT通过/总数 | SIFT通过/总数 | SIFT拟合有效/总数 |','|---|---|---|---|']
    for kind in ['synthetic_raw','synthetic_codec','synthetic_generated']:
        b=result['by_kind'][kind];lines.append(f'| {kind} | {b["flow_known_control_passed"]}/{b["total"]} | {b["sift_known_control_passed"]}/{b["total"]} | {b["sift_valid"]}/{b["total"]} |')
    lines+=['','SIFT只在真实参考门槛通过的位置作为独立复核；合成控制并非新增真实动作真值。原始冻结方向分母和阈值没有修改。','']
    (out/'RESULTS.md').write_text('\n'.join(lines));atomic(out/'complete.json',dict(status='passed',summary_sha256=digest(out/'summary.json'),
        cpu_rows_sha256=digest(out/'cpu_rows.json'),flow_rows_sha256=digest(out/'flow_rows.json'),new_world_model_predictions=0,new_training_updates=0))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--phase',choices=['cached','gpu','report'],required=True);a=ap.parse_args()
    cv2.setNumThreads(1);out=a.study/'v2/paper_confirmation_v1/rae_measurement_audit_v1';out.mkdir(exist_ok=True)
    if a.phase=='cached':cached(a.study,out)
    elif a.phase=='gpu':gpu(a.study,out)
    else:report(a.study,out)


if __name__=='__main__':main()
