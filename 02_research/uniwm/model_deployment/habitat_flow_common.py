"""CPU geometry and fixed-mask flow measurements for the Habitat probe."""
import numpy as np
import cv2
from habitat_study_common import scaled_K

CRITERIA=dict(resolution=256,raft_updates=12,roi_margin=26,texture_sobel_l1=.08,
    min_texture_pixels=400,min_valid_pixels=200,min_reference_coverage=.5,min_generated_coverage=.2,
    forward_backward_absolute_px=1.,forward_backward_relative=.05,near_zero_px=1.,direction_consensus=.7,
    real_geometry_median_epe_max_px=1.,real_geometry_p90_epe_max_px=2.,
    zero_median_magnitude_max_px=.5,shift_median_epe_absolute_px=.5,shift_median_epe_relative=.1)


def geometry_flow(camera,first,second):
    """Exact pinhole pure-rotation displacement, GL camera axes -> CV axes."""
    assert np.linalg.norm(np.asarray(first['camera_position_world'])-second['camera_position_world'])<1e-5
    K=scaled_K(camera['K'],camera['width'],camera['height'])
    D=np.diag([1.,-1.,-1.])
    R=D@np.asarray(second['camera_rotation_c2w']).T@np.asarray(first['camera_rotation_c2w'])@D
    assert np.allclose(R.T@R,np.eye(3),atol=1e-6,rtol=0)
    H=K@R@np.linalg.inv(K);yy,xx=np.mgrid[:256,:256]
    p=np.stack([xx,yy,np.ones_like(xx)],-1);q=p@H.T;z=q[...,2]
    q=q[...,:2]/z[...,None];flow=q-np.stack([xx,yy],-1)
    visible=(z>0)&(q[...,0]>=1)&(q[...,0]<255)&(q[...,1]>=1)&(q[...,1]<255)
    return flow.astype(np.float32),visible


def texture_mask(image):
    # Same RGB convention as RAFT: mean channels in [-1,1]. Fixed source only.
    gray=image.astype(np.float32).mean(-1)/127.5-1
    k=np.array([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]],np.float32)
    gx=cv2.filter2D(gray,-1,k,borderType=cv2.BORDER_CONSTANT)
    gy=cv2.filter2D(gray,-1,k.T,borderType=cv2.BORDER_CONSTANT)
    texture=cv2.dilate(((np.abs(gx)+np.abs(gy))>CRITERIA['texture_sobel_l1']).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    dark=cv2.dilate((gray<-.85).astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
    roi=np.zeros((256,256),bool);m=CRITERIA['roi_margin'];roi[m:-m,m:-m]=True
    return roi&texture&~dark


def vector_stats(flow,mask):
    vals=np.asarray(flow)[mask];n=len(vals)
    if not n:return dict(pixels=0,median_x=None,median_y=None,mean_magnitude=None,median_magnitude=None)
    mag=np.linalg.norm(vals,axis=-1)
    return dict(pixels=n,median_x=float(np.median(vals[:,0])),median_y=float(np.median(vals[:,1])),
        mean_magnitude=float(mag.mean()),median_magnitude=float(np.median(mag)))


def compare_flow(pred,reference,mask):
    n=int(mask.sum())
    if not n:return dict(pixels=0,mean_epe_px=None,median_epe_px=None,p90_epe_px=None,gain_projection=None,direction_consensus=None)
    x=pred[mask].astype(np.float64);y=reference[mask].astype(np.float64);error=np.linalg.norm(x-y,axis=-1)
    denom=float(np.square(y).sum());dot=np.sum(x*y,axis=-1)
    return dict(pixels=n,mean_epe_px=float(error.mean()),median_epe_px=float(np.median(error)),p90_epe_px=float(np.percentile(error,90)),
        gain_projection=float(dot.sum()/denom) if denom>1e-10 else None,
        direction_consensus=float((dot>0).mean()) if denom>1e-10 else None,
        predicted=vector_stats(pred,mask),reference=vector_stats(reference,mask))


def direction_label(pred,reference,mask,generated_valid,reference_gate):
    """Unknown on low measurement coverage; never select primary EPE pixels here."""
    base=int(mask.sum());valid=mask&generated_valid;n=int(valid.sum());coverage=n/base if base else 0.
    result=dict(label='unknown',coverage=coverage,valid_pixels=n,reference_pixels=base)
    if not reference_gate:return dict(result,reason='reference_calibration_failed')
    if base<CRITERIA['min_texture_pixels'] or n<CRITERIA['min_valid_pixels'] or coverage<CRITERIA['min_generated_coverage']:
        return dict(result,reason='insufficient_generated_flow_coverage')
    stats=vector_stats(pred,valid);metrics=compare_flow(pred,reference,valid);result.update(motion=stats,response=metrics)
    if stats['median_magnitude']<=CRITERIA['near_zero_px']:return dict(result,label='near_zero',reason='below_fixed_pixel_resolution')
    if metrics['gain_projection'] is None:return dict(result,reason='zero_reference_motion')
    consensus=metrics['direction_consensus']
    if consensus>=CRITERIA['direction_consensus']:return dict(result,label='correct',reason='motion_along_reference_flow')
    if 1-consensus>=CRITERIA['direction_consensus']:return dict(result,label='reverse',reason='motion_opposite_reference_flow')
    return dict(result,reason='mixed_motion_directions')


def direction_counts(rows):
    result={k:sum(row['label']==k for row in rows) for k in ['correct','reverse','near_zero','unknown']}
    result['total']=len(rows);assert sum(result[k] for k in ['correct','reverse','near_zero','unknown'])==len(rows)
    return result
