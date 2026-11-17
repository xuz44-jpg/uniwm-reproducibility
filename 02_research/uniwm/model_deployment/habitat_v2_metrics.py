"""CPU-only direct-image metric and tie-handling primitives."""
import math
import cv2
import numpy as np


def pixel_metrics(a,b):
    x=a.astype(np.float64)/255.;y=b.astype(np.float64)/255.
    mse=float(np.square(x-y).mean());psnr=None if mse==0 else float(-10*math.log10(mse))
    mx=cv2.GaussianBlur(x,(11,11),1.5);my=cv2.GaussianBlur(y,(11,11),1.5)
    vx=cv2.GaussianBlur(x*x,(11,11),1.5)-mx*mx;vy=cv2.GaussianBlur(y*y,(11,11),1.5)-my*my
    cov=cv2.GaussianBlur(x*y,(11,11),1.5)-mx*my
    ssim=((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    assert np.isfinite(ssim).all()
    return dict(ssim=float(ssim[5:-5,5:-5].mean()),psnr_db=psnr,perfect_pixel_match=mse==0,mse=mse)


def rank_match(values,expected,tolerance):
    best=min(values);ids=[i for i,v in enumerate(values) if v-best<=tolerance]
    if len(ids)!=1:return dict(status='tie',closest_indices=ids,correct_is_tied=expected in ids)
    return dict(status='correct' if ids[0]==expected else 'other_reference',closest_indices=ids)


