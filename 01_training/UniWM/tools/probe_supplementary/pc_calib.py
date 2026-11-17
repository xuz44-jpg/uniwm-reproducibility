import os

# Substitute: the archived phase-correlation estimator pc_flow.pc_shift (returns dx, dy; no peak value).
import sys, pickle, numpy as np
from PIL import Image
from pc_flow import gray256 as g, pc_shift
from _paths import DATA as root
def pc(a, b): dx, dy = pc_shift(a, b); return dx, dy, np.nan
names=sorted(d for d in os.listdir(root) if d.startswith('traj_'))[:60]
rows=[]
for t in names:
    d=np.asarray(pickle.load(open(f'{root}/{t}/traj_data.pkl','rb'))['delta'])
    for k in range(0,len(d),3):
        try:
            a=g(Image.open(f'{root}/{t}/{k}.jpg')); b=g(Image.open(f'{root}/{t}/{k+1}.jpg'))
        except FileNotFoundError: continue
        dx,dy,pk=pc(a,b); rows.append((t,k,dx,d[k,2],d[k-1,2] if k>0 else np.nan))
dx=np.array([r[2] for r in rows]); dy0=np.array([r[3] for r in rows]); dym=np.array([r[4] for r in rows])
def sp(a,b):
    m=~np.isnan(a)&~np.isnan(b); a,b=a[m],b[m]; ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b)); return np.corrcoef(ra,rb)[0,1]
print('pairs',len(dx),'Spearman pc vs dyaw[k]',round(sp(dx,dy0),3),'vs dyaw[k-1]',round(sp(dx,dym),3),'OLS slope vs dyaw[k]',round(np.polyfit(dy0,dx,1)[0],1))
for lo,hi in ((0,0.02),(0.02,0.05),(0.05,0.1),(0.1,0.2),(0.2,0.4)):
    m=(np.abs(dy0)>=lo)&(np.abs(dy0)<hi)
    if m.sum(): print(f' |dyaw[k]| [{lo},{hi}) n={m.sum()} median |pc dx| {np.median(np.abs(dx[m])):.2f} median |dx|/|dyaw| {np.median(np.abs(dx[m])/np.maximum(np.abs(dy0[m]),1e-3)):.0f} sign agr {np.mean(np.sign(dx[m])==np.sign(dy0[m])):.2f}')
