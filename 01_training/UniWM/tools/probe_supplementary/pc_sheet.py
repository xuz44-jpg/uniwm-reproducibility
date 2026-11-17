import numpy as np, json
from PIL import Image
def pc(a,b):
    h=a.shape[0]; a=a[h//4:3*h//4]; b=b[h//4:3*h//4]
    w=np.outer(np.hanning(a.shape[0]),np.hanning(a.shape[1]))
    A=np.fft.fft2((a-a.mean())*w); B=np.fft.fft2((b-b.mean())*w)
    R=np.conj(A)*B; R/=np.abs(R)+1e-9
    r=np.real(np.fft.ifft2(R)); iy,ix=np.unravel_index(np.argmax(r),r.shape)
    def sub(v,i,n):
        l,c,rr=v[(i-1)%n],v[i],v[(i+1)%n]; den=l-2*c+rr
        return i+(0.5*(l-rr)/den if den!=0 else 0)
    dx=sub(r[iy,:],ix,r.shape[1])
    if dx>r.shape[1]/2: dx-=r.shape[1]
    return dx*256/160
conds=['recorded','mirrored','zero','forward','left','right']
from _paths import FIGS
base=FIGS+'/'
for d in ['action_probe','action_probe_stopft','action_probe_de50','action_probe_de100','action_probe_de200','action_probe_de459']:
    im=np.asarray(Image.open(base+d+'/probe_sheet.jpg').convert('L'),float)
    rows=json.load(open(base+d+'/probe_rows.json')); turn=[r for r in rows if r['kind']=='turn'][:6]
    out=[]
    for i in range(im.shape[0]//160):
        tile=lambda j: im[i*160:(i+1)*160, j*160:(j+1)*160]
        cur=tile(0); v={c:pc(cur,tile(2+j)) for j,c in enumerate(conds)}; real=pc(cur,tile(1))
        r=turn[i]
        out.append(f"{r['traj'][-3:]}: real pc {real:+.2f}/far {r['real_hx']:+.2f}; L-R pc {v['left']-v['right']:+.2f} (L {v['left']:+.2f} R {v['right']:+.2f}) far(mean2) {r['left_hx']-r['right_hx']:+.2f}")
    print(d); print('   '+'\n   '.join(out))
