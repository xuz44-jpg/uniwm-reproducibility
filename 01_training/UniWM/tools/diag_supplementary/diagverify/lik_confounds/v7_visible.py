# visible horizontal image shift k->k+1 by phase correlation (numpy + PIL), on the 256x256 frames the model sees
from lc_common import *
from PIL import Image
import pickle
def img(t,k):
    return np.asarray(Image.open(f"{DATA}/{t}/{k}.jpg").convert("L").resize((256,256)),float)
def shift(a,b):
    w=np.outer(np.hanning(256),np.hanning(256)); A=np.fft.fft2((a-a.mean())*w); B=np.fft.fft2((b-b.mean())*w)
    R=A*np.conj(B); R/=np.abs(R)+1e-9; c=np.real(np.fft.ifft2(R)); c=np.fft.fftshift(c)
    i,j=np.unravel_index(np.argmax(c),c.shape)
    def sub(m1,m0,p1):
        d=m1-2*m0+p1; return 0.0 if d==0 else 0.5*(m1-p1)/d
    dj=sub(c[i,j-1],c[i,j],c[i,(j+1)%256]) if 0<j<255 else 0
    return (j+dj-128), (i-128)   # horizontal, vertical shift of a relative to b
R=nongoal(load("base"))
out={}
for r in R:
    t,k=r["traj"],r["k"]; out[(t,k)]=shift(img(t,k+1),img(t,k))[0]
pickle.dump(out,open(VIS_SHIFT,"wb"))
x=np.array([out[(r["traj"],r["k"])] for r in R]); y=np.array([r["dyaw_k"] for r in R]); y1=np.array([r["dyaw_km1"] if r["dyaw_km1"] is not None else np.nan for r in R])
def rank(v): return np.argsort(np.argsort(v))
m=~np.isnan(y1)
print(f"shift vs dyaw[k]: Pearson {np.corrcoef(x,y)[0,1]:+.3f}, Spearman {np.corrcoef(rank(x),rank(y))[0,1]:+.3f} (n={len(x)})")
print(f"shift vs dyaw[k-1]: Spearman {np.corrcoef(rank(x[m]),rank(y1[m]))[0,1]:+.3f} (n={m.sum()})")
T=np.abs(y)>=0.05; sgn=np.sign(np.polyfit(y,x,1)[0])
print(f"fitted slope {np.polyfit(y,x,1)[0]:.1f} px/rad; on turning frames median |shift| {np.median(np.abs(x[T])):.2f} px; sign agreement with dyaw[k] {np.mean(np.sign(x[T])==sgn*np.sign(y[T])):.2f} (n={T.sum()})")
S=np.abs(y)<0.02; print(f"straight frames median |shift| {np.median(np.abs(x[S])):.2f} px")
