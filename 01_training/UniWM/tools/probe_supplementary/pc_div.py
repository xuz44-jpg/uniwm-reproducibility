import sys, os, pickle, numpy as np, time
from PIL import Image
from _paths import OUT, DATA as root
names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
def gray256(img): return np.asarray(img.convert("L").resize((256, 256), Image.BILINEAR), dtype=np.float64)
W = None
def pcx(A, B):
    global W
    if W is None or W.shape != A.shape: W = np.outer(np.hanning(A.shape[0]), np.hanning(A.shape[1]))
    A = (A - A.mean()) * W; B = (B - B.mean()) * W
    R = np.fft.fft2(B) * np.conj(np.fft.fft2(A)); R /= np.abs(R) + 1e-9; r = np.real(np.fft.ifft2(R))
    iy, ix = np.unravel_index(np.argmax(r), r.shape); n = r.shape[1]
    m, z, p = r[iy, (ix-1) % n], r[iy, ix], r[iy, (ix+1) % n]; den = m - 2*z + p
    v = ix + (0.5 * (m - p) / den if abs(den) > 1e-12 else 0.0)
    return v - n if v > n / 2 else v
out = {}
for t in names:
    n = len(pickle.load(open(f"{root}/{t}/traj_data.pkl", "rb"))["yaw"])
    G = [gray256(Image.open(f"{root}/{t}/{k}.jpg"))[64:192] for k in range(n)]
    out[t] = np.array([(pcx(G[k][:, :96], G[k+1][:, :96]), pcx(G[k][:, 160:], G[k+1][:, 160:])) for k in range(n - 1)])
pickle.dump(out, open(f"{OUT}/pc_div_all.pkl", "wb")); print("done")
