import sys, os, pickle, numpy as np, time
sys.path.insert(0, os.path.dirname(__file__))
from pc_flow import gray256, pc_shift
from PIL import Image
from _paths import OUT, DATA as root
names = sorted(d for d in os.listdir(root) if d.startswith("traj_"))
out = {}
t0 = time.time()
for i, t in enumerate(names):
    n = len(pickle.load(open(f"{root}/{t}/traj_data.pkl", "rb"))["yaw"])
    G = [gray256(Image.open(f"{root}/{t}/{k}.jpg")) for k in range(n)]
    out[t] = np.array([pc_shift(G[k], G[k + 1]) for k in range(n - 1)])
    if i % 100 == 0: print(i, time.time() - t0, flush=True)
pickle.dump(out, open(os.path.join(OUT, "pc_flow_all.pkl"), "wb"))
print("done", time.time() - t0)
