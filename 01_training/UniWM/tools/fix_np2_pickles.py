"""Re-save traj_data.pkl files that were pickled with numpy 2 so that numpy 1.x can load them.
Usage: python tools/fix_np2_pickles.py data/1XHumanoid   (keeps a .bak of each file it rewrites)
"""
import glob, os, pickle, sys
import numpy as np
if not hasattr(np, "_core"):
    import numpy.core as _c
    sys.modules.setdefault("numpy._core", _c)
    for _s in ("multiarray", "numeric", "_multiarray_umath", "umath"):
        sys.modules.setdefault(f"numpy._core.{_s}", getattr(_c, _s, _c.multiarray))
root = sys.argv[1]
n = 0
for p in sorted(glob.glob(os.path.join(root, "*", "traj_data.pkl"))):
    with open(p, "rb") as f:
        raw = f.read()
    if b"numpy._core" not in raw:
        continue
    d = pickle.loads(raw)
    d = {k: (np.asarray(v).astype(np.float32) if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    os.replace(p, p + ".bak")
    with open(p, "wb") as f:
        pickle.dump(d, f, protocol=4)
    n += 1
print(f"rewrote {n} pickles under {root}")
