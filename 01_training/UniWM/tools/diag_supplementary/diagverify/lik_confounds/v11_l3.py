from lc_common import *
import re
R=nongoal(load("base"))
M=[r for r in R if r["dyaw_km1"] is not None and abs(r["dyaw_k"])>=0.05 and abs(r["dyaw_km1"])>=0.05 and np.sign(r["dyaw_k"])==np.sign(r["dyaw_km1"])]
yb=lambda r,c: re.search(r"<dyaw_(pos|neg)_bin_(\d+)>",text(r,c)).groups()
for c in ("prev2","prev1","next1"):
    rows=[r for r in M if c in r["scores"]]
    same=np.mean([yb(r,c)==yb(r,"recorded") for r in rows]); within1=np.mean([yb(r,c)[0]==yb(r,"recorded")[0] and abs(int(yb(r,c)[1])-int(yb(r,"recorded")[1]))<=1 for r in rows])
    print(f"mid-turn {c}: same dyaw bin token as recorded on {same:.2f}, within one 0.01-rad bin on {within1:.2f} (n={len(rows)})")
# informative-frame lag shares: frames where prev1 and recorded dyaw differ by >=0.05
for run in RUNS:
    RR=nongoal(load(run)); S=[r for r in RR if "prev1" in r["scores"] and "next1" in r["scores"] and "prev2" in r["scores"] and abs(act(r,"prev1")[2]-r["dyaw_k"])>=0.05]
    best=[min(("prev2","prev1","recorded","next1"),key=lambda c: nll(r,c)) for r in S]
    print(f"  {run:7s} frames with |dyaw[k-1]-dyaw[k]|>=0.05: n={len(S)}, first place prev2 {best.count('prev2')/len(S):.2f} prev1 {best.count('prev1')/len(S):.2f} rec {best.count('recorded')/len(S):.2f} next1 {best.count('next1')/len(S):.2f}")
