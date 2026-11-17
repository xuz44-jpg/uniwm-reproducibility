import io, contextlib, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from interv_verify import load, rows, acc, cboot, f, b
    from interv_verify2 import slope_ci
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    S = rows(load(run), key)
    C = [x for x in S if not x["stop"] and x["real"][0] != "stop" and x["gen"][0] != "stop"]
    print(run, "move steps with no Stop label and no greedy Stop (real or gen):", len(C))
    for tol in ((0.1,0.1,0.2),(0.05,0.05,0.1)):
        dv=[acc(x["real"][0],x["rec"],False,tol)-acc(x["gen"][0],x["rec"],False,tol) for x in C]
        bins=[]
        for bb in range(5):
            T=[(d,x) for d,x in zip(dv,C) if min(x["s"]//5,4)==bb]
            m,lo,hi,n,k=cboot([d for d,_ in T],[x["traj"] for _,x in T])
            bins.append(f"{m:+.2f}[{lo:+.2f},{hi:+.2f}]n{n}")
        s,lo,hi=slope_ci([x["s"] for x in C],dv,[x["traj"] for x in C])
        print(f"  tol {tol}: overall {f(cboot(dv,[x['traj'] for x in C]))}; bins {' '.join(bins)}; slope {s:+.4f} [{lo:+.4f},{hi:+.4f}]")
