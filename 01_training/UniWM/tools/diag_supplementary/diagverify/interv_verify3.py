import io, contextlib, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from interv_verify import load, rows, acc, cboot, f, b, TURN
ok = lambda p, r: float(p not in (None, "stop") and np.sign(p[2]) == np.sign(r[2]) and abs(p[2]) >= TURN / 2)
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    S = rows(load(run), key); M = [x for x in S if not x["stop"]]
    rec = np.array([x["rec"] for x in M])
    print("=="*10, run, "move steps", len(M))
    print(" |dyaw| label quantiles (50,90,99,max):", np.round(np.percentile(np.abs(rec[:,2]),[50,90,99,100]),3),
          "; share |binned dyaw|<=0.2 (dyaw=0 passes):", np.mean([abs(b(v))<=0.2+1e-9 for v in rec[:,2]]).round(3))
    print(" |dx| quantiles:", np.round(np.percentile(np.abs(rec[:,0]),[50,90,99,100]),3), " |dy| quantiles:", np.round(np.percentile(np.abs(rec[:,1]),[50,90,99,100]),3))
    # strict per-component failure rates by bin
    for comp, i, t in (("dx",0,0.05),("dy",1,0.05),("dyaw",2,0.1)):
        line=[]
        for bb in range(5):
            T=[x for x in M if min(x["s"]//5,4)==bb]
            fr=lambda nm: np.mean([not(isinstance(x[nm][0],list) and abs(x[nm][0][i]-b(x["rec"][i]))<=t+1e-9) for x in T])
            line.append(f"{fr('real'):.2f}/{fr('gen'):.2f}")
        print(f"  strict fail {comp} (tol {t}) real/gen by bin:", "  ".join(line))
    # turn sign at s=1,2,3 individually
    Tn=[x for x in M if abs(x["rec"][2])>=TURN]
    for s in range(0,6):
        T=[x for x in Tn if x["s"]==s]
        if T: print(f"  turn sign s={s}: n={len(T)} real {np.mean([ok(x['real'][0],x['rec']) for x in T]):.2f} vq {np.mean([ok(x['vq'][0],x['rec']) for x in T]):.2f} gen {np.mean([ok(x['gen'][0],x['rec']) for x in T]):.2f}")
    # accuracy at s=1 (depth 1) all move
    for s in (1,2):
        T=[x for x in M if x["s"]==s]
        print(f"  move acc s={s}: n={len(T)} real {np.mean([acc(x['real'][0],x['rec'],False) for x in T]):.2f} gen {np.mean([acc(x['gen'][0],x['rec'],False) for x in T]):.2f}")
    # real false stops on move-labelled steps: how far from end
    fs=[(x["traj"],x["n"]-1-x["s"]) for x in M if x["real"][0]=="stop"]
    print("  real greedy false Stops (traj, steps before end):", fs)
    # where stop-labelled steps fall in the 5 bins
    print("  stop-labelled steps per bin:", [sum(1 for x in S if x['stop'] and min(x['s']//5,4)==bb) for bb in range(5)])
    # bin-4 move steps only (stopft) : what does real predict wrongly
    if run=="stopft":
        T=[x for x in M if min(x["s"]//5,4)==4]
        wr=[(x["traj"],x["s"],x["n"],x["rec"] and [round(v,3) for v in x["rec"]],x["real"][0],x["gen"][0]) for x in T if not acc(x["real"][0],x["rec"],False)]
        print("  stopft bin4 move steps where real is wrong:", len(wr)); [print("   ",w) for w in wr]
