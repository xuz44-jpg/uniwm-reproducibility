from ivcommon import *
from ivcommon import _bin
from PIL import Image
for run, key in (("base", "stop_label_released"), ("stopft", "stop_label_dist")):
    R = load(run); rows = [(r["traj"], r["n"], st) for r in R for st in r["steps"]]
    print(f"\n== {run} ({key})")
    # determinism: at s=0 gen image == vq image (both sanitize(frame 0)); greedy should agree
    s0 = [st for _, _, st in rows if st["s"] == 0]
    print(f"  s=0: greedy gen==vq in {sum(st['gen'][0]==st['vq'][0] for st in s0)}/{len(s0)}; gen==real {sum(st['gen'][0]==st['real'][0] for st in s0)}/{len(s0)}; "
          f"sample-set gen==vq {sum(st['gen'][1:]==st['vq'][1:] for st in s0)}/{len(s0)}")
    # Stop split: final frame vs non-final Stop-labelled
    for lab, sel in (("final frame s=n-1", lambda n, st: st["s"] == n - 1),
                     ("dist-rule Stop, non-final", lambda n, st: st["stop_label_dist"] and st["s"] < n - 1),
                     ("not Stop-labelled (own rule)", lambda n, st: not st[key])):
        S = [(t, n, st) for t, n, st in rows if sel(n, st)]
        if not S: continue
        out = []
        for name in ("real", "vq", "gen"):
            g = np.mean([st[name][0] == "stop" for _, _, st in S])
            sm = np.mean([a == "stop" for _, _, st in S for a in st[name][1:]])
            out.append(f"{name} greedy {g:.3f} samples {sm:.3f}")
        print(f"  Stop rate on {lab} (n={len(S)}, depths {min(st['s'] for _,_,st in S)}-{max(st['s'] for _,_,st in S)}): " + "; ".join(out))
    # base-rule stop recall per depth for stopft non-final
    # turn sign paired
    T = [(t, st) for t, n, st in rows if st["recorded"] is not None and not st[key] and abs(st["recorded"][2]) >= TURN]
    for name in ("real", "vq", "gen"):
        print(f"  turn-sign {name}: {f3(cluster_ci([turn_ok(st[name][0], st['recorded']) for _, st in T], [t for t,_ in T]),3)} n={len(T)}")
    d = [int(turn_ok(st["real"][0], st["recorded"])) - int(turn_ok(st["gen"][0], st["recorded"])) for _, st in T]
    print(f"  turn-sign paired real-gen: {f3(cluster_ci(d, [t for t,_ in T]))}")
    dv = [int(turn_ok(st["real"][0], st["recorded"])) - int(turn_ok(st["vq"][0], st["recorded"])) for _, st in T]
    print(f"  turn-sign paired real-vq : {f3(cluster_ci(dv, [t for t,_ in T]))}")
    for b in range(5):
        B = [(t, st) for t, st in T if min(st["s"] // 5, 4) == b]
        print(f"    bin {5*b}-{5*b+4}: n={len(B)} real {np.mean([turn_ok(st['real'][0], st['recorded']) for _, st in B]):.2f} gen {np.mean([turn_ok(st['gen'][0], st['recorded']) for _, st in B]):.2f}")
    # breakdown of turn-sign failures: opposite sign vs too small vs stop/malformed
    for name in ("real", "gen"):
        opp = sum(1 for _, st in T if st[name][0] not in (None,"stop") and np.sign(st[name][0][2]) == -np.sign(st["recorded"][2]) and st[name][0][2] != 0)
        small = sum(1 for _, st in T if st[name][0] not in (None,"stop") and abs(st[name][0][2]) < TURN/2)
        oth = sum(1 for _, st in T if st[name][0] in (None,"stop"))
        print(f"    {name} turn failures: opposite sign (|pred|>0) {opp}, |pred dyaw|<0.025 {small}, stop/malformed {oth}")
# goal image identity check (confound b): goal in prompt = frame n-1 of the task
R = load("base"); same = 0
for r in R:
    n = r["n"]; d = f"{DATA}/{r['traj']}"; nfr = len([f for f in os.listdir(d) if f.endswith('.jpg')])
    a = np.asarray(Image.open(f"{d}/{n-1}.jpg").convert("RGB").resize((256, 256)))
    same += (nfr == n)
print(f"\n  tasks whose n equals the number of frames on disk (goal = last frame = real frame at s=n-1): {same}/{len(R)}")
