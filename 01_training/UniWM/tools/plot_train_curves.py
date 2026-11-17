"""Plot training curves from exported log rows (output/logs/*.rows.txt). Usage: python tools/plot_train_curves.py"""

import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
def load(name):
    rows = [eval(l.strip(), {"nan": float("nan"), "inf": float("inf")}) for l in open(f"output/logs/{name}.rows.txt") if l.startswith("{")]
    steps, loss, ep, sc_steps, sc = [], [], [], [], []; s = 0
    for r in rows:
        if "loss" in r and "grad_norm" in r and "train_runtime" not in r:
            s += 10; steps.append(s); loss.append(r["loss"]); ep.append(r["epoch"])
            if "loss_scale" in r: sc_steps.append(s); sc.append(r["loss_scale"])
        elif "loss_scale" in r:
            sc_steps.append(s); sc.append(r["loss_scale"])
    return np.array(steps), np.array(loss), np.array(ep), np.array(sc_steps), np.array(sc)
def smooth(y, w=25):
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for name, lab, c in [("FAILED_1x200", "run 1 (8 GPU, LoRA in fp16) - diverged at epoch 4.4", "#c0392b"), ("1x200", "run 2 (6 GPU, LoRA fp32 + grad downscale)", "#2c3e50")]:
    st, lo, ep, ss, sc = load(name); lo = np.where(np.isfinite(lo) & (lo > 0), lo, np.nan)
    axes[0].plot(ep[24:], smooth(lo), label=lab, color=c, lw=1.4)
    spe = st[-1] / ep[-1]; m = np.isfinite(sc) & (sc >= 1)
    axes[1].semilogy(ss[m] / spe, sc[m], label=lab.split(" (")[0] + f"  (median {np.median(sc[m]):.0f})", color=c, lw=1.4)
    if (~m).any(): axes[1].axvline(ss[~m][0] / spe, color=c, ls="--", lw=1.2, alpha=.8); axes[1].text(ss[~m][0] / spe + 0.3, 3, "scale -> 0,\nweights NaN", color=c, fontsize=8)
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss (smoothed, 250 steps)"); axes[0].set_title("1X-200 training loss"); axes[0].set_ylim(0, 20); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
axes[1].set_ylim(1, 2e4); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("fp16 loss scale"); axes[1].set_title("GradScaler loss scale (V100 fp16)"); axes[1].legend(fontsize=8, loc="lower right"); axes[1].grid(alpha=.3, which="both")
for name, lab, c in [("1x200_stopft", "Stop fine-tune (from run 2, 3 epochs)", "#8e44ad"), ("1x200_mempose", "pose-memory fine-tune (from Stop ft, running)", "#16a085")]:
    st, lo, ep, ss, sc = load(name); lo = np.where(np.isfinite(lo) & (lo > 0), lo, np.nan)
    axes[2].plot(ep[24:], smooth(lo), label=lab, color=c, lw=1.4)
axes[2].set_xlabel("epoch (of the fine-tune)"); axes[2].set_ylabel("train loss (smoothed)"); axes[2].set_title("fine-tunes"); axes[2].set_ylim(0, 16); axes[2].legend(fontsize=8); axes[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig("output/figs/train_curves.png", dpi=130); print("saved")
