"""Plot test accuracy during the second stage for every method (all start from the same SFT model)."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = lambda f: json.load(open(os.path.expanduser(f"~/.labs_data/{f}")))
runs = [("more SFT on the same labeled data", L("sftmore_log.json"), 2, "#7a5c2e"),
        ("GRPO", L("grpo_log.json"), 2, "#2e5c7a"),
        ("SFT on own correct answers", L("dpo_log_lr2e-05_nll1_nodpo.json"), 2, "#6a9a4a"),
        ("DPO + SFT loss", L("dpo_log_lr2e-05_nll1.json"), 2, "#9b6fb0"),
        ("DPO, lr 2e-6", L("dpo_log_lr2e-06_nll0.json"), 2, "#d19a3a"),
        ("DPO, lr 2e-5", L("dpo_log_lr2e-05_nll0.json"), 2, "#b03a2e")]
start = 100 * L("sft_log.json")[-1][2]          # accuracy of the SFT model every method starts from
fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
for name, log, k, c in runs:
    ax.plot([0] + [r[0] for r in log], [start] + [100 * r[k] for r in log], "o-", ms=3, color=c, label=name)
ax.set_xlabel("training steps after SFT"); ax.set_ylabel("test accuracy (%)")
ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "..", "..", "img", "rl_math_curves.png"))
print("saved")
