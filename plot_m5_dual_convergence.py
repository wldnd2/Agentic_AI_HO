from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(".").resolve()

CSV = (
    ROOT
    / "outputs"
    / "v3_convergence_anneal_2000"
    / "m5_dual_only_seed7"
    / "training_metrics.csv"
)

OUT = ROOT / "m5_dual_convergence.png"

WINDOW = 100
TARGET = 0.10


df = pd.read_csv(CSV)

iteration = np.arange(1, len(df) + 1)

outage = df["outage_ratio"].astype(float)

if "outage_dual_mu" not in df.columns:
    raise RuntimeError(
        "training_metrics.csv has no outage_dual_mu column"
    )

mu = df["outage_dual_mu"].astype(float)

outage_ma = outage.rolling(
    WINDOW,
    min_periods=1,
).mean()

mu_ma = mu.rolling(
    WINDOW,
    min_periods=1,
).mean()


fig, ax1 = plt.subplots(
    figsize=(9, 5),
)

ax1.plot(
    iteration,
    outage_ma,
    label=f"Outage ratio ({WINDOW}-iteration MA)",
)

ax1.axhline(
    TARGET,
    linestyle="--",
    label="Outage target = 0.10",
)

ax1.set_xlabel("Training iteration")
ax1.set_ylabel("Outage ratio")
ax1.set_ylim(bottom=0)

ax2 = ax1.twinx()

ax2.plot(
    iteration,
    mu_ma,
    linestyle="-.",
    label=f"Dual multiplier mu ({WINDOW}-iteration MA)",
)

ax2.set_ylabel("Dual multiplier mu")
ax2.set_ylim(bottom=0)


lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best",
)

fig.tight_layout()

fig.savefig(
    OUT,
    dpi=300,
    bbox_inches="tight",
)

print("=" * 70)
print("M5 DUAL CONVERGENCE")
print("=" * 70)

for window in [100, 200, 500]:
    tail = df.tail(window)

    print()
    print(f"Last {window} iterations")
    print(
        f"  outage mean = "
        f"{tail['outage_ratio'].mean():.9f}"
    )
    print(
        f"  outage std  = "
        f"{tail['outage_ratio'].std():.9f}"
    )
    print(
        f"  mu mean     = "
        f"{tail['outage_dual_mu'].mean():.9f}"
    )
    print(
        f"  mu std      = "
        f"{tail['outage_dual_mu'].std():.9f}"
    )
    print(
        f"  gap         = "
        f"{tail['outage_ratio'].mean() - TARGET:+.9f}"
    )

print()
print(f"Saved figure: {OUT}")
