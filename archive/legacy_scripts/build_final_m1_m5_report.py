from pathlib import Path
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(".").resolve()
OUTPUT_ROOT = ROOT / "outputs" / "v3_convergence_anneal_2000"
REPORT_DIR = ROOT / "final_reports_seed7"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def safe_read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def classify_model_from_npz(npz_path: Path):
    data = np.load(npz_path)

    use_memory = bool(int(np.asarray(data["model_use_memory"]).item()))
    use_reasoning = bool(int(np.asarray(data["model_use_reasoning"]).item()))
    use_coordination = bool(int(np.asarray(data["model_use_coordination"]).item()))
    use_constraint = bool(int(np.asarray(data["model_use_constraint"]).item()))

    if not use_memory and not use_reasoning and not use_coordination and not use_constraint:
        return "M1"
    if use_memory and not use_reasoning and not use_coordination and not use_constraint:
        return "M2"
    if use_memory and use_reasoning and not use_coordination and not use_constraint:
        return "M3"
    if use_memory and use_reasoning and use_coordination and not use_constraint:
        return "M4"
    if use_memory and use_reasoning and use_coordination and use_constraint:
        return "M5"

    return "UNKNOWN"


def model_sort_key(model_name: str):
    order = {"M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5}
    return order.get(model_name, 999)


def rolling_mean(series, window=100):
    return pd.Series(series).rolling(window, min_periods=1).mean()


def tail_mean(df, key, n):
    if key not in df.columns:
        return np.nan
    return float(df[key].tail(n).mean())


# ------------------------------------------------------------
# Discover model directories
# ------------------------------------------------------------
model_rows = []
model_dirs = {}

for subdir in sorted(OUTPUT_ROOT.iterdir()):
    if not subdir.is_dir():
        continue

    train_csv = subdir / "training_metrics.csv"
    npz = subdir / "mappo_weights.npz"

    if train_csv.exists() and npz.exists():
        model_name = classify_model_from_npz(npz)
        model_dirs[model_name] = subdir

        model_rows.append({
            "model": model_name,
            "directory": str(subdir),
        })

if not model_dirs:
    raise RuntimeError(
        f"No valid model directories found under: {OUTPUT_ROOT}"
    )

model_rows = sorted(model_rows, key=lambda x: model_sort_key(x["model"]))

pd.DataFrame(model_rows).to_csv(
    REPORT_DIR / "model_directory_map.csv",
    index=False,
)

print("=" * 70)
print("FOUND MODEL DIRECTORIES")
print("=" * 70)
for row in model_rows:
    print(f"{row['model']:>3s} -> {row['directory']}")
print()


# ------------------------------------------------------------
# Build tables
# ------------------------------------------------------------
det_rows = []
stoch_rows = []
train_rows = []

eval_keys = [
    "outage_ratio",
    "service_deficit",
    "service_ratio_mean",
    "resource_cost",
    "dro",
    "handover_started",
    "handover_failure_rate",
    "pingpong_rate",
    "deadline_violation_rate",
    "preparation_rejection_rate",
    "prepare_to_release_without_commit_rate",
    "prepare_to_commit_rate",
    "duplication_active_time_fraction",
    "resource_price_mean",
    "team_reward",
    "outage_dual_mu",
    "normalized_outage_dual_mu",
    "outage_constraint_gap",
    "constraint_satisfied",
]

for model in sorted(model_dirs.keys(), key=model_sort_key):
    subdir = model_dirs[model]

    det = safe_read_json(subdir / "evaluation.json")
    stoch = safe_read_json(subdir / "evaluation_stochastic.json")

    train_csv = subdir / "training_metrics.csv"
    df = pd.read_csv(train_csv)

    train_row = {
        "model": model,
        "directory": str(subdir),
        "iterations": len(df),
        "reward_last100": tail_mean(df, "team_reward", 100),
        "reward_last200": tail_mean(df, "team_reward", 200),
        "reward_last500": tail_mean(df, "team_reward", 500),
        "outage_last100": tail_mean(df, "outage_ratio", 100),
        "outage_last200": tail_mean(df, "outage_ratio", 200),
        "outage_last500": tail_mean(df, "outage_ratio", 500),
        "service_last100": tail_mean(df, "service_deficit", 100),
        "service_last200": tail_mean(df, "service_deficit", 200),
        "service_last500": tail_mean(df, "service_deficit", 500),
        "resource_last100": tail_mean(df, "resource_cost", 100),
        "resource_last200": tail_mean(df, "resource_cost", 200),
        "resource_last500": tail_mean(df, "resource_cost", 500),
        "handover_last100": tail_mean(df, "handover_started", 100),
        "handover_last200": tail_mean(df, "handover_started", 200),
        "handover_last500": tail_mean(df, "handover_started", 500),
        "pingpong_last100": tail_mean(df, "pingpong_rate", 100),
        "pingpong_last200": tail_mean(df, "pingpong_rate", 200),
        "pingpong_last500": tail_mean(df, "pingpong_rate", 500),
    }

    if "outage_dual_mu" in df.columns:
        train_row["mu_last100"] = tail_mean(df, "outage_dual_mu", 100)
        train_row["mu_last200"] = tail_mean(df, "outage_dual_mu", 200)
        train_row["mu_last500"] = tail_mean(df, "outage_dual_mu", 500)

    train_rows.append(train_row)

    if det is not None:
        row = {"model": model}
        for k in eval_keys:
            row[k] = det.get(k, np.nan)
        det_rows.append(row)

    if stoch is not None:
        row = {"model": model}
        for k in eval_keys:
            row[k] = stoch.get(k, np.nan)
        stoch_rows.append(row)

det_df = pd.DataFrame(det_rows).sort_values("model", key=lambda s: s.map(model_sort_key))
stoch_df = pd.DataFrame(stoch_rows).sort_values("model", key=lambda s: s.map(model_sort_key))
train_df = pd.DataFrame(train_rows).sort_values("model", key=lambda s: s.map(model_sort_key))

det_df.to_csv(REPORT_DIR / "summary_deterministic.csv", index=False)
stoch_df.to_csv(REPORT_DIR / "summary_stochastic.csv", index=False)
train_df.to_csv(REPORT_DIR / "summary_training_tail.csv", index=False)

print("=" * 70)
print("TABLE FILES CREATED")
print("=" * 70)
print(REPORT_DIR / "summary_deterministic.csv")
print(REPORT_DIR / "summary_stochastic.csv")
print(REPORT_DIR / "summary_training_tail.csv")
print()


# ------------------------------------------------------------
# Save markdown summary
# ------------------------------------------------------------
def df_to_markdown_file(df, path: Path, title: str):
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("No data available.\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")

df_to_markdown_file(det_df, REPORT_DIR / "summary_deterministic.md", "Deterministic Summary")
df_to_markdown_file(stoch_df, REPORT_DIR / "summary_stochastic.md", "Stochastic Summary")
df_to_markdown_file(train_df, REPORT_DIR / "summary_training_tail.md", "Training Tail Summary")


# ------------------------------------------------------------
# Plot 1: reward convergence
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))

for model in sorted(model_dirs.keys(), key=model_sort_key):
    subdir = model_dirs[model]
    df = pd.read_csv(subdir / "training_metrics.csv")

    if "team_reward" not in df.columns:
        continue

    x = np.arange(1, len(df) + 1)
    y = rolling_mean(df["team_reward"].astype(float), 100)

    plt.plot(x, y, label=model)

plt.xlabel("Training iteration")
plt.ylabel("Team reward (100-iteration moving average)")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "figure_reward_convergence.png", dpi=300, bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# Helper for KPI bar plots
# ------------------------------------------------------------
def save_bar_plot(df, key, ylabel, filename):
    if df.empty or key not in df.columns:
        return

    ordered = df.sort_values("model", key=lambda s: s.map(model_sort_key))

    plt.figure(figsize=(7, 4))
    plt.bar(ordered["model"], ordered[key].astype(float))
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# Plot 2~6: stochastic KPI compare
# ------------------------------------------------------------
save_bar_plot(stoch_df, "outage_ratio", "Outage ratio", "figure_outage_stochastic.png")
save_bar_plot(stoch_df, "service_deficit", "Service deficit", "figure_service_deficit_stochastic.png")
save_bar_plot(stoch_df, "resource_cost", "Resource cost", "figure_resource_stochastic.png")
save_bar_plot(stoch_df, "handover_started", "Handover started", "figure_handover_stochastic.png")
save_bar_plot(stoch_df, "pingpong_rate", "Ping-pong rate", "figure_pingpong_stochastic.png")
save_bar_plot(stoch_df, "duplication_active_time_fraction", "Duplication active time fraction", "figure_duplication_stochastic.png")


# ------------------------------------------------------------
# Plot 7: M5 dual convergence
# ------------------------------------------------------------
if "M5" in model_dirs:
    m5_dir = model_dirs["M5"]
    m5_df = pd.read_csv(m5_dir / "training_metrics.csv")

    if "outage_ratio" in m5_df.columns and "outage_dual_mu" in m5_df.columns:
        x = np.arange(1, len(m5_df) + 1)
        outage_ma = rolling_mean(m5_df["outage_ratio"].astype(float), 100)
        mu_ma = rolling_mean(m5_df["outage_dual_mu"].astype(float), 100)

        fig, ax1 = plt.subplots(figsize=(9, 5))

        ax1.plot(x, outage_ma, label="Outage ratio (100-iter MA)")
        ax1.axhline(0.10, linestyle="--", label="Outage target = 0.10")
        ax1.set_xlabel("Training iteration")
        ax1.set_ylabel("Outage ratio")

        ax2 = ax1.twinx()
        ax2.plot(x, mu_ma, linestyle="-.", label="Dual multiplier mu (100-iter MA)")
        ax2.set_ylabel("Dual multiplier mu")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "figure_m5_dual_convergence.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


print("=" * 70)
print("FIGURES CREATED")
print("=" * 70)
for name in [
    "figure_reward_convergence.png",
    "figure_outage_stochastic.png",
    "figure_service_deficit_stochastic.png",
    "figure_resource_stochastic.png",
    "figure_handover_stochastic.png",
    "figure_pingpong_stochastic.png",
    "figure_duplication_stochastic.png",
    "figure_m5_dual_convergence.png",
]:
    p = REPORT_DIR / name
    if p.exists():
        print(p)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
print(f"Report directory: {REPORT_DIR}")
