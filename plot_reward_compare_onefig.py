from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# user setting
# -----------------------------
ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "v3_convergence_anneal_2000"

MODEL_DIRS = {
    "M1": BASE / "m1_seed7",
    "M2": BASE / "m2_seed7",
    "M3": BASE / "m3_seed7",
    "M4": BASE / "m4_seed7",
    "M5": BASE / "m5_dual_only_seed7",
}

WINDOW = 50              # smoothing window
SHOW_RAW = False         # True면 raw reward도 옅게 같이 그림
RAW_ALPHA = 0.10
SAVE_PATH = BASE / "reward_compare_onefig.png"


def find_column(df, candidates, required=True):
    """
    df.columns 중에서 후보 이름과 일치/유사한 컬럼을 찾는다.
    """
    lower_map = {c.lower(): c for c in df.columns}

    # exact match
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    # partial match
    for c in df.columns:
        lc = c.lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c

    if required:
        raise KeyError(
            f"Could not find any of columns {candidates} in {list(df.columns)}"
        )
    return None


def load_reward_curve(model_dir: Path):
    csv_path = model_dir / "training_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"training_metrics.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # iteration column
    iter_col = find_column(
        df,
        ["iteration", "iter", "update", "updates", "step"],
        required=False,
    )

    # reward column
    reward_col = find_column(
        df,
        [
            "team_reward",
            "reward",
            "episode_reward",
            "mean_reward",
            "avg_reward",
        ],
        required=True,
    )

    if iter_col is None:
        x = pd.Series(range(1, len(df) + 1), name="iteration")
    else:
        x = pd.to_numeric(df[iter_col], errors="coerce")

    y = pd.to_numeric(df[reward_col], errors="coerce")

    out = pd.DataFrame({"iteration": x, "reward": y}).dropna()
    out["smoothed_reward"] = out["reward"].rolling(
        window=WINDOW, min_periods=1
    ).mean()

    return out, reward_col


def main():
    plt.figure(figsize=(12, 7))

    print("=== Reward column check ===")
    for model_name, model_dir in MODEL_DIRS.items():
        curve, reward_col = load_reward_curve(model_dir)
        print(f"{model_name}: {reward_col} | n={len(curve)} | dir={model_dir}")

        if SHOW_RAW:
            plt.plot(
                curve["iteration"],
                curve["reward"],
                alpha=RAW_ALPHA,
                linewidth=1.0,
            )

        plt.plot(
            curve["iteration"],
            curve["smoothed_reward"],
            linewidth=2.5,
            label=f"{model_name} (smoothed, window={WINDOW})",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Reward Comparison of M1~M5 in One Figure")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(SAVE_PATH, dpi=200)
    print(f"\nSaved: {SAVE_PATH}")

    plt.show()


if __name__ == "__main__":
    main()