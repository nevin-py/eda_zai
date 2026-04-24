from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_maintenance import (  # noqa: E402
    CYCLE_COL,
    ENGINE_ID_COL,
    RUL_COL,
    SENSOR_COLS,
    detect_near_zero_variance_sensors,
    prepare_train_test_with_rul,
)

sns.set_theme(style="whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA for CMAPSS FD001.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--summary-path", default="README_DATA_SUMMARY.md")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--variance-threshold", type=float, default=1e-4)
    return parser.parse_args()


def save_sensor_trend_plot(train_df, sensors: list[str], output_path: Path) -> None:
    sampled_engines = (
        train_df[ENGINE_ID_COL].drop_duplicates().sample(n=min(12, train_df[ENGINE_ID_COL].nunique()), random_state=42)
    )
    sampled_df = train_df[train_df[ENGINE_ID_COL].isin(sampled_engines)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes_flat = axes.flatten()

    for idx, sensor in enumerate(sensors[:6]):
        ax = axes_flat[idx]
        for engine_id, group in sampled_df.groupby(ENGINE_ID_COL):
            ax.plot(group[CYCLE_COL], group[sensor], alpha=0.35, linewidth=1.0)
        ax.set_title(f"{sensor} over engine lifetime")
        ax.set_xlabel("Cycle")
        ax.set_ylabel(sensor)

    for idx in range(len(sensors[:6]), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Sensor Trends Across Sampled Engines", fontsize=16)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_rul_histogram(train_df, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(train_df[RUL_COL], bins=40, kde=True, ax=ax, color="#1f77b4")
    ax.set_title("RUL Distribution (Train FD001)")
    ax.set_xlabel("Remaining Useful Life (cycles)")
    ax.set_ylabel("Count")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_sensor_rul_correlation(train_df, sensors: list[str], output_path: Path) -> None:
    corr_df = train_df[[*sensors, RUL_COL]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_df, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("Sensor and RUL Correlation Heatmap")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_data_summary(
    summary_path: Path,
    train_df,
    test_df,
    dropped_sensors: list[str],
    top_sensors: list[str],
) -> None:
    avg_life = train_df.groupby(ENGINE_ID_COL)[CYCLE_COL].max().mean()
    max_life = train_df.groupby(ENGINE_ID_COL)[CYCLE_COL].max().max()

    bullets = [
        "## Data Summary",
        f"- Train set has {len(train_df):,} rows across {train_df[ENGINE_ID_COL].nunique()} engines; test set has {len(test_df):,} rows across {test_df[ENGINE_ID_COL].nunique()} engines.",
        f"- Average observed lifecycle is {avg_life:.1f} cycles, with longest engine run of {int(max_life)} cycles in train data.",
        f"- Near-zero variance sensors removed: {', '.join(dropped_sensors) if dropped_sensors else 'none at current threshold'}.",
        f"- Most RUL-informative sensors by absolute correlation: {', '.join(top_sensors[:5])}.",
        "",
    ]
    summary_path.write_text("\n".join(bullets), encoding="utf-8")


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = prepare_train_test_with_rul(data_dir=args.data_dir, subset=args.subset)

    dropped_sensors = detect_near_zero_variance_sensors(train_df, threshold=args.variance_threshold)
    active_sensors = [sensor for sensor in SENSOR_COLS if sensor not in dropped_sensors]

    corr_series = (
        train_df[[*active_sensors, RUL_COL]]
        .corr(numeric_only=True)[RUL_COL]
        .drop(labels=[RUL_COL])
        .abs()
        .sort_values(ascending=False)
    )
    top_sensors = corr_series.head(8).index.tolist()

    save_sensor_trend_plot(train_df, top_sensors, output_dir / "eda_sensor_trends.png")
    save_rul_histogram(train_df, output_dir / "eda_rul_distribution.png")
    save_sensor_rul_correlation(train_df, active_sensors, output_dir / "eda_sensor_rul_corr_heatmap.png")

    write_data_summary(
        summary_path=Path(args.summary_path),
        train_df=train_df,
        test_df=test_df,
        dropped_sensors=dropped_sensors,
        top_sensors=top_sensors,
    )

    print("EDA artifacts generated:")
    print(f"- {output_dir / 'eda_sensor_trends.png'}")
    print(f"- {output_dir / 'eda_rul_distribution.png'}")
    print(f"- {output_dir / 'eda_sensor_rul_corr_heatmap.png'}")
    print(f"- {Path(args.summary_path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
