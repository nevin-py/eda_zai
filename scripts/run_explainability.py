from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_maintenance import (  # noqa: E402
    CYCLE_COL,
    ENGINE_ID_COL,
    FEATURE_WINDOWS,
    RUL_COL,
    clip_rul,
    latest_cycle_rows,
    prepare_feature_matrices,
    prepare_train_test_with_rul,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP explainability artifacts.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--sample-size", type=int, default=1200)
    return parser.parse_args()


def _expected_value_scalar(explainer: shap.TreeExplainer) -> float:
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        return float(base_value.flatten()[0])
    return float(base_value)


def _shap_values_2d(explainer: shap.TreeExplainer, X):
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return np.asarray(shap_values)


def _save_waterfall_plot(explainer, feature_cols, row, output_path: Path, title: str) -> None:
    row_values = row[feature_cols]
    shap_row = _shap_values_2d(explainer, row_values.to_frame().T)[0]

    explanation = shap.Explanation(
        values=shap_row,
        base_values=_expected_value_scalar(explainer),
        data=row_values.values,
        feature_names=feature_cols,
    )

    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()

    models_dir = Path(args.models_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
    reg_model = joblib.load(models_dir / "rul_model.pkl")

    train_df, test_df = prepare_train_test_with_rul(data_dir=args.data_dir, subset=args.subset)
    train_df = clip_rul(train_df, cap=metadata["rul_cap"])
    test_df = clip_rul(test_df, cap=metadata["rul_cap"])

    feature_pack = prepare_feature_matrices(
        train_df=train_df,
        test_df=test_df,
        dropped_sensors=metadata["dropped_sensors"],
        windows=tuple(metadata.get("feature_windows", FEATURE_WINDOWS)),
        include_raw_sensors=bool(metadata.get("include_raw_sensors", False)),
    )

    test_frame = feature_pack["test_frame"]
    X_test = feature_pack["X_test"]
    feature_cols = metadata["feature_columns"]

    latest_test = latest_cycle_rows(test_frame)
    latest_test["predicted_rul"] = np.clip(reg_model.predict(latest_test[feature_cols]), 0, metadata["rul_cap"])

    sample_size = min(args.sample_size, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    explainer = shap.TreeExplainer(reg_model)
    shap_values_sample = _shap_values_2d(explainer, X_sample)

    # Global explanation: top feature importance bar chart.
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_sample, X_sample, plot_type="bar", max_display=10, show=False)
    plt.tight_layout()
    summary_path = plots_dir / "shap_summary_bar.png"
    plt.savefig(summary_path, dpi=220, bbox_inches="tight")
    plt.close()

    # Select healthy, degrading, and near-failure engines by latest actual RUL.
    latest_sorted = latest_test.sort_values(RUL_COL).reset_index(drop=True)
    near_failure_row = latest_sorted.iloc[0]
    healthy_row = latest_sorted.iloc[-1]
    degrading_row = latest_sorted.iloc[len(latest_sorted) // 2]

    _save_waterfall_plot(
        explainer,
        feature_cols,
        healthy_row,
        plots_dir / "shap_waterfall_healthy.png",
        f"Healthy Engine {int(healthy_row[ENGINE_ID_COL])}",
    )
    _save_waterfall_plot(
        explainer,
        feature_cols,
        degrading_row,
        plots_dir / "shap_waterfall_degrading.png",
        f"Degrading Engine {int(degrading_row[ENGINE_ID_COL])}",
    )
    _save_waterfall_plot(
        explainer,
        feature_cols,
        near_failure_row,
        plots_dir / "shap_waterfall_near_failure.png",
        f"Near-Failure Engine {int(near_failure_row[ENGINE_ID_COL])}",
    )

    mean_abs_shap = np.abs(shap_values_sample).mean(axis=0)
    top_feature = feature_cols[int(np.argmax(mean_abs_shap))]

    # Track top feature SHAP contribution over cycle history for one near-failure engine.
    target_engine_id = int(near_failure_row[ENGINE_ID_COL])
    engine_history = (
        test_frame[test_frame[ENGINE_ID_COL] == target_engine_id]
        .sort_values(CYCLE_COL)
        .reset_index(drop=True)
    )
    shap_history = _shap_values_2d(explainer, engine_history[feature_cols])
    top_feature_idx = feature_cols.index(top_feature)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(engine_history[CYCLE_COL], shap_history[:, top_feature_idx], color="#d62728", linewidth=2)
    ax.axhline(y=0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"SHAP Contribution Over Time ({top_feature}) for Engine {target_engine_id}")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SHAP Value")
    fig.savefig(plots_dir / "shap_top_sensor_over_time.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    explainability_meta = {
        "healthy_engine_id": int(healthy_row[ENGINE_ID_COL]),
        "degrading_engine_id": int(degrading_row[ENGINE_ID_COL]),
        "near_failure_engine_id": int(near_failure_row[ENGINE_ID_COL]),
        "top_global_feature": top_feature,
        "summary_plot": str(summary_path),
    }
    (models_dir / "explainability_metadata.json").write_text(
        json.dumps(explainability_meta, indent=2),
        encoding="utf-8",
    )

    print("Explainability artifacts generated.")
    print(f"- {plots_dir / 'shap_summary_bar.png'}")
    print(f"- {plots_dir / 'shap_waterfall_healthy.png'}")
    print(f"- {plots_dir / 'shap_waterfall_degrading.png'}")
    print(f"- {plots_dir / 'shap_waterfall_near_failure.png'}")
    print(f"- {plots_dir / 'shap_top_sensor_over_time.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
