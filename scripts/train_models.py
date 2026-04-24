from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_maintenance import (  # noqa: E402
    ENGINE_ID_COL,
    FAILURE_THRESHOLD,
    FEATURE_WINDOWS,
    RUL_CLIP_VALUE,
    RUL_COL,
    SENSOR_COLS,
    clip_rul,
    detect_near_zero_variance_sensors,
    latest_cycle_rows,
    prepare_feature_matrices,
    prepare_train_test_with_rul,
)

sns.set_theme(style="whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RUL regression and failure classification models.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--rul-cap", type=int, default=RUL_CLIP_VALUE)
    parser.add_argument("--failure-threshold", type=int, default=FAILURE_THRESHOLD)
    parser.add_argument("--variance-threshold", type=float, default=1e-4)
    parser.add_argument("--include-raw-sensors", action="store_true")
    parser.add_argument("--reg-search-iter", type=int, default=18)
    parser.add_argument("--cls-search-iter", type=int, default=14)
    return parser.parse_args()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def risk_level(predicted_rul: float) -> str:
    if predicted_rul < 30:
        return "High"
    if predicted_rul <= 70:
        return "Medium"
    return "Low"


def save_confusion_matrix_plot(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix: Failure Within 30 Cycles")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    models_dir = Path(args.models_dir)
    plots_dir = Path(args.plots_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = prepare_train_test_with_rul(data_dir=args.data_dir, subset=args.subset)
    dropped_sensors = detect_near_zero_variance_sensors(train_df, threshold=args.variance_threshold)

    train_df = clip_rul(train_df, cap=args.rul_cap)
    test_df = clip_rul(test_df, cap=args.rul_cap)

    feature_pack = prepare_feature_matrices(
        train_df=train_df,
        test_df=test_df,
        dropped_sensors=dropped_sensors,
        windows=FEATURE_WINDOWS,
        include_raw_sensors=args.include_raw_sensors,
    )

    train_frame = feature_pack["train_frame"]
    test_frame = feature_pack["test_frame"]
    X_train = feature_pack["X_train"]
    X_test = feature_pack["X_test"]
    y_train = feature_pack["y_train"]
    y_test = feature_pack["y_test"]
    feature_cols = feature_pack["feature_cols"]

    groups = train_frame[ENGINE_ID_COL]

    baseline_reg = LinearRegression()
    baseline_reg.fit(X_train, y_train)
    baseline_pred = np.clip(baseline_reg.predict(X_test), 0, args.rul_cap)
    baseline_scores = regression_metrics(y_test, baseline_pred)

    reg_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    reg_params = {
        "n_estimators": [250, 400, 550, 700],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
    }
    reg_search = RandomizedSearchCV(
        estimator=reg_model,
        param_distributions=reg_params,
        n_iter=args.reg_search_iter,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        cv=GroupKFold(n_splits=5),
        random_state=42,
        verbose=1,
    )
    reg_search.fit(X_train, y_train, groups=groups)
    best_reg = reg_search.best_estimator_

    reg_pred = np.clip(best_reg.predict(X_test), 0, args.rul_cap)
    reg_scores = regression_metrics(y_test, reg_pred)

    y_train_cls = (y_train <= args.failure_threshold).astype(int)
    y_test_cls = (y_test <= args.failure_threshold).astype(int)

    cls_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    cls_params = {
        "n_estimators": [200, 350, 500, 700],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "scale_pos_weight": [1.0, 1.5, 2.0, 3.0],
    }
    cls_search = RandomizedSearchCV(
        estimator=cls_model,
        param_distributions=cls_params,
        n_iter=args.cls_search_iter,
        scoring="recall",
        n_jobs=-1,
        cv=GroupKFold(n_splits=5),
        random_state=42,
        verbose=1,
    )
    cls_search.fit(X_train, y_train_cls, groups=groups)
    best_cls = cls_search.best_estimator_

    cls_prob = best_cls.predict_proba(X_test)[:, 1]
    cls_pred = (cls_prob >= 0.5).astype(int)
    cls_scores = classification_metrics(y_test_cls, cls_pred, cls_prob)

    cm = confusion_matrix(y_test_cls, cls_pred)
    save_confusion_matrix_plot(cm, plots_dir / "confusion_matrix.png")

    joblib.dump(best_reg, models_dir / "rul_model.pkl")
    joblib.dump(best_cls, models_dir / "classifier.pkl")

    active_sensors = [sensor for sensor in SENSOR_COLS if sensor not in dropped_sensors]
    sensor_corr = (
        train_df[[*active_sensors, RUL_COL]]
        .corr(numeric_only=True)[RUL_COL]
        .drop(labels=[RUL_COL])
        .abs()
        .sort_values(ascending=False)
    )
    top_dashboard_sensors = sensor_corr.head(3).index.tolist()

    metadata = {
        "subset": args.subset,
        "rul_cap": args.rul_cap,
        "failure_threshold": args.failure_threshold,
        "variance_threshold": args.variance_threshold,
        "dropped_sensors": dropped_sensors,
        "feature_windows": list(FEATURE_WINDOWS),
        "include_raw_sensors": bool(args.include_raw_sensors),
        "feature_columns": feature_cols,
        "top_dashboard_sensors": top_dashboard_sensors,
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics_payload = {
        "dataset": {
            "subset": args.subset,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "engine_count_train": int(train_df[ENGINE_ID_COL].nunique()),
            "engine_count_test": int(test_df[ENGINE_ID_COL].nunique()),
        },
        "regression": {
            "baseline_linear_regression": baseline_scores,
            "xgboost": reg_scores,
            "best_params": reg_search.best_params_,
        },
        "classification": {
            "xgboost": cls_scores,
            "best_params": cls_search.best_params_,
            "threshold": 0.5,
            "failure_within_cycles": args.failure_threshold,
            "confusion_matrix": cm.astype(int).tolist(),
        },
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    latest_test = latest_cycle_rows(test_frame)
    latest_test = latest_test[[ENGINE_ID_COL, "cycle", RUL_COL, *feature_cols]].copy()
    latest_test["predicted_rul"] = np.clip(best_reg.predict(latest_test[feature_cols]), 0, args.rul_cap)
    latest_test["failure_probability_30"] = best_cls.predict_proba(latest_test[feature_cols])[:, 1]
    latest_test["failure_label_30"] = (latest_test["failure_probability_30"] >= 0.5).astype(int)
    latest_test["risk_level"] = latest_test["predicted_rul"].apply(risk_level)

    latest_test[[
        ENGINE_ID_COL,
        "cycle",
        RUL_COL,
        "predicted_rul",
        "risk_level",
        "failure_probability_30",
        "failure_label_30",
    ]].to_csv(models_dir / "fleet_predictions.csv", index=False)

    print("Training complete.")
    print(f"Baseline Regression: RMSE={baseline_scores['rmse']:.3f}, MAE={baseline_scores['mae']:.3f}")
    print(f"XGB Regression: RMSE={reg_scores['rmse']:.3f}, MAE={reg_scores['mae']:.3f}, R2={reg_scores['r2']:.3f}")
    print(
        "XGB Classification: "
        f"Precision={cls_scores['precision']:.3f}, Recall={cls_scores['recall']:.3f}, F1={cls_scores['f1']:.3f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
