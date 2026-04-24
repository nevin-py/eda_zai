from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .config import CYCLE_COL, ENGINE_ID_COL, RUL_COL, SENSOR_COLS, SETTING_COLS


def add_rolling_features(
    df: pd.DataFrame,
    sensors: Sequence[str],
    windows: Sequence[int],
) -> pd.DataFrame:
    enriched = df.sort_values([ENGINE_ID_COL, CYCLE_COL]).copy()

    for sensor in sensors:
        grouped_sensor = enriched.groupby(ENGINE_ID_COL, sort=False)[sensor]
        for window in windows:
            mean_col = f"{sensor}_roll_mean_{window}"
            std_col = f"{sensor}_roll_std_{window}"
            enriched[mean_col] = grouped_sensor.transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
            enriched[std_col] = grouped_sensor.transform(
                lambda s: s.rolling(window=window, min_periods=1).std().fillna(0.0)
            )

    return enriched


def build_feature_frame(
    df: pd.DataFrame,
    dropped_sensors: Sequence[str],
    windows: Sequence[int],
    include_raw_sensors: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    selected_sensors = [sensor for sensor in SENSOR_COLS if sensor not in set(dropped_sensors)]

    enriched = add_rolling_features(df, selected_sensors, windows)

    rolling_cols: list[str] = []
    for sensor in selected_sensors:
        for window in windows:
            rolling_cols.append(f"{sensor}_roll_mean_{window}")
            rolling_cols.append(f"{sensor}_roll_std_{window}")

    feature_cols = [*SETTING_COLS, *rolling_cols]
    if include_raw_sensors:
        feature_cols = [*SETTING_COLS, *selected_sensors, *rolling_cols]

    return enriched, feature_cols


def prepare_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dropped_sensors: Sequence[str],
    windows: Sequence[int],
    include_raw_sensors: bool = False,
) -> dict[str, pd.DataFrame | pd.Series | list[str]]:
    train_features, feature_cols = build_feature_frame(
        train_df,
        dropped_sensors=dropped_sensors,
        windows=windows,
        include_raw_sensors=include_raw_sensors,
    )
    test_features, _ = build_feature_frame(
        test_df,
        dropped_sensors=dropped_sensors,
        windows=windows,
        include_raw_sensors=include_raw_sensors,
    )

    return {
        "train_frame": train_features,
        "test_frame": test_features,
        "X_train": train_features[feature_cols],
        "X_test": test_features[feature_cols],
        "y_train": train_features[RUL_COL],
        "y_test": test_features[RUL_COL],
        "feature_cols": feature_cols,
    }
