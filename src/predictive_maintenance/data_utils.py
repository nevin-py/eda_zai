from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import CYCLE_COL, ENGINE_ID_COL, RUL_COL, SENSOR_COLS, SETTING_COLS


def get_column_names() -> list[str]:
    return [ENGINE_ID_COL, CYCLE_COL, *SETTING_COLS, *SENSOR_COLS]


def _read_cmapss_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    column_count = len(get_column_names())
    if df.shape[1] > column_count:
        df = df.iloc[:, :column_count]
    df.columns = get_column_names()
    return df


def load_cmapss_split(data_dir: str | Path, split: str, subset: str = "FD001") -> pd.DataFrame:
    file_path = Path(data_dir) / f"{split}_{subset}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Expected dataset file not found: {file_path}")
    return _read_cmapss_file(file_path)


def load_rul_targets(data_dir: str | Path, subset: str = "FD001") -> pd.Series:
    file_path = Path(data_dir) / f"RUL_{subset}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Expected target file not found: {file_path}")
    rul_df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    return rul_df.iloc[:, 0].astype(float)


def prepare_train_test_with_rul(data_dir: str | Path, subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = load_cmapss_split(data_dir, "train", subset=subset)
    test_df = load_cmapss_split(data_dir, "test", subset=subset)
    test_rul = load_rul_targets(data_dir, subset=subset)

    train_df = train_df.sort_values([ENGINE_ID_COL, CYCLE_COL]).reset_index(drop=True)
    max_cycle_train = train_df.groupby(ENGINE_ID_COL)[CYCLE_COL].transform("max")
    train_df[RUL_COL] = max_cycle_train - train_df[CYCLE_COL]

    test_df = test_df.sort_values([ENGINE_ID_COL, CYCLE_COL]).reset_index(drop=True)
    test_max_cycle = test_df.groupby(ENGINE_ID_COL)[CYCLE_COL].max().reset_index(name="max_cycle")

    if len(test_max_cycle) != len(test_rul):
        raise ValueError(
            "Mismatch between number of test engines and RUL targets: "
            f"{len(test_max_cycle)} engines vs {len(test_rul)} targets."
        )

    test_max_cycle["final_rul"] = test_rul.values
    test_df = test_df.merge(test_max_cycle, on=ENGINE_ID_COL, how="left")
    test_df[RUL_COL] = test_df["final_rul"] + (test_df["max_cycle"] - test_df[CYCLE_COL])
    test_df = test_df.drop(columns=["max_cycle", "final_rul"])

    return train_df, test_df


def detect_near_zero_variance_sensors(
    df: pd.DataFrame,
    threshold: float = 1e-4,
    sensors: Iterable[str] | None = None,
) -> list[str]:
    sensor_cols = list(sensors) if sensors is not None else SENSOR_COLS
    variances = df[sensor_cols].var(ddof=0)
    low_variance = variances[variances <= threshold].index.tolist()
    return sorted(low_variance)


def clip_rul(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    clipped = df.copy()
    clipped[RUL_COL] = clipped[RUL_COL].clip(upper=cap)
    return clipped


def latest_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    latest_idx = df.groupby(ENGINE_ID_COL)[CYCLE_COL].idxmax()
    latest = df.loc[latest_idx].sort_values(ENGINE_ID_COL).reset_index(drop=True)
    return latest
