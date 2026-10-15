"""Industrial 1-minute load -> 15-minute series for the project notebook.

Column mapping for industrial_load_1min.csv:
  a1      meter id (this file is only #1)
  ts      China local time, naive, interval start of that minute
  active  1-minute mean active power, kW, no storage charge/discharge

A 15-minute stamp 00:00 means the mean of minutes [00:00, 00:15).
A bin is kept only when all 15 minute samples exist; otherwise the 15-minute
value is NaN and later windows that touch it are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURE_TIMEZONE = "Asia/Shanghai"
MINUTE_TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
ID_COLUMN = "a1"
TIMESTAMP_COLUMN = "ts"
LOAD_COLUMN = "active"
TARGET_COLUMN = "load_kw"
SAMPLING_MINUTES = 15
REQUIRED_MINUTE_COUNT = 15


def read_industrial_1min(path: Path | str) -> pd.Series:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到分钟负荷文件: {path.resolve()}")
    frame = pd.read_csv(path)
    missing = {ID_COLUMN, TIMESTAMP_COLUMN, LOAD_COLUMN} - set(frame.columns)
    if missing:
        raise KeyError(f"{path.name} 缺少列: {sorted(missing)}")

    meter_ids = frame[ID_COLUMN].astype(str).unique().tolist()
    if len(meter_ids) != 1:
        raise ValueError(f"当前适配器只支持一个测点，实际有: {meter_ids}")

    parsed = pd.to_datetime(frame[TIMESTAMP_COLUMN], format=MINUTE_TIMESTAMP_FORMAT, errors="coerce")
    if parsed.isna().any():
        raise ValueError("存在无法按 '%Y/%m/%d %H:%M' 解析的时间戳。")

    load = pd.to_numeric(frame[LOAD_COLUMN], errors="coerce")
    series = pd.Series(load.to_numpy(), index=parsed, name=TARGET_COLUMN)
    series.index = series.index.tz_localize(FEATURE_TIMEZONE)
    series = series.sort_index()
    if series.index.has_duplicates:
        raise ValueError(f"发现 {int(series.index.duplicated().sum())} 个重复分钟时间戳。")
    if not series.index.equals(series.index.floor("1min")):
        raise ValueError("存在未对齐到整分钟的时间戳。")
    return series


def aggregate_1min_to_15min(minute_load: pd.Series) -> pd.Series:
    if minute_load.empty:
        raise ValueError("分钟序列为空。")
    if minute_load.index.tz is None:
        raise ValueError("分钟序列需要带时区的时间索引。")

    start = minute_load.index.min().floor(f"{SAMPLING_MINUTES}min")
    end = minute_load.index.max().floor(f"{SAMPLING_MINUTES}min")
    grid = pd.date_range(start, end, freq=f"{SAMPLING_MINUTES}min", name="timestamp")

    buckets = minute_load.index.floor(f"{SAMPLING_MINUTES}min")
    grouped = minute_load.groupby(buckets)
    counts = grouped.size()
    means = grouped.mean()
    complete = counts >= REQUIRED_MINUTE_COUNT
    values = means.where(complete)
    load_15 = values.reindex(grid).astype("float64").rename(TARGET_COLUMN)
    return load_15


def write_industrial_15min(load_15: pd.Series, path: Path | str) -> Path:
    path = Path(path)
    frame = load_15.to_frame(name=TARGET_COLUMN)
    frame.index.name = "timestamp"
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, encoding="utf-8-sig")
    else:
        frame.to_parquet(path)
    return path


def read_industrial_15min(path: Path | str, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Read the aggregated 15-minute industrial table for the project notebook."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到工业15分钟文件: {path.resolve()}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"工业15分钟文件只支持 parquet 或 csv，实际是 {path.suffix}")

    if target_column not in frame.columns:
        raise KeyError(f"缺少目标列 {target_column}，实际列: {list(frame.columns)}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("工业15分钟表的索引必须是时间。")
    if frame.index.tz is None:
        frame = frame.copy()
        frame.index = frame.index.tz_localize(FEATURE_TIMEZONE)
    else:
        frame = frame.copy()
        frame.index = frame.index.tz_convert(FEATURE_TIMEZONE)
    frame.index.name = "timestamp"
    return frame[[target_column]]


def build_industrial_15min(
    source_1min: Path | str = "industrial_load_1min.csv",
    output_parquet: Path | str = "industrial_load_15min.parquet",
    output_csv: Path | str | None = "industrial_load_15min.csv",
) -> dict:
    minute_load = read_industrial_1min(source_1min)
    load_15 = aggregate_1min_to_15min(minute_load)
    parquet_path = write_industrial_15min(load_15, output_parquet)
    csv_path = write_industrial_15min(load_15, output_csv) if output_csv else None
    expected_minutes = int((minute_load.index.max() - minute_load.index.min()).total_seconds() // 60) + 1
    report = {
        "meter_id": "#1",
        "timezone": FEATURE_TIMEZONE,
        "unit": "kW",
        "source_1min_rows": int(len(minute_load)),
        "missing_1min_in_span": int(expected_minutes - len(minute_load)),
        "n_15min": int(len(load_15)),
        "n_15min_complete": int(load_15.notna().sum()),
        "n_15min_incomplete_or_gap": int(load_15.isna().sum()),
        "zero_15min": int((load_15 == 0).sum()),
        "negative_15min": int((load_15 < 0).sum()),
        "first_local": str(load_15.index[0]),
        "last_local": str(load_15.index[-1]),
        "parquet": str(parquet_path),
        "csv": str(csv_path) if csv_path else None,
    }
    return report


if __name__ == "__main__":
    summary = build_industrial_15min()
    for key, value in summary.items():
        print(f"{key}: {value}")
