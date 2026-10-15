"""Assemble 672/5 + 96/4 direct-load windows using the project notebook rules.

Does not re-aggregate 1-minute data. Reads the already-built 15-minute table
and the same config / split / scaling / calendar logic as 01_load15_my_project.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_load_adapter import read_industrial_15min

PAST_FEATURE_NAMES = ["load_scaled", "time_sin", "time_cos", "dow_sin", "dow_cos"]
FUTURE_FEATURE_NAMES = ["time_sin", "time_cos", "dow_sin", "dow_cos"]
EXPECTED_LOOKBACK = 672
EXPECTED_HORIZON = 96
MIN_TRAIN_WINDOWS = 16
MIN_VAL_WINDOWS = 1


@dataclass
class DirectLoadBundle:
    config: dict
    load: pd.Series
    audit: dict
    lookback: int
    horizon: int
    train_end: int
    val_end: int
    train_mean: float
    train_std: float
    past_feature_names: list[str]
    future_feature_names: list[str]
    split_table: pd.DataFrame
    arrays: dict[str, dict[str, np.ndarray]]


def load_project_config(path: Path | str = "my_load15_config.json") -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到配置: {path.resolve()}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_configured_frame(config: dict) -> pd.DataFrame:
    kind = config["dataset_kind"]
    target = config["target_column"]
    if kind == "industrial_split_csv":
        frame, _ = read_target_split_frames(config)
        return frame
    data_path = Path(config["data_file"])
    if not data_path.is_file():
        raise FileNotFoundError(f"未找到数据文件: {data_path.resolve()}")
    if kind == "industrial_csv":
        return read_industrial_15min(data_path, target_column=target)
    if kind == "smard_demo":
        return pd.read_parquet(data_path, columns=[target])
    raise ValueError(f"未知 dataset_kind: {kind}")


def read_target_split_frames(config: dict) -> tuple[pd.DataFrame, dict[str, tuple[pd.Timestamp, pd.Timestamp]]]:
    """Read context-bearing business CSVs and validate their overlapping rows."""
    timezone_name = config["feature_timezone"]
    target = config["target_column"]
    pieces = []
    periods = {}
    for split in ("train", "validation", "test"):
        path = Path(config[f"{split}_file"])
        if not path.is_file():
            raise FileNotFoundError(f"未找到 {split} 业务输入: {path.resolve()}")
        frame = pd.read_csv(path)
        required = {"a1", "timestamp", target, "minute_count", "is_complete", "is_target_period"}
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{path.name} 缺少列: {sorted(missing)}")
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        if timestamps.isna().any():
            raise ValueError(f"{path.name} 有 {int(timestamps.isna().sum())} 个时间戳无法解析")
        local = pd.DatetimeIndex(timestamps).tz_convert(timezone_name)
        part = pd.DataFrame({
            target: pd.to_numeric(frame[target], errors="coerce").to_numpy(),
            "minute_count": pd.to_numeric(frame["minute_count"], errors="raise").to_numpy(),
            "is_complete": frame["is_complete"].astype(str).str.lower().eq("true").to_numpy(),
        }, index=local)
        part.index.name = "timestamp"
        pieces.append(part)
        raw_start, raw_end = config["target_periods"][split]
        periods[split] = (
            pd.Timestamp(raw_start, tz=timezone_name).tz_convert("UTC"),
            pd.Timestamp(raw_end, tz=timezone_name).tz_convert("UTC"),
        )

    combined = pd.concat(pieces).sort_index()
    for timestamp, group in combined.groupby(level=0):
        if len(group) > 1 and (
            group[target].nunique(dropna=False) > 1
            or group["minute_count"].nunique(dropna=False) > 1
            or group["is_complete"].nunique(dropna=False) > 1
        ):
            raise ValueError(f"业务 CSV 的重叠 context 在 {timestamp} 不一致")
    combined = combined[~combined.index.duplicated(keep="first")]
    if not combined.index.equals(pd.date_range(combined.index.min(), combined.index.max(), freq="15min")):
        raise ValueError("三个业务 CSV 合并后不是连续 15 分钟时间轴")
    complete_matches = combined["is_complete"].eq(combined["minute_count"].eq(15))
    if not complete_matches.all():
        raise ValueError("is_complete 与 minute_count == 15 不一致")
    if combined.loc[~combined["is_complete"], target].notna().any():
        raise ValueError("不完整 15 分钟桶的负荷必须为 NaN")
    result = combined[[target]].copy()
    result.index = result.index.tz_convert("UTC")
    return result, periods


def prepare_load(frame: pd.DataFrame, target: str, load_unit: str) -> tuple[pd.Series, dict]:
    if target not in frame.columns:
        raise KeyError(f"缺少目标列 {target}")
    if frame.empty:
        raise ValueError("数据为空。")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("索引不是 DatetimeIndex，需要检查文件结构。")
    if frame.index.hasnans:
        raise ValueError("时间索引含 NaT，不能自动推断。")

    frame = frame.copy()
    assumed_utc = frame.index.tz is None
    if assumed_utc:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame = frame.sort_index()
    duplicates = int(frame.index.duplicated().sum())
    if duplicates:
        raise ValueError(f"发现 {duplicates} 个重复时间戳；先查明来源，不自动合并。")
    if not frame.index.equals(frame.index.floor("15min")):
        raise ValueError("存在未对齐到 15 分钟网格的时间戳。")

    values = pd.to_numeric(frame[target], errors="coerce").astype("float64")
    values = values.replace([np.inf, -np.inf], np.nan)
    first, last = values.first_valid_index(), values.last_valid_index()
    if first is None:
        raise ValueError("目标列没有有效数值。")
    observed_span = values.loc[first:last]
    grid = pd.date_range(first, last, freq="15min", name="timestamp")
    load = observed_span.reindex(grid).rename("load_raw")
    audit = {
        "source_rows": len(frame),
        "assumed_utc_for_naive_index": assumed_utc,
        "trimmed_leading_rows": int((frame.index < first).sum()),
        "trimmed_trailing_rows": int((frame.index > last).sum()),
        "inserted_missing_timestamps": len(grid) - len(observed_span),
        "missing_load_on_grid": int(load.isna().sum()),
        "zero_load_on_grid": int((load == 0).sum()),
        "negative_load_on_grid": int((load < 0).sum()),
        "nonpositive_load_on_grid": int((load <= 0).sum()),
        "first_observed_utc": str(first),
        "last_observed_utc": str(last),
        "unit": load_unit,
    }
    return load, audit


def valid_origins(values: np.ndarray, lookback: int, horizon: int) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    if values.ndim != 1 or lookback < 1 or horizon < 1:
        raise ValueError("需要一维序列及正整数窗口长度。")
    if len(values) < lookback + horizon:
        raise ValueError("数据不足以生成一条完整样本。")
    candidates = np.arange(lookback, len(values) - horizon + 1)
    bad_prefix = np.concatenate(([0], np.cumsum(~np.isfinite(values))))
    missing_counts = bad_prefix[candidates + horizon] - bad_prefix[candidates - lookback]
    return candidates[missing_counts == 0]


def split_origins(
    origins: np.ndarray,
    n: int,
    horizon: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[dict, int, int]:
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("训练/验证比例必须为正，且合计小于1。")
    train_end = int(train_fraction * n)
    val_end = int((train_fraction + validation_fraction) * n)
    splits = {
        "train": origins[origins + horizon <= train_end],
        "validation": origins[(origins >= train_end) & (origins + horizon <= val_end)],
        "test": origins[(origins >= val_end) & (origins + horizon <= n)],
    }
    for name, positions in splits.items():
        if len(positions) == 0:
            raise ValueError(f"{name} 没有完整样本；检查数据长度和缺口分布。")
    return splits, train_end, val_end


def fit_train_scaling(values: np.ndarray, train_end: int) -> tuple[float, float]:
    observed_train = values[:train_end]
    observed_train = observed_train[np.isfinite(observed_train)]
    if len(observed_train) < 2:
        raise ValueError("训练期有效观测不足。")
    mean = float(observed_train.mean())
    std = float(observed_train.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("训练期标准差无效，无法标准化。")
    return mean, std


def calendar_features_for_project(index: pd.DatetimeIndex, timezone_name: str) -> pd.DataFrame:
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("时间特征要求带时区的 DatetimeIndex。")
    local = index.tz_convert(timezone_name)
    fraction_of_day = (local.hour.to_numpy() * 60 + local.minute.to_numpy()) / 1440.0
    fraction_of_week = local.dayofweek.to_numpy() / 7.0
    return pd.DataFrame({
        "time_sin": np.sin(2 * np.pi * fraction_of_day),
        "time_cos": np.cos(2 * np.pi * fraction_of_day),
        "dow_sin": np.sin(2 * np.pi * fraction_of_week),
        "dow_cos": np.cos(2 * np.pi * fraction_of_week),
    }, index=index, dtype="float32")


def build_xy_arrays(
    scaled_load: np.ndarray,
    calendar: np.ndarray,
    origins: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(origins)
    x_past = np.empty((n, lookback, 5), dtype="float32")
    x_future = np.empty((n, horizon, 4), dtype="float32")
    y = np.empty((n, horizon), dtype="float32")
    for i, cut in enumerate(origins):
        cut = int(cut)
        hist = scaled_load[cut - lookback:cut]
        fut = scaled_load[cut:cut + horizon]
        x_past[i, :, 0] = hist
        x_past[i, :, 1:] = calendar[cut - lookback:cut]
        x_future[i] = calendar[cut:cut + horizon]
        y[i] = fut
    return x_past, x_future, y


def inverse_load_kw(scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
    return scaled.astype("float64") * std + mean


def build_direct_load_bundle(config_path: Path | str = "my_load15_config.json") -> DirectLoadBundle:
    config = load_project_config(config_path)
    lookback = int(config["lookback_days"]) * (24 * 60 // int(config["sampling_minutes"]))
    horizon = int(config["horizon_hours"]) * 60 // int(config["sampling_minutes"])
    if lookback != EXPECTED_LOOKBACK or horizon != EXPECTED_HORIZON:
        raise ValueError(
            f"当前直接负荷图按 {EXPECTED_LOOKBACK}->{EXPECTED_HORIZON} 断言，"
            f"实际 lookback={lookback}, horizon={horizon}。"
        )

    periods = None
    if config["dataset_kind"] == "industrial_split_csv":
        frame, periods = read_target_split_frames(config)
    else:
        frame = read_configured_frame(config)
    load, audit = prepare_load(frame, config["target_column"], config["load_unit"])
    if audit["negative_load_on_grid"]:
        raise ValueError("存在负负荷，先检查符号和单位。")
    if config["dataset_kind"] == "smard_demo" and audit["zero_load_on_grid"]:
        raise ValueError("德国总负荷存在非正值；先检查异常。")

    values = load.to_numpy(dtype="float64")
    origins = valid_origins(values, lookback, horizon)
    if periods is None:
        splits, train_end, val_end = split_origins(
            origins, len(values), horizon,
            float(config["train_fraction"]), float(config["validation_fraction"]),
        )
    else:
        train_end = int(load.index.searchsorted(periods["validation"][0]))
        val_end = int(load.index.searchsorted(periods["test"][0]))
        splits = {}
        for name, (period_start, period_end) in periods.items():
            in_period = origins[
                (load.index[origins] >= period_start)
                & (load.index[origins + horizon - 1] <= period_end)
            ]
            if len(in_period) == 0:
                raise ValueError(f"{name} 没有 target 完整且无缺失的样本")
            splits[name] = in_period
    n_train = len(splits["train"])
    n_val = len(splits["validation"])
    if n_train < MIN_TRAIN_WINDOWS or n_val < MIN_VAL_WINDOWS:
        raise RuntimeError(
            f"完整窗口不足，停止组装。train={n_train} (至少 {MIN_TRAIN_WINDOWS})，"
            f"validation={n_val} (至少 {MIN_VAL_WINDOWS})。"
            "不会缩短历史窗口，也不会把验证/测试样本借给训练。"
        )

    train_mean, train_std = fit_train_scaling(values, train_end)
    scaled = ((values - train_mean) / train_std).astype("float64")
    calendar = calendar_features_for_project(load.index, config["feature_timezone"])
    if list(calendar.columns) != FUTURE_FEATURE_NAMES:
        raise ValueError(f"未来特征顺序被改变: {list(calendar.columns)}")
    cal_arr = calendar.to_numpy(dtype="float32")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, positions in splits.items():
        x_past, x_future, y = build_xy_arrays(scaled, cal_arr, positions, lookback, horizon)
        if x_past.shape[1:] != (lookback, 5) or x_future.shape[1:] != (horizon, 4) or y.shape[1:] != (horizon,):
            raise ValueError("样本形状与 (672,5)+(96,4)->(96,) 不一致。")
        if x_future.shape[-1] != 4:
            raise ValueError("未来已知输入特征数不是4。")
        if not (np.isfinite(x_past).all() and np.isfinite(x_future).all() and np.isfinite(y).all()):
            raise ValueError(f"{name} 样本含 NaN 或无穷值。")
        arrays[name] = {"X_past": x_past, "X_future_known": x_future, "y": y, "origins": positions}

    split_table = pd.DataFrame([
        {
            "split": name,
            "samples": len(pos),
            "first_target": str(load.index[pos[0]]),
            "last_target": str(load.index[pos[-1] + horizon - 1]),
        }
        for name, pos in splits.items()
    ])
    return DirectLoadBundle(
        config=config,
        load=load,
        audit=audit,
        lookback=lookback,
        horizon=horizon,
        train_end=train_end,
        val_end=val_end,
        train_mean=train_mean,
        train_std=train_std,
        past_feature_names=list(PAST_FEATURE_NAMES),
        future_feature_names=list(FUTURE_FEATURE_NAMES),
        split_table=split_table,
        arrays=arrays,
    )
