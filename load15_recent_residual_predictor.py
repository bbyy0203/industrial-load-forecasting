"""Production-style causal predictor based on last week plus recent residual.

Forecast definition:
    prediction = last_week_future + 0.40 * mean(recent 4h residual)
    residual_s = load_s - load_{s-7d}

Only timestamps strictly earlier than the prediction origin are read.  If the
recent residual window is incomplete, prediction safely falls back to the
last-week baseline.  Missing last-week baseline values are a hard error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from load15_direct_dataset import load_project_config, read_target_split_frames


HORIZON = 96
WEEK_STEPS = 672
RECENT_STEPS = 16
DEFAULT_BETA = 0.40
DEFAULT_MODEL_CONFIG = Path("model_checkpoints/recent_residual_v1/model.json")


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    metadata: dict


def normalize_origin(origin: str | pd.Timestamp, timezone_name: str) -> pd.Timestamp:
    value = pd.Timestamp(origin)
    if value.tzinfo is None:
        value = value.tz_localize(timezone_name)
    else:
        value = value.tz_convert(timezone_name)
    if value != value.floor("15min"):
        raise ValueError("origin 必须对齐到15分钟网格")
    return value.tz_convert("UTC")


def forecast_from_series(
    load: pd.Series,
    origin: str | pd.Timestamp,
    *,
    timezone_name: str = "Asia/Shanghai",
    beta: float = DEFAULT_BETA,
) -> ForecastResult:
    if not isinstance(load.index, pd.DatetimeIndex) or load.index.tz is None:
        raise ValueError("load 必须使用带时区的 DatetimeIndex")
    if load.index.has_duplicates:
        raise ValueError("load 时间索引存在重复")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta 必须是非负有限数")

    series = pd.to_numeric(load, errors="coerce").astype("float64").sort_index()
    series.index = series.index.tz_convert("UTC")
    origin_utc = normalize_origin(origin, timezone_name)
    target_index = pd.date_range(origin_utc, periods=HORIZON, freq="15min")
    last_week_index = target_index - pd.Timedelta(days=7)
    last_week = series.reindex(last_week_index).to_numpy(dtype="float64")
    if not np.isfinite(last_week).all():
        missing = last_week_index[~np.isfinite(last_week)]
        raise ValueError(f"上周同期96点不完整，无法预测；缺失点数={len(missing)}，首个={missing[0]}")

    recent_index = pd.date_range(end=origin_utc - pd.Timedelta(minutes=15), periods=RECENT_STEPS, freq="15min")
    recent_week_index = recent_index - pd.Timedelta(days=7)
    recent_load = series.reindex(recent_index).to_numpy(dtype="float64")
    recent_week_load = series.reindex(recent_week_index).to_numpy(dtype="float64")
    recent_complete = bool(np.isfinite(recent_load).all() and np.isfinite(recent_week_load).all())
    if recent_complete:
        recent_residual_mean_kw = float(np.mean(recent_load - recent_week_load))
        applied_correction_kw = float(beta * recent_residual_mean_kw)
        mode = "last_week_plus_recent_4h_residual"
        fallback_reason = None
    else:
        recent_residual_mean_kw = None
        applied_correction_kw = 0.0
        mode = "last_week_fallback"
        fallback_reason = "recent_4h_residual_incomplete"

    # Every source timestamp must be strictly earlier than the origin.
    latest_source = max(last_week_index.max(), recent_index.max(), recent_week_index.max())
    if not latest_source < origin_utc:
        raise AssertionError("预测读取了 origin 之后的数据")

    predicted = last_week + applied_correction_kw
    forecast = pd.DataFrame({
        "timestamp": target_index.tz_convert(timezone_name).astype(str),
        "horizon": np.arange(1, HORIZON + 1),
        "last_week_baseline_kw": last_week,
        "applied_correction_kw": applied_correction_kw,
        "predicted_load_kw": predicted,
    })
    metadata = {
        "model_name": f"last_week_plus_{beta:.2f}_recent_4h_residual",
        "model_kind": "causal_statistical_forecaster",
        "origin": str(origin_utc.tz_convert(timezone_name)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone": timezone_name,
        "horizon_steps": HORIZON,
        "sampling_minutes": 15,
        "beta": float(beta),
        "coefficient_provenance": "train-only grid search; confirmed by embargoed walk-forward validation",
        "mode": mode,
        "fallback_reason": fallback_reason,
        "recent_residual_mean_kw": recent_residual_mean_kw,
        "applied_correction_kw": applied_correction_kw,
        "last_week_source_start": str(last_week_index.min().tz_convert(timezone_name)),
        "last_week_source_end": str(last_week_index.max().tz_convert(timezone_name)),
        "recent_source_start": str(recent_index.min().tz_convert(timezone_name)),
        "recent_source_end": str(recent_index.max().tz_convert(timezone_name)),
        "recent_week_source_start": str(recent_week_index.min().tz_convert(timezone_name)),
        "recent_week_source_end": str(recent_week_index.max().tz_convert(timezone_name)),
        "all_sources_strictly_before_origin": True,
        "future_actual_load_read": False,
    }
    return ForecastResult(forecast=forecast, metadata=metadata)


def forecast_from_project(
    origin: str | pd.Timestamp,
    config_path: str | Path = "my_load15_config.json",
    *,
    beta: float | None = None,
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG,
) -> ForecastResult:
    config = load_project_config(config_path)
    if config["dataset_kind"] != "industrial_split_csv":
        raise ValueError("当前正式 recent-residual predictor 要求 industrial_split_csv 数据入口")
    frame, _ = read_target_split_frames(config)
    load = frame[config["target_column"]]
    model_config = json.loads(Path(model_config_path).read_text(encoding="utf-8"))
    expected_contract = {
        "recent_residual_steps": RECENT_STEPS,
        "week_lag_steps": WEEK_STEPS,
        "horizon_steps": HORIZON,
        "sampling_minutes": 15,
        "timezone": config["feature_timezone"],
        "load_unit": config["load_unit"],
    }
    mismatches = {
        key: (model_config.get(key), expected)
        for key, expected in expected_contract.items()
        if model_config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"模型配置与数据/代码接口不一致: {mismatches}")
    configured_beta = float(model_config["beta"])
    selected_beta = configured_beta if beta is None else float(beta)
    result = forecast_from_series(load, origin, timezone_name=config["feature_timezone"], beta=selected_beta)
    result.metadata["model_name"] = model_config["model_name"]
    result.metadata["model_config_path"] = str(Path(model_config_path).resolve())
    result.metadata["beta_override"] = beta is not None
    result.metadata["model_contract_validated"] = True
    return result


def save_forecast(result: ForecastResult, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    result.forecast.to_csv(output / "forecast.csv", index=False, encoding="utf-8-sig")
    (output / "metadata.json").write_text(json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
