"""Prepare the #3 industrial load data for direct 24-hour forecasting.

The five source CSV files are read-only inputs.  They are concatenated before
15-minute aggregation so that file boundaries cannot create artificial bins.
This script does not import TensorFlow and does not train or score an LSTM.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_load_adapter import aggregate_1min_to_15min
from load15_direct_dataset import calendar_features_for_project


INPUT_DIR = Path("input")
SOURCE_FILES = [INPUT_DIR / f"#3-{i}.csv" for i in range(1, 6)]
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
TIMEZONE = "Asia/Shanghai"
RANGE_START = pd.Timestamp("2024-08-01 00:00", tz=TIMEZONE)
RANGE_END = pd.Timestamp("2024-12-31 23:45", tz=TIMEZONE)
LOOKBACK = 672
HORIZON = 96
TARGET_PERIODS = {
    "train": (pd.Timestamp("2024-08-08 00:00", tz=TIMEZONE), pd.Timestamp("2024-10-31 23:45", tz=TIMEZONE)),
    "validation": (pd.Timestamp("2024-11-01 00:00", tz=TIMEZONE), pd.Timestamp("2024-11-30 23:45", tz=TIMEZONE)),
    "test": (pd.Timestamp("2024-12-01 00:00", tz=TIMEZONE), pd.Timestamp("2024-12-31 23:45", tz=TIMEZONE)),
}


@dataclass
class FileAudit:
    file: str
    a1: str
    rows: int
    start_ts: str
    end_ts: str
    active_dtype: str
    active_nan: int
    active_inf: int
    active_min: float
    active_max: float
    active_median: float
    active_negative: int
    ts_parse_failed: int
    duplicate_ts: int
    theoretical_minutes: int
    missing_minutes: int
    longest_missing_start: str | None
    longest_missing_end: str | None
    longest_missing_minutes: int
    overlap_with_previous_minutes: int
    gap_minutes_from_previous: int | None
    shared_ts_with_previous: int
    starts_mid_15min_bucket: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def missing_runs(index: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    if len(index) == 0:
        return []
    ordered = index.sort_values()
    groups = np.cumsum(np.r_[True, np.diff(ordered.asi8) != pd.Timedelta(minutes=1).value])
    runs = []
    for group in np.unique(groups):
        values = ordered[groups == group]
        runs.append((values[0], values[-1], len(values)))
    return runs


def longest_boolean_run(index: pd.DatetimeIndex, mask: np.ndarray) -> tuple[str | None, str | None, int]:
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return None, None, 0
    chunks = np.split(positions, np.where(np.diff(positions) > 1)[0] + 1)
    chunk = max(chunks, key=len)
    return str(index[chunk[0]]), str(index[chunk[-1]]), len(chunk)


def read_and_validate_sources() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    frames: list[tuple[Path, pd.DataFrame]] = []
    source_hashes = {}
    for path in SOURCE_FILES:
        if not path.is_file():
            raise FileNotFoundError(path.resolve())
        source_hashes[path.name] = sha256(path)
        frame = pd.read_csv(path)
        expected = {"a1", "ts", "active"}
        if set(frame.columns) != expected:
            raise ValueError(f"{path.name}: expected exactly {sorted(expected)}, got {list(frame.columns)}")
        if frame["a1"].isna().any() or frame["a1"].nunique(dropna=False) != 1:
            raise ValueError(f"{path.name}: a1 must contain one non-null station id")
        parsed = pd.to_datetime(frame["ts"], format=TIMESTAMP_FORMAT, errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"{path.name}: {int(parsed.isna().sum())} timestamps failed parsing")
        frame = frame.copy()
        frame["timestamp"] = pd.DatetimeIndex(parsed).tz_localize(TIMEZONE)
        numeric = pd.to_numeric(frame["active"], errors="coerce")
        if numeric.isna().sum() > frame["active"].isna().sum():
            raise ValueError(f"{path.name}: active contains non-numeric values")
        frame["active"] = numeric.astype("float64")
        frames.append((path, frame))

    station_ids = {str(frame["a1"].iloc[0]) for _, frame in frames}
    if len(station_ids) != 1:
        raise ValueError(f"Station ids differ across source files: {sorted(station_ids)}")

    frames.sort(key=lambda item: item[1]["timestamp"].min())
    audits = []
    previous: pd.DataFrame | None = None
    for path, frame in frames:
        ts = pd.DatetimeIndex(frame["timestamp"])
        start, end = ts.min(), ts.max()
        grid = pd.date_range(start, end, freq="1min")
        missing = grid.difference(ts.unique())
        runs = missing_runs(missing)
        longest = max(runs, key=lambda value: value[2]) if runs else (None, None, 0)
        overlap = 0
        gap = None
        shared = 0
        if previous is not None:
            previous_ts = pd.DatetimeIndex(previous["timestamp"])
            previous_end = previous_ts.max()
            if start <= previous_end:
                overlap = int((previous_end - start) / pd.Timedelta(minutes=1)) + 1
            else:
                gap = max(0, int((start - previous_end) / pd.Timedelta(minutes=1)) - 1)
            shared = len(ts.unique().intersection(previous_ts.unique()))
        values = frame["active"].to_numpy(dtype="float64")
        audits.append(FileAudit(
            file=path.name, a1=str(frame["a1"].iloc[0]), rows=len(frame), start_ts=str(start), end_ts=str(end),
            active_dtype=str(frame["active"].dtype), active_nan=int(np.isnan(values).sum()),
            active_inf=int(np.isinf(values).sum()), active_min=float(np.nanmin(values)),
            active_max=float(np.nanmax(values)), active_median=float(np.nanmedian(values)),
            active_negative=int((values < 0).sum()), ts_parse_failed=0,
            duplicate_ts=int(ts.duplicated().sum()), theoretical_minutes=len(grid),
            missing_minutes=len(missing), longest_missing_start=None if longest[0] is None else str(longest[0]),
            longest_missing_end=None if longest[1] is None else str(longest[1]),
            longest_missing_minutes=int(longest[2]), overlap_with_previous_minutes=overlap,
            gap_minutes_from_previous=gap, shared_ts_with_previous=shared,
            starts_mid_15min_bucket=bool(start.minute % 15),
        ))
        previous = frame

    combined = pd.concat([frame.assign(source_file=path.name) for path, frame in frames], ignore_index=True)
    combined = combined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    duplicates = combined["timestamp"].duplicated(keep=False)
    if duplicates.any():
        examples = combined.loc[duplicates, ["source_file", "timestamp"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate timestamps found after concatenation; no deduplication performed: {examples}")
    return combined, pd.DataFrame([asdict(row) for row in audits]), source_hashes


def aggregate_combined(combined: pd.DataFrame) -> pd.DataFrame:
    clipped = combined[(combined["timestamp"] >= RANGE_START) & (combined["timestamp"] <= RANGE_END + pd.Timedelta(minutes=14))]
    series = pd.Series(clipped["active"].to_numpy(), index=pd.DatetimeIndex(clipped["timestamp"]), name="load_kw")
    # Reuse the project's strict 15/15 mean aggregation implementation.
    strict_values = aggregate_1min_to_15min(series)
    full_grid = pd.date_range(max(RANGE_START, strict_values.index.min()), min(RANGE_END, strict_values.index.max()), freq="15min", name="timestamp")
    bucket = series.index.floor("15min")
    counts = series.groupby(bucket).size().reindex(full_grid, fill_value=0).astype("int64")
    values = strict_values.reindex(full_grid)
    result = pd.DataFrame({
        "a1": combined["a1"].iloc[0],
        "timestamp": full_grid,
        "load_kw": values.to_numpy(dtype="float64"),
        "minute_count": counts.to_numpy(),
        "is_complete": (counts.to_numpy() == 15),
    })
    return result


def make_business_csvs(aggregated: pd.DataFrame, stamp: str) -> dict[str, Path]:
    indexed = aggregated.set_index("timestamp")
    paths = {}
    names = {
        "train": "industrial_train_15min_202408_202410.csv",
        "validation": "industrial_validation_15min_202411.csv",
        "test": "industrial_test_15min_202412.csv",
    }
    for split, (target_start, target_end) in TARGET_PERIODS.items():
        context_start = target_start - LOOKBACK * pd.Timedelta(minutes=15)
        data_start = max(indexed.index.min(), context_start)
        part = indexed.loc[data_start:target_end].reset_index()
        part["is_target_period"] = part["timestamp"].between(target_start, target_end)
        path = INPUT_DIR / names[split]
        if path.exists():
            path = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        part.to_csv(path, index=False, encoding="utf-8-sig")
        paths[split] = path
    return paths


def build_samples(aggregated: pd.DataFrame, train_mean: float, train_std: float) -> tuple[dict, pd.DataFrame]:
    frame = aggregated.set_index("timestamp")
    values = frame["load_kw"].to_numpy(dtype="float64")
    finite = np.isfinite(values)
    scaled = (values - train_mean) / train_std
    calendar = calendar_features_for_project(frame.index.tz_convert("UTC"), TIMEZONE).to_numpy(dtype="float32")
    arrays = {}
    records = []
    for split, (target_start, target_end) in TARGET_PERIODS.items():
        theoretical = np.flatnonzero((frame.index >= target_start) & (frame.index <= target_end))
        boundary = theoretical[(theoretical + HORIZON > len(frame)) | (frame.index[np.minimum(theoretical + HORIZON - 1, len(frame)-1)] > target_end)]
        eligible = np.setdiff1d(theoretical, boundary, assume_unique=True)
        history_bad = np.array([o < LOOKBACK or not finite[o-LOOKBACK:o].all() for o in eligible])
        target_bad = np.array([not finite[o:o+HORIZON].all() for o in eligible])
        valid = eligible[~history_bad & ~target_bad]
        xp = np.empty((len(valid), LOOKBACK, 5), dtype="float32")
        xf = np.empty((len(valid), HORIZON, 4), dtype="float32")
        y = np.empty((len(valid), HORIZON), dtype="float32")
        y_raw = np.empty((len(valid), HORIZON), dtype="float64")
        for row, origin in enumerate(valid):
            xp[row, :, 0] = scaled[origin-LOOKBACK:origin]
            xp[row, :, 1:] = calendar[origin-LOOKBACK:origin]
            xf[row] = calendar[origin:origin+HORIZON]
            y[row] = scaled[origin:origin+HORIZON]
            y_raw[row] = values[origin:origin+HORIZON]
        arrays[split] = {"X_past": xp, "X_future_known": xf, "y": y, "y_raw": y_raw, "origins": valid}
        records.append({
            "split": split, "theoretical_origins": len(theoretical), "target_boundary_excluded": len(boundary),
            "eligible_after_boundary": len(eligible), "history_missing_only": int((history_bad & ~target_bad).sum()),
            "target_missing_only": int((~history_bad & target_bad).sum()),
            "history_and_target_missing": int((history_bad & target_bad).sum()),
            "history_missing_total": int(history_bad.sum()), "target_missing_total": int(target_bad.sum()),
            "valid_samples": len(valid), "X_past_shape": str(xp.shape),
            "X_future_known_shape": str(xf.shape), "y_shape": str(y.shape),
            "nan_count": int(np.isnan(xp).sum()+np.isnan(xf).sum()+np.isnan(y).sum()),
            "posinf_count": int(np.isposinf(xp).sum()+np.isposinf(xf).sum()+np.isposinf(y).sum()),
            "neginf_count": int(np.isneginf(xp).sum()+np.isneginf(xf).sum()+np.isneginf(y).sum()),
        })
    return arrays, pd.DataFrame(records)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    nonzero = actual != 0
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape_percent": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100),
        "mape_zero_targets_excluded": int((~nonzero).sum()),
        "evaluated_pairs": int(actual.size),
    }


def validation_baselines(arrays: dict, aggregated: pd.DataFrame) -> pd.DataFrame:
    values = aggregated["load_kw"].to_numpy(dtype="float64")
    origins = arrays["validation"]["origins"]
    actual = arrays["validation"]["y_raw"]
    predictions = {
        "yesterday": np.stack([values[o-96:o] for o in origins]),
        "last_week": np.stack([values[o-672:o-576] for o in origins]),
    }
    rows = []
    for name, prediction in predictions.items():
        rows.append({"baseline": name, "scope": "all_horizons", "horizon_step": "all", **metrics(actual, prediction)})
        for step in range(HORIZON):
            rows.append({"baseline": name, "scope": "by_horizon", "horizon_step": step + 1, **metrics(actual[:, step], prediction[:, step])})
    return pd.DataFrame(rows)


def main() -> None:
    run_at = datetime.now(timezone.utc)
    stamp = run_at.strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = Path("my_outputs") / f"data_refresh_aug_dec_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    combined, raw_summary, source_hashes_before = read_and_validate_sources()
    aggregated = aggregate_combined(combined)
    csv_paths = make_business_csvs(aggregated, stamp)

    train_end = TARGET_PERIODS["train"][1]
    train_values = aggregated.loc[aggregated["timestamp"] <= train_end, "load_kw"].dropna().to_numpy(dtype="float64")
    train_mean = float(train_values.mean())
    train_std = float(train_values.std(ddof=0))
    arrays, sample_counts = build_samples(aggregated, train_mean, train_std)
    baselines = validation_baselines(arrays, aggregated)

    ts = pd.DatetimeIndex(combined["timestamp"])
    minute_grid = pd.date_range(ts.min(), ts.max(), freq="1min")
    minute_missing = minute_grid.difference(ts)
    minute_longest = max(missing_runs(minute_missing), key=lambda value: value[2])
    minute_quality = pd.DataFrame([{
        "rows": len(combined), "start_ts": str(ts.min()), "end_ts": str(ts.max()),
        "theoretical_minutes": len(minute_grid), "missing_minutes": len(minute_missing),
        "duplicate_minutes": int(ts.duplicated().sum()), "active_nan": int(combined["active"].isna().sum()),
        "active_inf": int(np.isinf(combined["active"].to_numpy()).sum()),
        "longest_missing_start": str(minute_longest[0]), "longest_missing_end": str(minute_longest[1]),
        "longest_missing_minutes": minute_longest[2],
    }])
    valid = aggregated["is_complete"].to_numpy()
    invalid_run = longest_boolean_run(pd.DatetimeIndex(aggregated["timestamp"]), ~valid)
    valid_run = longest_boolean_run(pd.DatetimeIndex(aggregated["timestamp"]), valid)
    count_distribution = aggregated["minute_count"].value_counts().sort_index().to_dict()
    aggregation_summary = pd.DataFrame([{
        "start_ts": str(aggregated["timestamp"].min()), "end_ts": str(aggregated["timestamp"].max()),
        "total_points": len(aggregated), "complete_points": int(valid.sum()), "incomplete_points": int((~valid).sum()),
        "valid_rate": float(valid.mean()), "longest_incomplete_start": invalid_run[0],
        "longest_incomplete_end": invalid_run[1], "longest_incomplete_points": invalid_run[2],
        "longest_complete_start": valid_run[0], "longest_complete_end": valid_run[1],
        "longest_complete_points": valid_run[2], "minute_count_distribution": json.dumps(count_distribution),
    }])
    scaler_summary = pd.DataFrame([{
        "fit_start": str(aggregated["timestamp"].min()), "fit_end": str(train_end),
        "fit_valid_points": len(train_values), "train_mean": train_mean, "train_std_ddof0": train_std,
        "unit": "kW",
    }])
    config_snapshot = {
        "station": str(combined["a1"].iloc[0]), "timezone": TIMEZONE, "unit": "kW", "unit_verified": True,
        "timestamp_format": TIMESTAMP_FORMAT, "range_start": str(RANGE_START), "range_end": str(RANGE_END),
        "lookback": LOOKBACK, "horizon": HORIZON, "origin_stride_steps": 1,
        "target_periods": {key: [str(value[0]), str(value[1])] for key, value in TARGET_PERIODS.items()},
        "past_features": ["load_scaled", "time_sin", "time_cos", "dow_sin", "dow_cos"],
        "future_known_features": ["time_sin", "time_cos", "dow_sin", "dow_cos"],
        "source_files": [str(path) for path in SOURCE_FILES], "business_csvs": {key: str(path) for key, path in csv_paths.items()},
    }
    source_hashes_after = {path.name: sha256(path) for path in SOURCE_FILES}
    run_summary = {
        "kind": "industrial_data_refresh_aug_dec", "executed_at_utc": run_at.isoformat(),
        "python": __import__("sys").executable, "lstm_trained": False, "test_metrics_computed": False,
        "source_hashes_before": source_hashes_before, "source_hashes_after": source_hashes_after,
        "source_files_unchanged": source_hashes_before == source_hashes_after,
        "output_dir": str(output_dir.resolve()), "business_csvs": {key: str(path.resolve()) for key, path in csv_paths.items()},
    }

    raw_summary.to_csv(output_dir / "raw_files_summary.csv", index=False, encoding="utf-8-sig")
    minute_quality.to_csv(output_dir / "minute_quality_summary.csv", index=False, encoding="utf-8-sig")
    aggregation_summary.to_csv(output_dir / "aggregation_15min_summary.csv", index=False, encoding="utf-8-sig")
    sample_counts.to_csv(output_dir / "dataset_sample_counts.csv", index=False, encoding="utf-8-sig")
    scaler_summary.to_csv(output_dir / "scaler_summary.csv", index=False, encoding="utf-8-sig")
    baselines.to_csv(output_dir / "validation_baselines.csv", index=False, encoding="utf-8-sig")
    (output_dir / "config_snapshot.json").write_text(json.dumps(config_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_dir": str(output_dir.resolve()), "business_csvs": {key: str(path.resolve()) for key, path in csv_paths.items()},
        "samples": sample_counts[["split", "valid_samples"]].to_dict("records"),
        "validation_baselines": baselines[baselines["scope"] == "all_horizons"].to_dict("records"),
        "train_mean": train_mean, "train_std": train_std,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
