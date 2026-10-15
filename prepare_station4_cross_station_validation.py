"""Prepare station #4 and evaluate the frozen station-#3 correction zero-shot.

Confirmed conservative quality rules:
- every occurrence of a duplicated minute is invalidated;
- negative active values are invalidated until their physical meaning is known;
- invalid minutes are removed before strict 15/15 mean aggregation.

December test data are prepared/count-audited only and never scored.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_recent_residual_baselines import BETA_GRID, regression_metrics
from industrial_load_adapter import aggregate_1min_to_15min


INPUT_DIR = Path("input")
FILES = [INPUT_DIR / f"#4-{i}.csv" for i in range(1, 6)]
TIMEZONE = "Asia/Shanghai"
LOOKBACK = 672
HORIZON = 96
TARGETS = {
    "train": (pd.Timestamp("2024-08-08 00:00", tz=TIMEZONE), pd.Timestamp("2024-10-31 23:45", tz=TIMEZONE)),
    "validation": (pd.Timestamp("2024-11-01 00:00", tz=TIMEZONE), pd.Timestamp("2024-11-30 23:45", tz=TIMEZONE)),
    "test": (pd.Timestamp("2024-12-01 00:00", tz=TIMEZONE), pd.Timestamp("2024-12-31 23:45", tz=TIMEZONE)),
}
FROZEN_CROSS_STATION_BETA = 0.40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_and_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frames = []
    summaries = []
    hashes = {p.name: sha256(p) for p in FILES}
    for path in FILES:
        frame = pd.read_csv(path)
        if set(frame.columns) != {"a1", "ts", "active"}:
            raise ValueError(f"{path.name}: unexpected columns {list(frame.columns)}")
        if frame["a1"].isna().any() or frame["a1"].nunique(dropna=False) != 1 or str(frame["a1"].iloc[0]) != "#4":
            raise ValueError(f"{path.name}: station id is not one non-null #4")
        timestamps = pd.to_datetime(frame["ts"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        active = pd.to_numeric(frame["active"], errors="coerce")
        if timestamps.isna().any() or active.isna().any() or np.isinf(active.to_numpy()).any():
            raise ValueError(f"{path.name}: unparseable timestamp/load, NaN, or infinity")
        local = pd.DatetimeIndex(timestamps).tz_localize(TIMEZONE)
        duplicate_mask = local.duplicated(keep=False)
        negative_mask = active.to_numpy() < 0
        invalid_mask = duplicate_mask | negative_mask
        unique = local.unique().sort_values()
        grid = pd.date_range(unique.min(), unique.max(), freq="1min")
        summaries.append({
            "file": path.name, "station": "#4", "rows": len(frame),
            "start_ts": str(local.min()), "end_ts": str(local.max()),
            "duplicate_rows_invalidated": int(duplicate_mask.sum()),
            "duplicate_unique_minutes_invalidated": int(local[duplicate_mask].nunique()),
            "negative_rows_invalidated": int(negative_mask.sum()),
            "missing_minutes_before_quality_invalidation": int(len(grid.difference(unique))),
            "valid_rows_after_quality_rules": int((~invalid_mask).sum()),
        })
        valid = pd.DataFrame({
            "a1": "#4", "timestamp": local[~invalid_mask],
            "active": active.to_numpy(dtype="float64")[~invalid_mask], "source_file": path.name,
        })
        frames.append(valid)
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable")
    if combined["timestamp"].duplicated().any():
        raise RuntimeError("duplicates remain after conservative invalidation")
    return combined, pd.DataFrame(summaries), hashes


def aggregate(combined: pd.DataFrame) -> pd.DataFrame:
    series = pd.Series(combined["active"].to_numpy(), index=pd.DatetimeIndex(combined["timestamp"]), name="load_kw")
    values = aggregate_1min_to_15min(series)
    grid = pd.date_range(values.index.min(), values.index.max(), freq="15min", name="timestamp")
    counts = series.groupby(series.index.floor("15min")).size().reindex(grid, fill_value=0).astype("int64")
    values = values.reindex(grid)
    result = pd.DataFrame({
        "timestamp": grid, "a1": "#4", "load_kw": values.to_numpy(dtype="float64"),
        "minute_count": counts.to_numpy(), "is_complete": counts.to_numpy() == 15,
    })
    if result.loc[~result["is_complete"], "load_kw"].notna().any():
        raise RuntimeError("strict aggregation contract failed")
    return result


def origins_for_period(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[np.ndarray, dict]:
    index = pd.DatetimeIndex(frame["timestamp"])
    finite = np.isfinite(frame["load_kw"].to_numpy(dtype="float64"))
    theoretical = np.flatnonzero((index >= start) & (index <= end))
    boundary = theoretical[(theoretical + HORIZON > len(frame)) | (index[np.minimum(theoretical + HORIZON - 1, len(frame)-1)] > end)]
    eligible = np.setdiff1d(theoretical, boundary, assume_unique=True)
    history_bad = np.array([o < LOOKBACK or not finite[o-LOOKBACK:o].all() for o in eligible])
    target_bad = np.array([not finite[o:o+HORIZON].all() for o in eligible])
    valid = eligible[~history_bad & ~target_bad]
    audit = {
        "theoretical_origins": len(theoretical), "boundary_excluded": len(boundary),
        "history_missing_only": int((history_bad & ~target_bad).sum()),
        "target_missing_only": int((~history_bad & target_bad).sum()),
        "history_and_target_missing": int((history_bad & target_bad).sum()),
        "valid_samples": len(valid),
    }
    return valid, audit


def recent_4h(frame: pd.DataFrame, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = frame["load_kw"].to_numpy(dtype="float64")
    feature = np.full(len(origins), np.nan)
    available = np.zeros(len(origins), dtype=bool)
    for row, origin in enumerate(origins):
        current = raw[origin-16:origin]
        weekly = raw[origin-688:origin-672]
        if len(weekly) == 16 and np.isfinite(current).all() and np.isfinite(weekly).all():
            feature[row] = float(np.mean(current - weekly))
            available[row] = True
    return feature, available


def write_business_csvs(frame: pd.DataFrame, stamp: str) -> dict[str, Path]:
    indexed = frame.set_index("timestamp")
    names = {
        "train": "industrial_station4_train_15min_202408_202410.csv",
        "validation": "industrial_station4_validation_15min_202411.csv",
        "test": "industrial_station4_test_15min_202412.csv",
    }
    paths = {}
    for split, (start, end) in TARGETS.items():
        context_start = max(indexed.index.min(), start - pd.Timedelta(minutes=15 * LOOKBACK))
        part = indexed.loc[context_start:end].reset_index()
        part["is_target_period"] = part["timestamp"].between(start, end)
        path = INPUT_DIR / names[split]
        if path.exists():
            path = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        part.to_csv(path, index=False, encoding="utf-8-sig")
        paths[split] = path
    return paths


def main() -> None:
    run_at = datetime.now(timezone.utc)
    stamp = run_at.strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path("my_outputs") / f"station4_cross_station_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    combined, raw_summary, hashes_before = load_and_audit()
    aggregated = aggregate(combined)
    csv_paths = write_business_csvs(aggregated, stamp)
    raw = aggregated["load_kw"].to_numpy(dtype="float64")

    sample_rows = []
    split_origins = {}
    for split, period in TARGETS.items():
        origins, audit = origins_for_period(aggregated, *period)
        split_origins[split] = origins
        sample_rows.append({"split": split, **audit})
    pd.DataFrame(sample_rows).to_csv(out / "dataset_sample_counts.csv", index=False, encoding="utf-8-sig")

    train_origins = split_origins["train"]
    train_feature, train_available = recent_4h(aggregated, train_origins)
    train_actual = np.stack([raw[o:o+HORIZON] for o in train_origins])[train_available]
    train_last_week = np.stack([raw[o-672:o-576] for o in train_origins])[train_available]
    train_correction = train_feature[train_available, None]
    beta_rows = []
    for beta in BETA_GRID:
        beta_rows.append({"beta": float(beta), "train_origins": int(train_available.sum()), **regression_metrics(train_actual, train_last_week + beta * train_correction)})
    beta_search = pd.DataFrame(beta_rows)
    station4_beta = float(beta_search.loc[beta_search["MAE_kW"].idxmin(), "beta"])
    beta_search["selected_station4_train_only"] = beta_search["beta"].eq(station4_beta)
    beta_search.to_csv(out / "station4_beta_search_train.csv", index=False, encoding="utf-8-sig")

    val_origins = split_origins["validation"]
    val_feature, val_available = recent_4h(aggregated, val_origins)
    val_actual = np.stack([raw[o:o+HORIZON] for o in val_origins])[val_available]
    val_last_week = np.stack([raw[o-672:o-576] for o in val_origins])[val_available]
    val_correction = val_feature[val_available, None]
    predictions = {
        "Station4 Last-week baseline": val_last_week,
        "Station3 frozen beta=0.40 zero-shot": val_last_week + FROZEN_CROSS_STATION_BETA * val_correction,
        f"Station4 train-only beta={station4_beta:.2f} adapted upper bound": val_last_week + station4_beta * val_correction,
    }
    comparison = pd.DataFrame([
        {"method": method, "validation_origins": int(val_available.sum()), **regression_metrics(val_actual, prediction)}
        for method, prediction in predictions.items()
    ])
    comparison.to_csv(out / "station4_validation_comparison.csv", index=False, encoding="utf-8-sig")

    count_distribution = aggregated["minute_count"].value_counts().sort_index().to_dict()
    aggregation_summary = pd.DataFrame([{
        "station": "#4", "start_ts": str(aggregated["timestamp"].min()), "end_ts": str(aggregated["timestamp"].max()),
        "total_points": len(aggregated), "complete_points": int(aggregated["is_complete"].sum()),
        "incomplete_points": int((~aggregated["is_complete"]).sum()),
        "valid_rate": float(aggregated["is_complete"].mean()),
        "minute_count_distribution": json.dumps(count_distribution),
    }])
    raw_summary.to_csv(out / "raw_files_summary.csv", index=False, encoding="utf-8-sig")
    aggregation_summary.to_csv(out / "aggregation_15min_summary.csv", index=False, encoding="utf-8-sig")

    hashes_after = {p.name: sha256(p) for p in FILES}
    summary = {
        "kind": "station4_zero_shot_cross_station_validation",
        "executed_at_utc": run_at.isoformat(), "station": "#4",
        "quality_policy": {"duplicate_minute": "invalidate all occurrences", "negative_active": "invalidate minute"},
        "source_files_unchanged": hashes_before == hashes_after,
        "station3_frozen_beta": FROZEN_CROSS_STATION_BETA,
        "station4_train_only_beta": station4_beta,
        "station4_validation_origins_common": int(val_available.sum()),
        "test_samples_count_only": int(len(split_origins["test"])),
        "test_scored": False, "test_predictions_generated": False,
        "business_csvs": {key: str(path.resolve()) for key, path in csv_paths.items()},
        "output_dir": str(out.resolve()),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
