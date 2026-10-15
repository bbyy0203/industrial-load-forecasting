"""Prepare station #1 August-November data for the pre-registered blind test.

Quality policy was fixed before scoring: all occurrences of duplicate minutes
and all negative readings are invalid; 15-minute means require exactly 15
valid minute values. December (#1-1.csv) is not an input to this program.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_load_adapter import aggregate_1min_to_15min


FILES = [Path("input") / f"#1-{n}.csv" for n in (2, 3, 4, 5)]
OUTPUT = Path("input/industrial_station1_blind_15min_202408_202411.csv")
TIMEZONE = "Asia/Shanghai"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    run_at = datetime.now(timezone.utc)
    out = Path("my_outputs") / f"station1_blind_preparation_{run_at.strftime('%Y%m%dT%H%M%S_%fZ')}"
    out.mkdir(parents=True, exist_ok=False)
    hashes_before = {p.name: sha256(p) for p in FILES}
    frames, audit_rows = [], []

    for path in FILES:
        raw = pd.read_csv(path)
        if set(raw.columns) != {"a1", "ts", "active"}:
            raise ValueError(f"{path.name}: unexpected columns {list(raw.columns)}")
        if raw["a1"].isna().any() or set(raw["a1"].astype(str).unique()) != {"#1"}:
            raise ValueError(f"{path.name}: station identifier is not exclusively #1")
        timestamps = pd.to_datetime(raw["ts"], errors="coerce")
        active = pd.to_numeric(raw["active"], errors="coerce")
        bad_parse = timestamps.isna().to_numpy() | ~np.isfinite(active.to_numpy(float))
        if bad_parse.any():
            raise ValueError(f"{path.name}: {int(bad_parse.sum())} unparseable/non-finite rows")
        index = pd.DatetimeIndex(timestamps).tz_localize(TIMEZONE)
        duplicate = index.duplicated(keep=False)
        negative = active.to_numpy(float) < 0
        invalid = duplicate | negative
        unique = index.unique().sort_values()
        grid = pd.date_range(unique.min(), unique.max(), freq="1min")
        audit_rows.append({
            "file": path.name, "rows": len(raw), "start": str(index.min()), "end": str(index.max()),
            "duplicate_rows_invalidated": int(duplicate.sum()),
            "duplicate_minutes_invalidated": int(index[duplicate].nunique()),
            "negative_rows_invalidated": int(negative.sum()),
            "missing_minutes_before_invalidation": int(len(grid.difference(unique))),
            "valid_rows": int((~invalid).sum()),
        })
        frames.append(pd.DataFrame({"timestamp": index[~invalid], "load_kw": active.to_numpy(float)[~invalid]}))

    minute = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable")
    if minute["timestamp"].duplicated().any():
        raise ValueError("Cross-file duplicate timestamps remain")
    series = pd.Series(minute["load_kw"].to_numpy(float), index=pd.DatetimeIndex(minute["timestamp"]), name="load_kw")
    aggregated = aggregate_1min_to_15min(series)
    grid15 = pd.date_range(aggregated.index.min(), aggregated.index.max(), freq="15min", name="timestamp")
    counts = series.groupby(series.index.floor("15min")).size().reindex(grid15, fill_value=0).astype(int)
    aggregated = aggregated.reindex(grid15)
    complete = counts.eq(15)
    result = pd.DataFrame({"timestamp": grid15, "a1": "#1", "load_kw": aggregated.to_numpy(float),
                           "minute_count": counts.to_numpy(), "is_complete": complete.to_numpy()})
    if result.loc[~result["is_complete"], "load_kw"].notna().any():
        raise AssertionError("Incomplete buckets must have NaN load")
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    pd.DataFrame(audit_rows).to_csv(out / "raw_files_summary.csv", index=False, encoding="utf-8-sig")
    count_dist = {str(k): int(v) for k, v in counts.value_counts().sort_index().items()}
    aggregate_summary = {
        "station": "#1", "start": str(grid15.min()), "end": str(grid15.max()),
        "total_15min_points": len(result), "complete_points": int(complete.sum()),
        "incomplete_points": int((~complete).sum()), "complete_rate": float(complete.mean()),
        "minute_count_distribution": count_dist,
    }
    (out / "aggregation_summary.json").write_text(json.dumps(aggregate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    hashes_after = {p.name: sha256(p) for p in FILES}
    summary = {
        "kind": "station1_blind_data_preparation", "executed_at_utc": run_at.isoformat(),
        "source_files": [str(p) for p in FILES], "source_sha256": hashes_before,
        "source_files_unchanged": hashes_before == hashes_after,
        "quality_policy": {"duplicate_minute": "invalidate all occurrences", "negative_load": "invalidate"},
        "aggregation": "strict 15/15 minute arithmetic mean",
        "blind_target_period": ["2024-11-01 00:00:00+08:00", "2024-11-30 23:45:00+08:00"],
        "december_file_used": False, "parameters_searched": False,
        "output_file": str(OUTPUT.resolve()), "output_sha256": sha256(OUTPUT),
        "output_dir": str(out.resolve()), **aggregate_summary,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
