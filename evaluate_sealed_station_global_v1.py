"""One-shot evaluator for a sealed, unseen station using frozen global_v1.

This program never searches parameters. It reads exactly one explicitly named
15-minute CSV and evaluates beta=0.35 against Last-week on a declared target
period. The CSV must contain enough causal context before that period.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_recent_residual_baselines import regression_metrics


WEEK, RECENT, HORIZON, BETA = 672, 16, 96, 0.35


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boolean_column(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.lower().eq("true").to_numpy()


def load_declared_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"timestamp", "load_kw"}
    if missing := required - set(frame.columns):
        raise KeyError(f"Missing columns: {sorted(missing)}")
    index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], errors="raise", utc=True))
    if index.has_duplicates:
        raise ValueError("Duplicate 15-minute timestamps are forbidden")
    if not index.equals(pd.date_range(index[0], index[-1], freq="15min")):
        raise ValueError("Input must be a continuous 15-minute timestamp grid")
    load = pd.to_numeric(frame["load_kw"], errors="coerce").to_numpy(float)
    if "is_complete" in frame:
        load[~boolean_column(frame["is_complete"])] = np.nan
    return pd.DataFrame({"load": load}, index=index)


def build_examples(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    load = frame["load"].to_numpy(float)
    origins, actual, last_week, correction = [], [], [], []
    for origin in range(WEEK + RECENT, len(frame) - HORIZON + 1):
        first, last = frame.index[origin], frame.index[origin + HORIZON - 1]
        if first < start or last > end:
            continue
        pieces = [load[origin:origin + HORIZON], load[origin-WEEK:origin-WEEK+HORIZON],
                  load[origin-RECENT:origin], load[origin-WEEK-RECENT:origin-WEEK]]
        if not np.isfinite(np.concatenate(pieces)).all():
            continue
        recent = pieces[2] - pieces[3]
        origins.append(origin)
        actual.append(pieces[0])
        last_week.append(pieces[1])
        correction.append(float(recent.mean()))
    if not origins:
        raise ValueError("No usable origins in the declared target period")
    return {"origins": np.asarray(origins), "actual": np.stack(actual),
            "last_week": np.stack(last_week), "correction": np.asarray(correction)[:, None]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--target-start", required=True, help="ISO timestamp with timezone")
    parser.add_argument("--target-end", required=True, help="ISO timestamp with timezone")
    parser.add_argument("--station-label", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    path = Path(args.data_file)
    start, end = pd.Timestamp(args.target_start).tz_convert("UTC"), pd.Timestamp(args.target_end).tz_convert("UTC")
    if end <= start:
        raise ValueError("target-end must be later than target-start")
    frame = load_declared_file(path)
    data = build_examples(frame, start, end)
    prediction = data["last_week"] + BETA * data["correction"]

    base = regression_metrics(data["actual"], data["last_week"])
    candidate = regression_metrics(data["actual"], prediction)
    overall = pd.DataFrame([
        {"method": "Last-week baseline", **base},
        {"method": "frozen global_v1", **candidate},
    ])
    hrows = []
    for method, pred in (("Last-week baseline", data["last_week"]), ("frozen global_v1", prediction)):
        for h in range(HORIZON):
            hrows.append({"method": method, "horizon": h + 1, "minutes_ahead": (h + 1) * 15,
                          **regression_metrics(data["actual"][:, h], pred[:, h])})

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path(args.output_dir) if args.output_dir else Path("my_outputs") / f"sealed_station_global_v1_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    overall.to_csv(out / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(hrows).to_csv(out / "metrics_by_horizon.csv", index=False, encoding="utf-8-sig")

    mae_improvement = float((1 - candidate["MAE_kW"] / base["MAE_kW"]) * 100)
    rmse_improvement = float((1 - candidate["RMSE_kW"] / base["RMSE_kW"]) * 100)
    # Pre-registered decision rule: both primary MAE and safety RMSE must improve.
    passed = bool(candidate["MAE_kW"] < base["MAE_kW"] and candidate["RMSE_kW"] <= base["RMSE_kW"])
    first_origin, last_origin = data["origins"][0], data["origins"][-1]
    summary = {
        "kind": "sealed_unseen_station_one_shot_evaluation",
        "station_label": args.station_label,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(path.resolve()), "input_sha256": sha256(path),
        "declared_target_start_utc": start.isoformat(), "declared_target_end_utc": end.isoformat(),
        "usable_origins": int(len(data["origins"])),
        "first_origin_utc": frame.index[first_origin].isoformat(),
        "last_origin_utc": frame.index[last_origin].isoformat(),
        "beta": BETA, "parameters_searched": False,
        "MAE_improvement_pct": mae_improvement, "RMSE_improvement_pct": rmse_improvement,
        "pass_rule": "global_v1 MAE < Last-week MAE and global_v1 RMSE <= Last-week RMSE",
        "passed": passed,
        "causality": "recent residual sources end at origin-1; week-lag sources precede origin",
        "output_dir": str(out.resolve()),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
