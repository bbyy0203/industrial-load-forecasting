"""Diagnose station-scale invariance and leave-one-station-out transfer.

Only the configured train and validation CSVs for stations #3 and #4 are
opened. Test CSVs and any third-station data are deliberately not discovered.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_recent_residual_baselines import BETA_GRID, regression_metrics


STATIONS = {
    "station3": "my_load15_config.json",
    "station4": "station4_load15_config.json",
}
WEEK = 672
RECENT = 16
HORIZON = 96


def read_split(config: dict, split: str) -> pd.DataFrame:
    """Read one business split without touching any other configured file."""
    path = Path(config[f"{split}_file"])
    frame = pd.read_csv(path)
    required = {"timestamp", config["target_column"], "is_complete", "is_target_period"}
    if missing := required - set(frame.columns):
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    values = pd.to_numeric(frame[config["target_column"]], errors="coerce").to_numpy(float)
    complete = frame["is_complete"].astype(str).str.lower().eq("true").to_numpy()
    target = frame["is_target_period"].astype(str).str.lower().eq("true").to_numpy()
    if index.has_duplicates or not index.equals(pd.date_range(index[0], index[-1], freq="15min")):
        raise ValueError(f"{path} is not a unique continuous 15-minute grid")
    values[~complete] = np.nan
    return pd.DataFrame({"load": values, "target": target}, index=index)


def examples(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    load = frame["load"].to_numpy(float)
    target = frame["target"].to_numpy(bool)
    rows = []
    for origin in range(WEEK + RECENT, len(frame) - HORIZON + 1):
        required = np.r_[
            load[origin:origin + HORIZON],
            load[origin - WEEK:origin - WEEK + HORIZON],
            load[origin - RECENT:origin],
            load[origin - WEEK - RECENT:origin - WEEK],
        ]
        if target[origin:origin + HORIZON].all() and np.isfinite(required).all():
            residual = load[origin - RECENT:origin] - load[origin - WEEK - RECENT:origin - WEEK]
            rows.append((origin, load[origin:origin + HORIZON],
                         load[origin - WEEK:origin - WEEK + HORIZON], residual.mean()))
    if not rows:
        raise ValueError("No usable causal origins")
    return {
        "origins": np.asarray([r[0] for r in rows], int),
        "actual": np.stack([r[1] for r in rows]),
        "last_week": np.stack([r[2] for r in rows]),
        "correction": np.asarray([r[3] for r in rows], float)[:, None],
    }


def scales(train_frame: pd.DataFrame) -> dict[str, float]:
    x = train_frame.loc[train_frame["target"], "load"].dropna().to_numpy(float)
    result = {
        "median_abs_load": float(np.median(np.abs(x))),
        "mean_abs_load": float(np.mean(np.abs(x))),
        "load_std": float(np.std(x, ddof=0)),
        "load_iqr": float(np.quantile(x, .75) - np.quantile(x, .25)),
    }
    if any(not np.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError(f"Invalid train-only station scale: {result}")
    return result


def main() -> None:
    run_at = datetime.now(timezone.utc)
    out = Path("my_outputs") / f"scale_independent_transfer_{run_at.strftime('%Y%m%dT%H%M%S_%fZ')}"
    out.mkdir(parents=True, exist_ok=False)

    configs = {s: json.loads(Path(p).read_text(encoding="utf-8-sig")) for s, p in STATIONS.items()}
    frames = {s: {k: read_split(c, k) for k in ("train", "validation")} for s, c in configs.items()}
    data = {s: {k: examples(f) for k, f in parts.items()} for s, parts in frames.items()}
    station_scales = {s: scales(parts["train"]) for s, parts in frames.items()}

    scale_rows = [{"station": s, **v} for s, v in station_scales.items()]
    pd.DataFrame(scale_rows).to_csv(out / "station_train_scales.csv", index=False, encoding="utf-8-sig")

    dist_rows = []
    for station, parts in data.items():
        scale = station_scales[station]["median_abs_load"]
        for split, values in parts.items():
            z = values["correction"].ravel() / scale
            dist_rows.append({"station": station, "split": split, "scale": scale,
                              "mean": z.mean(), "std": z.std(), "median": np.median(z),
                              "p05": np.quantile(z, .05), "p95": np.quantile(z, .95),
                              "origins": len(z)})
    pd.DataFrame(dist_rows).to_csv(out / "relative_residual_distribution.csv", index=False, encoding="utf-8-sig")

    search_rows = []
    selected = {}
    for source, parts in data.items():
        base_mae = regression_metrics(parts["train"]["actual"], parts["train"]["last_week"])["MAE_kW"]
        for scale_name, scale in station_scales[source].items():
            maes = []
            for beta in BETA_GRID:
                # Normalize and restore explicitly. This must equal beta * correction.
                correction_kw = beta * (parts["train"]["correction"] / scale) * scale
                mae = regression_metrics(parts["train"]["actual"], parts["train"]["last_week"] + correction_kw)["MAE_kW"]
                maes.append(mae)
                search_rows.append({"source_station": source, "scale_name": scale_name,
                                    "beta": float(beta), "train_MAE_kW": mae,
                                    "MAE_ratio_to_last_week": mae / base_mae})
            selected[(source, scale_name)] = float(BETA_GRID[int(np.argmin(maes))])
    pd.DataFrame(search_rows).to_csv(out / "train_beta_search_by_scale.csv", index=False, encoding="utf-8-sig")

    transfer_rows, invariance_rows = [], []
    for source, target in (("station3", "station4"), ("station4", "station3")):
        target_data = data[target]["validation"]
        base = regression_metrics(target_data["actual"], target_data["last_week"])
        transfer_rows.append({"source_station": source, "target_station": target,
                              "scale_name": "baseline", "beta": 0.0,
                              "validation_origins": len(target_data["origins"]), **base})
        direct_predictions = {}
        for scale_name in station_scales[source]:
            beta = selected[(source, scale_name)]
            target_scale = station_scales[target][scale_name]
            normalized = target_data["correction"] / target_scale
            pred = target_data["last_week"] + beta * normalized * target_scale
            direct = target_data["last_week"] + beta * target_data["correction"]
            max_diff = float(np.max(np.abs(pred - direct)))
            if max_diff > 1e-10:
                raise AssertionError(f"Scale reconstruction changed predictions: {max_diff}")
            direct_predictions[scale_name] = pred
            transfer_rows.append({"source_station": source, "target_station": target,
                                  "scale_name": scale_name, "beta": beta,
                                  "validation_origins": len(target_data["origins"]),
                                  **regression_metrics(target_data["actual"], pred)})
            invariance_rows.append({"source_station": source, "target_station": target,
                                    "scale_name": scale_name, "max_abs_difference_kW": max_diff,
                                    "passed": max_diff <= 1e-10})
        reference = next(iter(direct_predictions.values()))
        if max(np.max(np.abs(p - reference)) for p in direct_predictions.values()) > 1e-10:
            raise AssertionError("Train-selected beta unexpectedly differs by scalar normalization")
    transfer = pd.DataFrame(transfer_rows)
    transfer.to_csv(out / "leave_one_station_out_validation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(invariance_rows).to_csv(out / "scale_invariance_checks.csv", index=False, encoding="utf-8-sig")

    summary = {
        "kind": "scale_independent_leave_one_station_out_diagnostic",
        "executed_at_utc": run_at.isoformat(),
        "stations_used": ["station3", "station4"],
        "files_opened": {s: [c["train_file"], c["validation_file"]] for s, c in configs.items()},
        "test_files_opened": False,
        "third_station_discovered_or_read": False,
        "normalization_finding": "scalar residual normalization cancels exactly after same-scale reconstruction",
        "origin_policy": "predictor-native: target 96, last-week 96, recent 4h and its week-lag must be finite; full 672-point LSTM history is not required",
        "selected_betas": {f"{s}:{n}": b for (s, n), b in selected.items()},
        "all_invariance_checks_passed": bool(pd.DataFrame(invariance_rows)["passed"].all()),
        "models_trained": False,
        "output_dir": str(out.resolve()),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nLeave-one-station-out validation:\n", transfer.to_string(index=False))


if __name__ == "__main__":
    main()
