"""Diagnose causal recent-residual corrections without training a model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from load15_direct_dataset import build_direct_load_bundle, inverse_load_kw


SOURCES = {
    "last_residual": 1,
    "residual_mean_1h": 4,
    "residual_mean_4h": 16,
    "residual_mean_24h": 96,
}
BETA_GRID = np.round(np.arange(0.0, 1.5001, 0.05), 2)
REPORT_HORIZONS = {1, 4, 16, 48, 96}


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    error = predicted - actual
    nonzero = actual != 0
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "MAE_kW": float(np.mean(np.abs(error))),
        "RMSE_kW": float(np.sqrt(np.mean(error ** 2))),
        "MAPE_pct": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100),
        "R2": float(1 - ss_res / ss_tot),
        "zero_targets_excluded": int((~nonzero).sum()),
        "pairs": int(actual.size),
    }


def recent_features(raw: np.ndarray, origins: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    values = {name: np.full(len(origins), np.nan, dtype="float64") for name in SOURCES}
    available = {name: np.zeros(len(origins), dtype=bool) for name in SOURCES}
    for row, origin_value in enumerate(origins):
        origin = int(origin_value)
        for name, length in SOURCES.items():
            current_positions = np.arange(origin - length, origin)
            weekly_positions = current_positions - 672
            if weekly_positions[0] < 0:
                continue
            current = raw[current_positions]
            weekly = raw[weekly_positions]
            if np.isfinite(current).all() and np.isfinite(weekly).all():
                values[name][row] = float(np.mean(current - weekly))
                available[name][row] = True
    return values, available


def distribution(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)),
        "median": float(np.median(values)), "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)), "origins": int(len(values)),
    }


def main() -> None:
    run_at = datetime.now(timezone.utc)
    run_id = run_at.strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path("my_outputs") / f"recent_residual_baseline_{run_id}"
    out.mkdir(parents=True, exist_ok=False)

    bundle = build_direct_load_bundle("my_load15_config.json")
    raw = bundle.load.to_numpy(dtype="float64")
    split_data = {}
    distribution_rows = []
    causality_rows = []
    for split_name in ("train", "validation"):
        direct = bundle.arrays[split_name]
        origins = direct["origins"]
        actual = inverse_load_kw(direct["y"], bundle.train_mean, bundle.train_std)
        last_week = np.stack([raw[o - 672:o - 576] for o in origins])
        features, available = recent_features(raw, origins)
        split_data[split_name] = {
            "origins": origins, "actual": actual, "last_week": last_week,
            "features": features, "available": available,
        }
        for source, length in SOURCES.items():
            usable_values = features[source][available[source]]
            distribution_rows.append({"split": split_name, "source": source, **distribution(usable_values)})
            usable_origins = origins[available[source]]
            earliest_current = usable_origins - length
            latest_current = usable_origins - 1
            earliest_weekly = earliest_current - 672
            latest_weekly = latest_current - 672
            causal = bool(
                len(usable_origins)
                and np.all(earliest_current < usable_origins)
                and np.all(latest_current < usable_origins)
                and np.all(earliest_weekly < usable_origins)
                and np.all(latest_weekly < usable_origins)
            )
            if not causal:
                raise RuntimeError(f"Causality failed for {split_name}/{source}")
            causality_rows.append({
                "split": split_name, "source": source, "required_residual_points": length,
                "total_origins": len(origins), "usable_origins": int(available[source].sum()),
                "unavailable_origins": int((~available[source]).sum()),
                "earliest_example_origin": str(bundle.load.index[int(usable_origins[0])]),
                "current_residual_source_start": str(bundle.load.index[int(usable_origins[0] - length)]),
                "current_residual_source_end": str(bundle.load.index[int(usable_origins[0] - 1)]),
                "weekly_residual_source_start": str(bundle.load.index[int(usable_origins[0] - length - 672)]),
                "weekly_residual_source_end": str(bundle.load.index[int(usable_origins[0] - 1 - 672)]),
                "all_source_timestamps_strictly_before_origin": causal,
            })

    pd.DataFrame(distribution_rows).to_csv(out / "recent_residual_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(causality_rows).to_csv(out / "causality_check.csv", index=False, encoding="utf-8-sig")

    # Select beta using train only, independently for each source.
    beta_rows = []
    selections = {}
    train = split_data["train"]
    for source in SOURCES:
        mask = train["available"][source]
        actual = train["actual"][mask]
        baseline = train["last_week"][mask]
        correction = train["features"][source][mask, None]
        for beta in BETA_GRID:
            result = regression_metrics(actual, baseline + beta * correction)
            beta_rows.append({"source": source, "beta": float(beta), "train_origins": int(mask.sum()), **result})
        source_rows = [row for row in beta_rows if row["source"] == source]
        selections[source] = min(source_rows, key=lambda row: row["MAE_kW"])["beta"]
    beta_search = pd.DataFrame(beta_rows)
    beta_search["selected_by_train_mae"] = [row.beta == selections[row.source] for row in beta_search.itertuples()]
    beta_search.to_csv(out / "beta_search_train.csv", index=False, encoding="utf-8-sig")

    validation = split_data["validation"]
    common = np.logical_and.reduce([validation["available"][source] for source in SOURCES])
    common_origins = validation["origins"][common]
    actual = validation["actual"][common]
    baseline = validation["last_week"][common]
    if len(common_origins) == 0:
        raise RuntimeError("No common validation origins")

    predictions = {"Last-week baseline": baseline}
    for source in SOURCES:
        correction = validation["features"][source][common, None]
        predictions[f"{source} direct beta=1.00"] = baseline + correction
        predictions[f"{source} train-selected beta={selections[source]:.2f}"] = baseline + selections[source] * correction

    metric_rows = []
    horizon_rows = []
    for method, prediction in predictions.items():
        source = "none" if method == "Last-week baseline" else method.split()[0]
        kind = "baseline" if method == "Last-week baseline" else ("direct" if "direct" in method else "train_selected_beta")
        metric_rows.append({
            "method": method, "source": source, "kind": kind,
            "selected_beta": 0.0 if kind == "baseline" else (1.0 if kind == "direct" else selections[source]),
            "validation_origins": len(common_origins), **regression_metrics(actual, prediction),
        })
        for step in range(bundle.horizon):
            horizon_rows.append({
                "method": method, "source": source, "kind": kind,
                "selected_beta": 0.0 if kind == "baseline" else (1.0 if kind == "direct" else selections[source]),
                "horizon": step + 1, **regression_metrics(actual[:, step], prediction[:, step]),
            })
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(out / "recent_residual_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(horizon_rows).to_csv(out / "recent_residual_metrics_by_horizon.csv", index=False, encoding="utf-8-sig")

    selected_rows = []
    for source, beta in selections.items():
        train_selected = beta_search[(beta_search["source"] == source) & (beta_search["selected_by_train_mae"])].iloc[0]
        validation_selected = metrics_frame[
            (metrics_frame["source"] == source) & (metrics_frame["kind"] == "train_selected_beta")
        ].iloc[0]
        selected_rows.append({
            "source": source, "selected_beta": beta, "train_origins": int(train_selected["train_origins"]),
            "train_MAE_kW": float(train_selected["MAE_kW"]),
            "validation_origins_common": len(common_origins),
            "validation_MAE_kW": float(validation_selected["MAE_kW"]),
            "validation_RMSE_kW": float(validation_selected["RMSE_kW"]),
            "validation_MAPE_pct": float(validation_selected["MAPE_pct"]),
            "validation_R2": float(validation_selected["R2"]),
        })
    pd.DataFrame(selected_rows).to_csv(out / "selected_beta_validation.csv", index=False, encoding="utf-8-sig")

    run_summary = {
        "kind": "causal_recent_residual_baseline_diagnostic",
        "executed_at_utc": run_at.isoformat(),
        "common_validation_origins": int(len(common_origins)),
        "common_validation_first_origin": str(bundle.load.index[int(common_origins[0])]),
        "common_validation_last_origin": str(bundle.load.index[int(common_origins[-1])]),
        "selected_betas_train_only": selections,
        "all_residual_sources_strictly_before_origin": True,
        "models_trained": False,
        "test_samples_count_only": int(bundle.arrays["test"]["X_past"].shape[0]),
        "test_values_used_in_diagnostic": False, "test_predict_called": False, "test_evaluate_called": False,
        "output_dir": str(out.resolve()),
    }
    (out / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
