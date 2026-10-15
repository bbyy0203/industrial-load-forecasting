"""Replay validation through the production inference function, never test."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_recent_residual_baselines import regression_metrics
from load15_direct_dataset import build_direct_load_bundle, inverse_load_kw
from load15_recent_residual_predictor import DEFAULT_MODEL_CONFIG, forecast_from_series


def main() -> None:
    run_at = datetime.now(timezone.utc)
    stamp = run_at.strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path("my_outputs") / f"recent_residual_runtime_validation_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    model_config = json.loads(DEFAULT_MODEL_CONFIG.read_text(encoding="utf-8"))
    beta = float(model_config["beta"])

    bundle = build_direct_load_bundle("my_load15_config.json")
    validation = bundle.arrays["validation"]
    origins = validation["origins"]
    actual = inverse_load_kw(validation["y"], bundle.train_mean, bundle.train_std)
    load = bundle.load
    predictions = []
    audit_rows = []
    latencies_ms = []
    for row, origin_position in enumerate(origins):
        origin = load.index[int(origin_position)]
        started = time.perf_counter()
        result = forecast_from_series(load, origin, timezone_name=bundle.config["feature_timezone"], beta=beta)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        prediction = result.forecast["predicted_load_kw"].to_numpy(dtype="float64")
        if prediction.shape != (96,) or not np.isfinite(prediction).all():
            raise RuntimeError(f"invalid runtime output at {origin}")
        if not result.metadata["all_sources_strictly_before_origin"] or result.metadata["future_actual_load_read"]:
            raise RuntimeError(f"causality contract failed at {origin}")
        baseline = result.forecast["last_week_baseline_kw"].to_numpy(dtype="float64")
        if result.metadata["mode"] == "last_week_fallback":
            np.testing.assert_array_equal(prediction, baseline)
        else:
            expected = baseline + beta * float(result.metadata["recent_residual_mean_kw"])
            np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-12)
        predictions.append(prediction)
        audit_rows.append({
            "origin": str(origin.tz_convert(bundle.config["feature_timezone"])),
            "mode": result.metadata["mode"],
            "fallback_reason": result.metadata["fallback_reason"],
            "recent_residual_mean_kw": result.metadata["recent_residual_mean_kw"],
            "applied_correction_kw": result.metadata["applied_correction_kw"],
            "all_sources_strictly_before_origin": result.metadata["all_sources_strictly_before_origin"],
            "output_points": len(prediction),
            "latency_ms": latencies_ms[-1],
        })

    predicted = np.stack(predictions)
    baseline = np.stack([load.to_numpy(dtype="float64")[o - 672:o - 576] for o in origins])
    metrics = pd.DataFrame([
        {"method": "Last-week baseline", **regression_metrics(actual, baseline)},
        {"method": "recent_residual_v1 runtime", **regression_metrics(actual, predicted)},
    ])
    metrics.to_csv(out / "runtime_validation_metrics.csv", index=False, encoding="utf-8-sig")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out / "origin_runtime_audit.csv", index=False, encoding="utf-8-sig")
    mode_counts = audit["mode"].value_counts().to_dict()
    summary = {
        "kind": "production_runtime_validation_replay",
        "executed_at_utc": run_at.isoformat(),
        "model_config": str(DEFAULT_MODEL_CONFIG.resolve()),
        "validation_origins": len(origins),
        "mode_counts": {key: int(value) for key, value in mode_counts.items()},
        "fallback_rate": float((audit["mode"] == "last_week_fallback").mean()),
        "latency_ms_mean": float(np.mean(latencies_ms)),
        "latency_ms_p95": float(np.quantile(latencies_ms, .95)),
        "all_outputs_96_finite_points": True,
        "all_sources_strictly_before_origin": True,
        "future_mutation_invariance_test": "covered by tests/test_recent_residual_predictor.py",
        "models_trained": False,
        "test_samples_count_only": int(bundle.arrays["test"]["X_past"].shape[0]),
        "test_values_used": False,
        "test_predict_called": False,
        "test_evaluate_called": False,
        "output_dir": str(out.resolve()),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
