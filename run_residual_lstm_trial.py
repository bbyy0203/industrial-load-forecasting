"""Lag diagnostics and two-epoch residual-LSTM pipeline trial (no test use)."""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from load15_direct_dataset import build_direct_load_bundle
from load15_residual_dataset import build_residual_splits, causality_table, reconstruct_load_kw


SEED = 42
TRAIN_LIMIT = 128
VALIDATION_LIMIT = 32
BATCH_SIZE = 16
EPOCHS = 2


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
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


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype="float64").ravel()
    return {
        "mean": float(values.mean()), "std": float(values.std(ddof=0)),
        "median": float(np.median(values)), "MAE_from_zero": float(np.mean(np.abs(values))),
        "p05": float(np.quantile(values, .05)), "p25": float(np.quantile(values, .25)),
        "p75": float(np.quantile(values, .75)), "p95": float(np.quantile(values, .95)),
        "observations": len(values),
    }


def unique_residual_series(bundle, split_name: str) -> pd.Series:
    raw = bundle.load
    origins = bundle.arrays[split_name]["origins"]
    positions = np.unique(np.concatenate([np.arange(o, o + bundle.horizon) for o in origins]))
    positions = positions[positions >= 672]
    current = raw.iloc[positions].to_numpy(dtype="float64")
    weekly = raw.iloc[positions - 672].to_numpy(dtype="float64")
    valid = np.isfinite(current) & np.isfinite(weekly)
    observed = pd.Series(current[valid] - weekly[valid], index=raw.index[positions[valid]], name="residual_kw")
    # Preserve missing/uncovered grid positions so a lag means elapsed 15-minute
    # steps rather than merely the next available row across a gap.
    grid = pd.date_range(observed.index.min(), observed.index.max(), freq="15min")
    return observed.reindex(grid)


def main() -> None:
    run_at = datetime.now(timezone.utc)
    run_id = run_at.strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path("my_outputs") / f"residual_lstm_trial_{run_id}"
    out.mkdir(parents=True, exist_ok=False)

    bundle = build_direct_load_bundle("my_load15_config.json")
    test_count_only = int(bundle.arrays["test"]["X_past"].shape[0])
    splits = build_residual_splits(bundle, splits=("train", "validation"))
    causal = causality_table(bundle, splits, samples_per_split=3, seed=SEED)
    causal.to_csv(out / "causality_check.csv", index=False, encoding="utf-8-sig")

    # Sign invariants: zero correction is identity; +30 kW reconstructs 500 -> 530 kW.
    identity = reconstruct_load_kw(np.array([500.0]), np.array([0.0]), bundle.train_std)
    assert np.array_equal(identity, np.array([500.0]))
    artificial_scaled = np.array([30.0 / bundle.train_std])
    artificial = reconstruct_load_kw(np.array([500.0]), artificial_scaled, bundle.train_std)
    assert np.allclose(artificial, np.array([530.0])) and not np.allclose(artificial, np.array([470.0]))

    comparison_rows = []
    for split_name, split in splits.items():
        comparison_rows.append({"split": split_name, "model": "Yesterday baseline", **metrics(split.actual_kw, split.yesterday_kw)})
        comparison_rows.append({"split": split_name, "model": "Last-week baseline", **metrics(split.actual_kw, split.last_week_kw)})

    alpha_rows = []
    train = splits["train"]
    for alpha in np.round(np.arange(0, 1.0001, .05), 2):
        prediction = alpha * train.last_week_kw + (1 - alpha) * train.yesterday_kw
        alpha_rows.append({"alpha": float(alpha), **metrics(train.actual_kw, prediction)})
    alpha_search = pd.DataFrame(alpha_rows)
    selected_alpha = float(alpha_search.loc[alpha_search["MAE_kW"].idxmin(), "alpha"])
    alpha_search["selected_by_train_mae"] = alpha_search["alpha"].eq(selected_alpha)
    alpha_search.to_csv(out / "blend_alpha_search.csv", index=False, encoding="utf-8-sig")
    for split_name, split in splits.items():
        blend = selected_alpha * split.last_week_kw + (1 - selected_alpha) * split.yesterday_kw
        comparison_rows.append({"split": split_name, "model": f"Train-selected blend alpha={selected_alpha:.2f}", **metrics(split.actual_kw, blend)})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out / "lag_baseline_comparison.csv", index=False, encoding="utf-8-sig")

    statistics_rows = []
    autocorrelation_rows = []
    for split_name, split in splits.items():
        residual_pairs = split.actual_kw - split.last_week_kw
        statistics_rows.append({"split": split_name, "scope": "all_origin_horizon_pairs", "group": "all", **describe(residual_pairs)})
        series = unique_residual_series(bundle, split_name)
        local = series.index.tz_convert(bundle.config["feature_timezone"])
        for hour in range(24):
            values = series[local.hour == hour].dropna().to_numpy()
            if len(values):
                statistics_rows.append({"split": split_name, "scope": "unique_targets_by_hour", "group": str(hour), **describe(values)})
        for dow in range(7):
            values = series[local.dayofweek == dow].dropna().to_numpy()
            if len(values):
                statistics_rows.append({"split": split_name, "scope": "unique_targets_by_dayofweek", "group": str(dow), **describe(values)})
        for label, lag in (("15min", 1), ("1h", 4), ("4h", 16), ("24h", 96), ("7d", 672)):
            autocorrelation_rows.append({
                "split": split_name, "lag": label, "lag_steps": lag,
                "autocorrelation": float(series.autocorr(lag=lag)),
                "unique_target_points": int(series.notna().sum()),
            })
    pd.DataFrame(statistics_rows).to_csv(out / "residual_statistics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(autocorrelation_rows).to_csv(out / "residual_autocorrelation.csv", index=False, encoding="utf-8-sig")

    import tensorflow as tf
    from tensorflow import keras
    from load15_residual_lstm import build_residual_lstm, compile_residual_lstm

    tf.keras.utils.set_random_seed(SEED)
    model = compile_residual_lstm(build_residual_lstm(hidden=32))
    summary = []
    model.summary(print_fn=summary.append)
    (out / "model_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    forward = model.predict([train.X_past[:1], train.X_future[:1]], verbose=0)
    assert forward.shape == (1, 96) and np.isfinite(forward).all()

    validation = splits["validation"]
    n_train = min(TRAIN_LIMIT, len(train.origins))
    n_validation = min(VALIDATION_LIMIT, len(validation.origins))
    started = time.perf_counter()
    history = model.fit(
        [train.X_past[:n_train], train.X_future[:n_train]], train.y_residual[:n_train],
        validation_data=(
            [validation.X_past[:n_validation], validation.X_future[:n_validation]],
            validation.y_residual[:n_validation],
        ),
        batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, verbose=2,
    )
    elapsed = time.perf_counter() - started
    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    history_frame.to_csv(out / "trial_history.csv", index=False, encoding="utf-8-sig")
    predicted_residual = model.predict(
        [validation.X_past[:n_validation], validation.X_future[:n_validation]], verbose=0,
    )
    reconstructed = reconstruct_load_kw(validation.last_week_kw[:n_validation], predicted_residual, bundle.train_std)
    assert np.isfinite(predicted_residual).all() and np.isfinite(reconstructed).all()

    baseline_validation = comparison[comparison["split"] == "validation"].set_index("model")
    run_summary = {
        "kind": "residual_lstm_two_epoch_trial_not_formal",
        "executed_at_utc": run_at.isoformat(), "python": platform.python_version(),
        "tensorflow": tf.__version__, "keras": keras.__version__, "seed": SEED,
        "selected_alpha_from_train_only": selected_alpha,
        "train_samples_total": len(train.origins), "validation_samples_total": len(validation.origins),
        "train_shapes": {"X_past": list(train.X_past.shape), "X_future": list(train.X_future.shape), "y_residual": list(train.y_residual.shape)},
        "validation_shapes": {"X_past": list(validation.X_past.shape), "X_future": list(validation.X_future.shape), "y_residual": list(validation.y_residual.shape)},
        "test_samples_count_only": test_count_only, "test_arrays_built_for_residual_model": False,
        "test_predict_called": False, "test_evaluate_called": False,
        "train_samples_used_trial": n_train, "validation_samples_used_trial": n_validation,
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "elapsed_seconds": elapsed,
        "forward_shape": list(forward.shape), "predicted_residual_finite": True,
        "reconstructed_load_finite": True, "reconstruction_formula": "last_week_kw + predicted_residual_scaled * train_std",
        "zero_residual_identity_assertion": True, "artificial_500_plus_30_equals_530_assertion": True,
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "validation_baselines": baseline_validation.reset_index().to_dict("records"),
        "output_dir": str(out.resolve()),
    }
    (out / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
