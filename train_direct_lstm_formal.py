"""Train the first formal direct-load LSTM and evaluate validation only."""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from load15_direct_dataset import build_direct_load_bundle, inverse_load_kw


SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 100
INITIAL_LR = 1e-3
EARLY_STOPPING_PATIENCE = 10
LR_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LR = 1e-6
TIMEZONE = "Asia/Shanghai"


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    error = predicted - actual
    nonzero = actual != 0
    ss_res = float(np.sum(error ** 2))
    centered = actual - float(np.mean(actual))
    ss_tot = float(np.sum(centered ** 2))
    return {
        "MAE_kW": float(np.mean(np.abs(error))),
        "RMSE_kW": float(np.sqrt(np.mean(error ** 2))),
        "MAPE_pct": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "zero_targets_excluded_from_mape": int((~nonzero).sum()),
        "evaluated_pairs": int(actual.size),
    }


def baseline_predictions(raw: np.ndarray, origins: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Yesterday baseline": np.stack([raw[o - 96:o] for o in origins]),
        "Last-week baseline": np.stack([raw[o - 672:o - 576] for o in origins]),
    }


def write_example(
    path: Path,
    origin: int,
    index: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    yesterday: np.ndarray,
    last_week: np.ndarray,
) -> None:
    timestamps = index[origin:origin + 96].tz_convert(TIMEZONE)
    pd.DataFrame({
        "timestamp": timestamps.astype(str),
        "actual_kw": actual,
        "predicted_kw": predicted,
        "yesterday_baseline_kw": yesterday,
        "last_week_baseline_kw": last_week,
    }).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    run_at = datetime.now(timezone.utc)
    run_id = run_at.strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = Path("my_outputs") / f"direct_lstm_formal_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    bundle = build_direct_load_bundle("my_load15_config.json")
    train = bundle.arrays["train"]
    validation = bundle.arrays["validation"]
    test_count = int(bundle.arrays["test"]["X_past"].shape[0])
    expected = {
        "train": ((1192, 672, 5), (1192, 96, 4), (1192, 96)),
        "validation": ((1169, 672, 5), (1169, 96, 4), (1169, 96)),
    }
    for name, pack in (("train", train), ("validation", validation)):
        shapes = (pack["X_past"].shape, pack["X_future_known"].shape, pack["y"].shape)
        if shapes != expected[name]:
            raise RuntimeError(f"{name} shapes changed: {shapes} != {expected[name]}")
        for key in ("X_past", "X_future_known", "y"):
            if not np.isfinite(pack[key]).all():
                raise RuntimeError(f"{name}.{key} contains NaN or infinity")
    if test_count != 1617:
        raise RuntimeError(f"test count changed: {test_count} != 1617")

    import tensorflow as tf
    from tensorflow import keras
    from load15_direct_lstm import build_direct_load_lstm, compile_direct_load_lstm

    tf.keras.utils.set_random_seed(SEED)
    model = compile_direct_load_lstm(
        build_direct_load_lstm(hidden=32), learning_rate=INITIAL_LR,
    )
    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    (output_dir / "model_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    training_config = {
        "kind": "first_formal_direct_load_lstm",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "shuffle": True,
        "optimizer": "Adam",
        "initial_learning_rate": INITIAL_LR,
        "loss": "mse_scaled_space",
        "early_stopping": {"monitor": "val_loss", "patience": EARLY_STOPPING_PATIENCE, "restore_best_weights": True},
        "reduce_lr_on_plateau": {"monitor": "val_loss", "factor": LR_FACTOR, "patience": LR_PATIENCE, "min_lr": MIN_LR},
        "lookback": bundle.lookback,
        "horizon": bundle.horizon,
        "hidden": 32,
        "test_policy": "count_only; never passed to fit, predict, or evaluate",
    }
    (output_dir / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")
    (output_dir / "scaler_snapshot.json").write_text(json.dumps({
        "train_mean": bundle.train_mean,
        "train_std_ddof0": bundle.train_std,
        "unit": "kW",
        "source": "rebuilt from configured training data",
    }, indent=2), encoding="utf-8")
    dataset_summary = {
        "train": {"X_past": list(train["X_past"].shape), "X_future_known": list(train["X_future_known"].shape), "y": list(train["y"].shape)},
        "validation": {"X_past": list(validation["X_past"].shape), "X_future_known": list(validation["X_future_known"].shape), "y": list(validation["y"].shape)},
        "test_samples_count_only": test_count,
        "train_validation_nan_count": 0,
        "train_validation_inf_count": 0,
        "split_table": bundle.split_table.to_dict(orient="records"),
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    environment = "\n".join([
        f"executed_at_utc={run_at.isoformat()}",
        f"python={platform.python_version()}",
        f"interpreter={sys.executable}",
        f"tensorflow={tf.__version__}",
        f"keras={keras.__version__}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"devices={[device.name for device in tf.config.list_physical_devices()]}",
    ]) + "\n"
    (output_dir / "environment.txt").write_text(environment, encoding="utf-8")

    checkpoint = output_dir / "best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR, verbose=1),
    ]
    start = time.perf_counter()
    history = model.fit(
        [train["X_past"], train["X_future_known"]], train["y"],
        validation_data=([validation["X_past"], validation["X_future_known"]], validation["y"]),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.perf_counter() - start

    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    if "learning_rate" not in history_frame and "lr" in history_frame:
        history_frame = history_frame.rename(columns={"lr": "learning_rate"})
    history_frame.to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    best_index = int(history_frame["val_loss"].idxmin())
    best_epoch = int(history_frame.loc[best_index, "epoch"])
    best_val_loss = float(history_frame.loc[best_index, "val_loss"])
    epochs_trained = len(history_frame)
    early_stopping_triggered = epochs_trained < MAX_EPOCHS
    final_lr = float(keras.backend.get_value(model.optimizer.learning_rate))

    # Validation is the only split passed to predict.  Test remains sealed.
    validation_pred_scaled = model.predict(
        [validation["X_past"], validation["X_future_known"]], batch_size=BATCH_SIZE, verbose=0,
    )
    actual_kw = inverse_load_kw(validation["y"], bundle.train_mean, bundle.train_std)
    predicted_kw = inverse_load_kw(validation_pred_scaled, bundle.train_mean, bundle.train_std)
    raw = bundle.load.to_numpy(dtype="float64")
    origins = validation["origins"]
    baselines = baseline_predictions(raw, origins)

    predictions = {**baselines, "Direct LSTM": predicted_kw}
    comparison_rows = []
    for name, prediction in predictions.items():
        comparison_rows.append({"model": name, **regression_metrics(actual_kw, prediction)})
    comparison = pd.DataFrame(comparison_rows)
    reference = comparison.set_index("model").loc["Last-week baseline"]
    comparison["MAE_improvement_vs_last_week_pct"] = (reference["MAE_kW"] - comparison["MAE_kW"]) / reference["MAE_kW"] * 100
    comparison["RMSE_improvement_vs_last_week_pct"] = (reference["RMSE_kW"] - comparison["RMSE_kW"]) / reference["RMSE_kW"] * 100
    comparison.to_csv(output_dir / "validation_model_comparison.csv", index=False, encoding="utf-8-sig")

    horizon_rows = []
    for name, prediction in predictions.items():
        for step in range(bundle.horizon):
            horizon_rows.append({"model": name, "horizon": step + 1, **regression_metrics(actual_kw[:, step], prediction[:, step])})
    horizon_metrics = pd.DataFrame(horizon_rows)
    horizon_metrics.to_csv(output_dir / "validation_metrics_by_horizon.csv", index=False, encoding="utf-8-sig")

    origin_dates = bundle.load.index[origins].tz_convert(TIMEZONE).date
    daily_rows = []
    for name, prediction in predictions.items():
        for date in sorted(set(origin_dates)):
            selected = np.asarray(origin_dates == date)
            daily_rows.append({"model": name, "origin_date": str(date), "origins": int(selected.sum()), **regression_metrics(actual_kw[selected], prediction[selected])})
    daily_metrics = pd.DataFrame(daily_rows)
    daily_metrics.to_csv(output_dir / "validation_metrics_by_day.csv", index=False, encoding="utf-8-sig")

    lstm_daily = daily_metrics[daily_metrics["model"] == "Direct LSTM"].copy()
    lstm_daily["weekday"] = pd.to_datetime(lstm_daily["origin_date"]).dt.dayofweek
    weekdays = lstm_daily[lstm_daily["weekday"] < 5].sort_values("MAE_kW")
    ordinary_date = weekdays.iloc[len(weekdays) // 2]["origin_date"] if len(weekdays) else lstm_daily.sort_values("MAE_kW").iloc[len(lstm_daily) // 2]["origin_date"]
    high_error_date = lstm_daily.sort_values("MAE_kW", ascending=False).iloc[0]["origin_date"]
    for label, date in (("ordinary_weekday", ordinary_date), ("high_error_date", high_error_date)):
        row = int(np.flatnonzero(np.asarray(origin_dates, dtype=str) == str(date))[0])
        origin = int(origins[row])
        write_example(
            output_dir / f"validation_example_{label}_{date}.csv", origin, bundle.load.index,
            actual_kw[row], predicted_kw[row], baselines["Yesterday baseline"][row], baselines["Last-week baseline"][row],
        )

    run_summary = {
        "kind": "first_formal_direct_load_lstm",
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "early_stopping_triggered": early_stopping_triggered,
        "final_learning_rate": final_lr,
        "elapsed_seconds": elapsed,
        "validation_origin_days": int(len(set(origin_dates))),
        "ordinary_example_date": str(ordinary_date),
        "high_error_example_date": str(high_error_date),
        "test_samples_count_only": test_count,
        "test_predict_called": False,
        "test_evaluate_called": False,
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_summary": run_summary, "comparison": comparison.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
