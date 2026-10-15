"""Finish validation artifacts after a completed formal training run.

This loads the already saved best model and never retrains or touches test
features beyond reporting the configured sample count.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from load15_direct_dataset import build_direct_load_bundle, inverse_load_kw
from train_direct_lstm_formal import TIMEZONE, baseline_predictions, regression_metrics, write_example


RUN_DIR = Path("my_outputs/direct_lstm_formal_20260928T120117_444358Z")
TRAINING_ELAPSED_SECONDS = 135.0  # reconstructed from the recorded epoch durations (rounded seconds)


def main() -> None:
    bundle = build_direct_load_bundle("my_load15_config.json")
    validation = bundle.arrays["validation"]
    test_count = int(bundle.arrays["test"]["X_past"].shape[0])
    model = keras.models.load_model(RUN_DIR / "best_model.keras")
    predicted_scaled = model.predict(
        [validation["X_past"], validation["X_future_known"]], batch_size=32, verbose=0,
    )
    actual_kw = inverse_load_kw(validation["y"], bundle.train_mean, bundle.train_std)
    predicted_kw = inverse_load_kw(predicted_scaled, bundle.train_mean, bundle.train_std)
    raw = bundle.load.to_numpy(dtype="float64")
    origins = validation["origins"]
    baselines = baseline_predictions(raw, origins)
    predictions = {**baselines, "Direct LSTM": predicted_kw}

    comparison = pd.DataFrame([{"model": name, **regression_metrics(actual_kw, pred)} for name, pred in predictions.items()])
    reference = comparison.set_index("model").loc["Last-week baseline"]
    comparison["MAE_improvement_vs_last_week_pct"] = (reference["MAE_kW"] - comparison["MAE_kW"]) / reference["MAE_kW"] * 100
    comparison["RMSE_improvement_vs_last_week_pct"] = (reference["RMSE_kW"] - comparison["RMSE_kW"]) / reference["RMSE_kW"] * 100
    comparison.to_csv(RUN_DIR / "validation_model_comparison.csv", index=False, encoding="utf-8-sig")

    horizon_rows = []
    for name, prediction in predictions.items():
        for step in range(bundle.horizon):
            horizon_rows.append({"model": name, "horizon": step + 1, **regression_metrics(actual_kw[:, step], prediction[:, step])})
    pd.DataFrame(horizon_rows).to_csv(RUN_DIR / "validation_metrics_by_horizon.csv", index=False, encoding="utf-8-sig")

    origin_dates = np.asarray(bundle.load.index[origins].tz_convert(TIMEZONE).date)
    daily_rows = []
    for name, prediction in predictions.items():
        for date in sorted(set(origin_dates)):
            selected = origin_dates == date
            daily_rows.append({"model": name, "origin_date": str(date), "origins": int(selected.sum()), **regression_metrics(actual_kw[selected], prediction[selected])})
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(RUN_DIR / "validation_metrics_by_day.csv", index=False, encoding="utf-8-sig")

    lstm_daily = daily[daily["model"] == "Direct LSTM"].copy()
    lstm_daily["weekday"] = pd.to_datetime(lstm_daily["origin_date"]).dt.dayofweek
    weekdays = lstm_daily[lstm_daily["weekday"] < 5].sort_values("MAE_kW")
    ordinary_date = str(weekdays.iloc[len(weekdays) // 2]["origin_date"])
    high_error_date = str(lstm_daily.sort_values("MAE_kW", ascending=False).iloc[0]["origin_date"])
    origin_date_strings = origin_dates.astype(str)
    for label, date in (("ordinary_weekday", ordinary_date), ("high_error_date", high_error_date)):
        row = int(np.flatnonzero(origin_date_strings == date)[0])
        origin = int(origins[row])
        write_example(
            RUN_DIR / f"validation_example_{label}_{date}.csv", origin, bundle.load.index,
            actual_kw[row], predicted_kw[row], baselines["Yesterday baseline"][row], baselines["Last-week baseline"][row],
        )

    history = pd.read_csv(RUN_DIR / "training_history.csv")
    best_row = history.loc[history["val_loss"].idxmin()]
    run_summary = {
        "kind": "first_formal_direct_load_lstm",
        "epochs_trained": int(len(history)),
        "best_epoch": int(best_row["epoch"]),
        "best_val_loss": float(best_row["val_loss"]),
        "early_stopping_triggered": len(history) < 100,
        "final_learning_rate": 0.0002500000118743628,
        "elapsed_seconds": TRAINING_ELAPSED_SECONDS,
        "elapsed_note": "reconstructed from per-epoch rounded durations after post-training export failed; uncertainty approximately +/- 1 second",
        "validation_origin_days": int(len(set(origin_dates))),
        "ordinary_example_date": ordinary_date,
        "high_error_example_date": high_error_date,
        "test_samples_count_only": test_count,
        "test_predict_called": False,
        "test_evaluate_called": False,
        "post_training_export_recovered_without_retraining": True,
        "output_dir": str(RUN_DIR.resolve()),
    }
    (RUN_DIR / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_summary": run_summary, "comparison": comparison.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
