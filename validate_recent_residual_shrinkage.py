"""Train-only walk-forward validation of causal 4h residual shrinkage.

No statistical or neural model is trained. Coefficients are selected by grid
search on chronologically prior train origins whose 96-step labels finish
before each evaluation block begins. Test data are not used.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_recent_residual_baselines import BETA_GRID, recent_features, regression_metrics
from load15_direct_dataset import build_direct_load_bundle, inverse_load_kw


SOURCE = "residual_mean_4h"
BANDS = {
    "0_1h": np.arange(0, 4),
    "1_4h": np.arange(4, 16),
    "4_12h": np.arange(16, 48),
    "12_24h": np.arange(48, 96),
}
FOLDS = [
    ("fold_1", "2024-09-30 00:00", "2024-10-02 23:45"),
    ("fold_2", "2024-10-03 00:00", "2024-10-05 23:45"),
    ("fold_3", "2024-10-06 00:00", "2024-10-08 23:45"),
    ("fold_4", "2024-10-18 17:45", "2024-10-19 23:45"),
]
TIMEZONE = "Asia/Shanghai"


def select_beta(actual: np.ndarray, baseline: np.ndarray, correction: np.ndarray) -> tuple[float, float]:
    rows = []
    for beta in BETA_GRID:
        mae = float(np.mean(np.abs(baseline + beta * correction - actual)))
        rows.append((mae, float(beta)))
    best_mae, best_beta = min(rows)
    return best_beta, best_mae


def prepare_split(bundle, name: str) -> dict:
    direct = bundle.arrays[name]
    origins = direct["origins"]
    raw = bundle.load.to_numpy(dtype="float64")
    features, available = recent_features(raw, origins)
    return {
        "origins": origins,
        "origin_local": bundle.load.index[origins].tz_convert(TIMEZONE),
        "actual": inverse_load_kw(direct["y"], bundle.train_mean, bundle.train_std),
        "last_week": np.stack([raw[o - 672:o - 576] for o in origins]),
        "correction": features[SOURCE],
        "available": available[SOURCE],
    }


def main() -> None:
    run_at = datetime.now(timezone.utc)
    run_id = run_at.strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path("my_outputs") / f"recent_residual_walkforward_{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    bundle = build_direct_load_bundle("my_load15_config.json")
    train = prepare_split(bundle, "train")

    fold_rows = []
    beta_rows = []
    oof_actual = []
    oof_predictions = {"Last-week baseline": [], "Expanding constant beta": [], "Expanding segmented beta": []}
    for fold_name, start_text, end_text in FOLDS:
        start = pd.Timestamp(start_text, tz=TIMEZONE)
        end = pd.Timestamp(end_text, tz=TIMEZONE)
        eval_mask = train["available"] & (train["origin_local"] >= start) & (train["origin_local"] <= end)
        if not eval_mask.any():
            raise RuntimeError(f"{fold_name} has no evaluation origins")
        eval_first_origin = int(train["origins"][eval_mask][0])
        # Strict target embargo: tuning target [origin, origin+95] ends before evaluation begins.
        tune_mask = train["available"] & (train["origins"] + bundle.horizon <= eval_first_origin)
        if not tune_mask.any():
            raise RuntimeError(f"{fold_name} has no embargoed tuning origins")
        tune_actual = train["actual"][tune_mask]
        tune_baseline = train["last_week"][tune_mask]
        tune_correction = train["correction"][tune_mask, None]
        constant_beta, constant_tune_mae = select_beta(tune_actual, tune_baseline, tune_correction)
        band_betas = {}
        for band, steps in BANDS.items():
            beta, band_mae = select_beta(tune_actual[:, steps], tune_baseline[:, steps], tune_correction)
            band_betas[band] = beta
            beta_rows.append({
                "fold": fold_name, "coefficient": band, "selected_beta": beta,
                "tuning_origins": int(tune_mask.sum()), "tuning_MAE_kW": band_mae,
            })
        beta_rows.append({
            "fold": fold_name, "coefficient": "constant_0_24h", "selected_beta": constant_beta,
            "tuning_origins": int(tune_mask.sum()), "tuning_MAE_kW": constant_tune_mae,
        })
        eval_actual = train["actual"][eval_mask]
        eval_baseline = train["last_week"][eval_mask]
        eval_correction = train["correction"][eval_mask, None]
        constant_pred = eval_baseline + constant_beta * eval_correction
        segmented_pred = eval_baseline.copy()
        for band, steps in BANDS.items():
            segmented_pred[:, steps] += band_betas[band] * eval_correction
        predictions = {
            "Last-week baseline": eval_baseline,
            "Expanding constant beta": constant_pred,
            "Expanding segmented beta": segmented_pred,
        }
        for method, prediction in predictions.items():
            fold_rows.append({
                "fold": fold_name, "method": method, "tuning_origins": int(tune_mask.sum()),
                "evaluation_origins": int(eval_mask.sum()),
                "evaluation_start": str(train["origin_local"][eval_mask][0]),
                "evaluation_end": str(train["origin_local"][eval_mask][-1]),
                "target_embargo_passed": bool(np.all(train["origins"][tune_mask] + 95 < eval_first_origin)),
                **regression_metrics(eval_actual, prediction),
            })
            oof_predictions[method].append(prediction)
        oof_actual.append(eval_actual)

    pd.DataFrame(beta_rows).to_csv(out / "walkforward_selected_betas.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(out / "walkforward_fold_metrics.csv", index=False, encoding="utf-8-sig")
    combined_actual = np.concatenate(oof_actual)
    oof_rows = []
    for method, parts in oof_predictions.items():
        oof_rows.append({"method": method, "evaluation_origins": int(sum(len(x) for x in oof_actual)), **regression_metrics(combined_actual, np.concatenate(parts))})
    pd.DataFrame(oof_rows).to_csv(out / "walkforward_oof_metrics.csv", index=False, encoding="utf-8-sig")

    # Freeze final coefficients using train only.
    usable = train["available"]
    actual = train["actual"][usable]
    baseline = train["last_week"][usable]
    correction = train["correction"][usable, None]
    final_constant, final_constant_mae = select_beta(actual, baseline, correction)
    final_bands = {}
    final_rows = [{"coefficient": "constant_0_24h", "selected_beta": final_constant, "train_origins": int(usable.sum()), "train_MAE_kW": final_constant_mae}]
    for band, steps in BANDS.items():
        beta, mae = select_beta(actual[:, steps], baseline[:, steps], correction)
        final_bands[band] = beta
        final_rows.append({"coefficient": band, "selected_beta": beta, "train_origins": int(usable.sum()), "train_MAE_kW": mae})
    pd.DataFrame(final_rows).to_csv(out / "final_train_coefficients.csv", index=False, encoding="utf-8-sig")

    # One validation evaluation after all choices and coefficients are frozen.
    validation = prepare_split(bundle, "validation")
    vmask = validation["available"]
    vactual = validation["actual"][vmask]
    vbaseline = validation["last_week"][vmask]
    vcorr = validation["correction"][vmask, None]
    vconstant = vbaseline + final_constant * vcorr
    vsegmented = vbaseline.copy()
    for band, steps in BANDS.items():
        vsegmented[:, steps] += final_bands[band] * vcorr
    validation_predictions = {
        "Last-week baseline": vbaseline,
        f"4h residual constant beta={final_constant:.2f}": vconstant,
        "4h residual segmented beta": vsegmented,
    }
    validation_rows = []
    horizon_rows = []
    for method, prediction in validation_predictions.items():
        validation_rows.append({"method": method, "validation_origins": int(vmask.sum()), **regression_metrics(vactual, prediction)})
        for step in range(bundle.horizon):
            horizon_rows.append({"method": method, "horizon": step + 1, **regression_metrics(vactual[:, step], prediction[:, step])})
    pd.DataFrame(validation_rows).to_csv(out / "frozen_validation_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(horizon_rows).to_csv(out / "frozen_validation_metrics_by_horizon.csv", index=False, encoding="utf-8-sig")

    summary = {
        "kind": "train_only_walkforward_and_horizon_shrinkage",
        "executed_at_utc": run_at.isoformat(), "source": SOURCE,
        "bands": {name: [int(steps[0] + 1), int(steps[-1] + 1)] for name, steps in BANDS.items()},
        "walkforward_folds": len(FOLDS), "walkforward_evaluation_origins": int(sum(len(x) for x in oof_actual)),
        "all_folds_target_embargo_passed": bool(all(row["target_embargo_passed"] for row in fold_rows)),
        "final_constant_beta_train_only": final_constant,
        "final_segmented_betas_train_only": final_bands,
        "validation_origins": int(vmask.sum()),
        "models_trained": False, "test_samples_count_only": int(bundle.arrays["test"]["X_past"].shape[0]),
        "test_values_used": False, "test_predict_called": False, "test_evaluate_called": False,
        "output_dir": str(out.resolve()),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
