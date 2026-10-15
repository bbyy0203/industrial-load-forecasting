"""Select a station-balanced shared beta using station #3/#4 train only.

The objective gives each station equal weight by averaging its MAE ratio to
its own last-week baseline. Validation is evaluated only after beta selection.
No December test values and no third-station files are used.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_recent_residual_baselines import BETA_GRID, recent_features, regression_metrics
from load15_direct_dataset import build_direct_load_bundle, inverse_load_kw


STATIONS = {
    "station3": "my_load15_config.json",
    "station4": "station4_load15_config.json",
}


def prepare(bundle, split: str) -> dict:
    direct = bundle.arrays[split]
    origins = direct["origins"]
    raw = bundle.load.to_numpy(dtype="float64")
    features, available = recent_features(raw, origins)
    usable = available["residual_mean_4h"]
    return {
        "origins": origins[usable],
        "actual": inverse_load_kw(direct["y"], bundle.train_mean, bundle.train_std)[usable],
        "last_week": np.stack([raw[o-672:o-576] for o in origins])[usable],
        "correction": features["residual_mean_4h"][usable, None],
    }


def main() -> None:
    run_at = datetime.now(timezone.utc)
    stamp = run_at.strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path("my_outputs") / f"global_beta_selection_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    bundles = {station: build_direct_load_bundle(config) for station, config in STATIONS.items()}
    train = {station: prepare(bundle, "train") for station, bundle in bundles.items()}
    validation = {station: prepare(bundle, "validation") for station, bundle in bundles.items()}

    station_train_baseline_mae = {
        station: regression_metrics(data["actual"], data["last_week"])["MAE_kW"]
        for station, data in train.items()
    }
    search_rows = []
    for beta in BETA_GRID:
        ratios = []
        row = {"beta": float(beta)}
        for station, data in train.items():
            prediction = data["last_week"] + beta * data["correction"]
            mae = regression_metrics(data["actual"], prediction)["MAE_kW"]
            ratio = mae / station_train_baseline_mae[station]
            row[f"{station}_train_origins"] = len(data["origins"])
            row[f"{station}_MAE_kW"] = mae
            row[f"{station}_MAE_ratio_to_last_week"] = ratio
            ratios.append(ratio)
        row["macro_mean_MAE_ratio"] = float(np.mean(ratios))
        row["macro_mean_improvement_pct"] = float((1 - np.mean(ratios)) * 100)
        search_rows.append(row)
    search = pd.DataFrame(search_rows)
    selected_beta = float(search.loc[search["macro_mean_MAE_ratio"].idxmin(), "beta"])
    search["selected"] = search["beta"].eq(selected_beta)
    search.to_csv(out / "shared_beta_train_search.csv", index=False, encoding="utf-8-sig")

    station_specific = {}
    for station in STATIONS:
        column = f"{station}_MAE_kW"
        station_specific[station] = float(search.loc[search[column].idxmin(), "beta"])

    candidate_betas = sorted(set([0.30, 0.40, selected_beta, *station_specific.values()]))
    validation_rows = []
    for station, data in validation.items():
        baseline_metrics = regression_metrics(data["actual"], data["last_week"])
        validation_rows.append({
            "station": station, "method": "Last-week baseline", "beta": 0.0,
            "validation_origins": len(data["origins"]), "is_station_specific": False,
            "is_shared_selected": False, **baseline_metrics,
        })
        for beta in candidate_betas:
            prediction = data["last_week"] + beta * data["correction"]
            validation_rows.append({
                "station": station, "method": f"recent residual beta={beta:.2f}", "beta": beta,
                "validation_origins": len(data["origins"]),
                "is_station_specific": beta == station_specific[station],
                "is_shared_selected": beta == selected_beta,
                **regression_metrics(data["actual"], prediction),
            })
    validation_frame = pd.DataFrame(validation_rows)
    validation_frame.to_csv(out / "two_station_validation_comparison.csv", index=False, encoding="utf-8-sig")

    macro_rows = []
    for beta in candidate_betas:
        ratios = []
        rmse_ratios = []
        mape_ratios = []
        for station in STATIONS:
            base = validation_frame[(validation_frame.station == station) & (validation_frame.method == "Last-week baseline")].iloc[0]
            candidate = validation_frame[(validation_frame.station == station) & (validation_frame.beta == beta) & (validation_frame.method != "Last-week baseline")].iloc[0]
            ratios.append(candidate.MAE_kW / base.MAE_kW)
            rmse_ratios.append(candidate.RMSE_kW / base.RMSE_kW)
            mape_ratios.append(candidate.MAPE_pct / base.MAPE_pct)
        macro_rows.append({
            "beta": beta, "selected_by_train": beta == selected_beta,
            "macro_MAE_ratio": float(np.mean(ratios)), "macro_MAE_improvement_pct": float((1-np.mean(ratios))*100),
            "macro_RMSE_ratio": float(np.mean(rmse_ratios)), "macro_RMSE_improvement_pct": float((1-np.mean(rmse_ratios))*100),
            "macro_MAPE_ratio": float(np.mean(mape_ratios)), "macro_MAPE_improvement_pct": float((1-np.mean(mape_ratios))*100),
        })
    pd.DataFrame(macro_rows).to_csv(out / "two_station_validation_macro.csv", index=False, encoding="utf-8-sig")

    summary = {
        "kind": "station_balanced_global_beta_selection",
        "executed_at_utc": run_at.isoformat(),
        "stations_used": list(STATIONS),
        "selection_data": "station3 train + station4 train only",
        "selection_objective": "equal-station macro mean of MAE / station Last-week MAE",
        "selected_shared_beta": selected_beta,
        "station_specific_train_betas": station_specific,
        "station3_to_station4_beta": 0.40,
        "station4_to_station3_beta": station_specific["station4"],
        "validation_used_for_selection": False,
        "models_trained": False,
        "station3_test_samples_count_only": len(bundles["station3"].arrays["test"]["origins"]),
        "station4_test_samples_count_only": len(bundles["station4"].arrays["test"]["origins"]),
        "test_values_used": False, "test_predict_called": False, "test_evaluate_called": False,
        "third_station_discovered_or_read": False,
        "output_dir": str(out.resolve()),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
