"""Train-only selection of a nonlinear, station-relative correction candidate.

The candidate clips the recent 4-hour weekly residual in units of each
station's train-only median absolute load, then restores kW. Only #3/#4 train
and validation CSVs are opened. Test and third-station data are not accessed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_recent_residual_baselines import BETA_GRID, regression_metrics
from diagnose_scale_independent_transfer import STATIONS, examples, read_split, scales


CAP_GRID = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00], dtype=float)
V1_BETA = 0.35


def prediction(data: dict[str, np.ndarray], scale: float, beta: float, cap: float) -> np.ndarray:
    relative = data["correction"] / scale
    clipped_kw = np.clip(relative, -cap, cap) * scale
    return data["last_week"] + beta * clipped_kw


def select_parameters(train: dict[str, dict], station_scales: dict[str, float], stations: list[str]) -> tuple[float, float, pd.DataFrame]:
    baseline = {
        s: regression_metrics(train[s]["actual"], train[s]["last_week"])["MAE_kW"]
        for s in stations
    }
    rows = []
    for cap in CAP_GRID:
        for beta in BETA_GRID:
            ratios, row = [], {"cap_relative_to_median_load": float(cap), "beta": float(beta)}
            for station in stations:
                pred = prediction(train[station], station_scales[station], float(beta), float(cap))
                mae = regression_metrics(train[station]["actual"], pred)["MAE_kW"]
                ratio = mae / baseline[station]
                row[f"{station}_train_MAE_kW"] = mae
                row[f"{station}_MAE_ratio"] = ratio
                ratios.append(ratio)
            row["macro_mean_MAE_ratio"] = float(np.mean(ratios))
            rows.append(row)
    search = pd.DataFrame(rows)
    best = search.loc[search["macro_mean_MAE_ratio"].idxmin()]
    search["selected"] = ((search["beta"] == best["beta"])
                          & (search["cap_relative_to_median_load"] == best["cap_relative_to_median_load"]))
    return float(best["beta"]), float(best["cap_relative_to_median_load"]), search


def main() -> None:
    run_at = datetime.now(timezone.utc)
    out = Path("my_outputs") / f"global_relative_correction_v2_{run_at.strftime('%Y%m%dT%H%M%S_%fZ')}"
    out.mkdir(parents=True, exist_ok=False)

    configs = {s: json.loads(Path(p).read_text(encoding="utf-8-sig")) for s, p in STATIONS.items()}
    frames = {s: {k: read_split(c, k) for k in ("train", "validation")} for s, c in configs.items()}
    all_data = {s: {k: examples(f) for k, f in parts.items()} for s, parts in frames.items()}
    train = {s: parts["train"] for s, parts in all_data.items()}
    validation = {s: parts["validation"] for s, parts in all_data.items()}
    station_scales = {s: scales(parts["train"])["median_abs_load"] for s, parts in frames.items()}

    beta, cap, search = select_parameters(train, station_scales, list(STATIONS))
    search.to_csv(out / "global_train_search.csv", index=False, encoding="utf-8-sig")

    validation_rows = []
    for station, data in validation.items():
        methods = {
            "Last-week baseline": data["last_week"],
            "global_v1 linear beta=0.35": data["last_week"] + V1_BETA * data["correction"],
            "global_v2 relative clipped": prediction(data, station_scales[station], beta, cap),
        }
        for method, pred in methods.items():
            validation_rows.append({"station": station, "method": method,
                                    "validation_origins": len(data["origins"]),
                                    **regression_metrics(data["actual"], pred)})
    validation_frame = pd.DataFrame(validation_rows)
    validation_frame.to_csv(out / "global_validation_comparison.csv", index=False, encoding="utf-8-sig")

    transfer_rows = []
    transfer_selected = {}
    for source, target in (("station3", "station4"), ("station4", "station3")):
        source_beta, source_cap, source_search = select_parameters(train, station_scales, [source])
        transfer_selected[source] = {"beta": source_beta, "cap": source_cap}
        source_search.to_csv(out / f"{source}_train_search.csv", index=False, encoding="utf-8-sig")
        target_data = validation[target]
        for method, pred in {
            "Last-week baseline": target_data["last_week"],
            "source-selected relative clipped": prediction(
                target_data, station_scales[target], source_beta, source_cap),
        }.items():
            transfer_rows.append({"source_station": source, "target_station": target,
                                  "method": method, "beta": 0.0 if method.startswith("Last") else source_beta,
                                  "cap_relative_to_median_load": np.nan if method.startswith("Last") else source_cap,
                                  "validation_origins": len(target_data["origins"]),
                                  **regression_metrics(target_data["actual"], pred)})
    pd.DataFrame(transfer_rows).to_csv(out / "leave_one_station_out_validation.csv", index=False, encoding="utf-8-sig")

    # Quantify how often the nonlinearity actually activates, without looking at validation outcomes for selection.
    activation_rows = []
    for station, parts in all_data.items():
        for split, data in parts.items():
            relative = np.abs(data["correction"].ravel() / station_scales[station])
            activation_rows.append({"station": station, "split": split, "cap": cap,
                                    "origins": len(relative),
                                    "clipped_origins": int((relative > cap).sum()),
                                    "clipped_fraction": float(np.mean(relative > cap))})
    pd.DataFrame(activation_rows).to_csv(out / "clipping_activation.csv", index=False, encoding="utf-8-sig")

    summary = {
        "kind": "global_relative_nonlinear_correction_candidate",
        "executed_at_utc": run_at.isoformat(),
        "formula": "last_week + beta * station_train_median_abs_load * clip(recent_residual / station_train_median_abs_load, -cap, cap)",
        "selection_data": "station3 train + station4 train only",
        "selection_objective": "equal-station macro mean of MAE / station Last-week MAE",
        "selected_beta": beta,
        "selected_cap_relative_to_median_load": cap,
        "station_train_scales_kw": station_scales,
        "leave_one_station_out_selected": transfer_selected,
        "validation_used_for_selection": False,
        "test_files_opened": False,
        "third_station_discovered_or_read": False,
        "models_trained": False,
        "candidate_not_frozen_automatically": True,
        "output_dir": str(out.resolve()),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nValidation comparison:\n", validation_frame.to_string(index=False))
    print("\nLeave-one-station-out:\n", pd.DataFrame(transfer_rows).to_string(index=False))


if __name__ == "__main__":
    main()
