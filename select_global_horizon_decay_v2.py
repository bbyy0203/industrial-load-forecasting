"""Select a shared horizon-decaying recent-residual correction using train only.

Formula: last_week[h] + beta * exp(-h / tau_steps) * recent_residual_mean_4h.
The constant global_v1 correction is included as tau=infinity. Only #3/#4
train and validation files are opened; test and third-station data are not.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_recent_residual_baselines import BETA_GRID, regression_metrics
from diagnose_scale_independent_transfer import STATIONS, examples, read_split


TAU_STEPS_GRID = [4, 8, 16, 32, 48, 96, 192, float("inf")]
V1_BETA = 0.35


def weights(tau_steps: float) -> np.ndarray:
    if np.isinf(tau_steps):
        return np.ones(96, dtype=float)
    return np.exp(-np.arange(96, dtype=float) / tau_steps)


def predict(data: dict[str, np.ndarray], beta: float, tau_steps: float) -> np.ndarray:
    return data["last_week"] + beta * data["correction"] * weights(tau_steps)[None, :]


def select(train: dict[str, dict], stations: list[str]) -> tuple[float, float, pd.DataFrame]:
    base_mae = {s: regression_metrics(train[s]["actual"], train[s]["last_week"])["MAE_kW"] for s in stations}
    rows = []
    for tau in TAU_STEPS_GRID:
        for beta in BETA_GRID:
            row = {"beta": float(beta), "tau_steps": tau,
                   "tau_hours": None if np.isinf(tau) else float(tau / 4)}
            ratios = []
            for station in stations:
                mae = regression_metrics(train[station]["actual"], predict(train[station], float(beta), tau))["MAE_kW"]
                ratio = mae / base_mae[station]
                row[f"{station}_train_MAE_kW"] = mae
                row[f"{station}_MAE_ratio"] = ratio
                ratios.append(ratio)
            row["macro_mean_MAE_ratio"] = float(np.mean(ratios))
            rows.append(row)
    frame = pd.DataFrame(rows)
    best = frame.loc[frame["macro_mean_MAE_ratio"].idxmin()]
    tau = float(best["tau_steps"])
    beta = float(best["beta"])
    frame["selected"] = frame["beta"].eq(beta) & frame["tau_steps"].eq(tau)
    return beta, tau, frame


def metric_rows(station: str, data: dict, beta: float, tau: float) -> list[dict]:
    methods = {
        "Last-week baseline": data["last_week"],
        "global_v1 constant beta=0.35": predict(data, V1_BETA, float("inf")),
        "horizon-decay candidate": predict(data, beta, tau),
    }
    return [{"station": station, "method": name, "origins": len(data["origins"]),
             **regression_metrics(data["actual"], pred)} for name, pred in methods.items()]


def horizon_metrics(station: str, data: dict, beta: float, tau: float) -> list[dict]:
    methods = {
        "Last-week baseline": data["last_week"],
        "global_v1 constant beta=0.35": predict(data, V1_BETA, float("inf")),
        "horizon-decay candidate": predict(data, beta, tau),
    }
    rows = []
    for name, pred in methods.items():
        for h in range(96):
            rows.append({"station": station, "method": name, "horizon": h + 1,
                         "minutes_ahead": (h + 1) * 15,
                         **regression_metrics(data["actual"][:, h], pred[:, h])})
    return rows


def main() -> None:
    run_at = datetime.now(timezone.utc)
    out = Path("my_outputs") / f"global_horizon_decay_v2_{run_at.strftime('%Y%m%dT%H%M%S_%fZ')}"
    out.mkdir(parents=True, exist_ok=False)
    configs = {s: json.loads(Path(p).read_text(encoding="utf-8-sig")) for s, p in STATIONS.items()}
    frames = {s: {k: read_split(c, k) for k in ("train", "validation")} for s, c in configs.items()}
    all_data = {s: {k: examples(f) for k, f in parts.items()} for s, parts in frames.items()}
    train = {s: d["train"] for s, d in all_data.items()}
    validation = {s: d["validation"] for s, d in all_data.items()}

    beta, tau, search = select(train, list(STATIONS))
    search.to_csv(out / "global_train_search.csv", index=False, encoding="utf-8-sig")

    rows, hrows = [], []
    for station, data in validation.items():
        rows.extend(metric_rows(station, data, beta, tau))
        hrows.extend(horizon_metrics(station, data, beta, tau))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "global_validation_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(hrows).to_csv(out / "validation_metrics_by_horizon.csv", index=False, encoding="utf-8-sig")

    transfer_rows, transfer_params = [], {}
    for source, target in (("station3", "station4"), ("station4", "station3")):
        sb, st, source_search = select(train, [source])
        transfer_params[source] = {"beta": sb, "tau_steps": st, "tau_hours": None if np.isinf(st) else st / 4}
        source_search.to_csv(out / f"{source}_train_search.csv", index=False, encoding="utf-8-sig")
        target_data = validation[target]
        for method, pred in {
            "Last-week baseline": target_data["last_week"],
            "source-selected horizon decay": predict(target_data, sb, st),
        }.items():
            transfer_rows.append({"source_station": source, "target_station": target,
                                  "method": method, "beta": 0.0 if method.startswith("Last") else sb,
                                  "tau_steps": None if method.startswith("Last") else st,
                                  "origins": len(target_data["origins"]),
                                  **regression_metrics(target_data["actual"], pred)})
    pd.DataFrame(transfer_rows).to_csv(out / "leave_one_station_out_validation.csv", index=False, encoding="utf-8-sig")

    # Predeclared promotion evidence: candidate must improve MAE over v1 on both stations.
    promotes = True
    improvements = {}
    for station in STATIONS:
        v1 = metrics[(metrics.station == station) & (metrics.method == "global_v1 constant beta=0.35")].iloc[0]
        candidate = metrics[(metrics.station == station) & (metrics.method == "horizon-decay candidate")].iloc[0]
        improvements[station] = float((1 - candidate.MAE_kW / v1.MAE_kW) * 100)
        promotes &= bool(candidate.MAE_kW < v1.MAE_kW)

    summary = {
        "kind": "global_horizon_decay_candidate",
        "executed_at_utc": run_at.isoformat(),
        "formula": "last_week[h] + beta * exp(-h/tau_steps) * recent_residual_mean_4h",
        "selected_beta": beta,
        "selected_tau_steps": None if np.isinf(tau) else tau,
        "selected_tau_hours": None if np.isinf(tau) else tau / 4,
        "constant_candidate_selected": bool(np.isinf(tau)),
        "selection_data": "station3 train + station4 train only",
        "validation_used_for_parameter_selection": False,
        "validation_MAE_improvement_vs_global_v1_pct": improvements,
        "predeclared_promote_if_MAE_improves_both_stations": promotes,
        "leave_one_station_out_selected": transfer_params,
        "test_files_opened": False,
        "third_station_discovered_or_read": False,
        "models_trained": False,
        "output_dir": str(out.resolve()),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nValidation:\n", metrics.to_string(index=False))
    print("\nTransfer:\n", pd.DataFrame(transfer_rows).to_string(index=False))


if __name__ == "__main__":
    main()
