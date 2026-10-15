"""Dataset helpers for correcting the causal last-week load baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from load15_direct_dataset import DirectLoadBundle, inverse_load_kw


PAST_FEATURE_NAMES = ["load_scaled", "time_sin", "time_cos", "dow_sin", "dow_cos"]
FUTURE_FEATURE_NAMES = [
    "time_sin", "time_cos", "dow_sin", "dow_cos",
    "last_week_load_scaled", "yesterday_load_scaled",
]


@dataclass
class ResidualSplit:
    X_past: np.ndarray
    X_future: np.ndarray
    y_residual: np.ndarray
    actual_kw: np.ndarray
    yesterday_kw: np.ndarray
    last_week_kw: np.ndarray
    origins: np.ndarray


def reconstruct_load_kw(last_week_kw: np.ndarray, predicted_residual_scaled: np.ndarray, train_std: float) -> np.ndarray:
    """Reconstruct load for residual=(actual-last_week), hence correction is additive."""
    return np.asarray(last_week_kw, dtype="float64") + np.asarray(predicted_residual_scaled, dtype="float64") * train_std


def build_residual_splits(bundle: DirectLoadBundle, splits: tuple[str, ...] = ("train", "validation")) -> dict[str, ResidualSplit]:
    raw = bundle.load.to_numpy(dtype="float64")
    result = {}
    for name in splits:
        source = bundle.arrays[name]
        origins = source["origins"]
        count = len(origins)
        yesterday = np.stack([raw[o - 96:o] for o in origins])
        last_week = np.stack([raw[o - 672:o - 576] for o in origins])
        # Use the same inverse-transformed y used by the formal validation
        # report, so baseline diagnostics reproduce that report bit-for-bit.
        actual = inverse_load_kw(source["y"], bundle.train_mean, bundle.train_std)
        future = np.empty((count, bundle.horizon, 6), dtype="float32")
        future[:, :, :4] = source["X_future_known"]
        future[:, :, 4] = ((last_week - bundle.train_mean) / bundle.train_std).astype("float32")
        future[:, :, 5] = ((yesterday - bundle.train_mean) / bundle.train_std).astype("float32")
        residual_scaled = ((actual - last_week) / bundle.train_std).astype("float32")
        split = ResidualSplit(
            X_past=source["X_past"], X_future=future, y_residual=residual_scaled,
            actual_kw=actual, yesterday_kw=yesterday, last_week_kw=last_week, origins=origins,
        )
        if split.X_past.shape[1:] != (672, 5) or split.X_future.shape[1:] != (96, 6) or split.y_residual.shape[1:] != (96,):
            raise ValueError(f"{name}: residual interface shape mismatch")
        if not all(np.isfinite(value).all() for value in (split.X_past, split.X_future, split.y_residual)):
            raise ValueError(f"{name}: residual arrays contain NaN or infinity")
        result[name] = split
    return result


def causality_table(bundle: DirectLoadBundle, splits: dict[str, ResidualSplit], samples_per_split: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name, split in splits.items():
        selected = rng.choice(len(split.origins), size=min(samples_per_split, len(split.origins)), replace=False)
        for row in selected:
            origin = int(split.origins[row])
            record = {
                "split": name,
                "origin": str(bundle.load.index[origin]),
                "last_historical_timestamp": str(bundle.load.index[origin - 1]),
                "first_target_timestamp": str(bundle.load.index[origin]),
                "last_target_timestamp": str(bundle.load.index[origin + 95]),
                "last_week_source_start": str(bundle.load.index[origin - 672]),
                "last_week_source_end": str(bundle.load.index[origin - 577]),
                "yesterday_source_start": str(bundle.load.index[origin - 96]),
                "yesterday_source_end": str(bundle.load.index[origin - 1]),
                "all_sources_available_at_origin": bool(
                    bundle.load.index[origin - 577] < bundle.load.index[origin]
                    and bundle.load.index[origin - 1] < bundle.load.index[origin]
                ),
            }
            rows.append(record)
    table = pd.DataFrame(rows)
    if not table["all_sources_available_at_origin"].all():
        raise RuntimeError("Lag baseline causality check failed")
    return table
