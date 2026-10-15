import numpy as np
import pandas as pd

from load15_recent_residual_predictor import forecast_from_series


def synthetic_load():
    index = pd.date_range("2024-01-01", periods=1200, freq="15min", tz="Asia/Shanghai")
    values = 500.0 + np.sin(np.arange(len(index)) / 20.0) * 10.0
    return pd.Series(values, index=index, name="load_kw")


def test_output_shape_and_causality():
    load = synthetic_load()
    origin = load.index[900]
    result = forecast_from_series(load, origin)
    assert result.forecast.shape == (96, 5)
    assert result.metadata["all_sources_strictly_before_origin"] is True
    assert result.metadata["future_actual_load_read"] is False
    assert np.isfinite(result.forecast["predicted_load_kw"]).all()


def test_future_mutation_cannot_change_forecast():
    load = synthetic_load()
    origin = load.index[900]
    original = forecast_from_series(load, origin).forecast["predicted_load_kw"].to_numpy()
    mutated = load.copy()
    mutated.loc[mutated.index >= origin] += 1_000_000.0
    changed = forecast_from_series(mutated, origin).forecast["predicted_load_kw"].to_numpy()
    np.testing.assert_array_equal(original, changed)


def test_recent_missing_falls_back_to_last_week():
    load = synthetic_load()
    origin = load.index[900]
    load.loc[origin - pd.Timedelta(minutes=15)] = np.nan
    result = forecast_from_series(load, origin)
    assert result.metadata["mode"] == "last_week_fallback"
    assert result.metadata["fallback_reason"] == "recent_4h_residual_incomplete"
    np.testing.assert_array_equal(
        result.forecast["predicted_load_kw"].to_numpy(),
        result.forecast["last_week_baseline_kw"].to_numpy(),
    )


def test_missing_last_week_is_hard_error():
    load = synthetic_load()
    origin = load.index[900]
    load.loc[origin - pd.Timedelta(days=7)] = np.nan
    try:
        forecast_from_series(load, origin)
    except ValueError as exc:
        assert "上周同期96点不完整" in str(exc)
    else:
        raise AssertionError("missing last-week source must fail")
