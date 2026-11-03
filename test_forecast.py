"""
test_forecast.py
----------------
Unit tests for the demand forecasting engine.

Run:
    pip install -r requirements.txt pytest
    pytest -v
"""

import numpy as np
import pandas as pd
import pytest

import forecast
import generate_data


def test_mape_perfect_forecast_is_zero():
    y = np.array([10, 20, 30], dtype=float)
    assert forecast.mape(y, y) == pytest.approx(0.0)


def test_wape_perfect_forecast_is_zero():
    y = np.array([5, 15, 25], dtype=float)
    assert forecast.wape(y, y) == pytest.approx(0.0)


def test_wape_handles_zero_demand_without_crashing():
    y = np.array([0, 0, 10], dtype=float)
    pred = np.array([1, 2, 9], dtype=float)
    assert np.isfinite(forecast.wape(y, pred))


def test_bias_sign_is_correct():
    y = np.array([10, 10, 10], dtype=float)
    over = np.array([12, 12, 12], dtype=float)
    under = np.array([8, 8, 8], dtype=float)
    assert forecast.bias(y, over) > 0
    assert forecast.bias(y, under) < 0


def test_rmse_matches_manual():
    y = np.array([0, 0], dtype=float)
    pred = np.array([3, 4], dtype=float)
    assert forecast.rmse(y, pred) == pytest.approx(np.sqrt(12.5))


def test_build_features_creates_lag_and_rolling_columns():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "units_sold": np.arange(40),
        "on_promo": 0,
        "is_holiday": 0,
    })
    out = forecast.build_features(df)
    for col in ("lag_1", "lag_7", "lag_28", "roll_mean_7", "roll_std_28"):
        assert col in out.columns
    assert out["lag_1"].iloc[1] == df["units_sold"].iloc[0]


def test_model_beats_seasonal_naive_on_generated_data():
    data = generate_data.generate()
    grp = data[(data["store_id"] == "STORE_ATX") &
               (data["sku"] == "SKU_COLA_12PK")]
    res = forecast.backtest_series(grp)
    assert res, "backtest should return results for a full series"
    m = res["metrics"]
    assert m["ml_wape"] <= m["naive_wape"]


def test_backtest_short_series_returns_empty():
    short = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "units_sold": np.arange(30),
        "on_promo": 0,
        "is_holiday": 0,
    })
    assert forecast.backtest_series(short) == {}