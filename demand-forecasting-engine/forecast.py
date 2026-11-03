"""
forecast.py
-----------
Demand forecasting engine for grocery SKUs.

Trains and backtests two forecasting approaches per store-SKU series:
  1. Seasonal-naive baseline (last week's same-day value) -- the honest benchmark.
  2. A machine-learning model on engineered calendar + lag features. The code
     auto-selects the best available backend: LightGBM -> sklearn
     HistGradientBoosting -> a dependency-free NumPy ridge regression, so the
     project runs anywhere (even before `pip install`).

Evaluation uses a time-based holdout (no leakage) and reports MAPE, WAPE,
bias, and RMSE -- the metrics supply-chain teams actually track.

Usage:
    python forecast.py                 # forecast all series, write metrics + charts
    python forecast.py --sku SKU_COLA_12PK --store STORE_ATX
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMRegressor
    _MODEL_NAME = "LightGBM"

    def _make_model():
        return LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=42, verbose=-1,
        )
except ImportError:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        _MODEL_NAME = "HistGradientBoosting"

        def _make_model():
            return HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.05, max_leaf_nodes=31, random_state=42,
            )
    except ImportError:
        # Final fallback: dependency-free ridge regression (NumPy only).
        _MODEL_NAME = "Ridge (NumPy fallback)"

        class _RidgeNP:
            def __init__(self, alpha=1.0):
                self.alpha, self.w, self.mu, self.sd = alpha, None, None, None

            def fit(self, X, y):
                X = np.asarray(X, float)
                self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
                Xs = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
                eye = np.eye(Xs.shape[1])
                eye[0, 0] = 0.0  # don't penalize the intercept
                self.w = np.linalg.solve(
                    Xs.T @ Xs + self.alpha * eye, Xs.T @ np.asarray(y, float)
                )
                return self

            def predict(self, X):
                X = np.asarray(X, float)
                Xs = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
                return Xs @ self.w

        def _make_model():
            return _RidgeNP(alpha=1.0)


HOLDOUT_DAYS = 90  # final 90 days are the test window


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true > 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def wape(y_true, y_pred):
    """Weighted absolute percentage error -- robust to zero-demand days."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.sum(np.abs(y_true))
    return np.sum(np.abs(y_true - y_pred)) / denom * 100 if denom else np.nan


def bias(y_true, y_pred):
    """Mean forecast error. Positive = over-forecasting."""
    return float(np.mean(np.asarray(y_pred, float) - np.asarray(y_true, float)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred, float) - np.asarray(y_true, float)) ** 2)))


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["day"] = df["date"].dt.day

    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = df["units_sold"].shift(lag)
    for win in (7, 28):
        df[f"roll_mean_{win}"] = df["units_sold"].shift(1).rolling(win).mean()
        df[f"roll_std_{win}"] = df["units_sold"].shift(1).rolling(win).std()

    return df


FEATURES = [
    "dow", "month", "weekofyear", "day", "on_promo", "is_holiday",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_mean_28", "roll_std_28",
]


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predict each test day using the value 7 days earlier (same weekday)."""
    full = pd.concat([train, test]).sort_values("date").reset_index(drop=True)
    pred = full["units_sold"].shift(7)
    return pred.iloc[-len(test):].fillna(train["units_sold"].mean()).values


def backtest_series(series: pd.DataFrame) -> dict:
    series = build_features(series).dropna(subset=["lag_28", "roll_std_28"])
    if len(series) < HOLDOUT_DAYS + 60:
        return {}

    train = series.iloc[:-HOLDOUT_DAYS]
    test = series.iloc[-HOLDOUT_DAYS:]

    model = _make_model()
    model.fit(train[FEATURES], train["units_sold"])
    ml_pred = np.clip(model.predict(test[FEATURES]), 0, None)

    naive_pred = seasonal_naive(train[["date", "units_sold"]], test[["date", "units_sold"]])

    y = test["units_sold"].values
    return {
        "dates": test["date"].values,
        "y_true": y,
        "ml_pred": ml_pred,
        "naive_pred": naive_pred,
        "metrics": {
            "model": _MODEL_NAME,
            "ml_mape": round(mape(y, ml_pred), 2),
            "ml_wape": round(wape(y, ml_pred), 2),
            "ml_bias": round(bias(y, ml_pred), 2),
            "ml_rmse": round(rmse(y, ml_pred), 2),
            "naive_mape": round(mape(y, naive_pred), 2),
            "naive_wape": round(wape(y, naive_pred), 2),
        },
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(sku: str | None = None, store: str | None = None):
    data = pd.read_csv("data/sales.csv", parse_dates=["date"])
    if sku:
        data = data[data["sku"] == sku]
    if store:
        data = data[data["store_id"] == store]

    results, rows = {}, []
    for (st, sk), grp in data.groupby(["store_id", "sku"]):
        res = backtest_series(grp)
        if not res:
            continue
        results[(st, sk)] = res
        m = res["metrics"]
        rows.append({"store_id": st, "sku": sk, **m})

    metrics_df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    metrics_df.to_csv("outputs/backtest_metrics.csv", index=False)

    if not metrics_df.empty:
        print(f"\nModel: {_MODEL_NAME}  |  Series evaluated: {len(metrics_df)}")
        print(f"Mean ML    WAPE: {metrics_df['ml_wape'].mean():.1f}%")
        print(f"Mean naive WAPE: {metrics_df['naive_wape'].mean():.1f}%")
        improvement = (1 - metrics_df["ml_wape"].mean() / metrics_df["naive_wape"].mean()) * 100
        print(f"ML beats seasonal-naive by {improvement:.1f}% on WAPE.\n")

    _plot(results)
    return metrics_df, results


def _plot(results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping charts.")
        return
    if not results:
        return

    os.makedirs("outputs", exist_ok=True)
    (st, sk), res = next(iter(results.items()))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(res["dates"], res["y_true"], label="Actual", color="#222", linewidth=1.6)
    ax.plot(res["dates"], res["ml_pred"], label=f"{_MODEL_NAME} forecast",
            color="#2563eb", linewidth=1.8)
    ax.plot(res["dates"], res["naive_pred"], label="Seasonal-naive",
            color="#f59e0b", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title(f"90-Day Backtest -- {sk} @ {st}")
    ax.set_ylabel("Units sold")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/forecast_backtest.png", dpi=130)
    print("Saved chart -> outputs/forecast_backtest.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sku", default=None)
    p.add_argument("--store", default=None)
    args = p.parse_args()
    run(sku=args.sku, store=args.store)
