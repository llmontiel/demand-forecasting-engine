"""
generate_data.py
----------------
Generate synthetic daily grocery sales data for the demand forecasting engine.

Produces realistic retail demand with:
  - Multiple stores x multiple products (SKUs)
  - Long-term growth trend
  - Yearly + weekly seasonality
  - Holiday/promo lift spikes
  - Random noise and occasional stockout (zero-sales) days

No real or confidential data is used. Everything is simulated.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

STORES = ["STORE_ATX", "STORE_HOU", "STORE_DAL", "STORE_SAT"]
PRODUCTS = {
    "SKU_COLA_12PK": {"base": 40, "price": 6.99, "promo_lift": 1.9},
    "SKU_CEREAL_OAT": {"base": 25, "price": 4.49, "promo_lift": 1.5},
    "SKU_CANDY_BAR": {"base": 60, "price": 1.29, "promo_lift": 2.3},
    "SKU_CHIPS_FAMILY": {"base": 35, "price": 3.99, "promo_lift": 1.7},
    "SKU_WATER_24PK": {"base": 50, "price": 5.49, "promo_lift": 1.4},
}

START = "2023-01-01"
END = "2025-12-31"

# US-ish holiday dates (month, day) that drive demand spikes
HOLIDAYS = [(1, 1), (2, 14), (5, 25), (7, 4), (9, 1), (11, 27), (12, 24), (12, 25), (12, 31)]


def _seasonal_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Yearly seasonality: higher in summer and december."""
    day_of_year = dates.dayofyear.values
    yearly = 1.0 + 0.18 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    # December holiday bump
    dec_bump = np.where(dates.month.values == 12, 0.15, 0.0)
    return yearly + dec_bump


def _weekly_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Weekends sell more."""
    dow = dates.dayofweek.values  # 0=Mon
    weekly = np.ones(len(dates))
    weekly[dow == 4] *= 1.15  # Fri
    weekly[dow == 5] *= 1.35  # Sat
    weekly[dow == 6] *= 1.25  # Sun
    return weekly


def _is_holiday(dates: pd.DatetimeIndex) -> np.ndarray:
    flags = np.zeros(len(dates), dtype=bool)
    for m, d in HOLIDAYS:
        flags |= (dates.month.values == m) & (dates.day.values == d)
    return flags


def generate() -> pd.DataFrame:
    dates = pd.date_range(START, END, freq="D")
    rows = []

    seasonal = _seasonal_multiplier(dates)
    weekly = _weekly_multiplier(dates)
    holiday = _is_holiday(dates)
    # linear growth over the window (newer = more demand)
    trend = np.linspace(1.0, 1.25, len(dates))

    for store in STORES:
        store_factor = RNG.uniform(0.8, 1.2)
        for sku, cfg in PRODUCTS.items():
            base = cfg["base"] * store_factor
            # random promo calendar: ~8% of days are on promo for this SKU
            promo = RNG.random(len(dates)) < 0.08
            lift = np.where(promo, cfg["promo_lift"], 1.0)
            holiday_lift = np.where(holiday, 1.6, 1.0)

            expected = base * seasonal * weekly * trend * lift * holiday_lift
            # Poisson noise around expected demand
            units = RNG.poisson(np.clip(expected, 0, None))

            # occasional stockouts: ~2% of days sell zero despite demand
            stockout = RNG.random(len(dates)) < 0.02
            units = np.where(stockout, 0, units)

            df = pd.DataFrame(
                {
                    "date": dates,
                    "store_id": store,
                    "sku": sku,
                    "units_sold": units,
                    "unit_price": cfg["price"],
                    "on_promo": promo.astype(int),
                    "is_holiday": holiday.astype(int),
                }
            )
            rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out["revenue"] = (out["units_sold"] * out["unit_price"]).round(2)
    return out


if __name__ == "__main__":
    data = generate()
    data.to_csv("data/sales.csv", index=False)
    print(f"Generated {len(data):,} rows across "
          f"{data['store_id'].nunique()} stores and {data['sku'].nunique()} SKUs.")
    print(f"Date range: {data['date'].min().date()} -> {data['date'].max().date()}")
    print(data.head())
