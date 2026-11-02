#!/usr/bin/env python3
"""Calculate market-reaction targets from historical daily prices.

Run from any directory with:
    python 1_calculate_impacts_values.py

The script reads datasets/historical_prices.csv and writes
datasets/historical_impacts_values.csv. The S&P 500 index must use the symbol
``^sp500`` (case-insensitive), as specified in the assignment.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
INPUT_FILE = DATA_DIR / "historical_prices.csv"
OUTPUT_FILE = DATA_DIR / "historical_impacts_values.csv"

PRICE_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "market_cap",
]
OUTPUT_COLUMNS = [
    *PRICE_COLUMNS,
    "daily_return",
    "daily_volatility",
    "market_return",
    "alpha",
    "beta",
    "AR",
    "CAR",
    "Z_CAR",
    "IIS",
    "MIV",
]
MARKET_SYMBOL = "^SP500"
ESTIMATION_WINDOW = 120
ESTIMATION_GAP = 20
EVENT_WINDOW = 3
VOLATILITY_WINDOW = 60
MIN_CAR_STD = 1e-12


def load_prices(path: Path = INPUT_FILE) -> pd.DataFrame:
    """Read and validate the historical price file."""
    if not path.is_file():
        raise ValueError(
            f"Missing input file: {path}. Put historical_prices.csv in datasets/."
        )

    try:
        prices = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Cannot read historical prices CSV {path}: {exc}") from exc

    missing = sorted(set(PRICE_COLUMNS) - set(prices.columns))
    if missing:
        raise ValueError(f"historical_prices.csv is missing columns: {', '.join(missing)}")
    if prices.empty:
        raise ValueError("historical_prices.csv contains no records")

    prices = prices[PRICE_COLUMNS].copy()
    parsed_dates = pd.to_datetime(prices["date"], errors="coerce", utc=True)
    if parsed_dates.isna().any():
        raise ValueError(
            f"historical_prices.csv has {int(parsed_dates.isna().sum())} invalid dates"
        )
    prices["date"] = parsed_dates.dt.normalize()

    if prices["symbol"].isna().any():
        raise ValueError("historical_prices.csv has blank ticker symbols")
    prices["symbol"] = prices["symbol"].astype(str).str.strip().str.upper()
    if prices["symbol"].eq("").any():
        raise ValueError("historical_prices.csv has blank ticker symbols")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "market_cap",
    ]
    for column in numeric_columns:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")

    index_mask = prices["symbol"].eq(MARKET_SYMBOL)
    for column in numeric_columns[:-1]:
        values = prices[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"historical_prices.csv has missing or non-numeric {column} values"
            )
    if not np.isfinite(prices["adj_close"].to_numpy(dtype=float)).all():
        raise ValueError("historical_prices.csv has missing or non-numeric adj_close values")
    if (prices["adj_close"] <= 0).any() or (prices["close"] <= 0).any():
        raise ValueError("close and adj_close prices must be greater than zero")

    stock_mask = ~index_mask
    stock_caps = prices.loc[stock_mask, "market_cap"].to_numpy(dtype=float)
    if not np.isfinite(stock_caps).all() or (stock_caps <= 0).any():
        raise ValueError(
            "Company rows must have finite, positive market_cap values to calculate MIV"
        )
    prices.loc[index_mask, "market_cap"] = np.nan

    if not index_mask.any():
        raise ValueError(
            "No S&P 500 index rows found. Add the market series with symbol '^sp500'."
        )

    duplicate_rows = prices.duplicated(["date", "symbol"], keep=False)
    if duplicate_rows.any():
        examples = prices.loc[duplicate_rows, ["date", "symbol"]].head(3)
        details = ", ".join(
            f"{symbol} on {date}"
            for date, symbol in examples.astype({"date": "string"}).itertuples(
                index=False, name=None
            )
        )
        raise ValueError(
            "historical_prices.csv must contain at most one row per date and symbol; "
            f"duplicates include {details}"
        )

    return prices.sort_values(["symbol", "date"]).reset_index(drop=True)


def information_impact_score(z_score: float) -> int:
    """Map a CAR z-score to the assignment's discrete IIS thresholds."""
    if z_score <= -3:
        return -3
    if z_score <= -2:
        return -2
    if z_score <= -1:
        return -1
    if z_score < 1:
        return 0
    if z_score < 2:
        return 1
    if z_score < 3:
        return 2
    return 3


def _prefix_sums(values: np.ndarray) -> np.ndarray:
    """Return a leading-zero cumulative sum for constant-time range sums."""
    return np.concatenate(([0.0], np.cumsum(values, dtype=float)))


def _range_sum(prefix: np.ndarray, start: int, stop: int) -> float:
    """Sum values in the half-open interval [start, stop)."""
    return float(prefix[stop] - prefix[start])


def calculate_impacts(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate returns, market-model abnormal returns, CAR, IIS, and MIV.

    For event index t, OLS parameters use returns t-140 through t-21,
    inclusive. The same parameters are used for the event window and for the
    historical three-day CAR distribution used to standardize that event.
    """
    missing = sorted(set(PRICE_COLUMNS) - set(prices.columns))
    if missing:
        raise ValueError(f"Price data is missing columns: {', '.join(missing)}")
    data = prices[PRICE_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True).dt.normalize()
    if data["date"].isna().any():
        raise ValueError("Price data contains invalid dates")
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    if data.duplicated(["date", "symbol"]).any():
        raise ValueError("Price data must contain at most one row per date and symbol")
    if not data["symbol"].eq(MARKET_SYMBOL).any():
        raise ValueError("Price data must include the S&P 500 index with symbol '^sp500'")

    data = data.sort_values(["symbol", "date"]).reset_index(drop=True)
    previous_adj_close = data.groupby("symbol", sort=False)["adj_close"].shift(1)
    data["daily_return"] = data["adj_close"].div(previous_adj_close) - 1
    data["daily_volatility"] = data.groupby("symbol", sort=False)["daily_return"].transform(
        lambda returns: returns.rolling(
            VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW
        ).std()
    )
    data["previous_market_cap"] = data.groupby("symbol", sort=False)["market_cap"].shift(1)

    market_returns = (
        data.loc[data["symbol"].eq(MARKET_SYMBOL), ["date", "daily_return"]]
        .rename(columns={"daily_return": "market_return"})
        .dropna(subset=["market_return"])
    )
    stocks = data.loc[~data["symbol"].eq(MARKET_SYMBOL)].merge(
        market_returns,
        on="date",
        how="inner",
        validate="many_to_one",
    )
    if stocks.empty:
        raise ValueError("No company returns align with available S&P 500 returns")

    records: list[dict[str, object]] = []
    skipped: Counter[str] = Counter()
    for symbol, company in stocks.groupby("symbol", sort=False):
        company = company.sort_values("date").reset_index(drop=True)
        stock_returns = company["daily_return"].to_numpy(dtype=float)
        market_values = company["market_return"].to_numpy(dtype=float)
        company_market_caps = company["market_cap"].to_numpy(dtype=float)
        count = len(company)
        candidate_count = max(count - EVENT_WINDOW + 1, 0)
        skipped["insufficient estimation history and gap"] += min(
            ESTIMATION_WINDOW + ESTIMATION_GAP, candidate_count
        )
        if count < ESTIMATION_WINDOW + ESTIMATION_GAP + EVENT_WINDOW:
            continue

        # Three-day rolling CARs within a 120-return estimation window.
        historical_stock_cars = np.convolve(
            stock_returns, np.ones(EVENT_WINDOW), mode="valid"
        )
        historical_market_cars = np.convolve(
            market_values, np.ones(EVENT_WINDOW), mode="valid"
        )
        stock_prefix = _prefix_sums(stock_returns)
        market_prefix = _prefix_sums(market_values)
        market_square_prefix = _prefix_sums(market_values**2)
        cross_product_prefix = _prefix_sums(stock_returns * market_values)
        car_stock_prefix = _prefix_sums(historical_stock_cars)
        car_market_prefix = _prefix_sums(historical_market_cars)

        for event_index in range(
            ESTIMATION_WINDOW + ESTIMATION_GAP,
            count - EVENT_WINDOW + 1,
        ):
            window_start = event_index - ESTIMATION_GAP - ESTIMATION_WINDOW
            window_stop = event_index - ESTIMATION_GAP
            window_count = ESTIMATION_WINDOW

            sum_market = _range_sum(market_prefix, window_start, window_stop)
            sum_stock = _range_sum(stock_prefix, window_start, window_stop)
            sum_market_squared = _range_sum(
                market_square_prefix, window_start, window_stop
            )
            sum_cross = _range_sum(
                cross_product_prefix, window_start, window_stop
            )
            centered_market_sum = (
                sum_market_squared - sum_market * sum_market / window_count
            )
            if centered_market_sum <= np.finfo(float).eps:
                skipped["constant market returns in estimation window"] += 1
                continue

            beta = (
                sum_cross - sum_market * sum_stock / window_count
            ) / centered_market_sum
            alpha = (sum_stock - beta * sum_market) / window_count

            # The 118 overlapping three-day CARs must remain inside the
            # estimation window; no event-window return enters this baseline.
            car_start = window_start
            car_stop = window_stop - EVENT_WINDOW + 1
            car_count = car_stop - car_start
            if car_count < 2:
                skipped["insufficient historical CARs"] += 1
                continue
            sum_stock_car = _range_sum(car_stock_prefix, car_start, car_stop)
            sum_market_car = _range_sum(car_market_prefix, car_start, car_stop)
            centered_stock_cars = (
                historical_stock_cars[car_start:car_stop]
                - sum_stock_car / car_count
            )
            centered_market_cars = (
                historical_market_cars[car_start:car_stop]
                - sum_market_car / car_count
            )
            centered_historical_cars = (
                centered_stock_cars - beta * centered_market_cars
            )
            historical_car_variance = float(
                np.dot(centered_historical_cars, centered_historical_cars)
                / (car_count - 1)
            )
            historical_car_std = float(np.sqrt(historical_car_variance))
            if not np.isfinite(historical_car_std) or historical_car_std <= MIN_CAR_STD:
                skipped["zero or negligible historical CAR standard deviation"] += 1
                continue

            event_stock_car = _range_sum(
                stock_prefix, event_index, event_index + EVENT_WINDOW
            )
            event_market_car = _range_sum(
                market_prefix, event_index, event_index + EVENT_WINDOW
            )
            car = event_stock_car - EVENT_WINDOW * alpha - beta * event_market_car
            historical_car_mean = (
                sum_stock_car - EVENT_WINDOW * alpha * car_count - beta * sum_market_car
            ) / car_count
            z_score = (car - historical_car_mean) / historical_car_std
            pre_event_market_cap = company_market_caps[event_index - 1]
            if not np.isfinite(pre_event_market_cap):
                skipped["missing pre-event market capitalization"] += 1
                continue

            event = company.iloc[event_index]
            records.append(
                {
                    "date": event["date"].strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    **{column: event[column] for column in PRICE_COLUMNS[2:]},
                    "daily_return": stock_returns[event_index],
                    "daily_volatility": event["daily_volatility"],
                    "market_return": market_values[event_index],
                    "alpha": alpha,
                    "beta": beta,
                    "AR": stock_returns[event_index]
                    - alpha
                    - beta * market_values[event_index],
                    "CAR": car,
                    "Z_CAR": z_score,
                    "IIS": information_impact_score(z_score),
                    "MIV": pre_event_market_cap * car,
                }
            )

    if skipped:
        print(
            "[INFO] Excluded events: "
            + ", ".join(f"{reason}: {count:,}" for reason, count in sorted(skipped.items()))
        )
    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)


def main() -> int:
    try:
        prices = load_prices(INPUT_FILE)
        impacts = calculate_impacts(prices)
        if impacts.empty:
            raise ValueError(
                "No events could be scored. Each company needs enough consecutive "
                "price history for the 120-day estimation window, 20-day gap, "
                "three-day event window, and a nonzero historical CAR standard deviation."
            )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        impacts.to_csv(OUTPUT_FILE, index=False)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(
        f"[OK] Wrote {len(impacts):,} scored company-day events to {OUTPUT_FILE}"
    )
    print(
        f"[OK] Columns: {', '.join(OUTPUT_COLUMNS)} "
        f"(120-return estimation window, 20-trading-day gap)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())