#!/usr/bin/env python3
"""Prepare production news and historical-price inputs for Assignment 3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "news_datasets"
DATA_DIR = ROOT / "datasets"
NEWS_OUTPUT = DATA_DIR / "news.csv"
PRICES_OUTPUT = DATA_DIR / "historical_prices.csv"
SP500_OUTPUT = DATA_DIR / "sp500.csv"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
PRICE_START = "2008-01-01"
PRICE_END = "2021-06-01"
NEWS_START = pd.Timestamp("2009-01-01", tz="UTC")
NEWS_END = pd.Timestamp("2021-01-01", tz="UTC")
NEWS_COLUMNS = ["date", "symbol", "headline", "URL", "publisher"]
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


def load_sp500_constituents() -> pd.DataFrame:
    tables = pd.read_html(
        SP500_URL,
        storage_options={"User-Agent": "Mozilla/5.0"},
    )
    constituents = tables[0][["Symbol", "Security"]].copy()
    constituents.columns = ["ticker", "company_name"]
    constituents["ticker"] = (
        constituents["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )
    constituents = constituents.drop_duplicates("ticker").sort_values("ticker")
    return constituents.reset_index(drop=True)


def current_shares_outstanding(ticker: yf.Ticker) -> float | None:
    shares = None
    try:
        shares = ticker.fast_info.get("shares")
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    if shares is None:
        try:
            shares = ticker.info.get("sharesOutstanding")
        except (KeyError, TypeError, ValueError, AttributeError):
            pass
    if shares is None or not np.isfinite(float(shares)) or float(shares) <= 0:
        return None
    return float(shares)


def split_adjusted_shares(
    dates: pd.Series,
    current_shares: float,
    splits: pd.Series,
) -> np.ndarray:
    if splits.empty:
        return np.full(len(dates), current_shares, dtype=float)

    split_series = splits.copy()
    split_series.index = pd.to_datetime(split_series.index, utc=True).tz_localize(None)
    split_series = pd.to_numeric(split_series, errors="coerce")
    split_series = split_series[np.isfinite(split_series) & (split_series > 0)]
    split_series = split_series.sort_index()
    if split_series.empty:
        return np.full(len(dates), current_shares, dtype=float)

    split_dates = split_series.index.to_numpy(dtype="datetime64[ns]")
    split_values = split_series.to_numpy(dtype=float)
    future_products = np.cumprod(split_values[::-1])[::-1]
    price_dates = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(split_dates, price_dates, side="right")
    factors = np.ones(len(price_dates), dtype=float)
    has_future_split = positions < len(split_values)
    factors[has_future_split] = future_products[positions[has_future_split]]
    return current_shares / factors


def download_symbol_prices(symbol: str) -> pd.DataFrame | None:
    downloaded = yf.download(
        symbol,
        start=PRICE_START,
        end=PRICE_END,
        auto_adjust=False,
        progress=False,
    )
    if downloaded.empty:
        return None
    if isinstance(downloaded.columns, pd.MultiIndex):
        downloaded.columns = downloaded.columns.get_level_values(0)

    downloaded = downloaded.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    required = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    if missing := sorted(set(required) - set(downloaded.columns)):
        raise ValueError(f"{symbol}: downloaded prices are missing {', '.join(missing)}")
    downloaded = downloaded[required].dropna().copy()
    if downloaded.empty:
        return None

    output_symbol = "^SP500" if symbol == "^GSPC" else symbol
    downloaded["symbol"] = output_symbol
    if symbol == "^GSPC":
        downloaded["market_cap"] = np.nan
    else:
        ticker = yf.Ticker(symbol)
        shares = current_shares_outstanding(ticker)
        if shares is None:
            return None
        try:
            splits = ticker.splits
        except (KeyError, TypeError, ValueError, AttributeError):
            splits = pd.Series(dtype=float)
        historical_shares = split_adjusted_shares(downloaded["date"], shares, splits)
        downloaded["market_cap"] = historical_shares * downloaded["close"].to_numpy()

    downloaded["date"] = pd.to_datetime(downloaded["date"]).dt.strftime("%Y-%m-%d")
    return downloaded[PRICE_COLUMNS]


def download_prices(symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    successful: list[str] = []
    failed: list[str] = []
    requested = [*symbols, "^GSPC"]
    for index, symbol in enumerate(requested, start=1):
        try:
            prices = download_symbol_prices(symbol)
        except Exception as exc:
            print(f"[WARNING] {symbol}: price download failed: {exc}", file=sys.stderr)
            prices = None
        if prices is None:
            failed.append(symbol)
        else:
            frames.append(prices)
            if symbol != "^GSPC":
                successful.append(symbol)
        print(f"[PRICE] {index}/{len(requested)} {symbol}")

    if "^GSPC" in failed:
        raise ValueError("S&P 500 index download failed")
    if not successful:
        raise ValueError("No company price histories could be prepared")
    if failed:
        print(f"[WARNING] Skipped {len(failed)} ticker(s): {', '.join(failed)}")
    return pd.concat(frames, ignore_index=True), successful


def join_unique(values: pd.Series) -> str:
    unique = pd.unique(values.dropna().astype(str).str.strip())
    return " || ".join(value for value in unique if value)


def load_news(symbols: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    source_columns = ["headline", "url", "publisher", "date", "stock"]
    for source in (SOURCE_DIR / "analyst_ratings.csv", SOURCE_DIR / "headlines.csv"):
        if not source.is_file():
            raise ValueError(f"Missing source news file: {source}")
        kept_rows = 0
        for chunk in pd.read_csv(source, usecols=source_columns, chunksize=200_000):
            chunk["symbol"] = (
                chunk["stock"]
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(".", "-", regex=False)
            )
            chunk = chunk.loc[chunk["symbol"].isin(symbols)].copy()
            if chunk.empty:
                continue
            chunk["date"] = pd.to_datetime(
                chunk["date"],
                format="mixed",
                errors="coerce",
                utc=True,
            )
            chunk = chunk.loc[
                chunk["date"].notna()
                & chunk["date"].ge(NEWS_START)
                & chunk["date"].lt(NEWS_END)
            ].copy()
            chunk = chunk.rename(columns={"url": "URL"})
            chunk["date"] = chunk["date"].dt.strftime("%Y-%m-%d")
            chunk = chunk[NEWS_COLUMNS]
            frames.append(chunk)
            kept_rows += len(chunk)
        print(f"[NEWS] {source.name}: retained {kept_rows:,} rows")

    if not frames:
        raise ValueError("No news rows matched the successfully downloaded tickers")
    news = pd.concat(frames, ignore_index=True)
    news = news.dropna(subset=["headline"])
    news = news.loc[news["headline"].astype(str).str.strip().ne("")].copy()
    news = news.drop_duplicates(["date", "symbol", "headline", "URL"])
    grouped = news.groupby(["date", "symbol"], as_index=False, sort=True).agg(
        headline=("headline", join_unique),
        URL=("URL", join_unique),
        publisher=("publisher", join_unique),
    )
    return grouped[NEWS_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare datasets/news.csv and datasets/historical_prices.csv"
    )
    parser.add_argument(
        "--ticker-limit",
        type=int,
        default=None,
        help="Limit companies for a practical smoke test; omit for all S&P 500 companies",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ticker_limit is not None and args.ticker_limit < 1:
        print("[ERROR] --ticker-limit must be at least 1", file=sys.stderr)
        return 1

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        constituents = load_sp500_constituents()
        if args.ticker_limit is not None:
            constituents = constituents.head(args.ticker_limit).copy()
        constituents.to_csv(SP500_OUTPUT, index=False)
        print(f"[OK] Selected {len(constituents):,} S&P 500 companies")

        prices, successful = download_prices(constituents["ticker"].tolist())
        news = load_news(set(successful))
        prices.to_csv(PRICES_OUTPUT, index=False)
        news.to_csv(NEWS_OUTPUT, index=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Wrote {len(prices):,} price rows to {PRICES_OUTPUT}")
    print(f"[OK] Wrote {len(news):,} company-day news rows to {NEWS_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
