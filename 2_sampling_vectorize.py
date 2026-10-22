"""CPSC 488 Assignment 3 — Increment 1: inspect and validate inputs.

Run from the Assignment_3 project directory:
    python 2_sampling_vectorize.py
    python 2_sampling_vectorize.py --test

Expected real inputs (place in datasets/):
    news.csv
    historical_impacts_values.csv

Test mode uses the professor's sample files, skipping rows with missing
ticker symbols with an explicit warning. Production mode stays strict.
This increment deliberately does not produce training data yet.
"""
from pathlib import Path
import argparse
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'datasets'
NEWS = DATA / 'news.csv'
IMPACTS = DATA / 'historical_impacts_values.csv'

REQUIRED = {
    'news': {'date', 'symbol', 'headline', 'URL', 'publisher'},
    'impacts': {'date', 'symbol', 'IIS', 'MIV'},
}


def read_checked(path: Path, name: str, *, test: bool = False) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f'{name}: missing file: {path}. Put the generated CSV in datasets/.')
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f'{name}: cannot read CSV {path}: {exc}') from exc
    missing = sorted(REQUIRED[name] - set(df.columns))
    if missing:
        raise ValueError(f'{name}: missing columns: {", ".join(missing)}')
    if df.empty:
        raise ValueError(f'{name}: file contains no records')
    if test and name == 'news':
        bad_symbol = df['symbol'].isna() | df['symbol'].astype(str).str.strip().eq('')
        if bad_symbol.any():
            print(f'[TEST WARNING] news: skipping {int(bad_symbol.sum())} sample row(s) '
                  'with blank symbol; these would be errors in production.')
            df = df.loc[~bad_symbol].copy()
            if df.empty:
                raise ValueError('news: no valid sample rows remain')
    for col in ('date', 'symbol'):
        if df[col].isna().any() or df[col].astype(str).str.strip().eq('').any():
            count = int((df[col].isna() | df[col].astype(str).str.strip().eq('')).sum())
            raise ValueError(f'{name}: {count} rows have blank {col}; review before proceeding')
    dates = pd.to_datetime(df['date'], errors='coerce', utc=True)
    if dates.isna().any():
        raise ValueError(f'{name}: {int(dates.isna().sum())} rows have unparseable dates')
    if name == 'news':
        if df['headline'].isna().any() or df['headline'].astype(str).str.strip().eq('').any():
            raise ValueError('news: one or more rows have blank headlines')
    else:
        for col in ('IIS', 'MIV'):
            values = pd.to_numeric(df[col], errors='coerce')
            if values.isna().any():
                raise ValueError(f'impacts: {int(values.isna().sum())} invalid or missing {col} values')
    print(f'[OK] {name}: {len(df):,} records, {len(df.columns)} columns, dates parsed')
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate Assignment 3 CSV inputs')
    parser.add_argument('--test', action='store_true',
                        help='read the professor sample CSVs (not production data)')
    args = parser.parse_args()
    news_path = DATA / 'sample_news.csv' if args.test else NEWS
    impacts_path = (DATA / 'sample_historical_impacts_values.csv' if args.test else IMPACTS)
    print(f"[MODE] {'SAMPLE TEST' if args.test else 'PRODUCTION'}")
    try:
        read_checked(news_path, 'news', test=args.test)
        read_checked(impacts_path, 'impacts', test=args.test)
    except ValueError as exc:
        print(f'[ACTION NEEDED] {exc}', file=sys.stderr)
        return 1
    print('[OK] Input validation complete; sampling/vectorization not implemented yet.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
