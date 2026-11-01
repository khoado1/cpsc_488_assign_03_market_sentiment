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
from bs4 import BeautifulSoup
import re
import contractions
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'datasets'
NEWS = DATA / 'news.csv'
IMPACTS = DATA / 'historical_impacts_values.csv'

REQUIRED = {
    'news': {'date', 'symbol', 'headline', 'URL', 'publisher'},
    'impacts': {'date', 'symbol', 'IIS', 'MIV'},
}

news_start_years = [2009, 2010, 2011, 2012, 2013, 2014]
news_period = 7

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

def preprocess_news(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the news DataFrame for vectorization.

    This function is a placeholder for future preprocessing steps

    """

    # Preprocessing steps would go here

    for col in ['headline', 'article']:
        if col == 'article':
            df[col] = df[col].apply(clean_html_text)

        df[col] = (
            df[col]
            .str.lower()
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )

        df[col] = df[col].apply(contractions.fix)

    df['text'] = df['headline'] + ' ' + df['article']

    return df

def clean_html_text(raw_text: str | None) -> str:
    """Remove HTML tags and unescape HTML entities from a string."""

    if (raw_text is None) or raw_text.strip() == '':
        return ''
    
    soup = BeautifulSoup(raw_text or '', 'html.parser')

    # Prefer the article/main content when the input is a full HTML document
    content = soup.find(['article', 'main' ]) or soup
    text = content.get_text(separator=' ', strip=True)

    # Remove non-content elements that are likely to be ads or navigation
    for ele in content.find_all(['script', 'style', 'noscript', 'nav', 'footer', 'header', 'aside', 'iframe', 'form']):
        ele.decompose()

    # Remove common ad/promotional markers in IDs and classes
    ad_marker = re.compile(
        r"\b(?:ad|ads|advert|advertisement|sponsored|promo)\b",
        re.IGNORECASE
    )

    for ele in list(content.find_all(True)):
        labels = [ele.get('id', '')]

        class_names = ele.get('class')
        if isinstance(class_names, str):
            labels.append(class_names)
        elif isinstance(class_names, (list, tuple)):
            labels.extend(class_names)
            
        if any(ad_marker.search(str(lbl)) for lbl in labels):
            ele.decompose()

    return content.get_text(separator=' ', strip=True)

def filter_on_range(news_df: pd.DataFrame, impacts_df: pd.DataFrame, start_years: list, period: int) -> tuple[pd.DataFrame, pd.DataFrame, Timestamp]:
    # Placeholder for range-based preprocessing logic
    np.random.seed(42)  # For reproducibility

    start_year = np.random.choice(start_years)
    end_year = start_year + period
    start = pd.Timestamp(year=start_year, month=1, day=1, tz='UTC')
    period_end = pd.Timestamp(year=end_year, month=1, day=1, tz='UTC')

    for df in (news_df, impacts_df):
        df['date'] = pd.to_datetime(df['date'], utc=True)

    news_df = news_df[(news_df['date'] >= start) & (news_df['date'] < period_end)]
    impacts_df = impacts_df[(impacts_df['date'] >= start) & (impacts_df['date'] < period_end)]

    return news_df, impacts_df, period_end

def match_news_to_impacts(news_df: pd.DataFrame, impacts_df: pd.DataFrame) -> pd.DataFrame:
    labels = impacts_df[['date', 'symbol', 'IIS', 'MIV']]

    return news_df.merge(
        labels,
        on=['date', 'symbol'],
        how='inner',
        validate='many_to_one'
    )

def main() -> int:
    parser = argparse.ArgumentParser(description='Validate Assignment 3 CSV inputs')
    parser.add_argument('--test', action='store_true',
                        help='read the professor sample CSVs (not production data)')
    args = parser.parse_args()
    news_path = DATA / 'sample_news.csv' if args.test else NEWS
    impacts_path = (DATA / 'sample_historical_impacts_values.csv' if args.test else IMPACTS)
    print(f"[MODE] {'SAMPLE TEST' if args.test else 'PRODUCTION'}")
    try:
        news_df = read_checked(news_path, 'news', test=args.test)
        news_df = preprocess_news(news_df)
        impacts_df = read_checked(impacts_path, 'impacts', test=args.test)
        news_df, impacts_df, period_end = filter_on_range(news_df, impacts_df, news_start_years, news_period)

        matched_df = match_news_to_impacts(news_df, impacts_df)

        matched_df = matched_df.sort_values(['date', 'symbol']).reset_index(drop=True)

        train_share = 0.8
        window_size = 3

        event_dates = matched_df['date'].drop_duplicates().sort_values().reset_index(drop=True)
        split_date = event_dates.iloc[int(train_share * len(event_dates))]

        date_positions = pd.Series(
            range(len(event_dates)),
            index=event_dates,
        )
        event_positions = matched_df['date'].map(date_positions)
        has_full_window = (
            event_positions.notna()
            & (event_positions + window_size - 1 < len(event_dates))
        )

        window_end = pd.Series(pd.NaT, index=matched_df.index, dtype='datetime64[ns, UTC]')
        end_positions = (event_positions.loc[has_full_window] + window_size - 1).astype(int)
        window_end.loc[has_full_window] = event_dates.take(end_positions.to_numpy())

        train_mask = (
            (matched_df['date'] < split_date)
            & window_end.notna()
            & (window_end < split_date)
        )
        test_mask = (
            (matched_df['date'] >= split_date)
            & window_end.notna()
            & (window_end < period_end)
        )

        train_df = matched_df.loc[train_mask].copy()
        test_df = matched_df.loc[test_mask].copy()


    except ValueError as exc:
        print(f'[ACTION NEEDED] {exc}', file=sys.stderr)
        return 1
    print('[OK] Input validation complete; sampling/vectorization not implemented yet.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
