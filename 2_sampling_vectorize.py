"""CPSC 488 Assignment 3 — sample, match, and vectorize news events.

Run from the Assignment_3 project directory:
    python 2_sampling_vectorize.py
    python 2_sampling_vectorize.py --test

Expected real inputs (place in datasets/):
    news.csv
    historical_impacts_values.csv

Test mode uses the professor's sample files, skipping rows with missing
ticker symbols with an explicit warning. Production mode stays strict.
The four vectorized train/test datasets are written to datasets/.
"""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd
from bs4 import BeautifulSoup
import re
import contractions
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
train_share = 0.8
event_window_size = 3
OUTPUT_COLUMNS = ['date', 'symbol', 'news_vector', 'IIS', 'MIV']

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
    df = df.copy()
    for col in ('date', 'symbol'):
        if df[col].isna().any() or df[col].astype(str).str.strip().eq('').any():
            count = int((df[col].isna() | df[col].astype(str).str.strip().eq('')).sum())
            raise ValueError(f'{name}: {count} rows have blank {col}; review before proceeding')
    dates = pd.to_datetime(df['date'], errors='coerce', utc=True)
    if dates.isna().any():
        raise ValueError(f'{name}: {int(dates.isna().sum())} rows have unparseable dates')
    df['date'] = dates.dt.normalize()
    df['symbol'] = df['symbol'].astype(str).str.strip().str.upper()
    if name == 'news':
        if df['headline'].isna().any() or df['headline'].astype(str).str.strip().eq('').any():
            raise ValueError('news: one or more rows have blank headlines')
    else:
        for col in ('IIS', 'MIV'):
            values = pd.to_numeric(df[col], errors='coerce')
            if values.isna().any():
                raise ValueError(f'impacts: {int(values.isna().sum())} invalid or missing {col} values')
            df[col] = values
    print(f'[OK] {name}: {len(df):,} records, {len(df.columns)} columns, dates parsed')
    return df

def preprocess_news(df: pd.DataFrame) -> pd.DataFrame:
    """Clean article/headline text and produce one document per company-day."""
    df = df.copy()
    if 'article' not in df.columns:
        df['article'] = ''

    for col in ('headline', 'article'):
        df[col] = df[col].fillna('').astype(str).apply(clean_html_text)
        df[col] = df[col].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
        df[col] = df[col].apply(contractions.fix)

    df['text'] = (df['headline'] + ' ' + df['article']).str.strip()
    df = df.groupby(['date', 'symbol'], as_index=False, sort=False).agg(
        text=('text', lambda documents: ' '.join(text for text in documents if text))
    )
    if df.empty or df['text'].str.strip().eq('').all():
        raise ValueError('news: no non-empty company-day text remains after preprocessing')

    return df

def clean_html_text(raw_text: str | None) -> str:
    """Remove HTML tags and unescape HTML entities from a string."""

    if (raw_text is None) or raw_text.strip() == '':
        return ''
    
    soup = BeautifulSoup(raw_text or '', 'html.parser')

    # Prefer the article/main content when the input is a full HTML document
    content = soup.find(['article', 'main' ]) or soup

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

def filter_on_range(news_df: pd.DataFrame, impacts_df: pd.DataFrame, start_years: list, period: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    np.random.seed(42)  # For reproducibility

    start_year = np.random.choice(start_years)
    end_year = start_year + period
    start = pd.Timestamp(year=start_year, month=1, day=1, tz='UTC')
    period_end = pd.Timestamp(year=end_year, month=1, day=1, tz='UTC')

    news_df = news_df.copy()
    impacts_df = impacts_df.copy()
    for df in (news_df, impacts_df):
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.normalize()

    news_df = news_df[(news_df['date'] >= start) & (news_df['date'] < period_end)]
    impacts_df = impacts_df[(impacts_df['date'] >= start) & (impacts_df['date'] < period_end)]

    return news_df, impacts_df, period_end

def match_news_to_impacts(news_df: pd.DataFrame, impacts_df: pd.DataFrame) -> pd.DataFrame:
    labels = impacts_df[['date', 'symbol', 'IIS', 'MIV']].copy()
    labels['date'] = pd.to_datetime(labels['date'], utc=True).dt.normalize()
    labels['symbol'] = labels['symbol'].astype(str).str.strip().str.upper()

    try:
        matched = news_df.merge(
            labels,
            on=['date', 'symbol'],
            how='inner',
            validate='many_to_one'
        )
    except pd.errors.MergeError as exc:
        raise ValueError(
            'impacts: expected at most one target row per date and symbol'
        ) from exc
    if matched.empty:
        raise ValueError('No news events matched historical impacts in the selected period')
    return matched.sort_values(['date', 'symbol']).reset_index(drop=True)

def split_on_event_window(
    matched_df: pd.DataFrame,
    impacts_df: pd.DataFrame,
    share: float = train_share,
    window_size: int = event_window_size,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split chronologically and keep event windows entirely within each split."""
    trading_dates = pd.Index(
        pd.to_datetime(impacts_df['date'], utc=True).dt.normalize().drop_duplicates().sort_values()
    )
    if len(trading_dates) < window_size:
        raise ValueError(
            f'Need at least {window_size} trading dates for the event window; '
            f'found {len(trading_dates)}'
        )

    split_index = int(share * len(trading_dates))
    if split_index < window_size or len(trading_dates) - split_index < window_size:
        raise ValueError(
            f'Not enough trading dates to keep {window_size}-day event windows '
            f'in both partitions ({split_index} training, '
            f'{len(trading_dates) - split_index} testing dates)'
        )

    date_positions = pd.Series(np.arange(len(trading_dates)), index=trading_dates)
    positions = pd.to_datetime(matched_df['date'], utc=True).dt.normalize().map(date_positions)
    if positions.isna().any():
        raise ValueError('Matched news contains dates absent from the trading calendar')

    train_mask = positions + window_size <= split_index
    test_mask = (
        (positions >= split_index)
        & (positions + window_size <= len(trading_dates))
    )
    train_df = matched_df.loc[train_mask].copy()
    test_df = matched_df.loc[test_mask].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            'No observations remain in one or both partitions after excluding '
            f'{window_size}-trading-day event windows that cross a boundary'
        )

    return train_df, test_df, trading_dates[split_index]

def vectorize_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Fit DTM and TF-IDF on training text only, then transform both splits."""
    outputs: dict[str, pd.DataFrame] = {}
    train_documents = train_df['text'].fillna('').astype(str).tolist()
    test_documents = test_df['text'].fillna('').astype(str).tolist()
    metadata_columns = ['date', 'symbol', 'IIS', 'MIV']

    vectorizers = {
        'dtm': CountVectorizer(),
        'tfidf': TfidfVectorizer(),
    }
    for name, vectorizer in vectorizers.items():
        try:
            train_matrix = vectorizer.fit_transform(train_documents)
            test_matrix = vectorizer.transform(test_documents)
        except ValueError as exc:
            raise ValueError(f'{name.upper()} vectorization failed: {exc}') from exc

        for split_name, source, matrix in (
            ('train', train_df, train_matrix),
            ('test', test_df, test_matrix),
        ):
            result = source[metadata_columns].copy()
            result['date'] = pd.to_datetime(result['date'], utc=True).dt.strftime('%Y-%m-%d')
            result['news_vector'] = [
                json.dumps(matrix.getrow(index).toarray().ravel().tolist(), separators=(',', ':'))
                for index in range(matrix.shape[0])
            ]
            outputs[f'{name}_{split_name}'] = result[OUTPUT_COLUMNS]

        print(
            f'[OK] {name.upper()}: {train_matrix.shape[1]:,} training vocabulary terms; '
            f'{train_matrix.shape[0]:,} training and {test_matrix.shape[0]:,} test documents'
        )
    return outputs

def main() -> int:
    parser = argparse.ArgumentParser(description='Sample and vectorize Assignment 3 news data')
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
        period_start = period_end - pd.DateOffset(years=news_period)
        print(
            f'[OK] Selected period: {period_start.date()} through '
            f'{(period_end - pd.Timedelta(days=1)).date()} '
            f'({news_period} years; np.random.seed(42), chronological dates)'
        )
        if news_df.empty or impacts_df.empty:
            raise ValueError(
                'No input records fall within the selected 7-year period '
                f'{period_start.date()} to {period_end.date()}'
            )
        matched_df = match_news_to_impacts(news_df, impacts_df)
        train_df, test_df, test_start = split_on_event_window(matched_df, impacts_df)
        print(
            f'[OK] Chronological split: {len(train_df):,} train and {len(test_df):,} test '
            f'observations; test period starts {test_start.date()}. '
            f'Excluded event windows crossing the split.'
        )
        vectorized = vectorize_splits(train_df, test_df)
        for dataset_name, output_df in vectorized.items():
            output_path = DATA / f'news_{dataset_name}.csv'
            output_df.to_csv(output_path, index=False)
            print(f'[OK] Wrote {output_path}')

    except ValueError as exc:
        print(f'[ACTION NEEDED] {exc}', file=sys.stderr)
        return 1
    print('[OK] Sampling, matching, and vectorization complete.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
