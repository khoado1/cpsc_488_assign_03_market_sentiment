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

def preprocess_news(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the news DataFrame for vectorization.

    This function is a placeholder for future preprocessing steps, such as:
    - Lowercasing text
    - Removing punctuation
    - Tokenization
    - Stopword removal
    - Lemmatization or stemming

    Currently, it returns the DataFrame unchanged.
    """
    # Placeholder for actual preprocessing logic
    df = df.copy()

    # Preprocessing steps would go here
    df['headline'] = df['headline'].astype(str).str.lower().str.strip()
    #df['headline'] = df['headline'].str.replace(r'[^\w\s]', '', regex=True)


    return df

def clean_html_text(raw_text: str) -> str:
    """Remove HTML tags and unescape HTML entities from a string."""
    soup = BeautifulSoup(raw_text or '', 'html.parser')

    # Prefer the article/main content when the input is a full HTML document
    main_content = soup.find(['article', 'main'])
    if main_content:
        text = main_content.get_text(separator=' ', strip=True)
    else:
        text = soup.get_text(separator=' ', strip=True)
    return text

    main_content = soup.find(['article', 'main' ])
    if main_content:
        text = main_content.get_text(seprator= ' ', strip=True)
    else:
        text = soup.get_text(separator=' ', strip=True)

    for ele in content.find_all(['script', 'style', 'noscript', 'nav', 'footer', 'header', 'aside', 'iframe', 'form']):
        ele.decompose()

    ad_marker = re.compile(
        r"\b(?:ad|ads|advert|advertisement|sponsored|promo)\b",
        re.IGNORECASE
    )

    for ele in list(content.find_all(True)):
        labels = [element.get('id', ''), *element.get('class', [])]
        if any(ad_marker.sear):
    
    return text
    

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
        preprocessed_news_df = preprocess_news(news_df)
        impacts_df = read_checked(impacts_path, 'impacts', test=args.test)
    except ValueError as exc:
        print(f'[ACTION NEEDED] {exc}', file=sys.stderr)
        return 1
    print('[OK] Input validation complete; sampling/vectorization not implemented yet.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
