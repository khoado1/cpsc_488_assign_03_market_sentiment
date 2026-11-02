import pandas as pd
import yfinance as yf
import numpy as np

#Import analyst_ratings.csv
ratings = pd.read_csv('datasets/news_datasets 2/analyst_ratings.csv')
#Gets rid of first column which is just index numbers
ratings2 = ratings.iloc[:, 1:]

#Import headlines.csv
headlines = pd.read_csv('datasets/news_datasets 2/headlines.csv')
#Gets rid of first column which is just index numbers
headlines2 = headlines.iloc[:, 1:]

#Create news.csv
news = pd.concat([ratings2, headlines2], ignore_index=True)

#Combine articles with the same date & company
news["date2"] = news["date"].str.split(" ", n=1).str[0]
grouped = news.groupby(["stock", "date2"]).agg(
    headline=("headline", " || ".join),
    url=("url", " || ".join)
)

#Create SP500 file
#sp500 = pd.read_csv('datasets/news_datasets 2/sp500.csv')
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})[0][["Symbol", "Security"]]
sp.columns = ["ticker", "company_name"]
sp.to_csv("datasets/news_datasets 2/sp500.csv", index=False)

#Select SP500 companies from news.csv
valid = set(sp["ticker"]) | set(sp["ticker"].str.replace(".", "-", regex=False))
news = news[news["stock"].isin(valid)]

#Get Historical prices from 2008-01-01 to 2021-06-01
tickers = sp["ticker"].str.replace(".", "-", regex=False).tolist()
tickers.append("^GSPC")
frames, failed = [], []
for i, t in enumerate(tickers):
    historical_prices = yf.download(t, start="2008-01-01", end="2021-06-01",
                     auto_adjust=False, progress=False)
    if historical_prices.empty:
        failed.append(t)
        continue
    historical_prices.columns = historical_prices.columns.get_level_values(0)   # flatten yfinance columns
    historical_prices = historical_prices.reset_index()
    historical_prices["symbol"] = "^sp500" if t == "^GSPC" else t
    frames.append(historical_prices)
    if i % 50 == 0:
        print(f"{i}/{len(tickers)} done")

prices = pd.concat(frames, ignore_index=True)
prices = prices.rename(columns={
    "Date": "date", "Open": "open", "High": "high", "Low": "low",
    "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
})

#Merket Cap
caps = []
for t in prices["symbol"].unique():
    p = prices[prices["symbol"] == t].copy()
    if t == "^sp500":
        p["market_cap"] = np.nan              # index has no market cap
        caps.append(p)
        continue

    tk = yf.Ticker(t)
    shares_now = tk.info.get("sharesOutstanding")
    splits = tk.splits                        # Series: split date -> ratio
    splits.index = splits.index.tz_localize(None)

    if shares_now is None:
        p["market_cap"] = np.nan
    else:
        # shares on date d = today's shares / product of all splits after d
        dates = pd.to_datetime(p["date"])
        factor = dates.apply(lambda d: splits[splits.index > d].prod() if len(splits) else 1.0)
        p["market_cap"] = (shares_now / factor) * p["close"]
    caps.append(p)

prices = pd.concat(caps, ignore_index=True)

prices = prices[["date", "symbol", "open", "high", "low", "close",
                 "adj_close", "volume", "market_cap"]]
prices.to_csv("historical_prices.csv", index=False)
