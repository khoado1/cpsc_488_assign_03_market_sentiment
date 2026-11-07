# cpsc_488_assign_03_market_sentiment


python3 -m venv .venv

#Step 5: Activate and use the environment
source .venv/bin/activate

python -m pip install -r requirements.txt
pip install --upgrade pip

#To run scripts

# Prepare datasets/news.csv and datasets/historical_prices.csv from the files
# in news_datasets/. This command downloads current S&P 500 constituents and
# historical prices, so it requires an internet connection.
python 0_Data_Prep.py

# For a quick end-to-end smoke test, limit preparation to three companies.
python 0_Data_Prep.py --ticker-limit 3

# Calculate market-reaction targets from historical_prices.csv
python 1_calculate_impacts_values.py

# The script reads datasets/historical_prices.csv and writes
# datasets/historical_impacts_values.csv. Include the S&P 500 index as ^sp500.

# To run against the sample CSVs
python 2_sampling_vectorize.py --test

# The sample fixtures are format examples only: their 2023 dates are outside the
# required 2009-2020 study period, and they do not contain enough trading days
# for the three-day event window. Sample mode therefore reports why it cannot
# produce training/test files.

# To run against actual data: news.csv and historical_impacts_values.csv
python 2_sampling_vectorize.py

# Successful runs write news_dtm_train.csv, news_dtm_test.csv,
# news_tfidf_train.csv, and news_tfidf_test.csv to datasets/.

# To classify:
python 3_classify.py --vectorizer both

# To run regression
python 4_regression.py --vectorizer both

# Test run, ensure that you have news_datasets.zip in your folder and unzipped, only run if you want the full path

python 0_Data_Prep.py
python 1_calculate_impacts_values.py
python 2_sampling_vectorize.py
python 3_classify.py --vectorizer both
python 4_regression.py --vectorizer both

# Complete run, ensure that you have news_datasets.zip in your folder and unzipped, only run if you want the full path

python 0_Data_Prep.py --ticker-limit 3
python 1_calculate_impacts_values.py
python 2_sampling_vectorize.py
python 3_classify.py --vectorizer both
python 4_regression.py --vectorizer both


