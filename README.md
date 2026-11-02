# cpsc_488_assign_03_market_sentiment


python3 -m venv .venv

#Step 5: Activate and use the environment
source .venv/bin/activate

python -m pip install -r requirements.txt

#To run scripts

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