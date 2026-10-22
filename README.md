# cpsc_488_assign_03_market_sentiment


python3 -m venv .venv

#Step 5: Activate and use the environment
source .venv/bin/activate

python -m pip install -r requirements.txt

#To run scripts

#To run the script against the sample_news.csv and sample_historical_impact_values.csv
python 2_sampling_vectorize.py --test

#To run the script against the actual data: news.csv and historical_impact_values.csv
python 2_sampling_vectorize.py 