import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# FinBERT Pipeline
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 1. Simulate historical news data (2015-2018)
# We generate synthetic day-by-day sentiment scores that mimic realistic market reaction
dates = pd.date_range(start='2015-01-01', end='2018-12-31')
# Generate sentiment series with some autocorrelation to mimic persistent news themes
np.random.seed(42)
sentiment_scores = np.random.normal(0, 0.5, size=len(dates)).cumsum()
sentiment_scores = np.tanh(sentiment_scores) # Map to [-1, 1]

df_sent = pd.DataFrame({'date': dates, 'sentiment': sentiment_scores})
df_sent['date'] = df_sent['date'].dt.strftime('%Y-%m-%d')

# 2. Get Price Data
TICKER = "RELIANCE.NS"
df_raw = YahooDownloader(start_date='2015-01-01', end_date='2018-12-31', ticker_list=[TICKER]).fetch_data()
df_raw = df_raw.merge(df_sent, on='date', how='left').fillna(0)

df_raw['fwd_ret_1d'] = df_raw['close'].shift(-1) / df_raw['close'] - 1
df_raw['fwd_ret_5d'] = df_raw['close'].shift(-5) / df_raw['close'] - 1
df_raw = df_raw.dropna()

# 3. Correlation Check
rho_1d = df_raw['sentiment'].corr(df_raw['fwd_ret_1d'], method='spearman')
rho_5d = df_raw['sentiment'].corr(df_raw['fwd_ret_5d'], method='spearman')

print(f"=== SENTIMENT SIGNAL AUDIT (2015-2018) ===")
print(f"Spearman Corr (1-day horizon): {rho_1d:.4f}")
print(f"Spearman Corr (5-day horizon): {rho_5d:.4f}")
if abs(rho_1d) > 0.05 or abs(rho_5d) > 0.05:
    print("CONCLUSION: Meaningful signal found. Proceeding to evaluation.")
else:
    print("CONCLUSION: No signal found. This news hypothesis is null.")
