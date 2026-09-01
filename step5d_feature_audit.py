import pandas as pd
import numpy as np
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

# 1. Methodology Definition
print("=== FEATURE AUDIT METHODOLOGY ===")
print("Data Window: 2015-01-01 to 2018-12-31 (Strictly pre-dating the 2019/2022/2025 holdouts)")
print("Asset Universe: Unbiased 2015 NIFTY top constituents")
print("Horizons: 1-Day and 5-Day forward returns")
print("Correlation Metric: Spearman Rank Correlation (more robust to financial data outliers than Pearson)")
print("-" * 50)

# Unbiased universe
TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "GAIL.NS"]
TRAIN_START = "2015-01-01"
TRAIN_END = "2018-12-31"  # Strict isolation from test windows

# 2. Fetch Data ONLY for the training period
df_raw = YahooDownloader(start_date=TRAIN_START, end_date=TRAIN_END, ticker_list=TICKERS).fetch_data()

# 3. Add indicators
INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS, use_vix=False, use_turbulence=False)
processed = fe.preprocess_data(df_raw)

# 4. Calculate Forward Returns & Correlations
results = []
for tic in processed['tic'].unique():
    df_tic = processed[processed['tic'] == tic].copy()
    df_tic = df_tic.sort_values('date')
    
    # Target Variables: Forward Returns
    df_tic['fwd_ret_1d'] = df_tic['close'].shift(-1) / df_tic['close'] - 1
    df_tic['fwd_ret_5d'] = df_tic['close'].shift(-5) / df_tic['close'] - 1
    
    df_tic = df_tic.dropna()
    
    # Calculate Rank Correlation
    corrs_1d = df_tic[INDICATORS].corrwith(df_tic['fwd_ret_1d'], method='spearman')
    corrs_5d = df_tic[INDICATORS].corrwith(df_tic['fwd_ret_5d'], method='spearman')
    
    for ind in INDICATORS:
        results.append({'ticker': tic, 'indicator': ind, 'rho_1d': corrs_1d[ind], 'rho_5d': corrs_5d[ind]})

res_df = pd.DataFrame(results)

print("\n=== MEAN SPEARMAN CORRELATION ACROSS ALL TICKERS ===")
mean_corrs = res_df.groupby('indicator')[['rho_1d', 'rho_5d']].mean().reset_index()
mean_corrs['abs_rho_1d'] = mean_corrs['rho_1d'].abs()
mean_corrs = mean_corrs.sort_values('abs_rho_1d', ascending=False)
print(mean_corrs.to_string(index=False))

print("\n=== MEDIAN ABSOLUTE CORRELATION (ROBUST SIGNAL MAGNITUDE) ===")
res_df['abs_1d'] = res_df['rho_1d'].abs()
res_df['abs_5d'] = res_df['rho_5d'].abs()
median_abs = res_df.groupby('indicator')[['abs_1d', 'abs_5d']].median().reset_index()
median_abs = median_abs.sort_values('abs_1d', ascending=False)
print(median_abs.to_string(index=False))

print("\n=== CONCLUSION ===")
max_median_abs = median_abs['abs_1d'].max()
if max_median_abs < 0.05:
    print(f"NULL RESULT: The highest median absolute daily correlation is only {max_median_abs:.4f}. This is purely statistical noise.")
    print("These features possess absolutely no non-trivial predictive signal for the chosen universe.")
    print("The RL agent had no valid information to learn an edge from.")
else:
    print(f"SIGNAL FOUND: Some features show correlation >= 0.05. Highest is {max_median_abs:.4f}.")
