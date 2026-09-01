import numpy as np
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

import warnings
warnings.filterwarnings("ignore")

# Selection Rules Documented
N_DAYS = 30   # Rebalance frequency
K_DAYS = 90   # Lookback momentum window
M_STOCKS = 5  # Top stocks to select
COST = 0.0015

print("=== STEP 10: MOMENTUM EDGE VALIDATION ACROSS REGIMES ===")
print("Testing pure, rule-based momentum (without RL intervention)")
print("Validating against two new untouched holdout windows: 2011-2013 (Sideways), 2008-2009 (Crash/Bear)")

UNIVERSE_WIDE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "ITC.NS", "GAIL.NS", "LT.NS",
    "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "M&M.NS",
    "WIPRO.NS", "AXISBANK.NS", "TITAN.NS", "ONGC.NS", "NTPC.NS", 
    "POWERGRID.NS", "HEROMOTOCO.NS", "TATASTEEL.NS", "HINDALCO.NS", 
    "GRASIM.NS", "CIPLA.NS", "DRREDDY.NS"
]

# Fetch exhaustive historical data since 2006
START_FETCH = '2006-01-01'  
df_raw = YahooDownloader(start_date=START_FETCH, end_date='2014-12-31', ticker_list=UNIVERSE_WIDE).fetch_data()
df_raw = df_raw.sort_values(['date', 'tic'])

# Pivot prices to calculate momentum easily
prices = df_raw.pivot(index='date', columns='tic', values='close').sort_index()

# Keep only columns with sufficient early history
valid_cols = prices.columns[prices.iloc[0].notna()].tolist()
prices = prices[valid_cols]

# Calculate log returns for momentum
mom_90d = (prices / prices.shift(K_DAYS)) - 1
trade_dates = prices.index.tolist()

windows = [
    {'name': 'Global Financial Crisis Bear (2008-2009)', 'start': '2008-01-01', 'end': '2009-12-31'},
    {'name': 'Sideways Choppy (2011-2013)', 'start': '2011-01-01', 'end': '2013-12-31'}
]

def backtest_pure_momentum(start_date, end_date, name):
    print(f"\n--- Testing Regime: {name} ---")
    
    test_dates = [d for d in trade_dates if start_date <= d <= end_date]
    if not test_dates: return None
        
    bh_portfolio_value = 1_000_000
    history_bh = []
    
    # Generate Equal-Weight Index baseline for this period using valid_cols
    index_prices = prices.loc[test_dates].copy()
    eq_weights = np.ones(len(index_prices.columns)) / len(index_prices.columns)
    index_vals = 1_000_000 * (index_prices / index_prices.iloc[0]).dot(eq_weights)
    
    # Rebalance loop
    for i in range(0, len(test_dates), N_DAYS):
        period_dates = test_dates[i : i + N_DAYS]
        start_t = period_dates[0]
        
        # Point-in-time momentum check
        past_dates = [d for d in trade_dates if d < start_t]
        if not past_dates: continue
        rebalance_date = past_dates[-1]
        
        current_mom = mom_90d.loc[rebalance_date]
        # Ignore NAs
        top_m = current_mom.dropna().nlargest(M_STOCKS).index.tolist()
        
        if len(top_m) < M_STOCKS: continue
            
        sub_prices = prices.loc[period_dates, top_m].copy()
        if len(sub_prices) == 0: continue
            
        # Allocate equal weight to the 5 top momentum stocks
        eq_w = np.ones(M_STOCKS) / M_STOCKS
        bh_vals = bh_portfolio_value * (sub_prices / sub_prices.iloc[0]).dot(eq_w)
        
        # Apply 0.15% friction cost for turnover 
        bh_vals = bh_vals * (1 - COST)
        history_bh.extend(bh_vals.values.tolist())
        bh_portfolio_value = bh_vals.iloc[-1]
            
    def calc_metrics(hist):
        df = pd.DataFrame({'val': hist})
        ret = (df['val'].iloc[-1] - df['val'].iloc[0]) / df['val'].iloc[0]
        dr = df['val'].pct_change()
        std = dr.std()
        sh = (dr.mean() / std) * np.sqrt(252) if std != 0 else 0
        dd = ((df['val'] - df['val'].cummax()) / df['val'].cummax()).min()
        return ret, sh, dd

    if history_bh:
        bh_ret, bh_sh, bh_dd = calc_metrics(history_bh)
        
        # Calculate Index Metrics
        df_index = pd.DataFrame({'val': index_vals.values})
        ix_ret = (df_index['val'].iloc[-1] - df_index['val'].iloc[0]) / df_index['val'].iloc[0]
        ix_dr = df_index['val'].pct_change()
        ix_std = ix_dr.std()
        ix_sh = (ix_dr.mean() / ix_std) * np.sqrt(252) if ix_std != 0 else 0
        ix_dd = ((df_index['val'] - df_index['val'].cummax()) / df_index['val'].cummax()).min()
        
        return {'ret': bh_ret, 'sh': bh_sh, 'dd': bh_dd}, {'ret': ix_ret, 'sh': ix_sh, 'dd': ix_dd}
    return None, None

print("\n| Regime Window | Momentum Strategy (Ret/Sh/DD) | Equal-Weight Index (Ret/Sh/DD) |")
for w in windows:
    mom_m, ix_m = backtest_pure_momentum(w['start'], w['end'], w['name'])
    if mom_m:
        print(f"| {w['name']} | {mom_m['ret']*100:.2f}% / {mom_m['sh']:.2f} / {mom_m['dd']*100:.2f}% | {ix_m['ret']*100:.2f}% / {ix_m['sh']:.2f} / {ix_m['dd']*100:.2f}% |")
    else:
        print(f"| {w['name']} | ERROR | - |")
