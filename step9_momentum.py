import numpy as np
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS, RESULTS_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure

import warnings
warnings.filterwarnings("ignore")

# 1. Selection Rules Documented
N_DAYS = 30   # Rebalance frequency
K_DAYS = 90   # Lookback momentum window
M_STOCKS = 5  # Top stocks to select
COST = 0.0015
INITIAL = 1_000_000
print("=== STEP 9: MOMENTUM UNIVERSE SELECTION ===")
print(f"Rule: Every {N_DAYS} days, rank by {K_DAYS}-day return, pick top {M_STOCKS}.")
print("Fresh Holdout reserved: 2020-01-01 to 2021-12-31")

# Universe of 30 established NSE stocks (ignoring newer listings to maintain data continuity)
UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "ITC.NS", "GAIL.NS", "LT.NS",
    "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "M&M.NS",
    "ULTRACEMCO.NS", "WIPRO.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "KOTAKBANK.NS", "TITAN.NS", "ONGC.NS", "COALINDIA.NS",
    "NTPC.NS", "POWERGRID.NS", "HEROMOTOCO.NS", "TATASTEEL.NS",
    "HINDALCO.NS", "GRASIM.NS", "CIPLA.NS", "DRREDDY.NS"
]

# Fetch exhaustive data
START_FETCH = '2014-06-01'  # Extra buffer for K_DAYS lookup before 2015
df_raw = YahooDownloader(start_date=START_FETCH, end_date='2026-03-20', ticker_list=UNIVERSE).fetch_data()
fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS, use_vix=False, use_turbulence=False)
processed = fe.preprocess_data(df_raw)

# Pivot prices to calculate momentum easily
prices = processed.pivot(index='date', columns='tic', values='close').sort_index()
# Drop columns that have missing data at the start to maintain uniform sizing
prices = prices.dropna(axis=1)

available_tickers = prices.columns.tolist()

# Calculate log returns for momentum
mom_90d = (prices / prices.shift(K_DAYS)) - 1

# Extract the available dates
trade_dates = prices.index.tolist()

def run_momentum_walk_forward(start_date, end_date, name, train_agent=False, PPO_trained=None):
    print(f"\n--- Running {name} ({start_date} to {end_date}) ---")
    
    # Filter test dates
    test_dates = [d for d in trade_dates if start_date <= d <= end_date]
    if not test_dates:
        return None, None
        
    portfolio_value = 1_000_000
    bh_portfolio_value = 1_000_000
    
    history_rl = []
    history_bh = []
    
    # Rebalance loop
    for i in range(0, len(test_dates), N_DAYS):
        period_dates = test_dates[i : i + N_DAYS]
        start_t = period_dates[0]
        
        # Point-in-time check: Momentum must use data strictly before start_t!
        # Find the latest date in prices that is < start_t
        past_dates = [d for d in trade_dates if d < start_t]
        if not past_dates: continue
        rebalance_date = past_dates[-1]
        
        # Get top M stocks
        current_mom = mom_90d.loc[rebalance_date]
        top_m = current_mom.nlargest(M_STOCKS).index.tolist()
        
        # Filter processed data for this sub-period and current top M stocks
        sub_df = processed[(processed['date'].isin(period_dates)) & (processed['tic'].isin(top_m))].copy()
        sub_df = sub_df.sort_values(['date', 'tic']).reset_index(drop=True)
        # Give mock integer dates to satisfy stocktrading env which expects indices 0..T-1
        dates_unique = sub_df['date'].unique()
        date_map = {d: i for i, d in enumerate(dates_unique)}
        sub_df.index = sub_df['date'].map(date_map)
        
        # Check if we have enough data (all M stocks must exist in this slice)
        if len(sub_df['tic'].unique()) < M_STOCKS:
            continue # Skip anomalous windows
            
        dim = M_STOCKS
        env_kwargs = {
            'hmax': 100, 'initial_amount': portfolio_value, 'num_stock_shares': [0]*dim,
            'buy_cost_pct': [COST]*dim, 'sell_cost_pct': [COST]*dim,
            'state_space': 1 + 2 * dim + len(INDICATORS) * dim, 'stock_dim': dim,
            'tech_indicator_list': INDICATORS, 'reward_scaling': 1e-4
        }
        
        env_test = StockTradingEnv(df=sub_df, action_space=dim, **env_kwargs)
        
        # Equal Weight Buy and Hold baseline for the current top M
        sub_prices = sub_df.pivot(index='date', columns='tic', values='close').sort_index()
        if len(sub_prices) == 0: continue
            
        eq_weights = np.ones(dim) / dim
        bh_vals = bh_portfolio_value * (sub_prices / sub_prices.iloc[0]).dot(eq_weights)
        # Apply 0.15% turnover cost to B&H once per rebalance! (since B&H changes stocks every 30 days)
        bh_vals = bh_vals * (1 - COST)
        history_bh.extend(bh_vals.values.tolist())
        bh_portfolio_value = bh_vals.iloc[-1]
        
        # RL execution
        df_acc, _ = DRLAgent.DRL_prediction(model=PPO_trained, environment=env_test)
        if df_acc is not None and not df_acc.empty:
            history_rl.extend(df_acc['account_value'].tolist())
            # For next month, deduct a 0.15% liquidation cost to simulate rolling cash over
            portfolio_value = df_acc['account_value'].iloc[-1] * (1 - COST)
            
    # Compute final metrics
    def calc_metrics(hist):
        df = pd.DataFrame({'val': hist})
        ret = (df['val'].iloc[-1] - df['val'].iloc[0]) / df['val'].iloc[0]
        dr = df['val'].pct_change()
        std = dr.std()
        sh = (dr.mean() / std) * np.sqrt(252) if std != 0 else 0
        dd = ((df['val'] - df['val'].cummax()) / df['val'].cummax()).min()
        return ret, sh, dd

    if history_rl and history_bh:
        rl_ret, rl_sh, rl_dd = calc_metrics(history_rl)
        bh_ret, bh_sh, bh_dd = calc_metrics(history_bh)
        return {'ret': rl_ret, 'sh': rl_sh, 'dd': rl_dd}, {'ret': bh_ret, 'sh': bh_sh, 'dd': bh_dd}
    return None, None

print("Training generalized PPO model on 5 random slots (2015-2018) so action/state spaces match M_STOCKS=5...")
rand_m = available_tickers[:M_STOCKS]
train_df = data_split(processed[processed['tic'].isin(rand_m)], '2015-01-01', '2018-12-31')
dim = M_STOCKS
env_kwargs_train = {
    'hmax': 100, 'initial_amount': INITIAL, 'num_stock_shares': [0]*dim,
    'buy_cost_pct': [COST]*dim, 'sell_cost_pct': [COST]*dim,
    'state_space': 1 + 2 * dim + len(INDICATORS) * dim, 'stock_dim': dim,
    'tech_indicator_list': INDICATORS, 'reward_scaling': 1e-4
}
env_train, _ = StockTradingEnv(df=train_df, action_space=dim, **env_kwargs_train).get_sb_env()
agent = DRLAgent(env=env_train)
PPO_PARAMS = {'n_steps': 2048, 'ent_coef': 0.05, 'learning_rate': 0.00025, 'batch_size': 128}
trained_model = agent.train_model(model=agent.get_model('ppo', model_kwargs=PPO_PARAMS), tb_log_name='ppo_mom', total_timesteps=50000)

windows = [
    # Retired Windows
    {'name': 'Window 1 (2019) *RETIRED*', 'start': '2019-01-01', 'end': '2019-12-31'},
    {'name': 'Window 2 (2022) *RETIRED*', 'start': '2022-01-01', 'end': '2022-12-31'},
    {'name': 'Window 3 (2025-26) *RETIRED*', 'start': '2025-01-01', 'end': '2026-03-20'},
    # Fresh Null Hypothesis Check Window
    {'name': 'FRESH HOLDOUT (2020-2021)', 'start': '2020-01-01', 'end': '2021-12-31'},
]

print("\n=== STEP 9 RESULTS ===")
print("| Window | PPO+Momentum (Ret/Sh/DD) | Rolling B&H Momentum (Ret/Sh/DD) |")
for w in windows:
    rl_m, bh_m = run_momentum_walk_forward(w['start'], w['end'], w['name'], PPO_trained=trained_model)
    if rl_m:
        print(f"| {w['name']} | {rl_m['ret']*100:.2f}% / {rl_m['sh']:.2f} / {rl_m['dd']*100:.2f}% | {bh_m['ret']*100:.2f}% / {bh_m['sh']:.2f} / {bh_m['dd']*100:.2f}% |")
    else:
        print(f"| {w['name']} | ERROR: Insufficient Data | - |")
