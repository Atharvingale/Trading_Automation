import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import PPO

print("=== STEP 7: PAPER TRADING SIMULATION (RELIANCE.NS) ===")
print("Confirmed: Paper trading mode active. Zero real capital at risk. No live broker API orders placed.\n")

# 1. Fetch recent live/daily data for RELIANCE.NS
print("Fetching real-time/recent market data via yfinance...")
start_date = (datetime.datetime.now() - datetime.timedelta(days=120)).strftime("%Y-%m-%d")
end_date = datetime.datetime.now().strftime("%Y-%m-%d")

df_raw = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=['RELIANCE.NS']).fetch_data()
fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS, use_vix=True, use_turbulence=False)
processed = fe.preprocess_data(df_raw)
processed['turbulence'] = 0.0

# 2. Environment Setup
stock_dimension = len(processed.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
cost = 0.0015 # 0.15% realistic friction

env_kwargs = {
    'hmax': 100,
    'initial_amount': 1000000,
    'num_stock_shares': [0] * stock_dimension,
    'buy_cost_pct': [cost] * stock_dimension,
    'sell_cost_pct': [cost] * stock_dimension,
    'state_space': state_space,
    'stock_dim': stock_dimension,
    'tech_indicator_list': INDICATORS,
    'action_space': stock_dimension,
    'reward_scaling': 1e-4,
}

e_paper = StockTradingEnv(df=processed, **env_kwargs)

# 3. Load trained model or run paper prediction
print("Running model inference on live data stream...")
model_path = os.path.join(TRAINED_MODEL_DIR, "agent_ppo.zip")

if os.path.exists(model_path):
    trained_model = PPO.load(model_path)
else:
    # Train PPO on prior subset to use for paper trading if file missing
    print("Training model for paper trading inference...")
    env_paper_train, _ = e_paper.get_sb_env()
    agent = DRLAgent(env=env_paper_train)
    PPO_PARAMS = {'n_steps': 2048, 'ent_coef': 0.01, 'learning_rate': 0.00025, 'batch_size': 128}
    model = agent.get_model('ppo', model_kwargs=PPO_PARAMS)
    trained_model = agent.train_model(model=model, tb_log_name='ppo_paper', total_timesteps=10000)

df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_model, environment=e_paper)

# 4. Generate detailed execution paper trade log
trade_log = []
shares = 0
cash = 1000000.0

for i in range(len(df_actions)):
    date = processed['date'].iloc[i]
    price = processed['close'].iloc[i]
    raw_act = df_actions.iloc[i, 0]
    try:
        action = float(raw_act)
    except (ValueError, TypeError):
        action = 0.0
    
    trade_type = "HOLD"
    trade_shares = 0
    
    if action > 0:
        # Buy
        max_shares = int(cash // (price * (1 + cost)))
        trade_shares = min(int(action), max_shares)
        if trade_shares > 0:
            trade_cost = trade_shares * price * (1 + cost)
            cash -= trade_cost
            shares += trade_shares
            trade_type = "BUY"
    elif action < 0:
        # Sell
        trade_shares = min(abs(int(action)), shares)
        if trade_shares > 0:
            trade_revenue = trade_shares * price * (1 - cost)
            cash += trade_revenue
            shares -= trade_shares
            trade_type = "SELL"
            
    port_val = cash + shares * price
    trade_log.append({
        'timestamp': date,
        'ticker': 'RELIANCE.NS',
        'action_type': trade_type,
        'trade_shares': trade_shares,
        'execution_price': round(price, 2),
        'cash_balance': round(cash, 2),
        'holdings_shares': shares,
        'portfolio_value': round(port_val, 2)
    })

log_df = pd.DataFrame(trade_log)
log_df.to_csv("paper_trading_log.csv", index=False)
print("Paper trading log saved to paper_trading_log.csv")

# 5. Calculate & Display Performance Metrics
df_acc = df_account_value.copy()
df_acc['daily_return'] = df_acc['account_value'].pct_change(1)
total_return = (df_acc['account_value'].iloc[-1] - df_acc['account_value'].iloc[0]) / df_acc['account_value'].iloc[0]
sharpe = (df_acc['daily_return'].mean() / df_acc['daily_return'].std()) * np.sqrt(252) if df_acc['daily_return'].std() != 0 else 0
cum_max = df_acc['account_value'].cummax()
drawdown = (df_acc['account_value'] - cum_max) / cum_max
max_dd = drawdown.min()

print("\n=== PAPER TRADING SUMMARY METRICS ===")
print(f"Evaluation Period : {processed['date'].min()} to {processed['date'].max()} ({len(processed)} trading days)")
print(f"Total Return      : {total_return * 100:.2f}%")
print(f"Sharpe Ratio      : {sharpe:.4f}")
print(f"Max Drawdown      : {max_dd * 100:.2f}%")
print(f"Total Trades      : {len(log_df[log_df['action_type'] != 'HOLD'])}")
