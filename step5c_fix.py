import numpy as np
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS, RESULTS_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO

# Custom Env with Holding Cost
class HoldingCostTradingEnv(StockTradingEnv):
    def __init__(self, **kwargs):
        self.holding_cost = kwargs.pop('holding_cost', 0.0)
        super().__init__(**kwargs)
    
    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        # Apply tiny penalty for holding positions (idle penalty)
        # Assuming position size is derived from state in standard FinRL env
        if np.sum(np.abs(self.actions)) < 1e-3: 
            reward -= self.holding_cost
        return obs, reward, terminated, truncated, info

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "GAIL.NS"]
COST = 0.0015
HOLDING_PENALTY = 0.01 # Small penalty to discourage static no-trading

df_raw = YahooDownloader(start_date='2015-01-01', end_date='2026-03-20', ticker_list=TICKERS).fetch_data()
fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS, use_vix=True, use_turbulence=True)
processed = fe.preprocess_data(df_raw)

test_w = {'name': 'Window 2 (2022)', 'train_start': '2018-01-01', 'train_end': '2021-12-31', 'test_start': '2022-01-01', 'test_end': '2022-12-31'}
train_df = data_split(processed, test_w['train_start'], test_w['train_end'])
test_df = data_split(processed, test_w['test_start'], test_w['test_end'])

    env_kwargs = {
        'hmax': 100, 'initial_amount': INITIAL, 'num_stock_shares': [0]*len(TICKERS),
        'buy_cost_pct': [COST]*len(TICKERS), 'sell_cost_pct': [COST]*len(TICKERS),
        'state_space': 1 + 2*len(TICKERS) + len(INDICATORS)*len(TICKERS), 'stock_dim': len(TICKERS), 'tech_indicator_list': INDICATORS, 'reward_scaling': 1e-4, 'holding_cost': HOLDING_PENALTY
    }

e_train = HoldingCostTradingEnv(df=train_df, action_space=10, **env_kwargs)
env_train, _ = e_train.get_sb_env()
agent = DRLAgent(env=env_train)
PPO_PARAMS = {'n_steps': 2048, 'ent_coef': 0.01, 'learning_rate': 0.00025, 'batch_size': 128}
model = agent.get_model('ppo', model_kwargs=PPO_PARAMS)
trained = agent.train_model(model=model, tb_log_name='ppo_holding_fix', total_timesteps=50000)

e_test = HoldingCostTradingEnv(df=test_df, action_space=10, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
df_account, df_actions = DRLAgent.DRL_prediction(model=trained, environment=e_test)

trades = (df_actions.abs().sum(axis=1) > 0.1).sum()
ret = (df_account['account_value'].iloc[-1] - df_account['account_value'].iloc[0]) / df_account['account_value'].iloc[0]
print(f"Window 2 | Return: {ret*100:.2f}% | Trades: {trades}")
