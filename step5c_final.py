import numpy as np
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS, RESULTS_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure

class HoldingCostTradingEnv(StockTradingEnv):
    def __init__(self, **kwargs):
        self.holding_cost = kwargs.pop('holding_cost', 0.0)
        super().__init__(**kwargs)
    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        if np.sum(np.abs(actions)) < 1e-3: reward -= self.holding_cost
        return obs, reward, terminated, truncated, info

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "GAIL.NS"]
COST = 0.0015
HOLDING_PENALTY = 0.01 
INITIAL = 1_000_000

df_raw = YahooDownloader(start_date='2015-01-01', end_date='2026-03-20', ticker_list=TICKERS).fetch_data()
fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS, use_vix=True, use_turbulence=True)
processed = fe.preprocess_data(df_raw)

windows = [
    {'name': 'Window 1 (2019)', 'train_start': '2015-01-01', 'train_end': '2018-12-31', 'test_start': '2019-01-01', 'test_end': '2019-12-31'},
    {'name': 'Window 2 (2022)', 'train_start': '2018-01-01', 'train_end': '2021-12-31', 'test_start': '2022-01-01', 'test_end': '2022-12-31'},
    {'name': 'Window 3 (2025-26)', 'train_start': '2021-01-01', 'train_end': '2024-12-31', 'test_start': '2025-01-01', 'test_end': '2026-03-20'}
]

dim = len(TICKERS)
env_kwargs = {
    'hmax': 100, 'initial_amount': INITIAL, 'num_stock_shares': [0]*dim,
    'buy_cost_pct': [COST]*dim, 'sell_cost_pct': [COST]*dim,
    'state_space': 1 + 2 * dim + len(INDICATORS) * dim, 'stock_dim': dim, 'tech_indicator_list': INDICATORS, 'reward_scaling': 1e-4, 'holding_cost': HOLDING_PENALTY
}

results = []
for w in windows:
    train_df = data_split(processed, w['train_start'], w['train_end'])
    test_df = data_split(processed, w['test_start'], w['test_end'])
    
    e_train = HoldingCostTradingEnv(df=train_df, action_space=dim, **env_kwargs)
    env_train, _ = e_train.get_sb_env()
    
    agent = DRLAgent(env=env_train)
    PPO_PARAMS = {'n_steps': 2048, 'ent_coef': 0.05, 'learning_rate': 0.00025, 'batch_size': 128}
    model = agent.get_model('ppo', model_kwargs=PPO_PARAMS)
    model.set_logger(configure(RESULTS_DIR + f"/ppo_step5c_{w['name']}", ['csv']))
    trained = agent.train_model(model=model, tb_log_name='ppo_fixed', total_timesteps=50000)
    
    e_test = HoldingCostTradingEnv(df=test_df, action_space=dim, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
    df_account, df_actions = DRLAgent.DRL_prediction(model=trained, environment=e_test)
    
    df_account['daily_return'] = df_account['account_value'].pct_change(1)
    ret = (df_account['account_value'].iloc[-1] - df_account['account_value'].iloc[0]) / df_account['account_value'].iloc[0]
    sh = (df_account['daily_return'].mean() / df_account['daily_return'].std()) * np.sqrt(252) if df_account['daily_return'].std() != 0 else 0
    trades = (df_actions.abs().sum(axis=1) > 0.1).sum()
    results.append({'name': w['name'], 'ret': ret, 'sh': sh, 'trades': trades})

print('| Window | Return | Sharpe | Day Trades |')
for r in results:
    print(f"| {r['name']} | {r['ret']*100:.2f}% | {r['sh']:.2f} | {r['trades']} |")
