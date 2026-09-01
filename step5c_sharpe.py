import numpy as np
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS, RESULTS_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure

class DiffSharpeTradingEnv(StockTradingEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.returns_history = []
    def step(self, actions):
        obs, reward_val, terminated, truncated, info = super().step(actions)
        # Calculate daily returns
        if len(self.asset_memory) > 1:
            daily_return = (self.asset_memory[-1] - self.asset_memory[-2]) / self.asset_memory[-2] 
            self.returns_history.append(daily_return)
        else:
            daily_return = 0
            
        if len(self.returns_history) > 30:
            vol = np.std(self.returns_history[-30:]) + 1e-6
            reward = daily_return / vol
        else:
            reward = daily_return * 10 
        
        return obs, reward, terminated, truncated, info

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "GAIL.NS"]
COST = 0.0015
INITIAL = 1_000_000

df_raw = YahooDownloader(start_date='2015-01-01', end_date='2026-03-20', ticker_list=TICKERS).fetch_data()
fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS, use_vix=True, use_turbulence=True)
processed = fe.preprocess_data(df_raw)

windows = [
    {'name': 'Window 1 (2019)', 'train_start': '2015-01-01', 'train_end': '2018-12-31', 'test_start': '2019-01-01', 'test_end': '2019-12-31'},
    {'name': 'Window 2 (2022)', 'train_start': '2018-01-01', 'train_end': '2021-12-31', 'test_start': '2022-01-01', 'test_end': '2022-12-31'},
    {'name': 'Window 3 (2025-26)', 'train_start': '2021-01-01', 'train_end': '2024-12-31', 'test_start': '2025-01-01', 'test_end': '2026-03-20'}
]

def get_metrics(df_account):
    df = df_account.copy()
    if 'account_value' not in df.columns: return 0, 0
    df['daily_return'] = df['account_value'].pct_change(1)
    tot_return = (df['account_value'].iloc[-1] - df['account_value'].iloc[0]) / df['account_value'].iloc[0]
    sharpe = (df['daily_return'].mean() / df['daily_return'].std()) * np.sqrt(252) if df['daily_return'].std() != 0 else 0
    return tot_return, sharpe

results = []
dim = len(TICKERS)
env_kwargs = {'hmax': 100, 'initial_amount': INITIAL, 'num_stock_shares': [0]*dim, 'buy_cost_pct': [COST]*dim, 'sell_cost_pct': [COST]*dim, 'state_space': 1 + 2*dim + len(INDICATORS)*dim, 'stock_dim': dim, 'tech_indicator_list': INDICATORS, 'reward_scaling': 1e-4}

for w in windows:
    train_df = data_split(processed, w['train_start'], w['train_end'])
    test_df = data_split(processed, w['test_start'], w['test_end'])
    
    e_train = DiffSharpeTradingEnv(df=train_df, action_space=dim, **env_kwargs)
    env_train, _ = e_train.get_sb_env()
    PPO_PARAMS = {'n_steps': 2048, 'ent_coef': 0.05, 'learning_rate': 0.00025, 'batch_size': 128}
    model = DRLAgent(env=env_train).get_model('ppo', model_kwargs=PPO_PARAMS)
    model.set_logger(configure(RESULTS_DIR + f"/ppo_step5c_sharpe_{w['name']}", ['csv']))
    trained = DRLAgent(env=env_train).train_model(model=model, tb_log_name='ppo_sharpe', total_timesteps=50000)
    
    e_test = DiffSharpeTradingEnv(df=test_df, action_space=dim, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
    df_account, df_actions = DRLAgent.DRL_prediction(model=trained, environment=e_test)
    
    ret, sh = get_metrics(df_account)
    trades = (df_actions.abs().sum(axis=1) > 0.1).sum()
    results.append({'name': w['name'], 'ret': ret, 'sh': sh, 'trades': trades})

print('\n| Window | Return | Sharpe | Day Trades |')
for r in results:
    print(f"| {r['name']} | {r['ret']*100:.2f}% | {r['sh']:.2f} | {r['trades']} |")
