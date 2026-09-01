import numpy as np
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS, RESULTS_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure

# Target: Window 2 (2022) only for the diagnosis
W2 = {'name': 'Window 2 (2022)', 'train_start': '2018-01-01', 'train_end': '2021-12-31', 'test_start': '2022-01-01', 'test_end': '2022-12-31'}

df_raw = YahooDownloader(start_date='2015-01-01', end_date='2026-03-20', ticker_list=['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'LT.NS']).fetch_data()
fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS, use_vix=True, use_turbulence=True)
processed = fe.preprocess_data(df_raw)
train_df = data_split(processed, W2['train_start'], W2['train_end'])
test_df = data_split(processed, W2['test_start'], W2['test_end'])

stock_dimension = len(train_df.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
env_kwargs = {
    'hmax': 100,
    'initial_amount': 1000000,
    'num_stock_shares': [0] * stock_dimension,
    'buy_cost_pct': [0.0015] * stock_dimension,
    'sell_cost_pct': [0.0015] * stock_dimension,
    'state_space': state_space,
    'stock_dim': stock_dimension,
    'tech_indicator_list': INDICATORS,
    'action_space': stock_dimension,
    'reward_scaling': 1e-4,
}

e_train = StockTradingEnv(df=train_df, **env_kwargs)
env_train, _ = e_train.get_sb_env()
agent = DRLAgent(env=env_train)

# Try different ent_coef
for coef in [0.01, 0.05]:
    print(f"\n--- Training Window 2 with ent_coef: {coef} ---")
    PPO_PARAMS = {'n_steps': 2048, 'ent_coef': coef, 'learning_rate': 0.00025, 'batch_size': 128}
    model = agent.get_model('ppo', model_kwargs=PPO_PARAMS)
    trained = agent.train_model(model=model, tb_log_name='ppo_test', total_timesteps=20000)
    
    e_test = StockTradingEnv(df=test_df, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
    df_account, df_actions = DRLAgent.DRL_prediction(model=trained, environment=e_test)
    
    # Calculate non-hold trades
    df_actions['sum_abs_actions'] = df_actions.abs().sum(axis=1)
    trades_count = (df_actions['sum_abs_actions'] > 0).sum()
    print(f"Total trading days with activity: {trades_count} out of {len(df_actions)}")
