import numpy as np
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS, RESULTS_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure

df_raw = YahooDownloader(start_date='2015-01-01', end_date='2026-03-20', ticker_list=['RELIANCE.NS']).fetch_data()
fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS, use_vix=True, use_turbulence=True)
processed = fe.preprocess_data(df_raw)

windows = [
    {'name': 'Window 1 (2019)', 'train_start': '2015-01-01', 'train_end': '2018-12-31', 'test_start': '2019-01-01', 'test_end': '2019-12-31'},
    {'name': 'Window 2 (2022)', 'train_start': '2018-01-01', 'train_end': '2021-12-31', 'test_start': '2022-01-01', 'test_end': '2022-12-31'},
    {'name': 'Window 3 (2025-2026)', 'train_start': '2021-01-01', 'train_end': '2024-12-31', 'test_start': '2025-01-01', 'test_end': '2026-03-20'}
]

def get_metrics(df_account):
    df = df_account.copy()
    df['daily_return'] = df['account_value'].pct_change(1)
    tot_return = (df['account_value'].iloc[-1] - df['account_value'].iloc[0]) / df['account_value'].iloc[0]
    sharpe = (df['daily_return'].mean() / df['daily_return'].std()) * np.sqrt(252) if df['daily_return'].std() != 0 else 0
    cum_max = df['account_value'].cummax()
    drawdown = (df['account_value'] - cum_max) / cum_max
    max_dd = drawdown.min()
    return tot_return, sharpe, max_dd

results = []

for w in windows:
    train_df = data_split(processed, w['train_start'], w['train_end'])
    test_df = data_split(processed, w['test_start'], w['test_end'])
    
    stock_dimension = len(train_df.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    env_kwargs = {
        'hmax': 100,
        'initial_amount': 1000000,
        'num_stock_shares': [0] * stock_dimension,
        'buy_cost_pct': [0.001] * stock_dimension,
        'sell_cost_pct': [0.001] * stock_dimension,
        'state_space': state_space,
        'stock_dim': stock_dimension,
        'tech_indicator_list': INDICATORS,
        'action_space': stock_dimension,
        'reward_scaling': 1e-4,
    }
    
    e_train = StockTradingEnv(df=train_df, **env_kwargs)
    env_train, _ = e_train.get_sb_env()
    
    agent = DRLAgent(env=env_train)
    PPO_PARAMS = {'n_steps': 2048, 'ent_coef': 0.01, 'learning_rate': 0.00025, 'batch_size': 128}
    model_ppo = agent.get_model('ppo', model_kwargs=PPO_PARAMS)
    model_ppo.set_logger(configure(RESULTS_DIR + f"/ppo_wf_{w['name']}", ['csv']))
    trained_ppo = agent.train_model(model=model_ppo, tb_log_name='ppo', total_timesteps=30000)
    
    e_test = StockTradingEnv(df=test_df, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
    df_account_ppo, _ = DRLAgent.DRL_prediction(model=trained_ppo, environment=e_test)
    
    rl_ret, rl_sh, rl_dd = get_metrics(df_account_ppo)
    
    init_p = test_df['close'].iloc[0]
    test_bh = test_df.copy()
    test_bh['account_value'] = 1000000 * (test_bh['close'] / init_p)
    bh_ret, bh_sh, bh_dd = get_metrics(test_bh)
    
    results.append({
        'window': w['name'],
        'test_dates': f"{test_df.date.min()} to {test_df.date.max()}",
        'rl_ret': rl_ret, 'rl_sh': rl_sh, 'rl_dd': rl_dd,
        'bh_ret': bh_ret, 'bh_sh': bh_sh, 'bh_dd': bh_dd
    })

print('\n=== STEP 4 WALK-FORWARD VALIDATION RESULTS ===')
for r in results:
    print(f"\n--- {r['window']} ({r['test_dates']}) ---")
    print(f"RL PPO : Return={r['rl_ret']*100:.2f}%, Sharpe={r['rl_sh']:.4f}, Max DD={r['rl_dd']*100:.2f}%")
    print(f"BuyHold: Return={r['bh_ret']*100:.2f}%, Sharpe={r['bh_sh']:.4f}, Max DD={r['bh_dd']*100:.2f}%")
