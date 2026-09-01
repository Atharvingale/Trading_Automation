import numpy as np
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS, RESULTS_DIR, TRAINED_MODEL_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure

print("=== STEP 5b: AUDIT & UNBIASED RE-EVALUATION ===")

# 1. Tickers as of January 2015 (Unbiased, including underperformers & collapses)
UNBIASED_2015_TICKERS = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "YESBANK.NS",
    "BHEL.NS",
    "NTPC.NS",
    "TATAMOTORS.NS",
    "GAIL.NS"
]

print("Unbiased 2015 NIFTY Universe:", UNBIASED_2015_TICKERS)

TRAIN_START = "2015-01-01"
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2026-03-20"
COST = 0.0015
INITIAL = 1_000_000

df_raw = YahooDownloader(
    start_date=TRAIN_START,
    end_date=TEST_END,
    ticker_list=UNBIASED_2015_TICKERS,
).fetch_data()

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=True,
)
processed = fe.preprocess_data(df_raw)

train_df = data_split(processed, TRAIN_START, TRAIN_END)
val_df = data_split(processed, VAL_START, VAL_END)
test_df = data_split(processed, TEST_START, TEST_END)

stock_dimension = len(train_df.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
env_kwargs = {
    "hmax": 100,
    "initial_amount": INITIAL,
    "num_stock_shares": [0] * stock_dimension,
    "buy_cost_pct": [COST] * stock_dimension,
    "sell_cost_pct": [COST] * stock_dimension,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

e_train = StockTradingEnv(df=train_df, **env_kwargs)
env_train, _ = e_train.get_sb_env()
agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
model.set_logger(configure(RESULTS_DIR + "/ppo_unbiased2015", ["csv"]))
trained = agent.train_model(model=model, tb_log_name="ppo", total_timesteps=50000)

e_test = StockTradingEnv(
    df=test_df,
    turbulence_threshold=70,
    risk_indicator_col="vix",
    **env_kwargs,
)
df_account, df_actions = DRLAgent.DRL_prediction(model=trained, environment=e_test)

def get_metrics(df_account):
    df = df_account.copy()
    df["daily_return"] = df["account_value"].pct_change(1)
    tot_return = (df["account_value"].iloc[-1] - df["account_value"].iloc[0]) / df["account_value"].iloc[0]
    std = df["daily_return"].std()
    sharpe = (df["daily_return"].mean() / std) * np.sqrt(252) if std and std != 0 else 0
    cum_max = df["account_value"].cummax()
    max_dd = ((df["account_value"] - cum_max) / cum_max).min()
    return tot_return, sharpe, max_dd

rl_ret, rl_sh, rl_dd = get_metrics(df_account)

price = test_df.pivot(index="date", columns="tic", values="close").sort_index()
eq_weights = np.ones(price.shape[1]) / price.shape[1]
bh_value = INITIAL * (price / price.iloc[0]).dot(eq_weights)
bh_df = pd.DataFrame({"account_value": bh_value.values})
bh_ret, bh_sh, bh_dd = get_metrics(bh_df)

print("\n=== UNBIASED (JAN 2015 CONSTITUENTS) EVALUATION RESULTS ===")
print(f"Test Period: {TEST_START} to {TEST_END}")
print(f"Metric             | RL PPO (Unbiased 2015) | Equal-weight Buy&Hold")
print(f"Total Return       | {rl_ret*100:.2f}%                    | {bh_ret*100:.2f}%")
print(f"Sharpe Ratio       | {rl_sh:.4f}                   | {bh_sh:.4f}")
print(f"Max Drawdown       | {rl_dd*100:.2f}%                   | {bh_dd*100:.2f}%")
