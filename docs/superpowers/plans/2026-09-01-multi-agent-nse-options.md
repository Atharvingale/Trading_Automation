# Multi-Agent Regime-Adaptive NSE Options Trading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-agent regime-adaptive NSE options trading simulation system (`nse_options_engine.py`, `options_regime_agent.py`, `paper_options_trader.py`) that prices contracts using Black-Scholes, maps LLM market regime signals into option spreads (Bull Call Spreads, Bear Put Spreads, Iron Condors), and executes paper trading under realistic 0.15% fee friction.

**Architecture:** 
1. `nse_options_engine.py`: Computes option premiums, Implied Volatility (IV), Delta, Gamma, Theta, Vega, and resolves NSE strike levels and Thursday expiration dates.
2. `options_regime_agent.py`: Queries `TradingAgentsGraph` to classify market regime (Bullish Trend, Bearish Trend, Rangebound, Volatility Spike) and selects optimal option spreads.
3. `paper_options_trader.py`: Manages portfolio account capital, option spread entry/exit, daily MTM valuation, -50% stop-loss risk management, and outputs `paper_options_log.csv` and `paper_options_valuation.csv`.

**Tech Stack:** Python 3.10+, `scipy.stats` (norm.cdf, norm.pdf), `yfinance`, `pandas`, `numpy`, `tradingagents` (LangGraph multi-agent), `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-01-multi-agent-nse-options-design.md`

## Global Constraints
- Python >= 3.10
- All tests run via `.\venv\Scripts\python.exe -m pytest <test_file>`
- Transaction fee friction: 0.15% per leg per trade (`COST_PCT = 0.0015`)
- Risk control: Maximum 20% capital per spread, -50% stop loss on spread risk budget

---

### Task 1: Black-Scholes Options Pricing & Greeks Engine (`nse_options_engine.py`)

**Files:**
- Create: `nse_options_engine.py`
- Test: `tests/test_options_engine.py`

**Interfaces:**
- Consumes: Underlying price $S$, strike $K$, time-to-expiry $T$ (in years), risk-free rate $r=0.07$, volatility $\sigma$.
- Produces: `black_scholes_price(S, K, T, r, sigma, option_type)`, `calculate_greeks(S, K, T, r, sigma, option_type)`, `resolve_nse_strikes(ticker, current_price)`.

- [ ] **Step 1: Write failing unit test for Black-Scholes pricing and Greeks**

Create `tests/test_options_engine.py`:
```python
import pytest
import math
from nse_options_engine import black_scholes_price, calculate_greeks, resolve_nse_strikes

def test_black_scholes_call_price():
    # S=24000, K=24000, T=30/365, r=0.07, sigma=0.15
    price = black_scholes_price(S=24000, K=24000, T=30/365, r=0.07, sigma=0.15, option_type="call")
    assert price > 0
    assert round(price, 2) == 407.48 or (390 < price < 420)

def test_greeks_calculation():
    greeks = calculate_greeks(S=24000, K=24000, T=30/365, r=0.07, sigma=0.15, option_type="call")
    assert "delta" in greeks and "theta" in greeks and "vega" in greeks and "gamma" in greeks
    assert 0.45 <= greeks["delta"] <= 0.60

def test_resolve_nse_strikes():
    strikes = resolve_nse_strikes("NIFTY.NS", 24123.45)
    assert strikes["atm"] == 24100
    assert strikes["otm_call_1"] == 24150
    assert strikes["otm_put_1"] == 24050
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_options_engine.py`
Expected: FAIL with "No module named 'nse_options_engine'"

- [ ] **Step 3: Implement minimal `nse_options_engine.py`**

Create `nse_options_engine.py`:
```python
import math
from scipy.stats import norm

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    if T <= 0:
        return max(0.0, S - K) if option_type.lower() == "call" else max(0.0, K - S)
    if sigma <= 0:
        sigma = 0.0001
        
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(0.01, float(price))

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> dict:
    if T <= 0 or sigma <= 0:
        return {"delta": 1.0 if S > K else 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    pdf_d1 = norm.pdf(d1)
    
    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (- (S * pdf_d1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (- (S * pdf_d1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0
        
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = (S * pdf_d1 * math.sqrt(T)) / 100.0
    
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega)
    }

def resolve_nse_strikes(ticker: str, current_price: float) -> dict:
    step = 50
    if "BANKNIFTY" in ticker.upper():
        step = 100
    elif "RELIANCE" in ticker.upper():
        step = 20
        
    atm = round(current_price / step) * step
    return {
        "atm": int(atm),
        "otm_call_1": int(atm + step),
        "otm_call_2": int(atm + 2 * step),
        "otm_put_1": int(atm - step),
        "otm_put_2": int(atm - 2 * step)
    }
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_options_engine.py`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add nse_options_engine.py tests/test_options_engine.py
git commit -m "feat: add Black-Scholes options pricing and Greeks engine for NSE"
```

---

### Task 2: Multi-Agent Regime Classifier & Option Strategy Selector (`options_regime_agent.py`)

**Files:**
- Create: `options_regime_agent.py`
- Test: `tests/test_regime_agent.py`

**Interfaces:**
- Consumes: Underlying ticker, current date, price series, IV rank, TradingAgents LLM provider settings.
- Produces: `select_option_strategy(ticker, date_str, current_price, iv_rank, ta_graph=None)`.

- [ ] **Step 1: Write failing unit test for strategy selector**

Create `tests/test_regime_agent.py`:
```python
import pytest
from options_regime_agent import select_option_strategy

def test_bullish_regime_strategy_selection():
    strat = select_option_strategy("NIFTY.NS", "2026-08-25", current_price=24000.0, iv_rank=30.0, signal_override="BUY")
    assert strat["strategy_type"] == "BULL_CALL_SPREAD"
    assert strat["long_leg"]["strike"] == 24000
    assert strat["short_leg"]["strike"] == 24050

def test_bearish_regime_strategy_selection():
    strat = select_option_strategy("NIFTY.NS", "2026-08-25", current_price=24000.0, iv_rank=30.0, signal_override="SELL")
    assert strat["strategy_type"] == "BEAR_PUT_SPREAD"
    assert strat["long_leg"]["strike"] == 24000
    assert strat["short_leg"]["strike"] == 23950

def test_rangebound_regime_strategy_selection():
    strat = select_option_strategy("NIFTY.NS", "2026-08-25", current_price=24000.0, iv_rank=55.0, signal_override="HOLD")
    assert strat["strategy_type"] == "IRON_CONDOR"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_regime_agent.py`
Expected: FAIL with "No module named 'options_regime_agent'"

- [ ] **Step 3: Implement `options_regime_agent.py`**

Create `options_regime_agent.py`:
```python
from nse_options_engine import resolve_nse_strikes

def select_option_strategy(
    ticker: str,
    date_str: str,
    current_price: float,
    iv_rank: float,
    ta_graph=None,
    signal_override: str = None
) -> dict:
    signal = "HOLD"
    if signal_override:
        signal = signal_override
    elif ta_graph is not None:
        try:
            _, raw_sig = ta_graph.propagate(ticker, date_str)
            signal = str(raw_sig).upper()
        except Exception:
            signal = "HOLD"

    strikes = resolve_nse_strikes(ticker, current_price)
    
    if "BUY" in signal or "OVERWEIGHT" in signal:
        strategy_type = "BULL_CALL_SPREAD"
        long_leg = {"type": "call", "strike": strikes["atm"]}
        short_leg = {"type": "call", "strike": strikes["otm_call_1"]}
        extra_legs = []
    elif "SELL" in signal or "UNDERWEIGHT" in signal:
        strategy_type = "BEAR_PUT_SPREAD"
        long_leg = {"type": "put", "strike": strikes["atm"]}
        short_leg = {"type": "put", "strike": strikes["otm_put_1"]}
        extra_legs = []
    elif iv_rank >= 40.0 or "HOLD" in signal or "NEUTRAL" in signal:
        strategy_type = "IRON_CONDOR"
        long_leg = {"type": "put", "strike": strikes["otm_put_2"]}
        short_leg = {"type": "put", "strike": strikes["otm_put_1"]}
        extra_legs = [
            {"type": "short_call", "strike": strikes["otm_call_1"]},
            {"type": "long_call", "strike": strikes["otm_call_2"]}
        ]
    else:
        strategy_type = "LONG_CALL"
        long_leg = {"type": "call", "strike": strikes["atm"]}
        short_leg = None
        extra_legs = []
        
    return {
        "ticker": ticker,
        "date": date_str,
        "strategy_type": strategy_type,
        "underlying_price": current_price,
        "long_leg": long_leg,
        "short_leg": short_leg,
        "extra_legs": extra_legs
    }
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_regime_agent.py`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add options_regime_agent.py tests/test_regime_agent.py
git commit -m "feat: add multi-agent regime classifier and options strategy selector"
```

---

### Task 3: Paper Options Execution & Risk Simulator (`paper_options_trader.py`)

**Files:**
- Create: `paper_options_trader.py`
- Test: `tests/test_paper_options.py`

**Interfaces:**
- Consumes: Multi-stock historical data, `nse_options_engine`, `options_regime_agent`.
- Produces: Daily valuation history, paper trade log, output CSVs (`paper_options_log.csv`, `paper_options_valuation.csv`).

- [ ] **Step 1: Write failing integration test for paper options simulator**

Create `tests/test_paper_options.py`:
```python
import pytest
import os
import pandas as pd
from paper_options_trader import run_paper_options_simulation

def test_paper_options_simulation_runs():
    res = run_paper_options_simulation(
        tickers=["NIFTY.NS"],
        days=30,
        initial_capital=1_000_000.0,
        mock=True
    )
    assert res is not None
    assert "total_return" in res
    assert os.path.exists("paper_options_log.csv")
    assert os.path.exists("paper_options_valuation.csv")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_paper_options.py`
Expected: FAIL with "No module named 'paper_options_trader'"

- [ ] **Step 3: Implement `paper_options_trader.py`**

Create `paper_options_trader.py`:
```python
import os
import sys
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "TradingAgents", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "TradingAgents")))

from nse_options_engine import black_scholes_price, calculate_greeks
from options_regime_agent import select_option_strategy
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

COST_PCT = 0.0015  # 0.15% friction per leg

def run_paper_options_simulation(
    tickers: list = None,
    days: int = 90,
    initial_capital: float = 1_000_000.0,
    mock: bool = False
) -> dict:
    if tickers is None:
        tickers = ["NIFTY.NS", "BANKNIFTY.NS", "RELIANCE.NS"]
        
    print(f"=== NSE OPTIONS PAPER TRADING SIMULATION ({days} Days) ===")
    print("Confirmed: Paper trading mode active. Zero real capital at risk.\n")
    
    end_date_dt = datetime.datetime.now()
    start_date_dt = end_date_dt - datetime.timedelta(days=days + 30)
    start_date = start_date_dt.strftime("%Y-%m-%d")
    end_date = end_date_dt.strftime("%Y-%m-%d")
    
    # Download underlying price data
    df_prices = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
    if df_prices.empty:
        print("Error downloading price data.")
        return None
        
    if isinstance(df_prices, pd.Series):
        df_prices = df_prices.to_frame(name=tickers[0])
        
    df_prices = df_prices.dropna(how="all").ffill().bfill()
    dates = df_prices.index.strftime("%Y-%m-%d").tolist()[-days:]
    
    # Calculate rolling 20-day volatility
    returns = df_prices.pct_change(1)
    rolling_vol = returns.rolling(20).std() * np.sqrt(252)
    rolling_vol = rolling_vol.fillna(0.18)
    
    # Initialize TradingAgents graph if not mock
    ta_graph = None
    if not mock:
        provider = os.getenv("TRADINGAGENTS_LLM_PROVIDER")
        if not provider:
            if os.getenv("OPENROUTER_API_KEY"):
                provider = "openrouter"
            elif os.getenv("GROQ_API_KEY"):
                provider = "groq"
            else:
                provider = "openai"
                
        try:
            config = DEFAULT_CONFIG.copy()
            config["llm_provider"] = provider
            config["max_debate_rounds"] = 1
            config["max_risk_discuss_rounds"] = 1
            ta_graph = TradingAgentsGraph(debug=False, config=config)
            print(f"TradingAgents Graph initialized with provider: {provider}")
        except Exception as e:
            print(f"Graph initialization fallback: {e}")
            ta_graph = None
            
    capital = initial_capital
    active_positions = []
    trade_log = []
    valuation_history = []
    
    rebalance_freq = 5  # Every 5 days
    T_expiry = 14 / 365.0  # 14 days to expiry
    r = 0.07
    
    for i, date_str in enumerate(dates):
        price_row = df_prices.loc[date_str]
        vol_row = rolling_vol.loc[date_str]
        
        # 1. Update active positions MTM & check stop loss / exit
        updated_positions = []
        for pos in active_positions:
            t = pos["ticker"]
            S = price_row[t]
            sigma = vol_row[t]
            pos["days_held"] += 1
            time_left = max(0.001, T_expiry - (pos["days_held"] / 365.0))
            
            # Reprice legs
            long_p = black_scholes_price(S, pos["long_strike"], time_left, r, sigma, pos["long_type"])
            short_p = 0.0
            if pos["short_strike"]:
                short_p = black_scholes_price(S, pos["short_strike"], time_left, r, sigma, pos["short_type"])
                
            curr_spread_val = long_p - short_p
            pnl = (curr_spread_val - pos["entry_cost"]) * pos["qty"]
            
            # Exit condition: 50% max profit, 50% stop loss, or 10 days held
            if pnl <= -0.50 * pos["risk_budget"] or pnl >= 0.50 * pos["risk_budget"] or pos["days_held"] >= 10:
                exit_cost = abs(pnl) * COST_PCT
                capital += (curr_spread_val * pos["qty"] - exit_cost)
                trade_log.append({
                    'timestamp': date_str,
                    'ticker': t,
                    'action': 'EXIT_' + pos["strategy_type"],
                    'entry_price': round(pos["entry_cost"], 2),
                    'exit_price': round(curr_spread_val, 2),
                    'pnl': round(pnl, 2),
                    'capital_after': round(capital, 2)
                })
            else:
                updated_positions.append(pos)
        active_positions = updated_positions
        
        # 2. Enter new position on rebalance days
        if i % rebalance_freq == 0 and len(active_positions) < len(tickers):
            for t in tickers:
                S = price_row[t]
                sigma = vol_row[t]
                
                strat = select_option_strategy(
                    ticker=t,
                    date_str=date_str,
                    current_price=S,
                    iv_rank=float(sigma * 100),
                    ta_graph=ta_graph
                )
                
                long_p = black_scholes_price(S, strat["long_leg"]["strike"], T_expiry, r, sigma, strat["long_leg"]["type"])
                short_p = 0.0
                short_strike = None
                short_type = None
                if strat["short_leg"]:
                    short_strike = strat["short_leg"]["strike"]
                    short_type = strat["short_leg"]["type"]
                    short_p = black_scholes_price(S, short_strike, T_expiry, r, sigma, short_type)
                    
                net_premium = max(5.0, long_p - short_p)
                risk_budget = capital * 0.15  # 15% per trade
                qty = max(1.0, risk_budget / net_premium)
                entry_cost = net_premium * qty
                fee = entry_cost * COST_PCT
                
                capital -= (entry_cost + fee)
                active_positions.append({
                    "ticker": t,
                    "strategy_type": strat["strategy_type"],
                    "long_strike": strat["long_leg"]["strike"],
                    "long_type": strat["long_leg"]["type"],
                    "short_strike": short_strike,
                    "short_type": short_type,
                    "entry_cost": net_premium,
                    "qty": qty,
                    "risk_budget": risk_budget,
                    "days_held": 0
                })
                
                trade_log.append({
                    'timestamp': date_str,
                    'ticker': t,
                    'action': 'ENTRY_' + strat["strategy_type"],
                    'entry_price': round(net_premium, 2),
                    'exit_price': 0.0,
                    'pnl': 0.0,
                    'capital_after': round(capital, 2)
                })
                
        # 3. Compute daily valuation
        mtm_val = capital
        for pos in active_positions:
            t = pos["ticker"]
            S = price_row[t]
            sigma = vol_row[t]
            time_left = max(0.001, T_expiry - (pos["days_held"] / 365.0))
            long_p = black_scholes_price(S, pos["long_strike"], time_left, r, sigma, pos["long_type"])
            short_p = black_scholes_price(S, pos["short_strike"], time_left, r, sigma, pos["short_type"]) if pos["short_strike"] else 0.0
            mtm_val += (long_p - short_p) * pos["qty"]
            
        valuation_history.append({"date": date_str, "account_value": mtm_val})
        
    val_df = pd.DataFrame(valuation_history)
    val_df.to_csv("paper_options_valuation.csv", index=False)
    
    log_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    log_df.to_csv("paper_options_log.csv", index=False)
    
    tot_ret = (val_df["account_value"].iloc[-1] - initial_capital) / initial_capital
    val_df["daily_return"] = val_df["account_value"].pct_change(1)
    std = val_df["daily_return"].std()
    sharpe = (val_df["daily_return"].mean() / std) * np.sqrt(252) if std and std != 0 else 0.0
    cum_max = val_df["account_value"].cummax()
    max_dd = ((val_df["account_value"] - cum_max) / cum_max).min()
    
    print("\n=== NSE OPTIONS SIMULATION METRICS ===")
    print(f"Total Return: {tot_ret * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_dd * 100:.2f}%")
    
    return {
        "total_return": tot_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "valuation_history": val_df
    }

if __name__ == "__main__":
    run_paper_options_simulation()
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_paper_options.py`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add paper_options_trader.py tests/test_paper_options.py
git commit -m "feat: add paper options execution and risk simulator for NSE options"
```

---

### Task 4: Full System Verification & Live Market Simulation Execution

**Files:**
- Execute: `paper_options_trader.py`
- Verify: Outputs `paper_options_log.csv` and `paper_options_valuation.csv`

- [ ] **Step 1: Run full test suite across options components**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_options_engine.py tests/test_regime_agent.py tests/test_paper_options.py`
Expected: PASS (5 passed)

- [ ] **Step 2: Execute full 90-day Paper Options Simulation**

Run: `.\venv\Scripts\python.exe paper_options_trader.py`
Expected: Outputs performance summary metrics, writes `paper_options_log.csv` and `paper_options_valuation.csv`.

- [ ] **Step 3: Commit final simulation outputs and scripts**

```bash
git add paper_options_trader.py paper_options_log.csv paper_options_valuation.csv
git commit -m "feat: complete multi-agent NSE options paper trading simulation"
```

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-01-multi-agent-nse-options.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
