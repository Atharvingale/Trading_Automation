# Multi-Agent Regime-Adaptive NSE Options Trading Strategy Design

**Date:** 2026-09-01  
**Target Market:** Indian Equity Index & Stock Options (NSE: `NIFTY.NS`, `BANKNIFTY.NS`, `RELIANCE.NS`)  
**Objective:** Maximize risk-adjusted returns by dynamically deploying regime-specific option strategies (Bull Call Spreads, Bear Put Spreads, Iron Condors, Long Options) using multi-agent LLM analysis and Black-Scholes Greeks pricing.

---

## 1. System Architecture

```
 ┌────────────────────────────────────────────────────────┐
 │            1. Data Fetcher & Greeks Engine             │
 │   - Fetch NSE Underlying (NIFTY, BANKNIFTY, RELIANCE) │
 │   - Black-Scholes Model: IV, Delta, Theta, Vega        │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │           2. Multi-Agent Regime Classifier             │
 │   - Technical, Fundamental, Volatility Analysts        │
 │   - Classifies: Strong Up, Strong Down, Range, Spike   │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │           3. Option Strategy Selector Engine           │
 │   - Strong Up   → Bull Call Spread / 0.60 Delta Call   │
 │   - Strong Down → Bear Put Spread / 0.60 Delta Put    │
 │   - Rangebound  → Iron Condor / Credit Spread          │
 │   - High IV     → Credit Spread (Sell Inflated Premium)│
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │        4. Execution & Risk Engine (0.15% Cost)        │
 │   - Position Sizing, Stop Loss (-50% of max loss)     │
 │   - Expiry Handling (Thursday Weekly / Monthly)       │
 │   - Paper Trading Log & Portfolio Valuation CSV        │
 └───────────────────────────┴────────────────────────────┘
```

---

## 2. Core Subsystems & Interfaces

### 2.1 Black-Scholes Options Engine (`nse_options_engine.py`)
- **Greeks Calculation**:
  - Implied Volatility (IV) estimation from underlying price series variance.
  - Call/Put Premium: $C(S, K, T, r, \sigma)$, $P(S, K, T, r, \sigma)$
  - Delta ($\Delta$), Gamma ($\Gamma$), Theta ($\Theta$), Vega ($\nu$)
- **Strike & Expiry Resolution**:
  - `NIFTY.NS`: 50-point strike steps, weekly Thursday expiration.
  - `BANKNIFTY.NS`: 100-point strike steps, weekly Thursday expiration.
  - `RELIANCE.NS`: 20-point strike steps, monthly last Thursday expiration.

### 2.2 Multi-Agent Regime Classifier (`options_regime_agent.py`)
- Interfaces with `TradingAgentsGraph`.
- Evaluates underlying technical momentum, historical volatility, and news context.
- Maps LLM signals (`BUY`, `OVERWEIGHT`, `NEUTRAL/HOLD`, `UNDERWEIGHT`, `SELL`) + IV Rank into 4 Market Regimes:
  1. **Bullish Trend** ($\Delta > +0.50$): Deploy **Bull Call Spread** (Buy ATM Call, Sell OTM Call).
  2. **Bearish Trend** ($\Delta < -0.50$): Deploy **Bear Put Spread** (Buy ATM Put, Sell OTM Put).
  3. **Rangebound / Neutral** ($|\Delta| \le 0.20$, IV Rank $\ge 40\%$): Deploy **Iron Condor** / **Credit Spread** (Harvest Theta $\Theta$).
  4. **High Volatility Expansion**: Deploy **Long Straddle/Strangle**.

### 2.3 Paper Options Execution Simulator (`paper_options_trader.py`)
- Simulates realistic option spread entry, mark-to-market (MTM) daily valuation, and expiry settlements.
- **Friction**: Enforces **0.15% per leg** transaction cost on entry and exit.
- **Risk Management**:
  - Maximum capital allocation per trade: 20% of account balance.
  - Stop loss: Exit spread if position drawdown exceeds -50% of initial premium/risk.
  - Profit target: Take profit at 50% of maximum potential gain on credit spreads.

---

## 3. Data Flow & Execution Sequence

1. Fetch 120 days of historical daily price & volatility data for NSE tickers via `yfinance`.
2. Compute rolling 20-day annualized Implied Volatility and IV Rank percentile.
3. At each 5-day rebalance interval:
   - Query TradingAgents multi-agent graph for underlying directional regime.
   - Select option strategy and exact strike levels (ATM, 1-step OTM, 2-step OTM).
   - Price option legs using Black-Scholes engine.
   - Deduct transaction fees (0.15% per leg).
4. Track daily portfolio mark-to-market account value until contract expiration or stop loss trigger.
5. Export detailed execution logs:
   - `paper_options_log.csv`: Individual trade entry/exit, strategy type, strikes, Greeks, and PnL.
   - `paper_options_valuation.csv`: Daily account value time-series.

---

## 4. Performance & Verification Metrics
- **Strategy Total Return (%)** vs **Underlying Benchmark Buy & Hold (%)**
- **Option Sharpe Ratio**
- **Win Rate (%)** across completed option spreads
- **Max Drawdown (%)**
- **Total Transaction Friction Paid (INR)**
