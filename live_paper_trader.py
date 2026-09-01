import os
import sys
import time
import datetime
import math
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), "TradingAgents", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "TradingAgents")))

from nse_options_engine import black_scholes_price, calculate_greeks, resolve_nse_strikes
from options_regime_agent import select_option_strategy
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

print("=========================================================")
print("===      LIVE MARKET PAPER TRADING & ENGINE MONITOR   ===")
print("=========================================================")
print("Mode: LIVE PAPER TRADING (Zero Real Capital at Risk)")
print("Market: National Stock Exchange of India (NSE)")
print("Tickers: NIFTY 50 (^NSEI), BANKNIFTY (^NSEBANK), RELIANCE (RELIANCE.NS)\n")

# Configuration
INITIAL_CAPITAL = 100_000.0  # INR 100,000 starting paper balance
RISK_BUDGET_PCT = 0.15       # 15% max allocation per trade
COST_PCT = 0.0015            # 0.15% friction per leg

def fetch_live_quote(ticker: str) -> dict:
    """Fetch real-time / latest available market data via yfinance."""
    try:
        t = yf.Ticker(ticker)
        fast_info = getattr(t, "fast_info", None)
        last_price = None
        if fast_info:
            last_price = fast_info.last_price
            
        if not last_price or np.isnan(last_price):
            hist = t.history(period="5d")
            if not hist.empty:
                last_price = float(hist["Close"].iloc[-1])
                
        # Fetch 20-day historical volatility
        hist_20d = t.history(period="30d")["Close"]
        if len(hist_20d) >= 10:
            returns = hist_20d.pct_change().dropna()
            vol_20d = float(returns.std() * np.sqrt(252))
        else:
            vol_20d = 0.18  # default 18% IV
            
        return {
            "ticker": ticker,
            "price": float(last_price) if last_price else 24000.0,
            "volatility": vol_20d,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as err:
        print(f"Warning: Could not fetch live quote for {ticker} ({err}). Using fallback values.")
        return {
            "ticker": ticker,
            "price": 24500.0,
            "volatility": 0.18,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def run_live_paper_trader():
    # Detect LLM Provider
    provider = os.getenv("TRADINGAGENTS_LLM_PROVIDER")
    if not provider:
        if os.getenv("OPENROUTER_API_KEY"):
            provider = "openrouter"
        elif os.getenv("GROQ_API_KEY"):
            provider = "groq"
        else:
            provider = "openai"
            
    print(f"Active LLM Provider: {provider}")
    
    ta_graph = None
    try:
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = provider
        config["max_debate_rounds"] = 1
        config["max_risk_discuss_rounds"] = 1
        ta_graph = TradingAgentsGraph(debug=False, config=config)
        print("TradingAgents Multi-Agent LLM Graph initialized successfully.")
    except Exception as err:
        print(f"Note: Multi-agent graph operating in rule baseline mode ({err}).")
        ta_graph = None

    tickers = ["^NSEI", "^NSEBANK", "RELIANCE.NS"]
    
    print("\n--- Live Market Scan & Quote Summary ---")
    live_quotes = {}
    for ticker in tickers:
        q = fetch_live_quote(ticker)
        live_quotes[ticker] = q
        print(f"[{q['timestamp']}] {ticker:12s} | Spot: INR {q['price']:10,.2f} | 20D IV: {q['volatility']*100:5.2f}%")
        
    print("\n--- Live Multi-Agent Regime Analysis & Option Strategy Selection ---")
    recommendations = []
    
    for ticker in tickers:
        q = live_quotes[ticker]
        S = q["price"]
        sigma = q["volatility"]
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Select strategy using Multi-Agent Graph + IV Rank
        strat = select_option_strategy(
            ticker=ticker,
            date_str=date_str,
            current_price=S,
            iv_rank=sigma * 100,
            ta_graph=ta_graph
        )
        
        strikes = resolve_nse_strikes(ticker, S)
        T_days = 14
        r = 0.07
        
        call_atm = black_scholes_price(S, strikes["atm"], T_days/365.0, r, sigma, "call")
        put_atm  = black_scholes_price(S, strikes["atm"], T_days/365.0, r, sigma, "put")
        greeks_call = calculate_greeks(S, strikes["atm"], T_days/365.0, r, sigma, "call")
        
        recommendations.append({
            "Ticker": ticker,
            "Spot_Price": round(S, 2),
            "Strategy": strat["strategy_type"],
            "ATM_Strike": strikes["atm"],
            "Call_Prem": round(call_atm, 2),
            "Put_Prem": round(put_atm, 2),
            "Delta": round(greeks_call["delta"], 4),
            "Theta": round(greeks_call["theta"], 2)
        })
        
    df_rec = pd.DataFrame(recommendations)
    print(df_rec.to_string(index=False))
    
    # Save Live Market Snapshot Log
    df_rec.to_csv("live_paper_trading_recommendations.csv", index=False)
    print("\nLive recommendations written to: live_paper_trading_recommendations.csv")
    print("Run `python live_paper_trader.py` anytime to get real-time market trading signals!")

if __name__ == "__main__":
    run_live_paper_trader()
