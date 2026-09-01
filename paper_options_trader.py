import os
import datetime
import math
import numpy as np
import pandas as pd
import yfinance as yf

from nse_options_engine import black_scholes_price
from options_regime_agent import select_option_strategy

COST_PCT = 0.0015  # 0.15% friction per leg per trade
RISK_BUDGET_PCT = 0.15  # 15% of account balance allocated per trade position
STOP_LOSS_PCT = -0.50  # -50% stop loss
PROFIT_TARGET_PCT = 0.50  # +50% profit target
REBALANCE_FREQ = 5  # Rebalance / enter strategy every 5 trading days

def extract_legs(strat_dict: dict) -> list:
    legs = []
    if strat_dict.get("long_leg"):
        leg = strat_dict["long_leg"]
        ltype = leg["type"].replace("long_", "").replace("short_", "")
        legs.append({"action": "BUY", "type": ltype, "strike": float(leg["strike"])})
    if strat_dict.get("short_leg"):
        leg = strat_dict["short_leg"]
        ltype = leg["type"].replace("long_", "").replace("short_", "")
        legs.append({"action": "SELL", "type": ltype, "strike": float(leg["strike"])})
    for leg in strat_dict.get("extra_legs", []):
        act = "SELL" if "short" in leg["type"] else "BUY"
        opt_type = "call" if "call" in leg["type"] else "put"
        legs.append({"action": act, "type": opt_type, "strike": float(leg["strike"])})
    return legs

def price_strategy_legs(legs: list, S: float, days_to_exp: float, r: float = 0.06, sigma: float = 0.20) -> tuple:
    T = max(0.001, days_to_exp) / 365.0
    unit_val = 0.0
    total_leg_sum = 0.0
    for leg in legs:
        p = black_scholes_price(S, leg["strike"], T, r, sigma, leg["type"])
        total_leg_sum += p
        if leg["action"] == "BUY":
            unit_val += p
        else:
            unit_val -= p
    return unit_val, total_leg_sum

def run_paper_options_simulation(
    tickers: list = None,
    days: int = 30,
    initial_capital: float = 1000000.0,
    mock: bool = True
) -> dict:
    if tickers is None:
        tickers = ["NIFTY.NS"]
        
    price_data = {}
    if not mock:
        try:
            end_dt = datetime.datetime.now()
            start_dt = end_dt - datetime.timedelta(days=days * 2)
            df_raw = yf.download(tickers, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), progress=False)
            if "Close" in df_raw and not df_raw["Close"].empty:
                close_df = df_raw["Close"]
                for t in tickers:
                    if len(tickers) == 1 and isinstance(close_df, pd.Series):
                        s = close_df.dropna()
                    else:
                        s = close_df[t].dropna() if t in close_df else pd.Series()
                    if not s.empty:
                        price_data[t] = s.iloc[-days:]
        except Exception:
            pass

    # Fallback to mock data for missing tickers or when mock=True
    end_dt = datetime.datetime.now()
    dates = [(end_dt - datetime.timedelta(days=days - i)).strftime("%Y-%m-%d") for i in range(days)]
    
    for t in tickers:
        if t not in price_data or len(price_data[t]) < days:
            np.random.seed(42 + hash(t) % 1000)
            start_price = 24000.0 if "NIFTY" in t.upper() else 2500.0
            returns = np.random.normal(0.0005, 0.01, size=days)
            p_path = start_price * np.cumprod(1 + returns)
            price_data[t] = pd.Series(p_path, index=dates)

    main_ticker = tickers[0]
    prices_series = price_data[main_ticker]
    date_strs = list(prices_series.index.astype(str))
    
    capital = float(initial_capital)
    active_position = None
    trade_log = []
    valuation_history = []

    for i, date_str in enumerate(date_strs):
        current_price = float(prices_series.iloc[i])
        
        # If we have an active position, evaluate daily valuation and exit triggers
        if active_position is not None:
            days_held = i - active_position["entry_idx"]
            remaining_days = max(1, active_position["initial_expiry_days"] - days_held)
            
            curr_unit_val, total_leg_sum = price_strategy_legs(
                active_position["legs"],
                current_price,
                days_to_exp=remaining_days,
                r=0.06,
                sigma=0.20
            )
            
            curr_pos_val = curr_unit_val * active_position["units"]
            entry_cost = active_position["entry_unit_cost"] * active_position["units"]
            
            if entry_cost != 0:
                pnl_pct = (curr_pos_val - entry_cost) / abs(entry_cost)
            else:
                pnl_pct = 0.0
                
            # Exit triggers
            exit_reason = None
            if pnl_pct <= STOP_LOSS_PCT:
                exit_reason = "STOP_LOSS"
            elif pnl_pct >= PROFIT_TARGET_PCT:
                exit_reason = "PROFIT_TARGET"
            elif days_held >= REBALANCE_FREQ:
                exit_reason = "REBALANCE_EXPIRY"
                
            if exit_reason is not None:
                # Close position
                exit_friction = total_leg_sum * active_position["units"] * COST_PCT * len(active_position["legs"])
                realized_cash = curr_pos_val - exit_friction
                capital += realized_cash
                
                trade_log.append({
                    "timestamp": date_str,
                    "ticker": main_ticker,
                    "action_type": "EXIT",
                    "strategy_type": active_position["strategy_type"],
                    "execution_price": round(current_price, 2),
                    "cost_incurred": round(exit_friction, 2),
                    "cash_balance": round(capital, 2),
                    "pnl_pct": round(pnl_pct * 100, 2),
                    "reason": exit_reason
                })
                active_position = None

        # Check if we should open a new position
        if active_position is None and (i % REBALANCE_FREQ == 0):
            # Select strategy using options_regime_agent
            strat_info = select_option_strategy(
                ticker=main_ticker,
                date_str=date_str,
                current_price=current_price,
                iv_rank=45.0,
                signal_override="BUY" if mock else None
            )
            
            legs = extract_legs(strat_info)
            if legs:
                initial_expiry_days = 30
                unit_val, total_leg_sum = price_strategy_legs(legs, current_price, days_to_exp=initial_expiry_days, r=0.06, sigma=0.20)
                
                cost_mag = max(abs(unit_val), 10.0)
                budget = capital * 0.05  # 5% prudent risk budget per trade
                units = budget / cost_mag
                
                entry_friction = total_leg_sum * units * COST_PCT * len(legs)
                entry_net_cost = unit_val * units
                
                capital -= (entry_net_cost + entry_friction)
                
                active_position = {
                    "entry_idx": i,
                    "strategy_type": strat_info["strategy_type"],
                    "legs": legs,
                    "units": units,
                    "entry_unit_cost": unit_val,
                    "initial_expiry_days": initial_expiry_days
                }
                
                trade_log.append({
                    "timestamp": date_str,
                    "ticker": main_ticker,
                    "action_type": "ENTRY",
                    "strategy_type": strat_info["strategy_type"],
                    "execution_price": round(current_price, 2),
                    "cost_incurred": round(entry_friction, 2),
                    "cash_balance": round(capital, 2),
                    "pnl_pct": 0.0,
                    "reason": "NEW_SIGNAL"
                })

        # Calculate current total account value
        pos_val = 0.0
        if active_position is not None:
            days_held = i - active_position["entry_idx"]
            remaining_days = max(1, active_position["initial_expiry_days"] - days_held)
            curr_unit_val, _ = price_strategy_legs(
                active_position["legs"],
                current_price,
                days_to_exp=remaining_days,
                r=0.06,
                sigma=0.20
            )
            pos_val = curr_unit_val * active_position["units"]
            
        account_val = capital + pos_val
        valuation_history.append({
            "date": date_str,
            "account_value": account_val,
            "cash_balance": capital,
            "position_value": pos_val
        })

    # Convert to DataFrames and compute metrics
    val_df = pd.DataFrame(valuation_history)
    val_df["daily_return"] = val_df["account_value"].pct_change().fillna(0.0)
    
    log_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(columns=[
        "timestamp", "ticker", "action_type", "strategy_type", "execution_price", "cost_incurred", "cash_balance", "pnl_pct", "reason"
    ])
    
    val_df.to_csv("paper_options_valuation.csv", index=False)
    log_df.to_csv("paper_options_log.csv", index=False)

    final_val = val_df["account_value"].iloc[-1]
    total_return = (final_val - initial_capital) / initial_capital
    
    daily_rets = val_df["daily_return"].iloc[1:]
    std_ret = daily_rets.std()
    sharpe = (daily_rets.mean() / std_ret * math.sqrt(252)) if (std_ret is not None and std_ret > 0) else 0.0
    
    cum_max = val_df["account_value"].cummax()
    drawdown = (val_df["account_value"] - cum_max) / cum_max
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "final_value": float(final_val),
        "trade_count": len(log_df)
    }

if __name__ == "__main__":
    res = run_paper_options_simulation(tickers=["^NSEI"], days=90, initial_capital=1000000.0, mock=False)
    print("\n=== MULTI-AGENT NSE OPTIONS PAPER TRADING PERFORMANCE ===")
    print(f"Initial Capital    : INR 1,000,000.00")
    print(f"Final Portfolio Val: INR {res['final_value']:,.2f}")
    print(f"Total Strategy Ret : {res['total_return'] * 100:.2f}%")
    print(f"Sharpe Ratio       : {res['sharpe']:.4f}")
    print(f"Max Drawdown       : {res['max_drawdown'] * 100:.2f}%")
    print(f"Total Trade Events : {res['trade_count']}")
    print(f"Log files written  : paper_options_log.csv, paper_options_valuation.csv")
