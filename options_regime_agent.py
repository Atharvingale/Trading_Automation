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
