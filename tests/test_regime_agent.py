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
