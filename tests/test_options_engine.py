import pytest
import math
from nse_options_engine import black_scholes_price, calculate_greeks, resolve_nse_strikes

def test_black_scholes_call_price():
    # S=24000, K=24000, T=30/365, r=0.07, sigma=0.15
    price = black_scholes_price(S=24000, K=24000, T=30/365, r=0.07, sigma=0.15, option_type="call")
    assert price > 0
    assert round(price, 2) in (407.48, 483.04) or (390 < price < 500)

def test_greeks_calculation():
    greeks = calculate_greeks(S=24000, K=24000, T=30/365, r=0.07, sigma=0.15, option_type="call")
    assert "delta" in greeks and "theta" in greeks and "vega" in greeks and "gamma" in greeks
    assert 0.45 <= greeks["delta"] <= 0.60

def test_resolve_nse_strikes():
    strikes = resolve_nse_strikes("NIFTY.NS", 24123.45)
    assert strikes["atm"] == 24100
    assert strikes["otm_call_1"] == 24150
    assert strikes["otm_put_1"] == 24050

def test_boundary_conditions():
    # Expired option
    assert black_scholes_price(S=24000, K=23500, T=0, r=0.07, sigma=0.15, option_type="call") == 500.0
    assert black_scholes_price(S=24000, K=24500, T=0, r=0.07, sigma=0.15, option_type="call") == 0.0
    # Zero volatility
    p = black_scholes_price(S=24000, K=24000, T=30/365, r=0.07, sigma=0.0, option_type="call")
    assert p > 0
    # Greeks boundary
    g = calculate_greeks(S=24000, K=24000, T=0, r=0.07, sigma=0.15, option_type="call")
    assert g["gamma"] == 0.0

