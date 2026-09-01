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
        delta_val = 1.0 if (option_type.lower() == "call" and S > K) else (-1.0 if option_type.lower() == "put" and K > S else 0.0)
        return {"delta": delta_val, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        
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
