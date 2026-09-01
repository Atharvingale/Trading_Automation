import os
import pytest
from paper_options_trader import run_paper_options_simulation

def test_paper_options_simulation_runs():
    res = run_paper_options_simulation(tickers=["NIFTY.NS"], days=30, initial_capital=1000000.0, mock=True)
    assert "total_return" in res
    assert "sharpe" in res
    assert "max_drawdown" in res
    assert os.path.exists("paper_options_log.csv")
    assert os.path.exists("paper_options_valuation.csv")
