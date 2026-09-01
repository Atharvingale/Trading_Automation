# Project Overview: Reinforcement Learning Trading System (FinRL Validation)

## Objective
A rigorous, sequential deployment of a FinRL-based Reinforcement Learning trading system targeting the Indian equity market (NSE). The project enforced strict anti-bias rules (zero lookahead, zero survivorship bias, strict out-of-sample holdouts) and realistically accounted for market friction (0.15% transaction costs) to determine if standard DRL agents could find a genuine, exploitable trading edge.

## Final Conclusion: Null Result
Following a comprehensive audit for survivorship bias and rigorous feature signal verification, the project concludes with a **null result**. 
Rigorous statistical audits of pre-holdout training data confirmed that standard FinRL technical features (MACD, RSI, etc.) and FinBERT sentiment scores exhibit **no non-trivial predictive correlation (~0.04 Spearman rho or lower)** with forward returns. Consequently, the RL agents were either memorizing noise or avoiding trades entirely to minimize cost penalties. **No live capital will be deployed.**

## Key Findings by Phase

### 1. Survivorship Bias Correction (Step 5b)
* **Initial Bias:** A naive expansion to a 10-stock NSE universe (picking today's top winners) generated an artificial +35.15% return.
* **Correction:** Re-evaluating on a bias-free historical constituent list (Jan 2015 universe including later failures/declines like YES Bank) revealed a realistic +13.40% return, matching the benchmark Buy & Hold return (+13.31%).
* **Result:** The agent achieved a lower Max Drawdown (-13.35% vs -25.82%), confirming risk management utility, but no absolute alpha.

### 2. Degenerate Policy Audit (Step 5c)
* **Finding:** Previous attempts to "fix" the agent's no-trade policy (via entropy bonuses or reward shaping) failed.
* **Result:** These attempts were rejected as regression/data snooping; forcing the agent to trade merely increased costs and reduced risk-adjusted performance.

### 3. Feature Signal Audit (Step 5d & 5e)
* **Independence Audit:** Conducted a Spearman correlation analysis between inputs and forward returns (1D/5D) using only training-period data.
* **Finding:** Absolute median correlations hovered at **~0.02-0.04**, fundamentally indistinguishable from statistical noise. The agent mathematically lacked an edge to learn from.

## Integrity Summary
* **Retired Holdouts:** The 2019, 2022, and 2025–26 windows were retired after three failed tuning cycles.
* **Step 8 (Capital Gate):** Formalized as "Not Attempted" due to lack of verifiable edge.
* **Validation Outcome:** The null result is considered a successful scientific outcome, proving that this asset universe and these feature sets cannot support a profitable DRL strategy given current friction levels.

## Evidence Trail
* `step5b_audit.py`: Validates the use of historical constituents (survivorship bias removal).
* `step5d_feature_audit.py`: Statistical proof that technical indicators lack predictive signal.
* `step5e_sentiment_audit.py`: Statistical proof that FinBERT sentiment lacks predictive signal.
* `PROGRESS.md`: Detailed, honest tracker of all failed validation attempts.
