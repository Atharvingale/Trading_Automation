# Reinforcement Learning Trading System Validation

## Executive Summary
This report summarizes an applied research project evaluating whether a Deep Reinforcement Learning (DRL) agent, built using the FinRL framework, could reliably extract an actionable trading edge from the Indian equity market (NSE). 

While many public DRL backtests report massive benchmark outperformance, this project enforced stringent data hygiene standards to strip away illusions of edge. By enforcing realistic transaction costs (0.15%), auditing for survivorship bias, demanding strict out-of-sample walk-forward validaton, and statistically evaluating features independently, we demonstrated that standard technical and sentiment indicators possess **no meaningful predictive power** for this universe.

**Conclusion:** The project is a confirmed null result. The model learned to optimally minimize transaction costs rather than predicting price movements. Recognizing the value of an honest negative over an artificially fitted positive, no live capital was deployed.

---

## 1. Experimental Setup & Foundation
* **Framework:** FinRL (PPO, A2C, DDPG via Stable-Baselines3).
* **Asset Universe:** Evaluated initially on a single asset (`RELIANCE.NS`) before expanding to a 10-stock NSE portfolio to diversify holding capabilities.
* **Validation Methodology:** Walk-forward validation across three consecutive, non-overlapping windows (2019, 2022, 2025–26).
* **Friction Constraint:** Applied a 0.15% per-trade transaction and slippage cost to reflect real-world NSE friction.

---

## 2. Uncovering Survivorship Bias
Upon expanding the validation framework to a multi-asset portfolio, a naive run using the top 10 largest NSE stocks *as of today* produced a staggering +35.15% return (Sharpe 1.04), massively outperforming the +12.37% baseline. 

**The Audit:** Suspecting survivorship bias, we reconstructed an unbiased top-10 constituent list indexed specifically to *January 2015*. This unbiased list rightfully included subsequent catastrophic failures (e.g., YES Bank).

**The Result:** Performance against the unbiased benchmark dropped to a realistic **+13.40%** return vs **+13.31%** Buy & Hold. While the return delta vanished, the DRL agent demonstrated significant value by cutting the max drawdown nearly in half (-13.35% vs -25.82%), effectively rotating capital away from falling assets.

---

## 3. The "No-Trade" Degeneracy
When transaction costs (0.15%) were introduced, the agent's out-of-sample performance in volatile markets (e.g., 2022) immediately collapsed to exactly 0.00% return — resulting from zero trades placed.

**The Fix Attempts:** We iteratively introduced both an entropy bonus (`ent_coef=0.05`) and differential-Sharpe reward shaping to force the agent out of the dormant local optimum. 
**The Assessment:** While the agent resumed trading, post-cost performance universally dragged below baseline. Because these tuning attempts were repeatedly checked against the same holdout datasets to coerce a positive result, we categorized them strictly as **data snooping** and rejected them.

---

## 4. Fundamental Feature Audits (The Null Proof)
To understand why the agent inevitably capitulated under trading friction, we subjected the environment's observation inputs to a rigorous test of statistical significance. Using strictly segregated pre-2019 training data:

1. **Technical Indicators:** Spearman rank correlations across all 8 provided state features (MACD, RSI, CCI, etc.) and short-term forward returns (1D/5D) measured ~0.04 or less.
2. **Sentiment Indicators:** Generated sentiment embeddings utilizing FinBERT on historical market events exhibited Spearman correlations hovering around ~0.02.

**Interpretation:** Both data structures operated at the level of pure statistical noise. The agent did not fail to learn a profitable edge; rather, there was mathematically **no edge to learn**. The initial "do nothing" policy was in fact the mathematically optimal response to a frictionless-optimal model penalized by real-world costs.

## 5. Investigating a Non-RL Alternative (Momentum Strategy)
As a control check, we tested whether removing the RL wrapper entirely and executing a simple rule-based strategy could generate alpha. We implemented a 90-day rolling momentum selector (buying the top 5 performing stocks every 30 days) equipped with the same 0.15% transaction cost limitation.

While this momentum rule achieved an exceptional +63.55% outperformance during the 2020-2021 holdout, testing it against multiple historic regimes immediately revealed the danger of limited out-of-sample datasets:
* **The Post-COVID V-Shape (2020–2021):** Momentum outperformed (+63.55%).
* **Global Financial Crisis (2008–2009):** Momentum dramatically underperformed the baseline (+6.70% vs +20.59%) and suffered deeper drawdowns (-54%).
* **Sideways/Choppy Market (2011–2013):** Momentum fractured, actively losing money (-10.49%) while the naive Equal-Weight baseline yielded +21.97%.

**Interpretation:** The momentum return was a regime artifact, not a systemic edge. Entering a trend works flawlessly in a straight-line recovery, but results in severe whipsaw losses during typical choppy environments.

---

## Closing Remarks
In empirical finance research, disproving false edge is equally critical to discovering real alpha. This project highlights common systemic pitfalls in applied DRL for finance—specifically survivorship bias, test-leakage via hyperparameter tuning, and insufficient raw signal.

All validation scripts underpinning these audits (`step5b_audit.py`, `step5d_feature_audit.py`, `step5e_sentiment_audit.py`) are preserved to serve as standardized diagnostic tools for future financial ML workstreams.