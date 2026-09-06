"""
Configuration for the dispersion trading pipeline.

Component tickers and weights are loaded from spy_weights.csv, which is
generated automatically by fetch_spy_weights.py (called from main.py on
a staleness check, see refresh_weights_if_stale). You shouldn't need to
run anything manually in normal use.
"""

import csv
import os

# Number of top-weighted SPY constituents to include in the basket.
# This is the single source of truth for basket size, referenced both
# by main.py's auto-refresh call and by any manual
# `python fetch_spy_weights.py --top ...` runs, so a manual override on
# the command line can't silently drift out of sync with what the
# auto-refresh rebuilds later.
NUM_COMPONENTS = 150

# The index whose options you're comparing against the basket.
# SPY is used instead of SPX because SPX options are cash-settled index
# options that are harder to source with free data. SPY is a liquid,
# easily accessible proxy.
INDEX_TICKER = "SPY"

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__),"data", "spy_weights.csv")

if not os.path.exists(_WEIGHTS_PATH):
    raise FileNotFoundError(
        "spy_weights.csv not found. This should be created automatically "
        "by main.py on first run via fetch_spy_weights.refresh_weights_if_stale(). "
        "If you're importing config.py directly without having run main.py "
        "first, run `python fetch_spy_weights.py --top 150` manually first."
    )

COMPONENT_WEIGHTS = {}
with open(_WEIGHTS_PATH) as f:
    for row in csv.DictReader(f):
        ticker = row["ticker"].strip()
        if not ticker or not any(c.isalpha() for c in ticker):
            continue  # skip malformed rows (e.g. a stray "-" from a bad parse upstream)
        COMPONENT_WEIGHTS[ticker] = float(row["weight"])

COMPONENT_TICKERS = list(COMPONENT_WEIGHTS.keys())

# Coverage: the fraction of the index this basket spans. The single-correlation
# identity needs this reasonably high — at 34.5% (the original 10 names) it
# broke down and produced rho = 1.25. Renormalization in main.py handles the
# math correctness regardless, but higher coverage means the renormalized
# basket is a closer proxy for the real, full index.
BASKET_COVERAGE = sum(COMPONENT_WEIGHTS.values())

# Risk-free rate used in Black-Scholes. Treated as constant for
# simplicity. A more careful version would pull a matching-maturity
# Treasury yield for each option's expiry.
RISK_FREE_RATE = 0.045

# How far out to look for expiries, in days. Options dispersion trades
# are typically done on monthly expiries around 30-60 days out.
MIN_DAYS_TO_EXPIRY = 15
MAX_DAYS_TO_EXPIRY = 50

# Lookback window (trading days) for realized correlation and realized
# volatility calculations.
REALIZED_LOOKBACK_DAYS = 30

# Rolling window (trading days) used when z-scoring the implied minus
# realized correlation spread to generate signals.
SIGNAL_ZSCORE_WINDOW = 60

# Entry/exit thresholds on the z-scored spread. These are placeholders,
# tune them once you have real historical output to look at.
ENTRY_ZSCORE = 1.25
EXIT_ZSCORE = -0.5
