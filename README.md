# Dispersion Trading Pipeline

An index-vs-component correlation dispersion strategy: sell index implied
volatility, buy a weighted basket of component implied volatility, betting that
realized correlation across the basket comes in lower than what index options
price in.

Based on Driessen, J., Maenhout, P. J., & Vilkov, G. (2009), "The Price of
Correlation Risk: Evidence from Equity Options." *Journal of Finance*, 64(3),
1377–1406.

## The core idea

Index variance is not just the weighted sum of component variances — it depends
on how the components co-move:

$$
Var(index) = Σ wᵢ² σᵢ²  +  ρ · [ (Σ wᵢ σᵢ)² − Σ wᵢ² σᵢ² ]
$$

Given an observed index IV and observed component IVs, everything in that
identity is known except `ρ`, so it can be solved for directly. That number is
the **implied correlation**: the average pairwise correlation the options market
is pricing in. The same identity applied to realized returns gives **realized
correlation**.

The trade monetizes the gap. Implied correlation has historically sat above
realized, which is the risk premium Driessen et al. document.

Assuming a single `ρ` across every pair is a simplification, but a necessary
one — solving for every pairwise correlation from one index IV and N component
IVs is badly underdetermined. Index providers' own implied-correlation indices
make the same assumption.

## Repo layout

The code splits into two tracks that **do not yet meet in the middle**. Knowing
which track a file belongs to is the fastest way to orient yourself.

### Shared math (used by both tracks)

| File | What it does |
|---|---|
| `black_scholes.py` | European option pricer, plus a Newton-Raphson IV solver with bisection fallback. BS is used purely as a quoting convention to turn prices into comparable vol numbers — nothing downstream assumes BS dynamics. |
| `correlation.py` | `implied_correlation()` solves the identity above for ρ. `realized_correlation()` computes the same quantity from historical returns. Has a self-test under `__main__` that recovers a known ρ from synthetic factor-model returns. |
| `config.py` | Basket size, index ticker, expiry window, lookback windows, signal thresholds. Loads component weights from `data/spy_weights.csv` at import time. |

### Track A — live snapshot (done)

| File | What it does |
|---|---|
| `fetch_spy_weights.py` | Downloads SPY's daily holdings file straight from State Street and rebuilds `data/spy_weights.csv`. Handles the `.`→`-` ticker convention (BRK.B → BRK-B) that yfinance needs. `refresh_weights_if_stale()` makes this self-maintaining. |
| `data_pipeline.py` | Pulls live option chains and price history from yfinance. Finds the near-ATM contract at a matching expiry per ticker. |
| `main.py` | Runs the whole live path end to end and prints today's implied correlation, realized correlation, and the spread. |

Track A produces **one data point** — today's. That's enough to prove the math
and plumbing work, and nothing more.

It is verified working end to end: it fetches live SPY plus 150-component
options via yfinance, solves IVs, computes implied correlation, fetches price
history, computes realized correlation, and prints the spread.

Recent live-path hardening:

- `config.py` filters out the malformed `-` ticker.
- `get_price_history()` uses `threads=False`, per-ticker retry, and
  `ffill`/`bfill` to survive Yahoo's flaky batch downloads.
- `get_atm_option()` rejects quotes below the no-arbitrage price floor before
  passing them to the IV solver, which catches stale `lastPrice` quotes.
- Noisy HTTP-layer stderr output is suppressed.

### Track B — historical data pipeline built

| File | What it does |
|---|---|
| `build_basket_universe.py` | Reads a CRSP daily constituents export, ranks by `DlyCap` on the last trading day of each quarter, keeps the top 150, and unions across quarters to get the full ticker superset needed for the options pull. |
| `historical_data_pipeline.py` | Turns a raw IvyDB/OptionMetrics export into a per-(ticker, date, expiry) ATM straddle table with call and put IVs. Caches to `data/atm_straddles.csv` and only rebuilds when the source file is newer. |
| `historical_correlation_series.py` | Builds `data/correlation_history.csv` from ATM straddles, point-in-time quarterly weights, and historical close prices. |
| `correlation_snapshot.py` | Shared one-date calculation: filters usable component IVs, renormalizes weights, and calls implied/realized correlation math. |

Track B's data preparation is done, and the historical glue script now turns
`atm_straddles.csv` plus `quarterly_weights.csv` into the implied/realized
correlation time series consumed by `signal.py` and `backtest.py`.

### Consumers of the generated correlation time series

| File | What it does |
|---|---|
| `signal.py` | Z-scores the implied-minus-realized spread over a rolling window and emits a binary in/out position. |
| `backtest.py` | Variance-notional P&L approximation with a flat transaction-cost charge. **Not** real option position P&L — see the module docstring. |

## Data files

None of these are in the repo. Three come from WRDS and can't be redistributed;
one is generated automatically.

| Path | Source | Read by |
|---|---|---|
| `data/spy_weights.csv` | Auto-generated by `fetch_spy_weights.py` from SSGA | `config.py`, live path |
| `data/Historical_SPY_Components.csv` | WRDS/CRSP daily S&P 500 constituents. Needs `PERMNO`, `Ticker`, `DlyCalDt`, `DlyCap` | `build_basket_universe.py` |
| `data/options_historical_data_full.csv` | WRDS/IvyDB (OptionMetrics): 20.8M rows, 265 tickers plus SPY, 2015-08-28 to 2025-08-29, delta-filtered, DTE-filtered, deduplicated, and verified | `historical_data_pipeline.py` |
| `data/atm_straddles.csv` | Cached output of `historical_data_pipeline.py`: 1.44M rows, one ATM call/put pair per ticker per day | `historical_correlation_series.py` |
| `data/quarterly_weights.csv` | Point-in-time 150-name basket weights per quarter from `build_basket_universe.py`: 6,150 rows, market-cap-weighted | `historical_correlation_series.py` |
| `data/superset_tickers.csv` | 271-ticker basket universe | Historical data sourcing and checks |
| `data/correlation_history.csv` | Generated historical implied/realized correlation series | `signal.py` |
| `data/signal_history.csv` | Generated z-score signal and position series | `backtest.py` |
| `data/backtest_results.csv` | Generated first-pass P&L approximation | Analysis/reporting |

The historical options file has been checked for crossed markets, missing
legs, duplicate rows, and stale entity labels. Nineteen renamed or delisted
entities were resolved by hand so the `secid` labels line up correctly.

## Setup

```bash
pip install -r requirements.txt
mkdir -p data
python fetch_spy_weights.py --top 150   # required before the first main.py run
python main.py
```

The explicit `fetch_spy_weights.py` call is not optional on a fresh clone.
`main.py` calls `import config` *before* it calls `refresh_weights_if_stale()`,
and `config.py` raises `FileNotFoundError` at import time when
`data/spy_weights.csv` is missing — so a clean checkout fails on
`python main.py` alone. Moving the refresh call above the `import config` line
(or making `config.py` lazy-load its weights) fixes this and lets the setup
collapse back to two commands.

Note also that `fetch_spy_weights.py`'s CLI defaults to `--top 50` while
`config.NUM_COMPONENTS` is 150, so passing `--top 150` explicitly is what keeps
a manual run in sync with what `main.py` rebuilds later.

Track A needs outbound access to Yahoo Finance and State Street, and behaves
best during US market hours. With 150 components and a 0.3s sleep per ticker,
a full run takes roughly a minute of wall clock just in fetching.

## Current status

**Track A is done:** the live snapshot path works end to end and now prints
today's implied correlation, realized correlation, and spread. Both self-tests
pass (`black_scholes.py` round-trips a known vol; `correlation.py` recovers a
known ρ from synthetic data).

**Track B data is done:** the full WRDS/IvyDB options export has been cleaned
and reduced to `data/atm_straddles.csv`, and the point-in-time quarterly
weights have been built in `data/quarterly_weights.csv`.

**The historical glue exists and has been debugged:** `historical_correlation_series.py`
produces the implied and realized correlation *series* that `signal.py` and
`backtest.py` take as input. It has been run end to end against real
historical closes and produces 2,493 daily rows from 2015-10-01 to
2025-08-29, with a plausible spike in implied correlation during the March
2020 COVID selloff.

For each historical date, the glue script:

1. Look up the quarter containing the date and pull the *prior* quarter's
   basket from `data/quarterly_weights.csv` (see the point-in-time note
   below for why it's the prior quarter, not the same one).
2. Pull each ticker's and SPY's call/put IVs from `data/atm_straddles.csv`.
3. Drop tickers with no usable quote that day and renormalize the remaining
   weights to sum to 1.
4. Call `implied_correlation()`.
5. Compute trailing 30-day realized correlation via `realized_correlation()`
   using actual historical close/spot prices.
6. Repeat across every available date to build the full time series.

Steps 3–5 are exactly what `main.py` already does for a single date, and
`correlation_snapshot.py` now factors that shared logic out so both entry
points call the same function rather than duplicating it.

**Two point-in-time bugs were found and fixed while validating this output:**

- **Quarter lookup was off by one.** `quarterly_weights.csv`'s market cap
  (`DlyCap`) for a quarter is measured on the *last trading day* of that
  quarter, so a naive `date -> same-quarter weights` lookup was using
  end-of-quarter market cap to weight trading days earlier in that same
  quarter, before the cap was observable. `_quarter_for_date()` now looks up
  the *prior* quarter's weights instead, so each weight vector only applies
  to dates after its snapshot was taken. This pushes the usable start of the
  series from 2015-08-28 to 2015-10-01, since there's no valid prior-quarter
  weight vector before that.
- **Ticker labels didn't match across renamed entities.** `quarterly_weights.csv`
  (CRSP-sourced) updates ticker labels at the actual corporate action date
  (FB → META, ANTM → ELV, PCLN → BKNG, FISV → FI, UTX/RTN → RTX post-merger,
  DWDP → DD), but `atm_straddles.csv` (OptionMetrics-sourced) keeps one fixed
  label per secid across its whole history. Without reconciling these,
  several large-cap positions silently dropped out of the implied-correlation
  basket for years at a time with no error raised. `load_quarterly_weights()`
  now applies a `TICKER_ALIASES` map and collapses any resulting
  `(quarter, ticker)` duplicates (e.g. the DowDuPont merger transition
  quarter, 2017Q3, where both `DWDP` and `DD` briefly appeared as separate
  rows) by summing weight rather than letting one silently overwrite the
  other.

Both issues are now confirmed fixed against the real data; see "Known
simplifications" for a third, related coverage effect (the DTE extraction
window) that was investigated and found to be a data-source limitation
rather than a bug.

## Known simplifications

- **Single average correlation.** Standard practice; see the note above.
- **ATM options only.** Real dispersion desks trade the whole skew. A
  skew-aware version (variance-swap replication across a strip of strikes) is
  the natural extension once the ATM version works.
- **Options data has a fixed DTE window from the source extraction.**
  `options_historical_data_full.csv` was pulled from WRDS with a 20-40 day
  days-to-expiry filter applied at extraction time, so `atm_straddles.csv`
  only ever contains DTE values between 21 and 39. Widening
  `config.MIN_DAYS_TO_EXPIRY`/`MAX_DAYS_TO_EXPIRY` beyond that range has no
  effect, since there's no data outside it to select from. Because
  monthly-only equity option expiries are spaced roughly 28-35 calendar days
  apart, wider than this 18-day window, names without weekly listings
  periodically have zero contracts available for a stretch of each monthly
  cycle. This shows up as `num_components` cycling between roughly 110 and
  143 (out of a 150-name basket) on a regular monthly rhythm throughout the
  full 2015-2025 sample. It's a structural property of the source data, not
  a code bug: coverage never drops below the `min_components=50` floor, so
  no dates are lost, and the effect is quantified and consistent rather than
  random. Closing it fully would require re-extracting
  `options_historical_data_full.csv` from WRDS with a wider DTE window
  (something like 15-55), which wasn't done here given the scope of the
  project; it's a natural next step if the data pipeline is revisited.
- **Backtest is a notional-scaled approximation.** It doesn't hold actual
  option positions, roll at expiry, or model assignment, margin, or bid-ask.
  Real position-level tracking is where most of the remaining difficulty lives.
- **Constant risk-free rate** (`RISK_FREE_RATE = 0.045`) rather than a
  maturity-matched Treasury yield per expiry.
- **150 names, not full replication.** Same simplification real desks make when
  full index replication isn't practical. Coverage is exposed as
  `config.BASKET_COVERAGE`; the renormalization in `main.py` keeps the math
  correct at any coverage level, but higher coverage means the basket is a
  closer proxy for the index. The original 10-name basket produced ρ = 1.25,
  which is what motivated the move to 150.
- **SPY, not SPX.** SPX options are cash-settled index options and harder to
  source freely. SPY is the liquid accessible proxy.

## Build order

7. ~~Run `historical_correlation_series.py` with real historical closes and
   validate the output.~~ Done. Found and fixed a look-ahead bias in the
   quarterly weight lookup and a ticker-labeling mismatch across renamed
   entities; documented a source-data DTE coverage limitation.
8. **← current step.** Run `signal.py` and `backtest.py` and tune the
   z-score thresholds, which are still placeholders.
9. Compute performance metrics comparable to the GARCH project: Sharpe, max
   drawdown, and Calmar.
10. Production-awareness stretch: convert the large CSVs to Parquet.
11. Longer-term stretch: extend the backtest toward real position-level P&L.
