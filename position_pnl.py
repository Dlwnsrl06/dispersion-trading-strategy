"""
position_pnl.py

Real position-level P&L for the dispersion trade: short one index (SPY)
ATM straddle, long a vega-matched basket of component ATM straddles,
held from entry to whichever comes first -- the z-score exit signal
firing, or the entered contracts' own expiry. Contracts are NOT rolled
if the signal remains active past expiry (per project decision); the
position simply closes at expiry and waits for the next distinct entry
signal in signal_history.csv.

This replaces backtest.py's notional-scaled correlation-spread proxy
with an actual option position, priced from real strikes and bid/ask
in atm_straddles.parquet rather than a synthetic notional times the
daily change in a correlation number.

Data constraint this module works around: options_historical_data_full.csv
was extracted from WRDS with a 20-40 day DTE filter, so any single
(ticker, exdate) contract has real market quotes only while its DTE is
between 21 and 39 -- roughly the middle three weeks of its life. There
are no quotes anywhere near actual expiry for any contract. Positions
held into this "dark zone" (DTE < 21) are marked using Black-Scholes
with the last observed market IV for that leg, consistent with how this
project already treats BS as a quoting convention rather than a
dynamics model (see black_scholes.py's docstring). At actual expiry, no
model is needed at all: settlement is pure intrinsic value from the
expiry-date spot price, which is model-free.

Entry and exit-by-signal transactions are costed using real bid-ask
spreads (sell short legs at bid, buy long legs at offer, reversed on
exit), unlike backtest.py's flat 5bps assumption. Expiry settlement has
no bid-ask cost, since it's a clearinghouse settlement, not a market
transaction.

Performance note: quote lookups are pre-indexed into plain dicts once
at startup (build_straddle_lookup), rather than filtering the full
straddle table on every leg/day/trade. With ~100+ legs per trade,
~20-30 days per trade, and 36 trades, the naive filter-per-lookup
approach meant hundreds of thousands of full-table scans over a
1.44M-row table; the dict-based lookups turn each of those into an O(1)
operation instead.
"""

import argparse

import numpy as np
import pandas as pd

import config
from black_scholes import bs_price, bs_vega
from correlation_snapshot import normalize_active_weights
from historical_correlation_series import (
    DEFAULT_STRADDLES_PATH,
    DEFAULT_WEIGHTS_PATH,
    _quarter_for_date,
    load_price_history,
    load_quarterly_weights,
)

DEFAULT_SIGNAL_PATH = "data/signal_history.csv"
DEFAULT_OUTPUT_PATH = "data/position_pnl_daily.csv"
DEFAULT_TRADES_OUTPUT_PATH = "data/position_pnl_trades.csv"


def load_raw_straddles(path=DEFAULT_STRADDLES_PATH):
    """
    Full straddle quote table -- strikes, bid, offer, IV for both legs --
    needed for actual position pricing. Unlike load_atm_iv_by_date() in
    historical_correlation_series.py, which collapses each leg to one
    averaged IV number for the correlation calculation, this keeps
    everything needed to actually enter and mark a real position.
    """
    df = pd.read_parquet(path)
    df["ticker"] = df["ticker"].astype("string")
    df["dte"] = (df["exdate"] - df["date"]).dt.days
    return df


def build_straddle_lookup(straddles):
    """
    One-time O(n) pass building the two lookup structures the whole
    simulation runs on. quote_by_key gives O(1) access to a specific
    contract's quote on a specific day (used by mark_leg and
    entry_exit_cashflow). by_ticker_date gives fast access to every
    exdate available for a ticker on a given day (used by
    select_entry_contract), instead of filtering the full table each time.
    """
    quote_by_key = {}
    for row in straddles.itertuples(index=False):
        quote_by_key[(row.ticker, row.exdate, row.date)] = row

    by_ticker_date = {
        key: group for key, group in straddles.groupby(["ticker", "date"], sort=False)
    }

    return quote_by_key, by_ticker_date


def get_trade_windows(signal_path=DEFAULT_SIGNAL_PATH):
    """
    Identify each entry date and the date the z-score signal would
    naturally exit, from signal.py's binary position column. The
    signal-exit date is an upper bound on how long the REAL position is
    held -- the actual close may come earlier if the entered contracts
    expire first (this project does not roll).
    """
    df = pd.read_csv(signal_path, parse_dates=["date"]).set_index("date").sort_index()
    position = df["position"]
    entries = df.index[(position == 1) & (position.shift(1).fillna(0) == 0)]

    trades = []
    for entry_date in entries:
        after_entry = position.loc[entry_date:]
        exit_candidates = after_entry.index[after_entry == 0]
        signal_exit_date = exit_candidates[0] if len(exit_candidates) else df.index[-1]
        trades.append((entry_date, signal_exit_date))
    return trades


def select_entry_contract(by_ticker_date, ticker, entry_date, target_dte):
    """
    Pick the exdate closest to target_dte for this ticker on entry_date --
    the same rule load_atm_iv_by_date() applies -- so the contract
    actually traded matches what the correlation/signal calc assumed.
    """
    day_rows = by_ticker_date.get((ticker, entry_date))
    if day_rows is None:
        return None
    day_rows = day_rows[
        day_rows["dte"].between(config.MIN_DAYS_TO_EXPIRY, config.MAX_DAYS_TO_EXPIRY)
    ]
    if day_rows.empty:
        return None
    idx = (day_rows["dte"] - target_dte).abs().idxmin()
    return day_rows.loc[idx]


def leg_vega(row, S, r):
    T = row["dte"] / 365
    call_vega = bs_vega(S, row["call_strike"], T, r, row["call_iv"])
    put_vega = bs_vega(S, row["put_strike"], T, r, row["put_iv"])
    return call_vega + put_vega


def build_position(by_ticker_date, price_history, weights_by_quarter, entry_date, min_components=50):
    """
    Assembles one trade's legs at entry: select the index contract,
    select each component's contract, drop components with no usable
    entry quote and renormalize (same convention correlation_snapshot.py
    already uses), then vega-match component sizing against the index.

    Returns a dict with 'index' (one leg dict) and 'components' (list of
    leg dicts), or None if the day can't assemble a usable basket.
    """
    quarter = _quarter_for_date(entry_date)
    base_weights = weights_by_quarter.get(quarter)
    if not base_weights:
        return None

    target_dte = config.REALIZED_LOOKBACK_DAYS

    if entry_date not in price_history.index:
        return None
    S_index = price_history.loc[entry_date, config.INDEX_TICKER]
    index_row = select_entry_contract(by_ticker_date, config.INDEX_TICKER, entry_date, target_dte)
    if index_row is None or pd.isna(S_index):
        return None

    entry_rows = {}
    for ticker in base_weights:
        if ticker not in price_history.columns or pd.isna(price_history.loc[entry_date, ticker]):
            continue
        row = select_entry_contract(by_ticker_date, ticker, entry_date, target_dte)
        if row is not None:
            entry_rows[ticker] = row

    if len(entry_rows) < min_components:
        return None

    _, active_weights, _ = normalize_active_weights(entry_rows, base_weights)

    index_vega = leg_vega(index_row, S_index, config.RISK_FREE_RATE)
    if index_vega <= 0:
        return None

    index_leg = {
        "ticker": config.INDEX_TICKER,
        "exdate": index_row["exdate"],
        "call_strike": index_row["call_strike"],
        "put_strike": index_row["put_strike"],
        "units": 1.0,
        "side": "short",
        "last_call_iv": index_row["call_iv"],
        "last_put_iv": index_row["put_iv"],
    }

    component_legs = []
    for ticker, weight in active_weights.items():
        row = entry_rows[ticker]
        S_i = price_history.loc[entry_date, ticker]
        comp_vega = leg_vega(row, S_i, config.RISK_FREE_RATE)
        if comp_vega <= 0:
            continue
        target_dollar_vega = weight * index_vega
        units = target_dollar_vega / comp_vega
        component_legs.append({
            "ticker": ticker,
            "exdate": row["exdate"],
            "call_strike": row["call_strike"],
            "put_strike": row["put_strike"],
            "units": units,
            "side": "long",
            "last_call_iv": row["call_iv"],
            "last_put_iv": row["put_iv"],
        })

    if not component_legs:
        return None

    return {"index": index_leg, "components": component_legs}


def mark_leg(quote_by_key, leg, date, S, r):
    """
    Marks one leg on `date`. Returns (call_mark, put_mark, quote_source),
    where quote_source is one of:
      "real"     - priced from an actual observed bid/ask that day
      "settled"  - at/after the leg's own expiry, priced at pure
                   intrinsic value from spot, no model involved
      "fallback" - no real quote exists (the DTE "dark zone"), priced
                   via Black-Scholes using the last observed IV

    Mutates leg's last_call_iv/last_put_iv in place whenever a real
    quote is found, so the "last observed" IV carries forward correctly
    across days spent in the dark zone.
    """
    dte_remaining = (leg["exdate"] - date).days

    if dte_remaining <= 0:
        call_mark = max(S - leg["call_strike"], 0.0)
        put_mark = max(leg["put_strike"] - S, 0.0)
        return call_mark, put_mark, "settled"

    row = quote_by_key.get((leg["ticker"], leg["exdate"], date))
    if row is not None:
        leg["last_call_iv"] = row.call_iv
        leg["last_put_iv"] = row.put_iv
        call_mark = (row.call_bid + row.call_offer) / 2
        put_mark = (row.put_bid + row.put_offer) / 2
        return call_mark, put_mark, "real"

    T = dte_remaining / 365
    call_mark = bs_price(S, leg["call_strike"], T, r, leg["last_call_iv"], "call")
    put_mark = bs_price(S, leg["put_strike"], T, r, leg["last_put_iv"], "put")
    return call_mark, put_mark, "fallback"


def position_value(quote_by_key, position, date, price_history, r):
    """
    Net mark-to-market value of the whole position on `date`: long
    component value minus short index value. Returns (value, in_dark_zone)
    where in_dark_zone is True if ANY leg was priced via the BS fallback
    that day (not counted for a leg that's simply settled at expiry).
    """
    in_dark_zone = False
    total = 0.0

    for leg in [position["index"]] + position["components"]:
        if leg["ticker"] not in price_history.columns or date not in price_history.index:
            return None, None
        S = price_history.loc[date, leg["ticker"]]
        if pd.isna(S):
            return None, None

        call_mark, put_mark, quote_source = mark_leg(quote_by_key, leg, date, S, r)
        if quote_source == "fallback":
            in_dark_zone = True

        leg_value = leg["units"] * (call_mark + put_mark)
        total += -leg_value if leg["side"] == "short" else leg_value

    return total, in_dark_zone


def entry_exit_cashflow(quote_by_key, position, date, is_entry):
    """
    Realistic entry/exit cash flow using actual bid-ask: short legs
    (index) sell at bid, long legs (components) buy at offer on entry;
    reversed on a signal-driven exit. Returns the net cash flow (positive
    = cash received), or None if a real quote isn't available to
    transact at (shouldn't happen for entry/signal-exit dates, since
    both fall inside the real-quote DTE window by construction).
    """
    total = 0.0
    for leg in [position["index"]] + position["components"]:
        row = quote_by_key.get((leg["ticker"], leg["exdate"], date))
        if row is None:
            return None

        if leg["side"] == "short":
            price = row.call_bid + row.put_bid if is_entry else row.call_offer + row.put_offer
            sign = 1 if is_entry else -1
        else:
            price = row.call_offer + row.put_offer if is_entry else row.call_bid + row.put_bid
            sign = -1 if is_entry else 1

        total += sign * leg["units"] * price
    return total


def simulate_trade(quote_by_key, by_ticker_date, price_history, weights_by_quarter,
                    entry_date, signal_exit_date, trade_id, min_components=50):
    position = build_position(by_ticker_date, price_history, weights_by_quarter, entry_date, min_components)
    if position is None:
        return None, None

    all_legs = [position["index"]] + position["components"]
    earliest_expiry = min(leg["exdate"] for leg in all_legs)
    close_date = min(signal_exit_date, earliest_expiry)
    close_reason = "contract_expiry" if earliest_expiry <= signal_exit_date else "signal_exit"

    trading_dates = price_history.loc[entry_date:close_date].index

    entry_cash = entry_exit_cashflow(quote_by_key, position, entry_date, is_entry=True)
    if entry_cash is None:
        return None, None

    daily_rows = []
    prev_value = None
    for date in trading_dates:
        value, in_dark_zone = position_value(quote_by_key, position, date, price_history, config.RISK_FREE_RATE)
        if value is None:
            continue
        daily_pnl = 0.0 if prev_value is None else value - prev_value
        daily_rows.append({
            "date": date,
            "trade_id": trade_id,
            "position_value": value,
            "daily_pnl": daily_pnl,
            "in_dark_zone": in_dark_zone,
        })
        prev_value = value

    if close_reason == "signal_exit":
        exit_cash = entry_exit_cashflow(quote_by_key, position, close_date, is_entry=False)
        exit_cash = exit_cash if exit_cash is not None else 0.0
    else:
        # settlement at expiry: intrinsic value, no bid-ask, already
        # captured as the final day's position_value in daily_rows
        exit_cash = 0.0

    trade_summary = {
        "trade_id": trade_id,
        "entry_date": entry_date,
        "close_date": close_date,
        "close_reason": close_reason,
        "num_components": len(position["components"]),
        "entry_cash": entry_cash,
        "exit_cash": exit_cash,
    }
    # total realized P&L = entry cashflow + sum of daily mark-to-market
    # moves (already includes the jump to intrinsic value at expiry, if
    # that's how the trade closed) + exit cashflow if closed via signal
    trade_summary["total_pnl"] = entry_cash + sum(r["daily_pnl"] for r in daily_rows) + exit_cash

    return daily_rows, trade_summary


def run_position_level_backtest(
    straddles_path=DEFAULT_STRADDLES_PATH,
    signal_path=DEFAULT_SIGNAL_PATH,
    price_path=None,
    min_components=50,
):
    straddles = load_raw_straddles(straddles_path)
    price_history = load_price_history(price_path)
    trade_windows = get_trade_windows(signal_path)

    print("Building fast lookup indexes (one-time cost)...")
    quote_by_key, by_ticker_date = build_straddle_lookup(straddles)

    weights = load_quarterly_weights(DEFAULT_WEIGHTS_PATH)
    weights_by_quarter = {
        q: dict(zip(g["ticker"], g["weight"])) for q, g in weights.groupby("quarter")
    }

    all_daily_rows = []
    all_trade_summaries = []

    for trade_id, (entry_date, signal_exit_date) in enumerate(trade_windows, start=1):
        daily_rows, trade_summary = simulate_trade(
            quote_by_key, by_ticker_date, price_history, weights_by_quarter,
            entry_date, signal_exit_date, trade_id, min_components,
        )
        if daily_rows is None:
            print(f"Trade {trade_id} ({entry_date.date()}): skipped, couldn't assemble a usable position")
            continue
        all_daily_rows.extend(daily_rows)
        all_trade_summaries.append(trade_summary)
        print(
            f"Trade {trade_id} ({entry_date.date()} -> {trade_summary['close_date'].date()}, "
            f"{trade_summary['close_reason']}): P&L = {trade_summary['total_pnl']:,.2f}"
        )

    daily_df = pd.DataFrame(all_daily_rows)
    trades_df = pd.DataFrame(all_trade_summaries)
    return daily_df, trades_df


def summarize(daily_df, trades_df):
    total_pnl = trades_df["total_pnl"].sum()
    win_rate = (trades_df["total_pnl"] > 0).mean()
    daily_pnl = daily_df.groupby("date")["daily_pnl"].sum()
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else np.nan

    cumulative = daily_pnl.cumsum()
    drawdown = cumulative - cumulative.cummax()
    max_drawdown = drawdown.min()
    calmar = (daily_pnl.mean() * 252) / abs(max_drawdown) if max_drawdown < 0 else np.nan

    expired_pct = (trades_df["close_reason"] == "contract_expiry").mean()
    dark_zone_pct = daily_df["in_dark_zone"].mean()

    print(f"\nTotal position-level P&L: {total_pnl:,.2f}")
    print(f"Number of trades: {len(trades_df)}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max drawdown: {max_drawdown:,.2f}")
    print(f"Calmar: {calmar:.2f}")
    print(f"Trades closed at contract expiry (not signal exit): {expired_pct:.1%}")
    print(f"Trading days marked via BS dark-zone fallback: {dark_zone_pct:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Real position-level P&L for the dispersion trade.")
    parser.add_argument("--straddles-path", default=DEFAULT_STRADDLES_PATH)
    parser.add_argument("--signal-path", default=DEFAULT_SIGNAL_PATH)
    parser.add_argument("--price-path", required=True)
    parser.add_argument("--min-components", type=int, default=50)
    parser.add_argument("--daily-output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--trades-output-path", default=DEFAULT_TRADES_OUTPUT_PATH)
    args = parser.parse_args()

    daily_df, trades_df = run_position_level_backtest(
        straddles_path=args.straddles_path,
        signal_path=args.signal_path,
        price_path=args.price_path,
        min_components=args.min_components,
    )

    daily_df.to_csv(args.daily_output_path, index=False)
    trades_df.to_csv(args.trades_output_path, index=False)
    print(f"\nSaved {len(daily_df):,} daily rows to {args.daily_output_path}")
    print(f"Saved {len(trades_df):,} trade summaries to {args.trades_output_path}")

    summarize(daily_df, trades_df)


if __name__ == "__main__":
    main()