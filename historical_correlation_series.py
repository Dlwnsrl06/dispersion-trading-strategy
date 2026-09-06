"""
Builds the historical implied/realized correlation series consumed by
signal.py and backtest.py.

By default this uses OptionMetrics' call_iv/put_iv columns from
data/atm_straddles.csv. Historical realized correlation requires a
close-price file supplied via --price-path. The price file may be either:

- wide: one row per date, with a date column and ticker columns
- long: columns date, ticker, close

An optional --download-prices flag can use yfinance as a temporary close
source, but a WRDS/CRSP close pull is the cleaner backtest input.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

import config
from correlation_snapshot import compute_correlation_snapshot


DEFAULT_STRADDLES_PATH = "data/atm_straddles.parquet"
DEFAULT_WEIGHTS_PATH = "data/quarterly_weights.csv"
DEFAULT_OUTPUT_PATH = "data/correlation_history.csv"
DEFAULT_PRICE_CACHE_PATH = "data/historical_closes_yfinance.csv"
TICKER_ALIASES = {
    "META": "FB",
    "ELV": "ANTM",
    "BKNG": "PCLN",
    "FISV": "FI",     # only one that flips direction
    "RTX": "UTX",      # UTX is the continuing entity post-merger; RTN
                        # correctly delists on 2020-04-02, no alias needed
    "DD": "DWDP",      # only affects post-2019 DD rows; pre-2017 DD rows
                        # have no options coverage under any label and
                        # will (correctly) still drop
}

def _resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(__file__), path)


def load_atm_iv_by_date(
    path=DEFAULT_STRADDLES_PATH,
    min_dte=None,
    max_dte=None,
    target_dte=None,
):
    """
    Load ATM straddles and select one tenor per ticker/date.

    If multiple expiries are available inside the target DTE window, keep
    the row closest to target_dte. The ticker-date IV is the simple
    average of call and put IV.
    """
    min_dte = config.MIN_DAYS_TO_EXPIRY if min_dte is None else min_dte
    max_dte = config.MAX_DAYS_TO_EXPIRY if max_dte is None else max_dte
    target_dte = config.REALIZED_LOOKBACK_DAYS if target_dte is None else target_dte

    df = pd.read_parquet(_resolve_path(path))
    df["ticker"] = df["ticker"].astype("string")

    df["dte"] = (df["exdate"] - df["date"]).dt.days
    df = df[df["dte"].between(min_dte, max_dte)].copy()
    df = df.dropna(subset=["call_iv", "put_iv"])
    df = df[(df["call_iv"] > 0) & (df["put_iv"] > 0)]

    df["iv"] = df[["call_iv", "put_iv"]].mean(axis=1)
    df["target_dte_dist"] = (df["dte"] - target_dte).abs()

    idx = df.groupby(["date", "ticker"], observed=True)["target_dte_dist"].idxmin()
    selected = df.loc[idx, ["date", "ticker", "exdate", "dte", "iv"]]
    return selected.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_quarterly_weights(path=DEFAULT_WEIGHTS_PATH):
    weights = pd.read_csv(
        _resolve_path(path),
        dtype={"quarter": "string", "ticker": "string"},
    )
    weights = weights.dropna(subset=["quarter", "ticker", "weight"])
    weights = weights[weights["weight"] > 0]

    weights["ticker"] = weights["ticker"].replace(TICKER_ALIASES)
    weights = (
        weights.groupby(["quarter", "ticker"], as_index=False)["weight"].sum()
    )

    return weights


def load_price_history(path):
    """
    Returns a wide close-price DataFrame indexed by date.
    """
    df = pd.read_csv(_resolve_path(path))
    lower_cols = {col.lower(): col for col in df.columns}

    if {"date", "ticker", "close"}.issubset(lower_cols):
        date_col = lower_cols["date"]
        ticker_col = lower_cols["ticker"]
        close_col = lower_cols["close"]

        df[date_col] = pd.to_datetime(df[date_col])
        wide = df.pivot_table(
            index=date_col,
            columns=ticker_col,
            values=close_col,
            aggfunc="last",
        )
    else:
        date_col = lower_cols.get("date") or df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        wide = df.set_index(date_col)

    wide = wide.sort_index()
    wide.columns = [str(col).strip() for col in wide.columns]
    return wide.apply(pd.to_numeric, errors="coerce")


def download_price_history(tickers, start, end, output_path=DEFAULT_PRICE_CACHE_PATH):
    """
    Temporary convenience path for development. Prefer WRDS/CRSP closes
    for the final historical backtest.
    """
    closes = {}
    failures = []

    for i, ticker in enumerate(tickers, start=1):
        yahoo_ticker = ticker.replace(".", "-")
        try:
            series = _download_yahoo_close_series(yahoo_ticker, start, end)
        except (urllib.error.URLError, ValueError, TimeoutError) as exc:
            failures.append((ticker, str(exc)))
            continue

        if not series.empty:
            series.name = ticker
            closes[ticker] = series

        if i % 25 == 0:
            print(
                f"Downloaded closes for {len(closes):,}/{i:,} attempted tickers..."
                ,
                flush=True,
            )

        time.sleep(0.1)

    if not closes:
        raise RuntimeError("No historical close prices downloaded.")

    closes = pd.DataFrame(closes).sort_index()
    output_path = _resolve_path(output_path)
    closes.to_csv(output_path, index_label="date")

    if failures:
        failure_path = os.path.splitext(output_path)[0] + "_failures.csv"
        pd.DataFrame(failures, columns=["ticker", "error"]).to_csv(
            failure_path, index=False
        )
        print(
            f"Saved {len(failures):,} Yahoo download failures to {failure_path}"
            ,
            flush=True,
        )

    print(f"Saved historical closes to {output_path}", flush=True)
    return closes


def _download_yahoo_close_series(ticker, start, end):
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    query = urllib.parse.urlencode(
        {
            "period1": start_ts,
            "period2": end_ts,
            "interval": "1d",
            "includePrePost": "false",
            "events": "history",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{query}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")

    payload = json.loads(body)
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(chart["error"])

    results = chart.get("result") or []
    if not results:
        raise ValueError("Yahoo chart response had no price result.")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjclose = (result.get("indicators", {}).get("adjclose") or [{}])[0]
    closes = adjclose.get("adjclose") or quote.get("close") or []

    if not timestamps or not closes:
        raise ValueError("Yahoo chart response missing close prices.")

    index = pd.to_datetime(timestamps, unit="s").normalize()
    series = pd.Series(closes, index=index, name=ticker)
    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()
    return series


def get_download_universe(straddles_path=DEFAULT_STRADDLES_PATH):
    df = pd.read_parquet(
        _resolve_path(straddles_path),
        columns=["date", "ticker"],
    )
    df["ticker"] = df["ticker"].astype("string")
    tickers = sorted(set(df["ticker"].dropna()) | {config.INDEX_TICKER})
    start = df["date"].min() - pd.Timedelta(days=config.REALIZED_LOOKBACK_DAYS * 3)
    end = df["date"].max() + pd.Timedelta(days=1)
    return tickers, start, end


def build_yfinance_price_cache(
    straddles_path=DEFAULT_STRADDLES_PATH,
    output_path=DEFAULT_PRICE_CACHE_PATH,
):
    tickers, start, end = get_download_universe(straddles_path)
    print(
        f"Downloading {len(tickers):,} tickers from {start.date()} to {end.date()}...",
        flush=True,
    )
    return download_price_history(tickers, start=start, end=end, output_path=output_path)


def _quarter_for_date(date):
    """
    Returns the PRIOR calendar quarter's weight vector for `date`.
    quarterly_weights.csv's DlyCap is measured on the last trading day of
    the quarter it's labeled with, so using that label's own quarter to
    weight trading days would use end-of-quarter market cap to trade days
    before that cap was observable. Shifting forward by one period makes
    each weight vector apply only to dates after its snapshot date.
    """
    period = pd.Timestamp(date).to_period("Q") - 1
    return str(period)


def _valid_price_window(price_history, date, tickers, lookback_days):
    available = price_history.loc[:date]
    if len(available) < lookback_days + 1:
        return None, []

    window = available.tail(lookback_days + 1)
    tickers = [ticker for ticker in tickers if ticker in window.columns]
    if not tickers:
        return None, []

    window = window[tickers].ffill().bfill()
    usable = [
        ticker
        for ticker in tickers
        if window[ticker].notna().all() and (window[ticker] > 0).all()
    ]

    if not usable:
        return None, []

    return window[usable], usable


def build_correlation_history(
    straddles_path=DEFAULT_STRADDLES_PATH,
    weights_path=DEFAULT_WEIGHTS_PATH,
    price_path=None,
    output_path=DEFAULT_OUTPUT_PATH,
    min_components=50,
    download_prices=False,
):
    ivs = load_atm_iv_by_date(straddles_path)
    weights = load_quarterly_weights(weights_path)

    dates = ivs["date"].drop_duplicates().sort_values()
    tickers = sorted(set(ivs["ticker"].dropna()) | {config.INDEX_TICKER})

    if price_path:
        price_history = load_price_history(price_path)
    elif download_prices:
        start = dates.min() - pd.Timedelta(days=config.REALIZED_LOOKBACK_DAYS * 3)
        end = dates.max() + pd.Timedelta(days=1)
        price_history = download_price_history(tickers, start=start, end=end)
    else:
        raise ValueError(
            "Historical close prices are required. Pass --price-path, or use "
            "--download-prices for a temporary yfinance approximation."
        )

    weights_by_quarter = {
        quarter: dict(zip(group["ticker"], group["weight"]))
        for quarter, group in weights.groupby("quarter")
    }

    rows = []
    grouped_ivs = {date: group for date, group in ivs.groupby("date")}

    for date in dates:
        day = grouped_ivs[date]
        iv_by_ticker = dict(zip(day["ticker"], day["iv"]))
        index_iv = iv_by_ticker.pop(config.INDEX_TICKER, None)
        if index_iv is None or pd.isna(index_iv):
            continue

        quarter = _quarter_for_date(date)
        base_weights = weights_by_quarter.get(quarter)
        if not base_weights:
            continue

        component_ivs = {
            ticker: iv
            for ticker, iv in iv_by_ticker.items()
            if ticker in base_weights and pd.notna(iv) and iv > 0
        }
        if len(component_ivs) < min_components:
            continue

        price_window, price_tickers = _valid_price_window(
            price_history,
            date,
            component_ivs.keys(),
            config.REALIZED_LOOKBACK_DAYS,
        )
        if price_window is None:
            continue

        component_ivs = {
            ticker: component_ivs[ticker]
            for ticker in price_tickers
        }
        if len(component_ivs) < min_components:
            continue

        try:
            snapshot = compute_correlation_snapshot(
                index_iv=index_iv,
                component_ivs=component_ivs,
                base_weights=base_weights,
                price_history=price_window,
            )
        except ValueError:
            continue

        implied = snapshot["implied"]["implied_correlation"]
        realized = snapshot["realized"]["realized_correlation"]

        rows.append(
            {
                "date": date,
                "quarter": quarter,
                "implied_correlation": implied,
                "realized_correlation": realized,
                "spread": implied - realized,
                "num_components": snapshot["num_components"],
                "raw_weight_coverage": snapshot["raw_weight_coverage"],
                "index_iv": index_iv,
                "out_of_bounds": snapshot["implied"]["out_of_bounds"],
            }
        )

    history = pd.DataFrame(rows)
    if not history.empty:
        history = history.sort_values("date")
    output_path = _resolve_path(output_path)
    history.to_csv(output_path, index=False)
    return history


def main():
    
    parser = argparse.ArgumentParser(
        description="Build historical implied/realized correlation series."
    )
    parser.add_argument("--straddles-path", default=DEFAULT_STRADDLES_PATH)
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--price-path")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--price-cache-path", default=DEFAULT_PRICE_CACHE_PATH)
    parser.add_argument("--min-components", type=int, default=50)
    parser.add_argument(
        "--prices-only",
        action="store_true",
        help="Only download and save the yfinance historical close cache.",
    )
    parser.add_argument(
        "--download-prices",
        action="store_true",
        help="Use yfinance closes as a temporary approximation.",
    )
    args = parser.parse_args()
    if args.prices_only:
        build_yfinance_price_cache(
            straddles_path=args.straddles_path,
            output_path=args.price_cache_path,
        )
        return

    if not args.price_path and not args.download_prices:
        parser.error(
            "historical close prices are required: pass --price-path, "
            "or use --download-prices for a temporary yfinance approximation"
        )

    history = build_correlation_history(
        straddles_path=args.straddles_path,
        weights_path=args.weights_path,
        price_path=args.price_path,
        output_path=args.output_path,
        min_components=args.min_components,
        download_prices=args.download_prices,
    )

    print(f"Built {len(history):,} daily correlation rows.")
    if not history.empty:
        print(
            history[
                [
                    "date",
                    "implied_correlation",
                    "realized_correlation",
                    "spread",
                    "num_components",
                ]
            ].tail().to_string(index=False)
        )


if __name__ == "__main__":
    main()
