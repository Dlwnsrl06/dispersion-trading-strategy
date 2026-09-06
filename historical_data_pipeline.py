"""
Processes the historical WRDS/IvyDB options file into a per-day,
per-ticker ATM straddle series. Only rebuilds if the source file is
newer than the cached straddle output, or if the cache doesn't exist,
so downstream scripts (correlation.py, backtest.py) can just load
atm_straddles.csv directly without re-running this every time.
"""

import os
import pandas as pd


def load_options_data(filepath: str) -> pd.DataFrame:
    df = pd.read_parquet(filepath)
    df["optionid"] = df["optionid"].astype("int64")
    df["secid"] = df["secid"].astype("int64")
    return df


def filter_valid_quotes(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["best_bid"] <= df["best_offer"]].copy()


def select_atm_straddle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized version: for each (ticker, date, exdate, cp_flag),
    finds the row closest to the target delta (0.5 for calls, -0.5
    for puts) without an explicit Python loop over every group.
    """
    df = df.copy()

    # target delta depends on side: calls target +0.5, puts target -0.5
    df["delta_target"] = df["cp_flag"].map({"C": 0.5, "P": -0.5})
    df["delta_dist"] = (df["delta"] - df["delta_target"]).abs()

    # within each (ticker, date, exdate, cp_flag) group, keep only the
    # row with the smallest delta_dist
    idx = df.groupby(["ticker", "date", "exdate", "cp_flag"])["delta_dist"].idxmin()
    best_rows = df.loc[idx]

    # pivot so calls and puts for the same (ticker, date, exdate) sit
    # side by side in one row
    calls = best_rows[best_rows["cp_flag"] == "C"].set_index(["ticker", "date", "exdate"])
    puts = best_rows[best_rows["cp_flag"] == "P"].set_index(["ticker", "date", "exdate"])

    straddles = calls[["strike_price", "best_bid", "best_offer", "impl_volatility"]].join(
        puts[["strike_price", "best_bid", "best_offer", "impl_volatility"]],
        lsuffix="_call", rsuffix="_put",
        how="inner",  # only keep rows where BOTH a call and put exist
    ).reset_index()

    straddles = straddles.rename(columns={
        "strike_price_call": "call_strike", "best_bid_call": "call_bid",
        "best_offer_call": "call_offer", "impl_volatility_call": "call_iv",
        "strike_price_put": "put_strike", "best_bid_put": "put_bid",
        "best_offer_put": "put_offer", "impl_volatility_put": "put_iv",
    })

    return straddles

def get_atm_straddles(
    source_path: str = "data/options_historical_data_full.parquet",
    cache_path: str = "data/atm_straddles.parquet",
    force_rebuild: bool = False,
) -> pd.DataFrame:
    needs_rebuild = (
        force_rebuild
        or not os.path.exists(cache_path)
        or os.path.getmtime(source_path) > os.path.getmtime(cache_path)
    )

    if not needs_rebuild:
        print(f"Using cached straddle file: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Rebuilding ATM straddle file from raw options data...")
    df = load_options_data(source_path)
    df = filter_valid_quotes(df)
    straddles = select_atm_straddle(df)
    straddles.to_parquet(cache_path, index=False)
    print(f"Saved {len(straddles):,} straddle rows to {cache_path}")
    return straddles


if __name__ == "__main__":
    straddles = get_atm_straddles()
    print(f"Total straddle rows: {len(straddles):,}")
    print(f"Unique tickers: {straddles['ticker'].nunique()}")