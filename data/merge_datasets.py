import pandas as pd

SECID_TO_TICKER = {
    109820: "SPY", 154402: "FB", 112254: "ACE", 111907: "AGN",
    113119: "ANTM", 102362: "BRCM", 207402: "BXLT", 102822: "CELG",
    211899: "DWDP", 104049: "EMC", 104870: "FI", 109182: "PCLN",
    109148: "PCP", 109621: "RAI", 109497: "RTN", 122337: "TFCFA",
    129054: "TWC", 111459: "UTX", 112180: "YHOO",
}


def merge_options_files(main_filepath: str, batch2_filepath: str) -> pd.DataFrame:
    main_df = pd.read_csv(
        main_filepath,
        dtype={"optionid": "int64", "secid": "int64"},
        parse_dates=["date", "exdate"],
        date_format="%Y-%m-%d",
    )
    main_df["strike_price"] = main_df["strike_price"] / 1000.0
    main_df["dte"] = (main_df["exdate"] - main_df["date"]).dt.days

    batch2_df = pd.read_csv(
        batch2_filepath,
        dtype={"optionid": "int64", "secid": "int64"},
        parse_dates=["date", "exdate"],
        date_format="%Y-%m-%d",
    )
    batch2_df["strike_price"] = batch2_df["strike_price"] / 1000.0
    batch2_df["dte"] = (batch2_df["exdate"] - batch2_df["date"]).dt.days

    combined = pd.concat([main_df, batch2_df], ignore_index=True)

    # Apply the ticker remap AFTER concatenation, to every row matching
    # one of these secids, regardless of which file it came from
    for secid, ticker in SECID_TO_TICKER.items():
        combined.loc[combined["secid"] == secid, "ticker"] = ticker

    dedup_keys = ["secid", "date", "exdate", "cp_flag", "strike_price"]
    dupes = combined.duplicated(subset=dedup_keys).sum()
    if dupes > 0:
        print(f"Found {dupes} true duplicate rows, dropping.")
        combined = combined.drop_duplicates(subset=dedup_keys)

    return combined


if __name__ == "__main__":
    combined = merge_options_files(
        "data/options_historical_data.csv",
        "data/options_historical_data2.csv",
    )

    print(f"Combined total rows: {len(combined):,}")
    print(f"Combined unique tickers: {combined['ticker'].nunique()}")
    print(f"Combined unique secids: {combined['secid'].nunique()}")

    combined.to_csv("data/options_historical_data_full.csv", index=False)
    expected_tickers = set(SECID_TO_TICKER.values())
    actual_tickers = set(combined["ticker"].unique())

    missing = expected_tickers - actual_tickers
    print(f"Batch2 tickers missing from combined ticker set: {missing}")

    for t in ["FB", "ACE", "META", "CB"]:
        n = (combined["ticker"] == t).sum()
        print(f"{t}: {n:,} rows")

    print(f"\nsecid dtype: {combined['secid'].dtype}")
    print(f"Sample secid values: {combined['secid'].head(3).tolist()}")
    print("Saved to data/options_historical_data_full.csv")