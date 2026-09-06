import pandas as pd

SECID_TO_TICKER = {
    109820: "SPY", 154402: "FB", 112254: "ACE", 111907: "AGN",
    113119: "ANTM", 102362: "BRCM", 207402: "BXLT", 102822: "CELG",
    211899: "DWDP", 104049: "EMC", 104870: "FI", 109182: "PCLN",
    109148: "PCP", 109621: "RAI", 109497: "RTN", 122337: "TFCFA",
    129054: "TWC", 111459: "UTX", 112180: "YHOO",
}

# Load the already-merged output, not the raw main file
main_df_check = pd.read_csv(
    "data/options_historical_data.csv",
    dtype={"secid": "int64"},
    usecols=["secid", "ticker"],
)

for secid, expected_ticker in SECID_TO_TICKER.items():
    original_labels = main_df_check[main_df_check["secid"] == secid]["ticker"].unique()
    if len(original_labels) > 0:
        print(f"secid {secid} ({expected_ticker}) already existed in main file under: {original_labels}")