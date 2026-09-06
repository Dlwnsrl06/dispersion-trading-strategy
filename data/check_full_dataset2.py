import pandas as pd

full_df = pd.read_csv(
    "data/options_historical_data_full.csv",
    dtype={"optionid": "int64", "secid": "int64"},
    parse_dates=["date", "exdate"],
)

# 1. Confirm the delta filter actually held across the full merged file
print("Delta range check:")
print(f"  Calls: min={full_df[full_df['cp_flag']=='C']['delta'].min():.3f}, "
      f"max={full_df[full_df['cp_flag']=='C']['delta'].max():.3f}")
print(f"  Puts:  min={full_df[full_df['cp_flag']=='P']['delta'].min():.3f}, "
      f"max={full_df[full_df['cp_flag']=='P']['delta'].max():.3f}")

# 2. Confirm DTE filter held
print(f"\nDTE range: min={full_df['dte'].min()}, max={full_df['dte'].max()}")

# 3. Check for negative or zero bid/ask, which would break straddle pricing
bad_quotes = full_df[(full_df['best_bid'] < 0) | (full_df['best_offer'] <= 0)]
print(f"\nRows with negative bid or non-positive offer: {len(bad_quotes)}")

# 4. Check for crossed markets (bid > ask), a data quality red flag
crossed = full_df[full_df['best_bid'] > full_df['best_offer']]
print(f"Crossed markets (bid > offer): {len(crossed)}")

# 5. Check every ticker has at least some minimum coverage,
# flag any suspiciously thin ones beyond the known short-lived names
coverage = full_df.groupby("ticker")["date"].nunique().sort_values()
print(f"\nTickers with fewest distinct trading days (bottom 10):")
print(coverage.head(10))

# 6. Confirm no ticker has calls but zero puts or vice versa
# (would break straddle formation entirely for that name)
cp_counts = full_df.groupby(["ticker", "cp_flag"]).size().unstack(fill_value=0)
missing_leg = cp_counts[(cp_counts.get("C", 0) == 0) | (cp_counts.get("P", 0) == 0)]
print(f"\nTickers missing one entire option side (call or put): {len(missing_leg)}")
if len(missing_leg) > 0:
    print(missing_leg)