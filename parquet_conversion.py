# convert_to_parquet.py
import pandas as pd
import time
import os

def convert_and_compare(csv_path, parquet_path, parse_dates=None):
    print(f"\n{csv_path}")
    t0 = time.time()
    df = pd.read_csv(csv_path, parse_dates=parse_dates)
    csv_read_time = time.time() - t0
    csv_size = os.path.getsize(csv_path) / 1e6

    df.to_parquet(parquet_path, index=False)
    parquet_size = os.path.getsize(parquet_path) / 1e6

    t0 = time.time()
    _ = pd.read_parquet(parquet_path)
    parquet_read_time = time.time() - t0

    print(f"  CSV:     {csv_size:,.1f} MB, read in {csv_read_time:.2f}s")
    print(f"  Parquet: {parquet_size:,.1f} MB, read in {parquet_read_time:.2f}s")
    print(f"  Size reduction: {(1 - parquet_size/csv_size)*100:.1f}%")
    print(f"  Read speedup: {csv_read_time/parquet_read_time:.1f}x")

convert_and_compare(
    "data/options_historical_data_full.csv",
    "data/options_historical_data_full.parquet",
    parse_dates=["date", "exdate"],
)
convert_and_compare(
    "data/atm_straddles.csv",
    "data/atm_straddles.parquet",
    parse_dates=["date", "exdate"],
)