import pandas as pd
old = pd.read_csv("data/correlation_history.csv")
new = pd.read_csv("data/correlation_history_parquet_check.csv")
print(old.equals(new))