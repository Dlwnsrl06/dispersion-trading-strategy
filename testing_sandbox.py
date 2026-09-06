import pandas as pd
from signal import build_signal_table
from backtest import run_backtest
import numpy as np

correlation_history = pd.read_csv("data/correlation_history.csv")
correlation_history["date"] = pd.to_datetime(correlation_history["date"])
correlation_history = correlation_history.sort_values("date").reset_index(drop=True)
split_idx = int(len(correlation_history) * 0.8)
train = correlation_history.iloc[:split_idx]
test = correlation_history.iloc[split_idx:]

def evaluate(df, entry_z, exit_z):
    signal_df = build_signal_table(df, entry_z=entry_z, exit_z=exit_z).dropna(subset=["zscore"])
    if signal_df["position"].sum() == 0:
        return np.nan, 0
    results = run_backtest(signal_df)
    daily_pnl = results["pnl"].fillna(0)
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else np.nan
    num_trades = int(results["position_change"].sum() / 2)
    return sharpe, num_trades

entry_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
exit_grid = [-0.5, -0.25, 0.0, 0.25, 0.5]

print("=== Sharpe heatmap (TRAIN) ===")
sharpe_table = pd.DataFrame(index=entry_grid, columns=exit_grid, dtype=float)
trade_table = pd.DataFrame(index=entry_grid, columns=exit_grid, dtype=float)
for e in entry_grid:
    for x in exit_grid:
        if x >= e:
            continue
        s, n = evaluate(train, e, x)
        sharpe_table.loc[e, x] = s
        trade_table.loc[e, x] = n
print(sharpe_table.round(2))
print()
print("=== Trade counts (TRAIN) ===")
print(trade_table)
print()

print("=== entry=1.00, exit=-0.50 on TEST (higher trade-count neighbor) ===")
s, n = evaluate(test, 1.00, -0.50)
print(f"Sharpe: {s:.2f}, trades: {n}")

s, n = evaluate(test, 1.25, -0.50)
print(f"entry=1.25, exit=-0.50 on TEST -> Sharpe: {s:.2f}, trades: {n}")