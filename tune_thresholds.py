import pandas as pd
import numpy as np
from signal import build_signal_table
from backtest import run_backtest, summarize_backtest

correlation_history = pd.read_csv("data/correlation_history.csv")
correlation_history["date"] = pd.to_datetime(correlation_history["date"])
correlation_history = correlation_history.sort_values("date").reset_index(drop=True)

split_idx = int(len(correlation_history) * 0.8)
train = correlation_history.iloc[:split_idx]
test = correlation_history.iloc[split_idx:]
print(f"Train: {train['date'].min().date()} to {train['date'].max().date()} ({len(train)} rows)")
print(f"Test:  {test['date'].min().date()} to {test['date'].max().date()} ({len(test)} rows)")
print()

def evaluate(df, entry_z, exit_z):
    signal_df = build_signal_table(df, entry_z=entry_z, exit_z=exit_z)
    signal_df = signal_df.dropna(subset=["zscore"])
    if signal_df["position"].sum() == 0:
        return None
    results = run_backtest(signal_df)

    active = results[results["position"] == 1]
    active_sharpe = np.nan
    if len(active) > 1 and active["pnl"].std() > 0:
        active_sharpe = active["pnl"].mean() / active["pnl"].std() * np.sqrt(252)

    total_pnl = results["pnl"].sum()
    daily_pnl = results["pnl"].fillna(0)
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else np.nan
    num_trades = int(results["position_change"].sum() / 2)

    return {
        "entry_z": entry_z, "exit_z": exit_z,
        "sharpe_all_days": sharpe, "sharpe_active_days": active_sharpe,
        "total_pnl": total_pnl, "num_trades": num_trades,
    }

entry_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
exit_grid = [-0.5, -0.25, 0.0, 0.25, 0.5]

rows = []
for entry_z in entry_grid:
    for exit_z in exit_grid:
        if exit_z >= entry_z:
            continue
        result = evaluate(train, entry_z, exit_z)
        if result:
            rows.append(result)

grid_results = pd.DataFrame(rows).sort_values("sharpe_all_days", ascending=False)
print("=== Top 10 combos on TRAIN ===")
print(grid_results.head(10).to_string(index=False))

best = grid_results.iloc[0]
print(f"\n=== Best combo (entry={best['entry_z']}, exit={best['exit_z']}) evaluated on TEST (holdout) ===")
test_result = evaluate(test, best["entry_z"], best["exit_z"])
print(test_result)

print(f"\n=== Placeholder (entry=1.0, exit=0.0) on TEST for comparison ===")
print(evaluate(test, 1.0, 0.0))