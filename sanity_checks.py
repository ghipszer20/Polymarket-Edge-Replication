import pandas as pd

df = pd.read_csv(r"C:\Users\24GHi\PycharmProjects\PythonProject2\live_bot_trades.csv")
print(f"Total trades: {len(df)}")
print(f"\npnl_usdc stats:")
print(df["pnl_usdc"].describe())
print(f"\nSample trades:")
print(df[["signal","entry_price","exit_price","exit_reason","pnl_usdc"]].head(20).to_string())
print(f"\nWin rate (pnl > 0): {(df['pnl_usdc'] > 0).mean()*100:.1f}%")
print(f"\nExit reasons:\n{df['exit_reason'].value_counts()}")