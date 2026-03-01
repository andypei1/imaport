"""
Entry script: load transactions -> positions -> cash -> prices -> NAV -> returns -> save.
"""

import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.portfolio.load_transactions import load_transactions
from src.portfolio.positions import build_positions
from src.portfolio.cash import build_cash_ledger
from src.portfolio.prices import get_prices
from src.portfolio.nav import compute_daily_nav


def main() -> None:
    repo = Path(__file__).resolve().parent
    transactions_path = repo / "Transactions.csv"
    if not transactions_path.exists():
        print(f"Transactions file not found: {transactions_path}")
        sys.exit(1)

    # 1. Load and classify transactions (re-run after editing Transactions.csv to pick up fixes)
    trades_df, cashflows_df = load_transactions(str(transactions_path))
    if trades_df.empty and cashflows_df.empty:
        print("No transactions loaded.")
        sys.exit(0)

    # 2. Build security positions
    positions_df = build_positions(trades_df)

    out_dir = repo / "outputs"
    out_dir.mkdir(exist_ok=True)

    # Daily holdings (date, symbol, shares) — only non-zero so we don't track positions after they're sold
    holdings_path = out_dir / "holdings.csv"
    holdings_out = positions_df[positions_df["shares"] != 0]
    holdings_out.to_csv(holdings_path, index=False)
    print(f"Wrote {holdings_path} ({len(holdings_out)} rows)")

    # Negative shares: write for review so user can fix trade log
    neg = positions_df[positions_df["shares"] < 0]
    if not neg.empty:
        neg_path = out_dir / "negative_shares.csv"
        neg.to_csv(neg_path, index=False)
        print(f"{len(neg)} rows with negative shares written to {neg_path}; review and fix trade log.")

    # 3. Build cash ledger
    cash_df = build_cash_ledger(trades_df, cashflows_df)

    # 4. Fetch prices for date range
    date_min = cash_df["date"].min()
    date_max = cash_df["date"].max()
    symbols = positions_df["symbol"].unique().tolist()
    inputs_dir = repo / "inputs"
    alias_path = inputs_dir / "symbol_aliases.csv" if inputs_dir.exists() else None
    manual_prices_path = inputs_dir / "manual_prices.csv" if inputs_dir.exists() else None
    prices_df = get_prices(
        symbols,
        date_min,
        date_max,
        alias_path=alias_path,
        manual_prices_path=manual_prices_path,
        trades_for_fallback=trades_df,
    )

    # 5 & 6. Compute daily NAV and returns
    nav_df = compute_daily_nav(positions_df, cash_df, prices_df)

    # 7. Save output
    out_path = out_dir / "nav.csv"
    nav_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(nav_df)} rows)")

    # One-line summary (latest NAV breakdown)
    last = nav_df.iloc[-1]
    print(f"Latest: securities_value={last['securities_value']:,.0f}, cash_balance={last['cash_balance']:,.0f}, nav={last['nav']:,.0f}")


if __name__ == "__main__":
    main()
