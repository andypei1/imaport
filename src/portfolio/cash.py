"""
Build daily cash ledger from trade cash impacts and cash-flow transactions.

Cash is moved only by:
- Buy/Sale of actual securities (trades_df): Buy decreases cash, Sale increases cash.
- Dividend, Interest: increase cash.
- Cash Deposit, Transfer In (of cash-equiv): external inflow.
- Cash Withdrawal, Transfer Out (of cash-equiv): external outflow.

Buy/Sale of FEDXX/SWEEP0206/CASH have zero cash impact (they are the cash pool; we already count security buys/sales).
Transfer In/Out of securities (non-cash-equiv) have zero cash impact (shares moving, not cash).
"""

import pandas as pd

CASH_EQUIVALENT_SYMBOLS = {"FEDXX", "SWEEP0206", "CASH"}


def _parse_cash_impact(row: pd.Series) -> tuple[float, float]:
    """
    Return (cash_delta, external_flow) for a cashflows row.
    external_flow is non-zero only for deposits/withdrawals/transfers (external flows).
    """
    amount = float(row.get("amount", 0) or 0)
    txn = str(row.get("txn_type", "")).strip()
    # Dividend, Interest: increase cash; not external.
    if txn in ("Dividend", "Interest"):
        return (amount, 0.0)
    # Cash Deposit: inflow, external. Transfer In: only if cash-equivalent (e.g. FEDXX); security transfers are shares, not cash.
    if txn == "Cash Deposit":
        return (amount, amount)
    if txn == "Transfer In":
        symbol = str(row.get("symbol", "")).strip().upper()
        if symbol not in CASH_EQUIVALENT_SYMBOLS:
            return (0.0, 0.0)
        return (amount, amount)
    # Cash Withdrawal: outflow, external. Transfer Out: only if cash-equivalent.
    if txn == "Cash Withdrawal":
        return (amount, amount)  # amount is typically negative for outflow
    if txn == "Transfer Out":
        symbol = str(row.get("symbol", "")).strip().upper()
        if symbol not in CASH_EQUIVALENT_SYMBOLS:
            return (0.0, 0.0)
        return (amount, amount)  # amount is typically negative for outflow
    # Buy/Sale of cash-equivalent (FEDXX, SWEEP0206, CASH): net zero effect on cash.
    # Selling FEDXX converts shares to cash that is used for security buys; buying FEDXX is cash moving into the fund.
    # We already count security Buy/Sale in the trade loop, so we do not also add/subtract for cash-equiv Buy/Sale.
    if txn in ("Buy", "Sale"):
        return (0.0, 0.0)
    return (0.0, 0.0)


def build_cash_ledger(
    trades_df: pd.DataFrame,
    cashflows_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cumulative daily cash balance and external flow.

    Trade impacts: Buy decreases cash by amount, Sale increases cash by |amount|.
    Cash-flow rows: apply dividend/interest/deposit/withdrawal/transfer rules.

    Returns:
        DataFrame with columns: date, cash_balance, external_flow.
    """
    rows = []

    # Trade cash impacts: only Buy and Sale move cash. Transfer In/Out are in cashflows_df; do not double-count.
    for _, r in trades_df.iterrows():
        if r["txn_type"] not in ("Buy", "Sale"):
            continue
        amt = float(r["amount"])
        if r["txn_type"] == "Buy":
            delta = -abs(amt)
        else:
            delta = abs(amt)
        rows.append({"date": r["trade_date"], "cash_delta": delta, "external_flow": 0.0})

    # Cash-flow impacts
    for _, r in cashflows_df.iterrows():
        cash_delta, ext = _parse_cash_impact(r)
        rows.append({"date": r["trade_date"], "cash_delta": cash_delta, "external_flow": ext})

    if not rows:
        return pd.DataFrame(columns=["date", "cash_balance", "external_flow"])

    df = pd.DataFrame(rows)
    df = df.groupby("date", as_index=False).agg({"cash_delta": "sum", "external_flow": "sum"})
    df = df.sort_values("date")
    df["cash_balance"] = df["cash_delta"].cumsum()
    df = df.rename(columns={"date": "date"})[["date", "cash_balance", "external_flow"]]

    # Expand to every calendar day: carry forward cash balance and zero external_flow on non-event days.
    date_min = df["date"].min()
    date_max = df["date"].max()
    all_dates = pd.date_range(date_min, date_max, freq="D")
    full = pd.DataFrame({"date": all_dates})
    full = full.merge(df[["date", "cash_balance", "external_flow"]], on="date", how="left")
    full["external_flow"] = full["external_flow"].fillna(0)
    full["cash_balance"] = full["cash_balance"].ffill()
    full["cash_balance"] = full["cash_balance"].fillna(0)
    return full.sort_values("date").reset_index(drop=True)
