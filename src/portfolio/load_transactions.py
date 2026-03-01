"""
Load and classify BlackDiamond transaction exports.

TRADE: Buy/Sale of investable securities (changes share counts).
CASH_FLOW: Dividends, interest, deposits, withdrawals, transfers; also
Buy/Sale of cash-equivalent symbols (FEDXX, SWEEP0206, CASH) are treated
as cash movements only, not positions.
"""

import re
import pandas as pd


# Column name mapping: common BlackDiamond export names -> canonical snake_case.
COLUMN_ALIASES = {
    "date": "trade_date",
    "trade date": "trade_date",
    "type": "txn_type",
    "transaction type": "txn_type",
    "symbol": "symbol",
    "ticker": "symbol",
    "units": "units",
    "shares": "units",
    "quantity": "units",
    "price": "price",
    "amount": "amount",
    "value": "amount",
}

REQUIRED_COLUMNS = ["trade_date", "symbol", "txn_type", "units", "price", "amount"]

# Cash-equivalent symbols: treat as cash movements only, not investable positions.
CASH_EQUIVALENT_SYMBOLS = {"FEDXX", "SWEEP0206", "CASH"}

# Position-affecting: Buy/Sale/Transfer In/Out change share counts; corporate actions add shares only.
TRADE_TYPES = {"Buy", "Sale", "Transfer In", "Transfer Out"}
POSITION_ADDITION_TYPES = {"Corporate Action", "Stock Split", "Adjustment"}
CASH_FLOW_TYPES = {
    "Dividend",
    "Interest",
    "Cash Deposit",
    "Cash Withdrawal",
    "Transfer In",
    "Transfer Out",
}


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase snake_case and map to canonical names."""
    def to_snake(s: str) -> str:
        s = str(s).strip().lower()
        s = re.sub(r"[\s\-]+", "_", s)
        return s

    rename = {}
    for col in df.columns:
        snake = to_snake(col)
        canonical = COLUMN_ALIASES.get(snake, snake)
        if canonical != col:
            rename[col] = canonical
    return df.rename(columns=rename)


def _parse_numeric_series(ser: pd.Series) -> pd.Series:
    """Parse numeric values; strip commas and handle quoted numbers."""
    return pd.to_numeric(ser.astype(str).str.replace(",", "", regex=False), errors="coerce")


def load_transactions(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read CSV, parse dates and numbers, classify rows into trades vs cash flows.

    Returns:
        trades_df: rows that change security share counts (Buy, Sale, Transfer In, Transfer Out,
                   Corporate Action/Stock Split/Adjustment, excluding cash-equivalent symbols).
        cashflows_df: rows that affect cash (dividends, interest, deposits, withdrawals,
                     transfers, and Buy/Sale of cash-equivalent symbols).
    """
    df = pd.read_csv(path)
    df = _normalize_column_names(df)

    # Ensure required columns exist (support common aliases already mapped).
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Parse trade_date as datetime.
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["units"] = _parse_numeric_series(df["units"])
    df["price"] = _parse_numeric_series(df["price"])
    df["amount"] = _parse_numeric_series(df["amount"])

    # Drop rows with missing key data.
    df = df.dropna(subset=["trade_date", "symbol", "txn_type"]).copy()
    df["txn_type"] = df["txn_type"].astype(str).str.strip()

    # Classify: trades_df = position-affecting (Buy, Sale, Transfer In, Transfer Out, Corporate Action, etc.) for non–cash-equivalent.
    # cashflows_df = Dividend, Interest, Deposit, Withdrawal, Transfer In, Transfer Out, and Buy/Sale of cash-equivalent only.
    is_cash_equiv = df["symbol"].astype(str).str.upper().str.strip().isin(CASH_EQUIVALENT_SYMBOLS)
    is_trade_type = df["txn_type"].isin(TRADE_TYPES) | df["txn_type"].isin(POSITION_ADDITION_TYPES)
    is_trade = is_trade_type & ~is_cash_equiv
    is_cash_flow = df["txn_type"].isin(CASH_FLOW_TYPES) | (is_cash_equiv & df["txn_type"].isin({"Buy", "Sale"}))

    trades_df = df.loc[is_trade, REQUIRED_COLUMNS].sort_values("trade_date").reset_index(drop=True)
    cashflows_df = df.loc[is_cash_flow, REQUIRED_COLUMNS].sort_values("trade_date").reset_index(drop=True)

    return trades_df, cashflows_df
