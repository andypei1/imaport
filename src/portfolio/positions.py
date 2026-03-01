"""
Build security positions over time from trade history.

Cumulative shares per symbol; positions evolve daily after each trade.
Holdings must never go negative (warn if they do).
"""

import warnings
import pandas as pd

# Transaction types that add shares (Buy, Transfer In, corporate actions). Others (Sale, Transfer Out) subtract.
POSITION_ADD_TYPES = {"Buy", "Transfer In", "Corporate Action", "Stock Split", "Adjustment"}


def build_positions(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    From trades_df (trade_date, symbol, txn_type, units, price, amount),
    compute cumulative shares per symbol through time.

    Returns:
        DataFrame with columns: date, symbol, shares.
    """
    if trades_df.empty:
        return pd.DataFrame(columns=["date", "symbol", "shares"])

    df = trades_df.copy()
    # Signed units: Buy/Transfer In/Corporate Action add shares; Sale/Transfer Out subtract.
    df["signed_units"] = df.apply(
        lambda r: r["units"] if r["txn_type"] in POSITION_ADD_TYPES else -r["units"],
        axis=1,
    )
    # Aggregate by date and symbol (multiple lots same day).
    daily = df.groupby(["trade_date", "symbol"], as_index=False)["signed_units"].sum()
    daily = daily.rename(columns={"trade_date": "date"})

    # Cumulative shares per symbol (order by date).
    daily = daily.sort_values(["symbol", "date"])
    daily["shares"] = daily.groupby("symbol")["signed_units"].cumsum()
    daily = daily.drop(columns=["signed_units"])

    # Warn if any holding goes negative.
    if (daily["shares"] < 0).any():
        neg = daily[daily["shares"] < 0]
        warnings.warn(
            f"Negative shares detected: {neg[['date', 'symbol', 'shares']].to_dict('records')}",
            UserWarning,
            stacklevel=2,
        )

    # Expand to every calendar day: for each symbol, forward-fill position.
    date_min = daily["date"].min()
    date_max = daily["date"].max()
    all_dates = pd.date_range(date_min, date_max, freq="D")

    result_list = []
    for symbol, grp in daily.groupby("symbol"):
        s = grp.set_index("date")["shares"].reindex(all_dates).ffill()
        # Drop leading NaNs (before first trade).
        s = s.dropna()
        if s.empty:
            continue
        for date, shares in s.items():
            result_list.append({"date": date, "symbol": symbol, "shares": shares})

    if not result_list:
        return pd.DataFrame(columns=["date", "symbol", "shares"])

    out = pd.DataFrame(result_list)
    out["shares"] = out["shares"].astype(float)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)
