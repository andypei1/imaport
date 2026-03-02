"""
Compute daily NAV and time-weighted returns.

NAV = securities_value + cash_balance.
Daily return excludes external cash flows: return_t = (NAV_t - NAV_{t-1} - external_flow_t) / NAV_{t-1}.
"""

import pandas as pd


def compute_daily_nav(
    positions_df: pd.DataFrame,
    cash_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each day: securities_value = sum(shares * price) per symbol; portfolio_nav = securities_value + cash_balance.

    Positions and prices are joined on (date, symbol). Cash is joined on date.
    Returns DataFrame with: date, nav, securities_value, cash_balance, daily_return, external_flow.
    """
    cash_df = cash_df.copy()
    cash_df["date"] = pd.to_datetime(cash_df["date"]).dt.normalize()
    cash_df = cash_df.sort_values("date").reset_index(drop=True)

    # Use trading calendar from observed price dates; fallback to business days.
    if prices_df.empty:
        trading_dates = pd.date_range(cash_df["date"].min(), cash_df["date"].max(), freq="B")
    else:
        px_dates = pd.to_datetime(prices_df["date"]).dt.normalize().drop_duplicates().sort_values()
        px_dates = px_dates[px_dates.dt.weekday < 5]
        trading_dates = pd.DatetimeIndex(
            [d for d in px_dates if cash_df["date"].min() <= d <= cash_df["date"].max()]
        )
        if trading_dates.empty:
            trading_dates = pd.date_range(cash_df["date"].min(), cash_df["date"].max(), freq="B")

    if positions_df.empty or prices_df.empty:
        # No positions: NAV = cash only, sampled on trading days.
        cash_td = (
            cash_df.set_index("date")
            .reindex(trading_dates)
            .ffill()
            .reset_index()
            .rename(columns={"index": "date"})
        )
        cash_td["external_cum"] = cash_td["external_flow"].fillna(0).cumsum()
        cash_td["external_prev"] = cash_td["external_cum"].shift(1).fillna(0)
        cash_td["external_flow"] = cash_td["external_cum"] - cash_td["external_prev"]
        cash_td["nav"] = cash_td["cash_balance"]
        cash_td["securities_value"] = 0.0
        cash_td["daily_return"] = float("nan")
        return cash_td[["date", "nav", "securities_value", "cash_balance", "daily_return", "external_flow"]]

    # Value positions at latest available price (forward-fill; no look-ahead).
    pos = positions_df.copy()
    pos["date"] = pd.to_datetime(pos["date"]).dt.normalize()
    prices_df = prices_df.copy()
    prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.normalize()

    # Pivot prices to (date x symbol), then reindex to trading dates and ffill per symbol.
    all_dates = trading_dates
    price_pivot = prices_df.pivot_table(index="date", columns="symbol", values="price")
    price_pivot = price_pivot.reindex(all_dates).ffill()
    price_pivot = price_pivot.reset_index()
    price_pivot = price_pivot.rename(columns={price_pivot.columns[0]: "date"})
    price_long = price_pivot.melt(id_vars=["date"], var_name="symbol", value_name="price")
    merged = pos.merge(price_long, on=["date", "symbol"], how="left")
    merged["price"] = merged.groupby("symbol")["price"].ffill()
    merged = merged.dropna(subset=["price"])
    # Cap shares at 0 for valuation only (negative positions not valued until data is fixed).
    merged["shares_val"] = merged["shares"].clip(lower=0)
    merged["market_value"] = merged["shares_val"] * merged["price"]

    # Daily securities value = sum of market_value per date.
    securities_value = merged.groupby("date", as_index=False)["market_value"].sum()

    # Align with cash on trading dates. Aggregate non-trading-day flows into next trading date.
    cash_td = (
        cash_df.set_index("date")
        .reindex(trading_dates)
        .ffill()
        .reset_index()
        .rename(columns={"index": "date"})
    )
    cash_daily = cash_df[["date", "external_flow"]].copy()
    cash_daily["external_flow"] = cash_daily["external_flow"].fillna(0.0)
    cash_daily["external_cum"] = cash_daily["external_flow"].cumsum()
    ext_cum = cash_daily.set_index("date")["external_cum"]
    ext_at_t = ext_cum.reindex(trading_dates, method="ffill").fillna(0.0)
    prev_dates = pd.Series(trading_dates).shift(1)
    ext_prev = ext_cum.reindex(pd.DatetimeIndex(prev_dates.dropna()), method="ffill").fillna(0.0)
    ext_prev.index = prev_dates.dropna().index
    ext_prev_full = pd.Series(0.0, index=pd.RangeIndex(len(trading_dates)))
    ext_prev_full.loc[ext_prev.index] = ext_prev.values
    cash_td["external_flow"] = (ext_at_t.values - ext_prev_full.values)

    nav_df = cash_td[["date", "cash_balance", "external_flow"]].merge(
        securities_value.rename(columns={"market_value": "securities_value"}),
        on="date",
        how="left",
    )
    nav_df["securities_value"] = nav_df["securities_value"].fillna(0)
    nav_df["nav"] = nav_df["securities_value"] + nav_df["cash_balance"]

    # Time-weighted daily return: (NAV_t - NAV_{t-1} - external_flow_t) / NAV_{t-1}.
    nav_df = nav_df.sort_values("date").reset_index(drop=True)
    nav_df["nav_prev"] = nav_df["nav"].shift(1)
    nav_df["daily_return"] = (
        (nav_df["nav"] - nav_df["nav_prev"] - nav_df["external_flow"]) / nav_df["nav_prev"]
    )
    nav_df.loc[nav_df["nav_prev"] == 0, "daily_return"] = float("nan")
    nav_df = nav_df.drop(columns=["nav_prev"])

    return nav_df[["date", "nav", "securities_value", "cash_balance", "daily_return", "external_flow"]]
