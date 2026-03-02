from __future__ import annotations

from pathlib import Path
import math

import pandas as pd
import streamlit as st

from attribution_poc import run_attribution_for_window
from src.portfolio.load_transactions import load_transactions
from src.portfolio.cash import build_cash_ledger
from src.portfolio.positions import POSITION_ADD_TYPES, build_positions
from src.portfolio.prices import get_latest_market_date, get_prices
from attribution_poc import BloombergClient, BloombergConfig, _extend_positions_and_cash_to_date, _portfolio_security_panel
from src.portfolio.nav import compute_daily_nav


def _load_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _format_percent_like(df: pd.DataFrame, unit: str, skip_cols: set[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    skip_cols = skip_cols or set()
    scale = 100.0 if unit == "%" else 10000.0
    suffix = "%" if unit == "%" else " bps"
    for col in out.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x * scale:,.2f}{suffix}" if pd.notna(x) else "")
    return out


def _load_alias_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "symbol" not in df.columns:
        return {}
    ticker_col = "bloomberg_ticker" if "bloomberg_ticker" in df.columns else None
    if ticker_col is None and "yfinance_ticker" in df.columns:
        ticker_col = "yfinance_ticker"
    if ticker_col is None:
        return {}
    return df.set_index("symbol")[ticker_col].astype(str).to_dict()


def _xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float | None:
    """Simple XIRR solver using Newton iterations."""
    if not cashflows:
        return None
    cashflows = sorted(cashflows, key=lambda x: x[0])
    t0 = cashflows[0][0]
    amounts = [cf for _, cf in cashflows]
    if not (any(a < 0 for a in amounts) and any(a > 0 for a in amounts)):
        return None

    def f(rate: float) -> float:
        val = 0.0
        for d, cf in cashflows:
            years = (d - t0).days / 365.2425
            val += cf / ((1.0 + rate) ** years)
        return val

    def df(rate: float) -> float:
        val = 0.0
        for d, cf in cashflows:
            years = (d - t0).days / 365.2425
            if years == 0:
                continue
            val += -years * cf / ((1.0 + rate) ** (years + 1.0))
        return val

    r = 0.10
    for _ in range(100):
        denom = df(r)
        if abs(denom) < 1e-14:
            break
        step = f(r) / denom
        r2 = r - step
        if r2 <= -0.9999 or not math.isfinite(r2):
            r2 = (r - 0.9999) / 2.0
        if abs(r2 - r) < 1e-10:
            return r2
        r = r2
    return r if math.isfinite(r) else None


def _most_recent_open_cycle_start(sym_trades: pd.DataFrame) -> pd.Timestamp | None:
    """Return start date of the most recent currently-open position cycle."""
    if sym_trades.empty:
        return None
    g = sym_trades.sort_values("trade_date").copy()
    g["signed_units"] = g.apply(
        lambda r: float(r["units"]) if r["txn_type"] in POSITION_ADD_TYPES else -float(r["units"]),
        axis=1,
    )
    g["cum_shares"] = g["signed_units"].cumsum()
    if g["cum_shares"].iloc[-1] <= 0:
        return None
    zero_or_neg = g[g["cum_shares"] <= 1e-12]
    last_flat_date = zero_or_neg["trade_date"].max() if not zero_or_neg.empty else None
    if last_flat_date is None:
        return pd.to_datetime(g["trade_date"].min()).normalize()
    after = g[g["trade_date"] > last_flat_date]
    if after.empty:
        return None
    return pd.to_datetime(after["trade_date"].min()).normalize()


@st.cache_data(show_spinner=False, ttl=300)
def _compute_live_holdings(repo: str, as_of_date: str) -> pd.DataFrame:
    repo_path = Path(repo)
    as_of = pd.to_datetime(as_of_date).normalize()
    trades_df, _ = load_transactions(str(repo_path / "Transactions.csv"))
    if trades_df.empty:
        return pd.DataFrame(columns=["ticker", "company", "shares", "market_value", "cost_basis", "weight", "pnl_dollar", "pnl_pct", "irr"])

    trades = trades_df.copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"]).dt.normalize()
    trades = trades[trades["trade_date"] <= as_of].copy()
    if trades.empty:
        return pd.DataFrame(columns=["ticker", "company", "shares", "market_value", "cost_basis", "weight", "pnl_dollar", "pnl_pct", "irr"])

    # Current shares by symbol.
    signed = trades.apply(
        lambda r: float(r["units"]) if r["txn_type"] in POSITION_ADD_TYPES else -float(r["units"]),
        axis=1,
    )
    trades["signed_units"] = signed
    shares = trades.groupby("symbol", as_index=False)["signed_units"].sum().rename(columns={"signed_units": "shares"})
    shares = shares[shares["shares"] > 0].copy()
    if shares.empty:
        shares = pd.DataFrame(columns=["ticker", "shares"])

    # Approximate tax-lot tracking with moving-average cost basis per symbol.
    cost_rows: list[dict] = []
    for sym, g in trades.sort_values(["trade_date"]).groupby("symbol"):
        sh = 0.0
        cost = 0.0
        for _, r in g.iterrows():
            qty = float(r.get("units", 0) or 0)
            px = float(r.get("price", 0) or 0)
            if qty <= 0:
                continue
            if r["txn_type"] in POSITION_ADD_TYPES:
                sh += qty
                cost += qty * px
            else:
                if sh <= 0:
                    continue
                avg = cost / sh if sh > 0 else 0.0
                q = min(qty, sh)
                cost -= avg * q
                sh -= q
                if sh <= 1e-12:
                    sh = 0.0
                    cost = 0.0
        if sh > 0:
            cost_rows.append({"ticker": sym, "cost_basis": cost})
    cb = pd.DataFrame(cost_rows) if cost_rows else pd.DataFrame(columns=["ticker", "cost_basis"])

    # Latest prices for market value.
    tickers = shares["symbol"].tolist() if "symbol" in shares.columns else []
    inputs_dir = repo_path / "inputs"
    alias_path = inputs_dir / "symbol_aliases.csv" if inputs_dir.exists() else None
    manual_prices_path = inputs_dir / "manual_prices.csv" if inputs_dir.exists() else None
    if tickers:
        px = get_prices(
            tickers,
            as_of - pd.Timedelta(days=10),
            as_of,
            alias_path=alias_path,
            manual_prices_path=manual_prices_path,
            trades_for_fallback=trades_df,
        )
        px = px.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1)
        px = px.rename(columns={"symbol": "ticker", "price": "last_price"})[["ticker", "last_price"]]
    else:
        px = pd.DataFrame(columns=["ticker", "last_price"])

    # Company names from Bloomberg.
    alias_map = _load_alias_map(alias_path)
    if tickers:
        sec_map = {t: alias_map.get(t, f"{t} US Equity") for t in tickers}
        with BloombergClient(BloombergConfig()) as bbg:
            ref = bbg.get_reference(list(sec_map.values()), ["NAME"])
        names = []
        for t, sec in sec_map.items():
            names.append({"ticker": t, "company": str(ref.get(sec, {}).get("NAME", t))})
        nm = pd.DataFrame(names)
    else:
        nm = pd.DataFrame(columns=["ticker", "company"])

    out = shares.rename(columns={"symbol": "ticker"}).merge(px, on="ticker", how="left").merge(cb, on="ticker", how="left").merge(nm, on="ticker", how="left")
    out["last_price"] = out["last_price"].fillna(0.0)
    out["cost_basis"] = out["cost_basis"].fillna(0.0)
    out["market_value"] = out["shares"] * out["last_price"]
    out["pnl_dollar"] = out["market_value"] - out["cost_basis"]
    out["pnl_pct"] = out["pnl_dollar"] / out["cost_basis"].replace(0, pd.NA)
    out["irr"] = pd.NA

    # IRR by symbol from most recent open cycle only.
    if tickers:
        for t in tickers:
            sym_trades = trades[trades["symbol"] == t].copy()
            cycle_start = _most_recent_open_cycle_start(sym_trades)
            if cycle_start is None:
                continue
            cycle = sym_trades[sym_trades["trade_date"] >= cycle_start].sort_values("trade_date")
            cfs: list[tuple[pd.Timestamp, float]] = []
            for _, r in cycle.iterrows():
                amt = float(r.get("amount", 0) or 0)
                txn = str(r.get("txn_type", ""))
                if txn in POSITION_ADD_TYPES:
                    cfs.append((pd.to_datetime(r["trade_date"]).normalize(), -abs(amt)))
                else:
                    cfs.append((pd.to_datetime(r["trade_date"]).normalize(), abs(amt)))
            mv = float(out.loc[out["ticker"] == t, "market_value"].iloc[0]) if (out["ticker"] == t).any() else 0.0
            cfs.append((as_of, mv))
            irr = _xirr(cfs)
            if irr is not None:
                out.loc[out["ticker"] == t, "irr"] = irr

    # Add cash row.
    _, cashflows_df = load_transactions(str(repo_path / "Transactions.csv"))
    cash_ledger = build_cash_ledger(trades_df, cashflows_df)
    cash_ledger["date"] = pd.to_datetime(cash_ledger["date"]).dt.normalize()
    cash_ledger = cash_ledger[cash_ledger["date"] <= as_of]
    cash_val = float(cash_ledger.sort_values("date")["cash_balance"].iloc[-1]) if not cash_ledger.empty else 0.0
    cash_row = pd.DataFrame(
        [
            {
                "ticker": "CASH",
                "company": "Cash",
                "shares": 0.0,
                "market_value": cash_val,
                "cost_basis": cash_val,
                "pnl_dollar": 0.0,
                "pnl_pct": 0.0,
                "irr": pd.NA,
            }
        ]
    )
    out = pd.concat([out, cash_row], ignore_index=True)
    total_mv = float(out["market_value"].sum())
    out["weight"] = out["market_value"] / total_mv if total_mv != 0 else 0.0
    out = out.sort_values("market_value", ascending=False)
    # Total row at bottom.
    total_cost = float(out["cost_basis"].sum())
    total_pnl = float(out["pnl_dollar"].sum())
    total_row = pd.DataFrame(
        [
            {
                "ticker": "TOTAL",
                "company": "",
                "shares": out["shares"].sum(),
                "market_value": out["market_value"].sum(),
                "cost_basis": total_cost,
                "weight": 1.0,
                "pnl_dollar": total_pnl,
                "pnl_pct": (total_pnl / total_cost) if total_cost != 0 else pd.NA,
                "irr": pd.NA,
            }
        ]
    )
    out = pd.concat([out, total_row], ignore_index=True)
    return out[["ticker", "company", "shares", "market_value", "cost_basis", "weight", "pnl_dollar", "pnl_pct", "irr"]]


@st.cache_data(show_spinner=False, ttl=300)
def _compute_security_decomp(repo: str, start_date: str, end_date: str) -> pd.DataFrame:
    repo_path = Path(repo)
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()

    security = _load_if_exists(repo_path / "outputs" / "attribution_by_security.csv")
    if security is None or security.empty:
        return pd.DataFrame(columns=["symbol", "sector", "selection_effect", "avg_weight", "local_return"])

    trades_df, cashflows_df = load_transactions(str(repo_path / "Transactions.csv"))
    positions_df = build_positions(trades_df)
    positions_df["date"] = pd.to_datetime(positions_df["date"]).dt.normalize()
    positions_df = positions_df[positions_df["date"].between(start, end)]
    cash_df = build_cash_ledger(trades_df, cashflows_df)
    cash_df["date"] = pd.to_datetime(cash_df["date"]).dt.normalize()
    cash_df = cash_df[cash_df["date"].between(start, end)]
    positions_df, cash_df = _extend_positions_and_cash_to_date(positions_df, cash_df, end)

    symbols = sorted(positions_df["symbol"].unique().tolist())
    if not symbols:
        return security
    inputs_dir = repo_path / "inputs"
    alias_path = inputs_dir / "symbol_aliases.csv" if inputs_dir.exists() else None
    manual_prices_path = inputs_dir / "manual_prices.csv" if inputs_dir.exists() else None
    prices_df = get_prices(
        symbols,
        start,
        end,
        alias_path=alias_path,
        manual_prices_path=manual_prices_path,
        trades_for_fallback=trades_df,
    )
    nav_df = compute_daily_nav(positions_df, cash_df, prices_df)

    with BloombergClient(BloombergConfig()) as bbg:
        bench = bbg.get_historical_px_last({"SML Index": "SML Index"}, start - pd.Timedelta(days=10), end)
        bench = bench.sort_values("date")
        bench["ret"] = bench["price"].pct_change()
        trading_dates = pd.DatetimeIndex(bench.loc[bench["ret"].notna(), "date"])
        trading_dates = trading_dates[(trading_dates >= start) & (trading_dates <= end)]
        sec_map = {s: f"{s} US Equity" for s in symbols}
        ref = bbg.get_reference(list(sec_map.values()), ["GICS_SECTOR_NAME"])
    sector_by_symbol = {s: str(ref.get(f"{s} US Equity", {}).get("GICS_SECTOR_NAME", "Unknown")) for s in symbols}
    panel = _portfolio_security_panel(positions_df, prices_df, nav_df, sector_by_symbol, trading_dates=trading_dates)
    if panel.empty:
        return security

    panel_held = panel[panel["w_prev"] > 0].copy()
    grp = panel_held.groupby("symbol", as_index=False).agg(
        avg_weight=("w_prev", "mean"),
        local_return=("sec_ret", lambda s: (1.0 + s).prod() - 1.0),
        holding_days=("date", "count"),
    )
    out = security.merge(grp, on="symbol", how="left")
    out["avg_weight"] = out["avg_weight"].fillna(0.0)
    out["local_return"] = out["local_return"].fillna(0.0)
    out["holding_days"] = out["holding_days"].fillna(0).astype(int)
    return out[out["holding_days"] > 0].copy()


def main() -> None:
    repo = Path(__file__).resolve().parent
    outputs = repo / "outputs"

    st.set_page_config(page_title="IMA Portfolio Tool", layout="wide")
    
    # Inject custom CSS for Source Sans Pro font and specific header styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"], [class*="st-"] {
            font-family: 'Source Sans Pro', sans-serif !important;
        }
        .ima-header {
            color: #13294B;
            font-weight: 700;
            margin-bottom: 0px;
        }
        .ima-subheader {
            color: #E84A27;
            font-weight: 600;
            margin-top: -10px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('<h1 class="ima-header">Portfolio Tool</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="ima-subheader">UIUC Investment Management Academy</h3>', unsafe_allow_html=True)
    st.caption("Bloomberg-backed portfolio analytics and attribution with adjustable date windows.")

    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Start Date", value=pd.Timestamp("2025-01-01").date())
    with c2:
        end = st.date_input("End Date", value=pd.Timestamp.today().date())
    holdings_as_of_default = get_latest_market_date() or pd.Timestamp.today().normalize()
    holdings_as_of = st.date_input("Holdings As Of", value=holdings_as_of_default.date())

    c3, c4 = st.columns([1, 2])
    with c3:
        use_lookback = st.checkbox("Use Lookback Days", value=False)
    with c4:
        lookback_days = st.number_input("Lookback Days", min_value=1, max_value=10000, value=365, step=1)
    unit = st.radio("Display Unit", options=["%", "bps"], horizontal=True, index=0)

    if st.button("Run Attribution", type="primary"):
        with st.spinner("Running attribution..."):
            if use_lookback:
                run_attribution_for_window(repo, lookback_days=int(lookback_days))
            else:
                run_attribution_for_window(
                    repo,
                    start_date=pd.Timestamp(start),
                    end_date=pd.Timestamp(end),
                    lookback_days=None,
                )
        st.success("Attribution run complete.")

    summary = _load_if_exists(outputs / "attribution_summary.csv")
    daily = _load_if_exists(outputs / "attribution_daily.csv")
    sector = _load_if_exists(outputs / "attribution_by_sector.csv")
    security = _load_if_exists(outputs / "attribution_by_security.csv")
    holdings = _compute_live_holdings(str(repo), str(holdings_as_of))

    if holdings is not None and not holdings.empty:
        st.subheader("Live Holdings")
        h = holdings.copy()
        h["shares"] = h["shares"].map(lambda x: f"{x:,.2f}")
        h["market_value"] = h["market_value"].map(lambda x: f"${x:,.0f}")
        h["cost_basis"] = h["cost_basis"].map(lambda x: f"${x:,.0f}")
        h["weight"] = h["weight"].map(lambda x: f"{x*100:,.2f}%")
        h["irr"] = holdings["irr"].map(lambda x: f"{x*100:,.2f}%" if pd.notna(x) else "")
        h["pnl"] = holdings.apply(
            lambda r: f"${r['pnl_dollar']:,.0f}, {r['pnl_pct']*100:,.2f}%"
            if pd.notna(r["pnl_pct"])
            else f"${r['pnl_dollar']:,.0f}",
            axis=1,
        )
        st.dataframe(h[["ticker", "company", "shares", "market_value", "cost_basis", "weight", "pnl", "irr"]], use_container_width=True, hide_index=True)

    if summary is not None:
        st.subheader("Summary")
        st.dataframe(_format_percent_like(summary, unit, skip_cols={"metric"}), use_container_width=True, hide_index=True)

    if daily is not None:
        st.subheader("Daily Returns / Effects")
        cols = [
            "date",
            "daily_return",
            "benchmark_return",
            "active_return",
            "alloc_effect",
            "select_effect",
            "model_gap",
        ]
        daily_view = daily[[c for c in cols if c in daily.columns]].tail(30).copy()
        st.dataframe(_format_percent_like(daily_view, unit, skip_cols={"date"}), use_container_width=True, hide_index=True)

    if sector is not None:
        st.subheader("Sector Out/Underperformance")
        sv = sector.copy()
        sv["status"] = sv["total_effect"].map(lambda x: "Outperforming" if x > 0 else "Underperforming")
        sv["dominant_driver"] = sv.apply(
            lambda r: "Security Selection" if abs(r["select_effect"]) >= abs(r["alloc_effect"]) else "Asset Allocation",
            axis=1,
        )
        sv = sv.sort_values("total_effect", ascending=False)
        st.dataframe(
            _format_percent_like(
                sv[["sector", "status", "dominant_driver", "alloc_effect", "select_effect", "total_effect"]],
                unit,
                skip_cols={"sector", "status", "dominant_driver"},
            ),
            use_container_width=True,
            hide_index=True,
        )
        chart_df = sv.set_index("sector")[["alloc_effect", "select_effect", "total_effect"]]
        st.bar_chart(chart_df)

    if security is not None:
        st.subheader("Top Security Selection")
        if daily is not None and not daily.empty and "date" in daily.columns:
            d0 = pd.to_datetime(daily["date"]).min().strftime("%Y-%m-%d")
            d1 = pd.to_datetime(daily["date"]).max().strftime("%Y-%m-%d")
            sec = _compute_security_decomp(str(repo), d0, d1)
        else:
            sec = _compute_security_decomp(str(repo), str(start), str(end))
        if "selection_effect" in sec.columns:
            sec = sec[sec["selection_effect"].abs() > 1e-12]
        if "holding_days" in sec.columns:
            sec = sec[sec["holding_days"] > 0]
        cols = ["symbol", "sector", "selection_effect", "avg_weight", "local_return", "holding_days"]
        sec = sec[[c for c in cols if c in sec.columns]].sort_values("selection_effect", ascending=False)
        st.dataframe(
            _format_percent_like(sec.head(50), unit, skip_cols={"symbol", "sector", "holding_days"}),
            use_container_width=True,
            hide_index=True,
        )

    p1 = outputs / "attribution_cum_returns.png"
    p2 = outputs / "attribution_cum_effects.png"
    if p1.exists():
        st.subheader("Cumulative Returns")
        st.image(str(p1))
    if p2.exists():
        st.subheader("Cumulative Effects")
        st.image(str(p2))


if __name__ == "__main__":
    main()
