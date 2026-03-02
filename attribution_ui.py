from __future__ import annotations

from pathlib import Path
import math
import sys

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


def _scaled_percent_like(df: pd.DataFrame, unit: str, skip_cols: set[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    skip_cols = skip_cols or set()
    scale = 100.0 if unit == "%" else 10000.0
    for col in out.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col] * scale
    return out


def _percent_unit_format(unit: str) -> str:
    return "%.2f%%" if unit == "%" else "%.2f bps"


def _effect_number_config(df: pd.DataFrame, unit: str, skip_cols: set[str] | None = None) -> dict[str, st.column_config.NumberColumn]:
    skip_cols = skip_cols or set()
    fmt = _percent_unit_format(unit)
    out: dict[str, st.column_config.NumberColumn] = {}
    for col in df.columns:
        if col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = st.column_config.NumberColumn(format=fmt)
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


def _bbg_security_candidates(symbol: str, alias_map: dict[str, str] | None = None) -> list[str]:
    alias_map = alias_map or {}
    sym = str(symbol).strip()
    out: list[str] = []
    aliased = alias_map.get(sym)
    if aliased:
        out.append(str(aliased).strip())
    out.append(f"{sym} US Equity")
    parts = sym.split()
    if len(parts) == 2 and len(parts[1]) == 1 and parts[1].isalpha():
        out.append(f"{parts[0]}/{parts[1]} US Equity")
    if "." in sym:
        out.append(f"{sym.replace('.', '/')} US Equity")
    seen: set[str] = set()
    uniq: list[str] = []
    for sec in out:
        if sec and sec not in seen:
            seen.add(sec)
            uniq.append(sec)
    return uniq


def _resolve_bbg_field_by_symbol(
    bbg: BloombergClient,
    symbols: list[str],
    field: str,
    alias_map: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    alias_map = alias_map or {}
    symbol_to_candidates = {s: _bbg_security_candidates(s, alias_map) for s in symbols}
    securities: list[str] = []
    for cands in symbol_to_candidates.values():
        for sec in cands:
            if sec not in securities:
                securities.append(sec)
    ref = bbg.get_reference(securities, [field]) if securities else {}

    field_by_symbol: dict[str, str] = {}
    chosen_security: dict[str, str] = {}
    for s in symbols:
        val = "Unknown"
        chosen = ""
        for sec in symbol_to_candidates.get(s, []):
            raw = ref.get(sec, {}).get(field)
            txt = str(raw).strip() if raw is not None else ""
            if txt and txt.lower() not in {"nan", "none"}:
                val = txt
                chosen = sec
                break
        field_by_symbol[s] = val
        chosen_security[s] = chosen
    return field_by_symbol, chosen_security


def _resolve_sector_by_symbol(
    bbg: BloombergClient,
    symbols: list[str],
    alias_map: dict[str, str] | None = None,
) -> dict[str, str]:
    alias_map = alias_map or {}
    out: dict[str, str] = {}
    for s in symbols:
        cands = _bbg_security_candidates(s, alias_map)
        ref = bbg.get_reference(cands, ["GICS_SECTOR_NAME", "INDUSTRY_SECTOR"])
        sector = "Unknown"
        for sec in cands:
            rec = ref.get(sec, {})
            gics = str(rec.get("GICS_SECTOR_NAME", "")).strip()
            ind = str(rec.get("INDUSTRY_SECTOR", "")).strip()
            if gics and gics.lower() not in {"nan", "none"}:
                sector = gics
                break
            if ind and ind.lower() not in {"nan", "none"}:
                sector = ind
                break
        out[s] = sector
    return out


@st.cache_data(show_spinner=False, ttl=60)
def _check_bloomberg_status() -> tuple[bool, str]:
    try:
        with BloombergClient(BloombergConfig()) as bbg:
            ref = bbg.get_reference(["SML Index"], ["NAME"])
        name = str(ref.get("SML Index", {}).get("NAME", "")).strip()
        if name:
            return True, f"Bloomberg API connected ({name}). Python: {sys.executable}"
        return False, f"Bloomberg session opened, but reference test returned empty data. Python: {sys.executable}"
    except Exception as e:
        return False, f"Bloomberg API unavailable: {type(e).__name__}: {e}. Python: {sys.executable}"


def _outputs_last_updated(outputs_dir: Path) -> pd.Timestamp | None:
    files = [
        outputs_dir / "attribution_summary.csv",
        outputs_dir / "attribution_daily.csv",
        outputs_dir / "attribution_by_sector.csv",
        outputs_dir / "attribution_by_security.csv",
    ]
    existing = [p for p in files if p.exists()]
    if not existing:
        return None
    latest = max(p.stat().st_mtime for p in existing)
    return pd.to_datetime(latest, unit="s")


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
def _compute_live_holdings(repo: str, as_of_date: str) -> tuple[pd.DataFrame, dict[str, float | None]]:
    repo_path = Path(repo)
    as_of = pd.to_datetime(as_of_date).normalize()
    trades_df, cashflows_df = load_transactions(str(repo_path / "Transactions.csv"))
    if trades_df.empty:
        empty = pd.DataFrame(columns=["ticker", "company", "shares", "avg_price", "today_price", "ret_1d", "ret_1w", "ret_1m", "market_value", "cost_basis", "weight", "pnl_dollar", "pnl_pct", "irr"])
        return empty, {"market_value": 0.0, "cost_basis": 0.0, "pnl_dollar": 0.0, "pnl_pct": None}

    trades = trades_df.copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"]).dt.normalize()
    trades = trades[trades["trade_date"] <= as_of].copy()
    if trades.empty:
        empty = pd.DataFrame(columns=["ticker", "company", "shares", "avg_price", "today_price", "ret_1d", "ret_1w", "ret_1m", "market_value", "cost_basis", "weight", "pnl_dollar", "pnl_pct", "irr"])
        return empty, {"market_value": 0.0, "cost_basis": 0.0, "pnl_dollar": 0.0, "pnl_pct": None}

    # Current shares by symbol.
    trades["signed_units"] = trades.apply(
        lambda r: float(r["units"]) if r["txn_type"] in POSITION_ADD_TYPES else -float(r["units"]),
        axis=1,
    )
    shares = trades.groupby("symbol", as_index=False)["signed_units"].sum().rename(columns={"signed_units": "shares"})
    shares = shares[shares["shares"] > 0].copy()
    if shares.empty:
        shares = pd.DataFrame(columns=["ticker", "shares"])

    # Moving-average cost basis per symbol.
    cost_rows: list[dict] = []
    for sym, g in trades.sort_values(["trade_date"]).groupby("symbol"):
        sh = 0.0
        cost = 0.0
        first_buy_date: pd.Timestamp | None = None
        for _, r in g.iterrows():
            d = pd.to_datetime(r["trade_date"]).normalize()
            qty = float(r.get("units", 0) or 0)
            px = float(r.get("price", 0) or 0)
            if qty <= 0:
                continue
            if r["txn_type"] in POSITION_ADD_TYPES:
                if sh <= 1e-12:
                    first_buy_date = d
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
            cost_rows.append({"ticker": sym, "cost_basis": cost, "first_buy_date": first_buy_date})
    cb = pd.DataFrame(cost_rows) if cost_rows else pd.DataFrame(columns=["ticker", "cost_basis", "first_buy_date"])

    # Latest prices for market value.
    tickers = shares["symbol"].tolist() if "symbol" in shares.columns else []
    inputs_dir = repo_path / "inputs"
    alias_path = inputs_dir / "symbol_aliases.csv" if inputs_dir.exists() else None
    manual_prices_path = inputs_dir / "manual_prices.csv" if inputs_dir.exists() else None
    if tickers:
        px_hist = get_prices(
            tickers,
            as_of - pd.Timedelta(days=60),
            as_of,
            alias_path=alias_path,
            manual_prices_path=manual_prices_path,
            trades_for_fallback=trades_df,
        )
        px_hist["date"] = pd.to_datetime(px_hist["date"]).dt.normalize()
        px_hist = px_hist.sort_values(["symbol", "date"])
        px = px_hist.groupby("symbol", as_index=False).tail(1)
        px = px.rename(columns={"symbol": "ticker", "price": "today_price"})[["ticker", "today_price"]]

        def _price_asof(df: pd.DataFrame, d: pd.Timestamp) -> float | None:
            sub = df[df["date"] <= d]
            if sub.empty:
                return None
            return float(sub["price"].iloc[-1])

        ret_rows: list[dict] = []
        for sym, g in px_hist.groupby("symbol"):
            p0 = _price_asof(g, as_of)
            p1d = _price_asof(g, as_of - pd.Timedelta(days=1))
            p1w = _price_asof(g, as_of - pd.Timedelta(days=7))
            p1m = _price_asof(g, as_of - pd.Timedelta(days=30))

            def _ret(curr: float | None, prev: float | None) -> float | None:
                if curr is None or prev is None or prev == 0:
                    return None
                return (curr / prev) - 1.0

            ret_rows.append(
                {
                    "ticker": str(sym),
                    "ret_1d": _ret(p0, p1d),
                    "ret_1w": _ret(p0, p1w),
                    "ret_1m": _ret(p0, p1m),
                }
            )
        ret_df = pd.DataFrame(ret_rows) if ret_rows else pd.DataFrame(columns=["ticker", "ret_1d", "ret_1w", "ret_1m"])
    else:
        px = pd.DataFrame(columns=["ticker", "today_price"])
        ret_df = pd.DataFrame(columns=["ticker", "ret_1d", "ret_1w", "ret_1m"])

    # Company names from Bloomberg.
    alias_map = _load_alias_map(alias_path)
    if tickers:
        names = []
        try:
            sec_map = {t: alias_map.get(t, f"{t} US Equity") for t in tickers}
            with BloombergClient(BloombergConfig()) as bbg:
                ref = bbg.get_reference(list(sec_map.values()), ["NAME"])
            for t, sec in sec_map.items():
                names.append({"ticker": t, "company": str(ref.get(sec, {}).get("NAME", t))})
        except Exception:
            for t in tickers:
                names.append({"ticker": t, "company": t})
        nm = pd.DataFrame(names)
    else:
        nm = pd.DataFrame(columns=["ticker", "company"])

    out = shares.rename(columns={"symbol": "ticker"}).merge(px, on="ticker", how="left").merge(ret_df, on="ticker", how="left").merge(cb, on="ticker", how="left").merge(nm, on="ticker", how="left")
    out["today_price"] = out["today_price"].fillna(0.0)
    out["cost_basis"] = out["cost_basis"].fillna(0.0)
    out["first_buy_date"] = pd.to_datetime(out["first_buy_date"], errors="coerce")
    out["avg_price"] = out["cost_basis"] / out["shares"].replace(0, pd.NA)
    out["avg_price"] = out["avg_price"].fillna(0.0)
    out["market_value"] = out["shares"] * out["today_price"]
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
                "avg_price": pd.NA,
                "today_price": pd.NA,
                "ret_1d": pd.NA,
                "ret_1w": pd.NA,
                "ret_1m": pd.NA,
                "first_buy_date": pd.NaT,
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

    totals = {
        "market_value": float(out["market_value"].sum()),
        "cost_basis": float(out["cost_basis"].sum()),
        "pnl_dollar": float(out["pnl_dollar"].sum()),
        "pnl_pct": None,
    }
    if totals["cost_basis"] != 0:
        totals["pnl_pct"] = totals["pnl_dollar"] / totals["cost_basis"]
    return out[["ticker", "company", "shares", "first_buy_date", "avg_price", "today_price", "ret_1d", "ret_1w", "ret_1m", "market_value", "cost_basis", "weight", "pnl_dollar", "pnl_pct", "irr"]], totals


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
        alias_map = _load_alias_map(alias_path)
        sector_by_symbol = _resolve_sector_by_symbol(bbg, symbols, alias_map=alias_map)
    panel = _portfolio_security_panel(positions_df, prices_df, nav_df, sector_by_symbol, trading_dates=trading_dates)
    if panel.empty:
        return security

    panel_held = panel[panel["w_prev"] > 0].copy()
    grp = panel_held.groupby("symbol", as_index=False).agg(
        avg_weight=("w_prev", "mean"),
        local_return=("sec_ret", lambda s: (1.0 + s).prod() - 1.0),
        first_held=("date", "min"),
        last_held=("date", "max"),
    )
    grp["holding_days"] = (pd.to_datetime(grp["last_held"]) - pd.to_datetime(grp["first_held"])).dt.days + 1
    grp["holding_days"] = grp["holding_days"].fillna(0).astype(int)
    grp = grp.drop(columns=["first_held", "last_held"])
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
        
        html, body, .stApp {
            font-family: 'Source Sans Pro', sans-serif !important;
        }
        .material-icons, .material-symbols-outlined, .material-symbols-rounded, .material-symbols-sharp {
            font-family: "Material Symbols Outlined", "Material Icons" !important;
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
    bbg_ok, bbg_msg = _check_bloomberg_status()
    if bbg_ok:
        st.success(bbg_msg)
    else:
        st.warning(bbg_msg)
        st.info("Offline mode: live Bloomberg pulls are disabled. Displaying saved outputs and local fallback data where available.")

    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Start Date", value=pd.Timestamp("2025-01-01").date())
    with c2:
        end = st.date_input("End Date", value=pd.Timestamp.today().date())
    try:
        holdings_as_of_default = get_latest_market_date() if bbg_ok else None
    except Exception:
        holdings_as_of_default = None
    holdings_as_of_default = holdings_as_of_default or pd.Timestamp.today().normalize()
    holdings_as_of = st.date_input("Holdings As Of", value=holdings_as_of_default.date())

    c3, c4 = st.columns([1, 2])
    with c3:
        use_lookback = st.checkbox("Use Lookback Days", value=False)
    with c4:
        lookback_days = st.number_input("Lookback Days", min_value=1, max_value=10000, value=365, step=1)
    unit = st.radio("Display Unit", options=["%", "bps"], horizontal=True, index=0)

    if st.button("Run Attribution", type="primary", disabled=not bbg_ok):
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
    last_updated = _outputs_last_updated(outputs)
    if last_updated is not None:
        st.caption(f"Loaded saved outputs last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    holdings, holdings_totals = _compute_live_holdings(str(repo), str(holdings_as_of))
    sec_decomp = None
    if security is not None:
        if bbg_ok:
            if daily is not None and not daily.empty and "date" in daily.columns:
                d0 = pd.to_datetime(daily["date"]).min().strftime("%Y-%m-%d")
                d1 = pd.to_datetime(daily["date"]).max().strftime("%Y-%m-%d")
                sec_decomp = _compute_security_decomp(str(repo), d0, d1)
            else:
                sec_decomp = _compute_security_decomp(str(repo), str(start), str(end))
        else:
            sec_decomp = security.copy()
        if sec_decomp is not None and not sec_decomp.empty and "selection_effect" in sec_decomp.columns:
            sec_decomp = sec_decomp[sec_decomp["selection_effect"].abs() > 1e-12]
        if sec_decomp is not None and not sec_decomp.empty and "holding_days" in sec_decomp.columns:
            sec_decomp = sec_decomp[sec_decomp["holding_days"] > 0]

    if holdings is not None and not holdings.empty:
        st.subheader("Live Holdings")
        h_view = holdings[["ticker", "company", "shares", "first_buy_date", "avg_price", "today_price", "ret_1d", "ret_1w", "ret_1m", "market_value", "cost_basis", "weight", "pnl_dollar", "pnl_pct", "irr"]].copy()
        h_view["first_buy_date"] = pd.to_datetime(h_view["first_buy_date"], errors="coerce")
        for col in ["ret_1d", "ret_1w", "ret_1m", "weight", "pnl_pct", "irr"]:
            h_view[col] = h_view[col] * 100.0
        st.dataframe(
            h_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": "Ticker",
                "company": "Company",
                "shares": st.column_config.NumberColumn("Shares", format="%.2f"),
                "first_buy_date": st.column_config.DateColumn("First Buy"),
                "avg_price": st.column_config.NumberColumn("Avg Price", format="$%.2f"),
                "today_price": st.column_config.NumberColumn("Today Price", format="$%.2f"),
                "ret_1d": st.column_config.NumberColumn("1D", format="%.2f%%"),
                "ret_1w": st.column_config.NumberColumn("1W", format="%.2f%%"),
                "ret_1m": st.column_config.NumberColumn("1M", format="%.2f%%"),
                "market_value": st.column_config.NumberColumn("Market Value", format="$%.0f"),
                "cost_basis": st.column_config.NumberColumn("Cost Basis", format="$%.0f"),
                "weight": st.column_config.NumberColumn("Weight", format="%.2f%%"),
                "pnl_dollar": st.column_config.NumberColumn("PnL ($)", format="$%.0f"),
                "pnl_pct": st.column_config.NumberColumn("PnL (%)", format="%.2f%%"),
                "irr": st.column_config.NumberColumn("IRR", format="%.2f%%"),
            },
        )
        total_pnl_pct = holdings_totals.get("pnl_pct")
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Total Market Value", f"${float(holdings_totals.get('market_value') or 0.0):,.0f}")
        tc2.metric("Total Cost Basis", f"${float(holdings_totals.get('cost_basis') or 0.0):,.0f}")
        tc3.metric("Total PnL ($)", f"${float(holdings_totals.get('pnl_dollar') or 0.0):,.0f}")
        tc4.metric("Total PnL (%)", f"{float(total_pnl_pct) * 100.0:,.2f}%" if total_pnl_pct is not None else "N/A")

    if summary is not None:
        st.subheader("Summary")
        summary_scaled = _scaled_percent_like(summary, unit, skip_cols={"metric"})
        st.dataframe(
            summary_scaled,
            use_container_width=True,
            hide_index=True,
            column_config=_effect_number_config(summary_scaled, unit, skip_cols={"metric"}),
        )

    if sector is not None:
        st.subheader("Sector Out/Underperformance")
        sv = sector.copy()
        sv = sv[sv["sector"].astype(str).str.strip().str.lower() != "cash"].copy()
        unknown_in_sector = sv["sector"].astype(str).str.strip().str.lower() == "unknown"
        if unknown_in_sector.any() and sec_decomp is not None and not sec_decomp.empty:
            unknown_symbols = (
                sec_decomp[sec_decomp["sector"].astype(str).str.strip().str.lower() == "unknown"]["symbol"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if unknown_symbols:
                st.caption(f"Unknown sector currently contains: {', '.join(unknown_symbols)}")
        sv["status"] = sv["total_effect"].map(lambda x: "Outperforming" if x > 0 else "Underperforming")
        sv["dominant_driver"] = sv.apply(
            lambda r: "Security Selection" if abs(r["select_effect"]) >= abs(r["alloc_effect"]) else "Asset Allocation",
            axis=1,
        )
        sv = sv.sort_values("total_effect", ascending=False)
        sv_view = sv[["sector", "status", "dominant_driver", "alloc_effect", "select_effect", "total_effect"]]
        sv_scaled = _scaled_percent_like(sv_view, unit, skip_cols={"sector", "status", "dominant_driver"})
        st.dataframe(
            sv_scaled,
            use_container_width=True,
            hide_index=True,
            column_config=_effect_number_config(sv_scaled, unit, skip_cols={"sector", "status", "dominant_driver"}),
        )
        chart_df = sv.set_index("sector")[["alloc_effect", "select_effect", "total_effect"]]
        st.bar_chart(chart_df)
        if sec_decomp is not None and not sec_decomp.empty:
            st.caption("Selection effect by sector (expand to view constituents).")
            sec_group = (
                sec_decomp.groupby("sector", as_index=False)
                .agg(sector_selection_effect=("selection_effect", "sum"), count=("symbol", "count"))
                .sort_values("sector_selection_effect", ascending=False)
            )
            unit_scale = 100.0 if unit == "%" else 10000.0
            unit_suffix = "%" if unit == "%" else "bps"
            for _, g in sec_group.iterrows():
                sector_name = str(g["sector"])
                with st.expander(sector_name, expanded=False):
                    st.caption(
                        f"Selection Effect: {float(g['sector_selection_effect']) * unit_scale:,.2f} {unit_suffix}  |  Names: {int(g['count'])}"
                    )
                    sub = sec_decomp[sec_decomp["sector"] == sector_name].copy().sort_values("selection_effect", ascending=False)
                    sub_scaled = _scaled_percent_like(sub, unit, skip_cols={"symbol", "sector", "holding_days"})
                    st.dataframe(
                        sub_scaled,
                        use_container_width=True,
                        hide_index=True,
                        column_config=_effect_number_config(sub_scaled, unit, skip_cols={"symbol", "sector", "holding_days"}),
                    )

    if security is not None:
        st.subheader("Top Security Selection")
        sec = sec_decomp.copy() if sec_decomp is not None else pd.DataFrame()
        cols = ["symbol", "sector", "selection_effect", "avg_weight", "local_return", "holding_days"]
        sec = sec[[c for c in cols if c in sec.columns]].sort_values("selection_effect", ascending=False)
        sec_view = sec.head(50).copy()
        sec_scaled = _scaled_percent_like(sec_view, unit, skip_cols={"symbol", "sector", "holding_days"})
        st.dataframe(
            sec_scaled,
            use_container_width=True,
            hide_index=True,
            column_config=_effect_number_config(sec_scaled, unit, skip_cols={"symbol", "sector", "holding_days"}),
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
