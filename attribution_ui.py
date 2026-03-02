from __future__ import annotations

import base64
from pathlib import Path
import math
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

from attribution_poc import run_attribution_for_window, refresh_bloomberg_cache
from src.portfolio.load_transactions import load_transactions, POSITION_ADDITION_TYPES
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


def _load_yf_alias_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "symbol" not in df.columns:
        return {}
    if "yfinance_ticker" not in df.columns:
        return {}
    out = (
        df[["symbol", "yfinance_ticker"]]
        .dropna()
        .assign(symbol=lambda x: x["symbol"].astype(str).str.strip(), yfinance_ticker=lambda x: x["yfinance_ticker"].astype(str).str.strip())
    )
    out = out[out["symbol"] != ""]
    out = out[out["yfinance_ticker"] != ""]
    return out.drop_duplicates("symbol").set_index("symbol")["yfinance_ticker"].to_dict()


def _last_on_or_before(series: pd.Series, target: pd.Timestamp) -> float | None:
    if series.empty:
        return None
    sub = series[series.index <= target]
    if sub.empty:
        return None
    v = sub.iloc[-1]
    if pd.isna(v):
        return None
    return float(v)


def _first_on_or_after(series: pd.Series, target: pd.Timestamp) -> float | None:
    if series.empty:
        return None
    sub = series[series.index >= target]
    if sub.empty:
        return None
    v = sub.iloc[0]
    if pd.isna(v):
        return None
    return float(v)


@st.cache_data(show_spinner=False, ttl=20)
def _fetch_live_yf_quotes(symbols: tuple[str, ...], alias_csv: Optional[str]) -> pd.DataFrame:
    cols = ["ticker", "today_price", "ret_1d", "ret_1w", "ret_1m", "ret_ytd"]
    if yf is None or not symbols:
        return pd.DataFrame(columns=cols)

    alias_path = Path(alias_csv) if alias_csv else None
    yf_alias = _load_yf_alias_map(alias_path)
    ticker_to_yf = {s: yf_alias.get(s, s) for s in symbols}
    unique_yf = tuple(sorted({v for v in ticker_to_yf.values() if str(v).strip()}))
    if not unique_yf:
        return pd.DataFrame(columns=cols)

    daily_rows: list[dict] = []
    try:
        daily = yf.download(
            tickers=list(unique_yf),
            period="3mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        daily = pd.DataFrame()

    intraday_last: dict[str, float] = {}
    try:
        intraday = yf.download(
            tickers=list(unique_yf),
            period="1d",
            interval="1m",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
            prepost=False,
        )
    except Exception:
        intraday = pd.DataFrame()

    def _extract_series(df: pd.DataFrame, yf_ticker: str, field: str) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype="float64")
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if yf_ticker in df.columns.get_level_values(0):
                    s = df[(yf_ticker, field)]
                elif field in df.columns.get_level_values(0):
                    s = df[field]
                else:
                    return pd.Series(dtype="float64")
            else:
                if field not in df.columns:
                    return pd.Series(dtype="float64")
                s = df[field]
            s = pd.to_numeric(s, errors="coerce").dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s
        except Exception:
            return pd.Series(dtype="float64")

    now = pd.Timestamp.now(tz="America/Chicago").tz_localize(None)
    d_1 = now - pd.Timedelta(days=1)
    d_7 = now - pd.Timedelta(days=7)
    d_30 = now - pd.Timedelta(days=30)
    ytd_start = pd.Timestamp(year=now.year, month=1, day=1)

    for ticker, yf_ticker in ticker_to_yf.items():
        ds = _extract_series(daily, yf_ticker, "Close")
        ids = _extract_series(intraday, yf_ticker, "Close")
        today_price = float(ids.iloc[-1]) if not ids.empty else (_last_on_or_before(ds, now))
        prev_1d = _last_on_or_before(ds, d_1)
        prev_1w = _last_on_or_before(ds, d_7)
        prev_1m = _last_on_or_before(ds, d_30)
        prev_ytd = _first_on_or_after(ds, ytd_start)

        def _ret(curr: float | None, prev: float | None) -> float | None:
            if curr is None or prev is None or prev == 0:
                return None
            return (curr / prev) - 1.0

        daily_rows.append(
            {
                "ticker": ticker,
                "today_price": today_price,
                "ret_1d": _ret(today_price, prev_1d),
                "ret_1w": _ret(today_price, prev_1w),
                "ret_1m": _ret(today_price, prev_1m),
                "ret_ytd": _ret(today_price, prev_ytd),
            }
        )

    if not daily_rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(daily_rows)[cols]


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
    dt_central = datetime.fromtimestamp(latest, tz=timezone.utc).astimezone(ZoneInfo("America/Chicago"))
    return pd.Timestamp(dt_central)


def _outputs_date_window(outputs_dir: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    daily_path = outputs_dir / "attribution_daily.csv"
    if not daily_path.exists():
        return None, None
    try:
        df = pd.read_csv(daily_path, usecols=["date"])
    except Exception:
        return None, None
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    if d.empty:
        return None, None
    return d.min().normalize(), d.max().normalize()


def _portfolio_date_bounds(repo_path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    today = pd.Timestamp.today().normalize()
    tx_path = repo_path / "Transactions.csv"
    if not tx_path.exists():
        return pd.Timestamp("2025-01-01"), today
    try:
        df = pd.read_csv(tx_path, usecols=["Date"])
    except Exception:
        try:
            df = pd.read_csv(tx_path, usecols=["trade_date"])
        except Exception:
            return pd.Timestamp("2025-01-01"), today
    col = "Date" if "Date" in df.columns else "trade_date"
    d = pd.to_datetime(df[col], errors="coerce").dropna()
    if d.empty:
        return pd.Timestamp("2025-01-01"), today
    inception = d.min().normalize()
    return inception, today


def _clamp_date_to_bounds(value: object, min_date: pd.Timestamp, max_date: pd.Timestamp, fallback: pd.Timestamp) -> pd.Timestamp:
    try:
        d = pd.to_datetime(value).normalize()
    except Exception:
        d = fallback
    if d < min_date:
        return min_date
    if d > max_date:
        return max_date
    return d


def _has_attribution_cache(repo_path: Path) -> bool:
    cdir = repo_path / "outputs" / "cache"
    required = [
        cdir / "portfolio_prices.csv",
        cdir / "benchmark_px.csv",
        cdir / "benchmark_sector_model.csv",
        cdir / "portfolio_sector_map.csv",
    ]
    return all(p.exists() for p in required)


def _load_nav_history(outputs_dir: Path) -> pd.DataFrame:
    nav_path = outputs_dir / "nav.csv"
    if not nav_path.exists():
        return pd.DataFrame(columns=["date", "nav", "external_flow"])
    nav = pd.read_csv(nav_path)
    nav["date"] = pd.to_datetime(nav.get("date"), errors="coerce").dt.normalize()
    nav = nav.dropna(subset=["date"]).sort_values("date")
    nav["nav"] = pd.to_numeric(nav.get("nav"), errors="coerce")
    nav["external_flow"] = pd.to_numeric(nav.get("external_flow"), errors="coerce").fillna(0.0)
    return nav[["date", "nav", "external_flow"]].dropna(subset=["nav"]).copy()


def _load_benchmark_history(cache_dir: Path) -> pd.DataFrame:
    bench_path = cache_dir / "benchmark_px.csv"
    if not bench_path.exists():
        return pd.DataFrame(columns=["date", "benchmark_return"])
    bench = pd.read_csv(bench_path)
    bench["date"] = pd.to_datetime(bench.get("date"), errors="coerce").dt.normalize()
    bench = bench.dropna(subset=["date"]).sort_values("date")
    if "benchmark_return" in bench.columns:
        bench["benchmark_return"] = pd.to_numeric(bench["benchmark_return"], errors="coerce")
    elif "tri" in bench.columns:
        bench["tri"] = pd.to_numeric(bench["tri"], errors="coerce")
        bench["benchmark_return"] = bench["tri"].pct_change()
    elif "price" in bench.columns:
        bench["price"] = pd.to_numeric(bench["price"], errors="coerce")
        bench["benchmark_return"] = bench["price"].pct_change()
    else:
        bench["benchmark_return"] = pd.NA
    return bench[["date", "benchmark_return"]].copy()


def _annualize_return(total_return: float, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float | None:
    if pd.isna(total_return):
        return None
    days = int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days)
    if days <= 0:
        return None
    if total_return <= -1.0:
        return None
    years = days / 365.25
    if years <= 0:
        return None
    return float((1.0 + float(total_return)) ** (1.0 / years) - 1.0)


def _period_start(end_date: pd.Timestamp, label: str, inception: pd.Timestamp) -> pd.Timestamp:
    label = str(label).strip()
    if label == "YTD":
        start = pd.Timestamp(year=end_date.year, month=1, day=1)
    elif label == "6M":
        start = end_date - pd.DateOffset(months=6)
    elif label == "1Y":
        start = end_date - pd.DateOffset(years=1)
    elif label == "2Y":
        start = end_date - pd.DateOffset(years=2)
    elif label == "3Y":
        start = end_date - pd.DateOffset(years=3)
    elif label == "5Y":
        start = end_date - pd.DateOffset(years=5)
    else:
        start = inception
    return max(pd.to_datetime(start).normalize(), pd.to_datetime(inception).normalize())


def _build_performance_panel(repo_path: Path, outputs_dir: Path) -> pd.DataFrame:
    nav = _load_nav_history(outputs_dir)
    bench = _load_benchmark_history(outputs_dir / "cache")
    if nav.empty:
        return pd.DataFrame(columns=["date", "nav", "portfolio_return", "benchmark_return"])
    nav = nav.sort_values("date").drop_duplicates("date")
    bench = bench.sort_values("date").drop_duplicates("date")
    bench = bench[bench["benchmark_return"].notna()].copy()
    if bench.empty:
        return pd.DataFrame(columns=["date", "nav", "portfolio_return", "benchmark_return"])

    nav_idx = nav.set_index("date").sort_index()
    flow_series = nav_idx["external_flow"].copy()

    rows: list[dict[str, object]] = []
    prev_date: pd.Timestamp | None = None
    for _, br in bench.iterrows():
        d = pd.to_datetime(br["date"]).normalize()
        if d not in nav_idx.index:
            continue
        nav_d = float(nav_idx.at[d, "nav"])
        if prev_date is None:
            prev_date = d
            continue
        if prev_date not in nav_idx.index:
            prev_date = d
            continue
        nav_prev = float(nav_idx.at[prev_date, "nav"])
        if nav_prev == 0 or pd.isna(nav_prev):
            prev_date = d
            continue
        flows_between = float(flow_series[(flow_series.index > prev_date) & (flow_series.index <= d)].sum())
        p_ret = (nav_d - nav_prev - flows_between) / nav_prev
        rows.append(
            {
                "date": d,
                "nav": nav_d,
                "portfolio_return": float(p_ret),
                "benchmark_return": float(br["benchmark_return"]),
            }
        )
        prev_date = d

    if not rows:
        return pd.DataFrame(columns=["date", "nav", "portfolio_return", "benchmark_return"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _render_performance_section(repo_path: Path, outputs_dir: Path) -> None:
    perf = _build_performance_panel(repo_path, outputs_dir)
    if perf.empty:
        return

    st.subheader("Portfolio Performance")
    period_options = ["YTD", "6M", "1Y", "2Y", "5Y", "Since Inception"]
    period = st.radio("Performance Window", options=period_options, horizontal=True, index=0, key="perf_window")

    end_date = pd.to_datetime(perf["date"].max()).normalize()
    inception = pd.to_datetime(perf["date"].min()).normalize()
    start_date = _period_start(end_date, period, inception)
    window = perf[(perf["date"] >= start_date) & (perf["date"] <= end_date)].copy()
    if window.empty:
        st.caption("No performance data available for the selected window.")
        return

    window["cum_portfolio"] = (1.0 + window["portfolio_return"]).cumprod() - 1.0
    window["cum_benchmark"] = (1.0 + window["benchmark_return"]).cumprod() - 1.0
    window["cum_active"] = window["cum_portfolio"] - window["cum_benchmark"]

    aum_min = float(window["nav"].min())
    aum_max = float(window["nav"].max())
    aum_pad = max((aum_max - aum_min) * 0.08, max(aum_max, 1.0) * 0.02)
    aum_chart = (
        alt.Chart(window)
        .mark_line(color="#13294B", strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(
                "nav:Q",
                title="AUM ($)",
                axis=alt.Axis(format="$,.0f"),
                scale=alt.Scale(zero=False, domain=[aum_min - aum_pad, aum_max + aum_pad]),
            ),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("nav:Q", title="AUM", format=",.0f")],
        )
        .properties(height=360)
    )
    st.altair_chart(aum_chart, use_container_width=True)

    ret_lines = pd.concat(
        [
            window[["date", "cum_portfolio"]].rename(columns={"cum_portfolio": "value"}).assign(series="Portfolio"),
            window[["date", "cum_benchmark"]].rename(columns={"cum_benchmark": "value"}).assign(series="SML Index"),
        ],
        ignore_index=True,
    )
    ret_lines["value_pct"] = ret_lines["value"] * 100.0
    ret_lines["opacity"] = ret_lines["series"].map({"Portfolio": 1.0, "SML Index": 0.35}).fillna(1.0)

    y_min = float(min(ret_lines["value_pct"].min(), (window["cum_active"] * 100.0).min(), 0.0))
    y_max = float(max(ret_lines["value_pct"].max(), (window["cum_active"] * 100.0).max(), 0.0))
    y_pad = max((y_max - y_min) * 0.08, 1.0)
    y_scale = alt.Scale(domain=[y_min - y_pad, y_max + y_pad])

    return_line_chart = (
        alt.Chart(ret_lines)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value_pct:Q", title="Cumulative Return / Outperformance (%)", scale=y_scale),
            color=alt.Color("series:N", scale=alt.Scale(domain=["Portfolio", "SML Index"], range=["#E84A27", "#4C78A8"])),
            opacity=alt.Opacity("opacity:Q", legend=None),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value_pct:Q", title="Cumulative Return", format=".2f"),
            ],
        )
        .properties(height=380)
    )

    bars = window[["date", "cum_active"]].copy()
    bars["cum_active_pct"] = bars["cum_active"] * 100.0
    outperf_chart = (
        alt.Chart(bars)
        .mark_bar(color="#8DA0BC", opacity=0.22)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("cum_active_pct:Q", scale=y_scale),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("cum_active_pct:Q", title="Cumulative Outperformance", format=".2f"),
            ],
        )
        .properties(height=380)
    )
    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9AA0A6", strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart(outperf_chart + zero_line + return_line_chart, use_container_width=True)

    rows: list[dict[str, object]] = []
    inception = pd.to_datetime(perf["date"].min()).normalize()
    years = sorted(perf["date"].dt.year.unique().tolist())
    current_year = int(perf["date"].max().year)
    for y in years:
        ydf = perf[perf["date"].dt.year == y]
        if ydf.empty:
            continue
        period_start = pd.to_datetime(ydf["date"].min()).normalize()
        period_end = pd.to_datetime(ydf["date"].max()).normalize()
        yr_p = float((1.0 + ydf["portfolio_return"]).prod() - 1.0)
        yr_b = float((1.0 + ydf["benchmark_return"]).prod() - 1.0)
        si = perf[(perf["date"] >= inception) & (perf["date"] <= period_end)].copy()
        si_p = float((1.0 + si["portfolio_return"]).prod() - 1.0)
        si_b = float((1.0 + si["benchmark_return"]).prod() - 1.0)
        si_p_ann = _annualize_return(si_p, inception, period_end)
        si_b_ann = _annualize_return(si_b, inception, period_end)
        label = f"{y} YTD" if y == current_year else str(y)
        rows.append(
            {
                "period": label,
                "portfolio_return": yr_p * 100.0,
                "benchmark_return": yr_b * 100.0,
                "active_return": (yr_p - yr_b) * 100.0,
                "portfolio_since_inception_cumulative": si_p * 100.0,
                "sml_since_inception_cumulative": si_b * 100.0,
            }
        )

    p_total = float((1.0 + perf["portfolio_return"]).prod() - 1.0)
    b_total = float((1.0 + perf["benchmark_return"]).prod() - 1.0)
    p_ann = _annualize_return(p_total, perf["date"].min(), perf["date"].max())
    b_ann = _annualize_return(b_total, perf["date"].min(), perf["date"].max())
    rows.append(
        {
            "period": "Since Inception",
            "portfolio_return": p_total * 100.0,
            "benchmark_return": b_total * 100.0,
            "active_return": (p_total - b_total) * 100.0,
            "portfolio_since_inception_cumulative": p_total * 100.0,
            "sml_since_inception_cumulative": b_total * 100.0,
        }
    )
    rows.append(
        {
            "period": "Since Inception (Annualized)",
            "portfolio_return": (p_ann * 100.0) if p_ann is not None else pd.NA,
            "benchmark_return": (b_ann * 100.0) if b_ann is not None else pd.NA,
            "active_return": ((p_ann - b_ann) * 100.0) if (p_ann is not None and b_ann is not None) else pd.NA,
            "portfolio_since_inception_cumulative": p_total * 100.0,
            "sml_since_inception_cumulative": b_total * 100.0,
        }
    )

    metrics = pd.DataFrame(rows)
    display = metrics.copy()
    pct_cols = [
        "portfolio_return",
        "benchmark_return",
        "active_return",
        "portfolio_since_inception_cumulative",
        "sml_since_inception_cumulative",
    ]
    for c in pct_cols:
        display[c] = pd.to_numeric(display[c], errors="coerce").map(lambda v: f"{v:,.2f}%" if pd.notna(v) else "")
    display = display.rename(
        columns={
            "period": "Period",
            "portfolio_return": "Portfolio Return",
            "benchmark_return": "SML Return",
            "active_return": "Active Return",
            "portfolio_since_inception_cumulative": "Portfolio Since Inception Cumulative",
            "sml_since_inception_cumulative": "SML Since Inception Cumulative",
        }
    )
    mask_bottom = display["Period"].isin(["Since Inception", "Since Inception (Annualized)"])
    display.loc[mask_bottom, "Portfolio Since Inception Cumulative"] = ""
    display.loc[mask_bottom, "SML Since Inception Cumulative"] = ""
    st.dataframe(display, use_container_width=True, hide_index=True)
    inception_year = int(inception.year)
    st.caption(f"* {inception_year} return starts at fund inception date ({inception.strftime('%Y-%m-%d')}), not Jan 1.")


def _first_non_blank(values: list[object]) -> object | None:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", "nat"}:
            return v
    return None


def _normalize_earnings_session(raw: object) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    if any(k in s for k in ["before", "bmo", "pre", "open"]):
        return "Pre-Market"
    if any(k in s for k in ["after", "amc", "post", "close"]):
        return "After-Market"
    if "during" in s:
        return "During Market"
    return str(raw)


@st.cache_data(show_spinner=False, ttl=3600)
def _compute_upcoming_earnings(repo: str, as_of_date: str, tickers: tuple[str, ...]) -> pd.DataFrame:
    cols = ["ticker", "company", "report_date", "days_until"]
    if not tickers:
        return pd.DataFrame(columns=cols)
    as_of = pd.to_datetime(as_of_date).normalize()
    repo_path = Path(repo)
    inputs_dir = repo_path / "inputs"
    alias_path = inputs_dir / "symbol_aliases.csv" if inputs_dir.exists() else None
    alias_map = _load_alias_map(alias_path)

    fields = [
        "NAME",
        "EXPECTED_REPORT_DT",
        "ANNOUNCEMENT_DT",
        "LATEST_ANNOUNCEMENT_DT",
    ]
    rows: list[dict] = []
    try:
        with BloombergClient(BloombergConfig()) as bbg:
            for t in tickers:
                cands = _bbg_security_candidates(t, alias_map)
                ref = bbg.get_reference(cands, fields)
                chosen: dict[str, object] = {}
                for sec in cands:
                    rec = ref.get(sec, {})
                    if rec:
                        chosen = rec
                        date_probe = _first_non_blank(
                            [
                                rec.get("EXPECTED_REPORT_DT"),
                                rec.get("EARNINGS_ANNOUNCEMENT_DT"),
                                rec.get("ANNOUNCEMENT_DT"),
                                rec.get("LATEST_ANNOUNCEMENT_DT"),
                            ]
                        )
                        if date_probe is not None:
                            break
                if not chosen:
                    continue
                d_raw = _first_non_blank(
                    [
                        chosen.get("EXPECTED_REPORT_DT"),
                        chosen.get("ANNOUNCEMENT_DT"),
                        chosen.get("LATEST_ANNOUNCEMENT_DT"),
                    ]
                )
                if d_raw is None:
                    continue
                d = pd.to_datetime(d_raw, errors="coerce")
                if pd.isna(d):
                    continue
                d = pd.to_datetime(d).normalize()
                if d < as_of:
                    continue
                rows.append(
                    {
                        "ticker": t,
                        "company": str(chosen.get("NAME", t)),
                        "report_date": d,
                        "days_until": int((d - as_of).days),
                    }
                )
    except Exception:
        return pd.DataFrame(columns=cols)

    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "report_date"], keep="first")
    out = out.sort_values(["report_date", "ticker"]).reset_index(drop=True)
    return out[cols]


@st.cache_data(show_spinner=False, ttl=60)
def _get_benchmark_1d_return(repo: str, as_of_date: str) -> tuple[str, float | None]:
    as_of = pd.to_datetime(as_of_date).normalize()
    try:
        with BloombergClient(BloombergConfig()) as bbg:
            px = bbg.get_historical_field({"SML": "SML Index"}, "PX_LAST", as_of - pd.Timedelta(days=14), as_of)
        if not px.empty:
            px["date"] = pd.to_datetime(px["date"]).dt.normalize()
            px = px.sort_values("date")
            px = px[(px["date"] <= as_of) & (px["value"].notna())]
            if len(px) >= 2:
                p1 = float(px["value"].iloc[-1])
                p0 = float(px["value"].iloc[-2])
                if p0 != 0:
                    return "SML 1D Return", (p1 / p0) - 1.0
    except Exception:
        pass

    # Offline/non-Bloomberg fallback: yfinance IJR (S&P SmallCap 600 ETF proxy).
    try:
        q = _fetch_live_yf_quotes(("IJR",), None)
        if not q.empty and "ret_1d" in q.columns:
            v = pd.to_numeric(q["ret_1d"], errors="coerce").dropna()
            if not v.empty:
                return "IJR 1D Return", float(v.iloc[-1])
    except Exception:
        pass

    try:
        bench_path = Path(repo) / "outputs" / "cache" / "benchmark_px.csv"
        if bench_path.exists():
            b = pd.read_csv(bench_path)
            b["date"] = pd.to_datetime(b.get("date"), errors="coerce").dt.normalize()
            b = b[b["date"] <= as_of].sort_values("date")
            if "benchmark_return" in b.columns:
                s = pd.to_numeric(b["benchmark_return"], errors="coerce").dropna()
                if not s.empty:
                    return "SML 1D Return", float(s.iloc[-1])
            if "tri" in b.columns:
                tri = pd.to_numeric(b["tri"], errors="coerce")
                ret = tri.pct_change().dropna()
                if not ret.empty:
                    return "SML 1D Return", float(ret.iloc[-1])
    except Exception:
        pass
    return "Benchmark 1D Return", None


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


def _compute_symbol_irr(sym_trades: pd.DataFrame, as_of: pd.Timestamp, terminal_value: float) -> float | None:
    if sym_trades.empty:
        return None
    cycle_start = _most_recent_open_cycle_start(sym_trades)
    if cycle_start is None:
        return None
    cycle = sym_trades[sym_trades["trade_date"] >= cycle_start].sort_values("trade_date")
    if cycle.empty:
        return None
    holding_days = int((as_of - pd.to_datetime(cycle_start).normalize()).days)
    if holding_days < 365:
        return None

    cfs: list[tuple[pd.Timestamp, float]] = []
    for _, r in cycle.iterrows():
        amt = float(r.get("amount", 0) or 0)
        txn = str(r.get("txn_type", ""))
        if txn in POSITION_ADD_TYPES:
            cfs.append((pd.to_datetime(r["trade_date"]).normalize(), -abs(amt)))
        else:
            cfs.append((pd.to_datetime(r["trade_date"]).normalize(), abs(amt)))
    cfs.append((as_of, float(terminal_value)))
    return _xirr(cfs)


def _return_cell_color(v: object) -> str:
    try:
        x = float(v)
    except Exception:
        return ""
    if pd.isna(x):
        return ""
    if x > 0:
        return "color: #1b7f3a; font-weight: 600;"
    if x < 0:
        return "color: #b42318; font-weight: 600;"
    return "color: #667085;"


def _live_flash_styles(df: pd.DataFrame, changed_cells: set[tuple[str, str]]) -> pd.DataFrame:
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for ticker, col in changed_cells:
        if ticker in styles.index and col in styles.columns:
            styles.at[ticker, col] = "background-color: #fff3bf;"
    return styles


@st.fragment(run_every=5)
def _render_live_holdings_fragment(repo: str, as_of_date: str) -> None:
    holdings, holdings_totals, changed_count = _compute_live_holdings(repo, as_of_date)
    if holdings is None or holdings.empty:
        return

    st.subheader("Live Holdings")
    h_view = holdings[
        [
            "ticker",
            "company",
            "today_price",
            "ret_1d",
            "ret_1w",
            "ret_1m",
            "ret_ytd",
            "market_value",
            "weight",
            "avg_price",
            "pnl_dollar",
            "pnl_pct",
            "first_buy_date",
            "irr",
        ]
    ].copy()
    h_view["first_buy_date"] = pd.to_datetime(h_view["first_buy_date"], errors="coerce")
    for col in ["ret_1d", "ret_1w", "ret_1m", "ret_ytd", "weight", "pnl_pct", "irr"]:
        h_view[col] = h_view[col] * 100.0

    h_display = h_view.rename(
        columns={
            "ticker": "Ticker",
            "company": "Company",
            "today_price": "Today Price",
            "ret_1d": "1D",
            "ret_1w": "1W",
            "ret_1m": "1M",
            "ret_ytd": "YTD",
            "market_value": "Market Value",
            "weight": "Weight",
            "avg_price": "Average Price",
            "pnl_dollar": "PnL ($)",
            "pnl_pct": "PnL (%)",
            "first_buy_date": "First Buy",
            "irr": "IRR",
        }
    )
    h_display = h_display.set_index("Ticker", drop=False)

    prev_key = f"live_quote_prev_{as_of_date}"
    prev = st.session_state.get(prev_key, {})
    current = {}
    changed_cells: set[tuple[str, str]] = set()
    for _, r in h_display.iterrows():
        t = str(r["Ticker"])
        current[t] = {
            "Today Price": pd.to_numeric(r["Today Price"], errors="coerce"),
            "1D": pd.to_numeric(r["1D"], errors="coerce"),
            "1W": pd.to_numeric(r["1W"], errors="coerce"),
            "1M": pd.to_numeric(r["1M"], errors="coerce"),
            "YTD": pd.to_numeric(r["YTD"], errors="coerce"),
        }
        p = prev.get(t)
        if isinstance(p, dict):
            for c in ["Today Price", "1D", "1W", "1M", "YTD"]:
                old_v = p.get(c)
                new_v = current[t][c]
                if pd.notna(old_v) and pd.notna(new_v) and abs(float(new_v) - float(old_v)) > 1e-9:
                    changed_cells.add((t, c))
    st.session_state[prev_key] = current

    h_styled = (
        h_display.style.format(
            {
                "Today Price": lambda x: f"${x:,.2f}" if pd.notna(x) else "NA",
                "1D": lambda x: f"{x:.2f}%" if pd.notna(x) else "NA",
                "1W": lambda x: f"{x:.2f}%" if pd.notna(x) else "NA",
                "1M": lambda x: f"{x:.2f}%" if pd.notna(x) else "NA",
                "YTD": lambda x: f"{x:.2f}%" if pd.notna(x) else "NA",
                "Market Value": lambda x: f"${x:,.0f}" if pd.notna(x) else "NA",
                "Weight": lambda x: f"{x:.2f}%" if pd.notna(x) else "NA",
                "Average Price": lambda x: f"${x:,.2f}" if pd.notna(x) else "NA",
                "PnL ($)": lambda x: f"${x:,.0f}" if pd.notna(x) else "NA",
                "PnL (%)": lambda x: f"{x:.2f}%" if pd.notna(x) else "NA",
                "First Buy": lambda x: pd.to_datetime(x).strftime("%Y-%m-%d") if pd.notna(x) else "NA",
                "IRR": lambda x: f"{x:.2f}%" if pd.notna(x) else "NA",
            }
        )
        .map(_return_cell_color, subset=["1D", "1W", "1M", "YTD"])
        .apply(lambda df: _live_flash_styles(df, changed_cells), axis=None)
    )
    st.dataframe(h_styled, use_container_width=True, hide_index=True)
    st.caption("Market data for Today Price / 1D / 1W / 1M is from yfinance and is typically delayed by ~15 minutes.")
    st.caption("IRR is only calculated for holdings held longer than a calendar year.")
    update_text = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S %Z")
    st.caption(f"Last quote update: {update_text} | Symbols updated this cycle: {changed_count}")

    total_pnl_pct = holdings_totals.get("pnl_pct")
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Total Market Value", f"${float(holdings_totals.get('market_value') or 0.0):,.0f}")
    tc2.metric("Total Cost Basis", f"${float(holdings_totals.get('cost_basis') or 0.0):,.0f}")
    tc3.metric("Total PnL ($)", f"${float(holdings_totals.get('pnl_dollar') or 0.0):,.0f}")
    tc4.metric("Total PnL (%)", f"{float(total_pnl_pct) * 100.0:,.2f}%" if total_pnl_pct is not None else "N/A")

    port_1d = None
    try:
        numer = (holdings["market_value"].fillna(0.0) * holdings["ret_1d"].fillna(0.0)).sum()
        denom = holdings["market_value"].fillna(0.0).sum()
        if denom != 0:
            port_1d = float(numer / denom)
    except Exception:
        port_1d = None
    bench_label, bench_1d = _get_benchmark_1d_return(repo, as_of_date)
    r1, r2 = st.columns(2)
    r1.metric("Portfolio 1D Return", f"{port_1d * 100.0:,.2f}%" if port_1d is not None else "N/A")
    r2.metric(bench_label, f"{bench_1d * 100.0:,.2f}%" if bench_1d is not None else "N/A")


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
def _compute_live_holdings_base(repo: str, as_of_date: str) -> tuple[pd.DataFrame, dict[str, float | None]]:
    repo_path = Path(repo)
    as_of = pd.to_datetime(as_of_date).normalize()
    trades_df, cashflows_df = load_transactions(str(repo_path / "Transactions.csv"))
    if trades_df.empty:
        empty = pd.DataFrame(columns=["ticker", "company", "shares", "avg_price", "today_price", "ret_1d", "ret_1w", "ret_1m", "ret_ytd", "market_value", "cost_basis", "weight", "pnl_dollar", "pnl_pct", "irr"])
        return empty, {"market_value": 0.0, "cost_basis": 0.0, "pnl_dollar": 0.0, "pnl_pct": None}

    trades = trades_df.copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"]).dt.normalize()
    trades = trades[trades["trade_date"] <= as_of].copy()
    if trades.empty:
        empty = pd.DataFrame(columns=["ticker", "company", "shares", "avg_price", "today_price", "ret_1d", "ret_1w", "ret_1m", "ret_ytd", "market_value", "cost_basis", "weight", "pnl_dollar", "pnl_pct", "irr"])
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
            ytd_start = pd.Timestamp(year=as_of.year, month=1, day=1)
            g_ytd = g[g["date"] >= ytd_start]
            p_ytd = float(g_ytd["price"].iloc[0]) if not g_ytd.empty else None

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
                    "ret_ytd": _ret(p0, p_ytd),
                }
            )
        ret_df = pd.DataFrame(ret_rows) if ret_rows else pd.DataFrame(columns=["ticker", "ret_1d", "ret_1w", "ret_1m", "ret_ytd"])
    else:
        px = pd.DataFrame(columns=["ticker", "today_price"])
        ret_df = pd.DataFrame(columns=["ticker", "ret_1d", "ret_1w", "ret_1m", "ret_ytd"])

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
            mv = float(out.loc[out["ticker"] == t, "market_value"].iloc[0]) if (out["ticker"] == t).any() else 0.0
            irr = _compute_symbol_irr(sym_trades, as_of, mv)
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
                "ret_ytd": pd.NA,
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
    return out[["ticker", "company", "shares", "first_buy_date", "avg_price", "today_price", "ret_1d", "ret_1w", "ret_1m", "ret_ytd", "market_value", "cost_basis", "weight", "pnl_dollar", "pnl_pct", "irr"]], totals


def _compute_live_holdings(repo: str, as_of_date: str) -> tuple[pd.DataFrame, dict[str, float | None], int]:
    out, totals = _compute_live_holdings_base(repo, as_of_date)
    if out.empty:
        return out, totals, 0

    as_of = pd.to_datetime(as_of_date).normalize()
    repo_path = Path(repo)
    inputs_dir = repo_path / "inputs"
    alias_path = inputs_dir / "symbol_aliases.csv" if inputs_dir.exists() else None

    live_universe = tuple(
        sorted(
            out.loc[(out["ticker"] != "CASH") & (out["shares"].fillna(0) > 0), "ticker"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )
    alias_csv = str(alias_path) if alias_path is not None and alias_path.exists() else None
    live_quotes = _fetch_live_yf_quotes(live_universe, alias_csv)
    if live_quotes.empty:
        return out, totals, 0

    old_px = out.set_index("ticker")["today_price"].copy()
    out = out.merge(
        live_quotes.rename(
            columns={
                "today_price": "today_price_live",
                "ret_1d": "ret_1d_live",
                "ret_1w": "ret_1w_live",
                "ret_1m": "ret_1m_live",
                "ret_ytd": "ret_ytd_live",
            }
        ),
        on="ticker",
        how="left",
    )
    out["today_price"] = out["today_price_live"].where(out["today_price_live"].notna(), out["today_price"])
    out["ret_1d"] = out["ret_1d_live"].where(out["ret_1d_live"].notna(), out["ret_1d"])
    out["ret_1w"] = out["ret_1w_live"].where(out["ret_1w_live"].notna(), out["ret_1w"])
    out["ret_1m"] = out["ret_1m_live"].where(out["ret_1m_live"].notna(), out["ret_1m"])
    out["ret_ytd"] = out["ret_ytd_live"].where(out["ret_ytd_live"].notna(), out.get("ret_ytd"))
    out = out.drop(columns=["today_price_live", "ret_1d_live", "ret_1w_live", "ret_1m_live", "ret_ytd_live"], errors="ignore")

    # Recompute valuation metrics from live prices.
    out["market_value"] = out["shares"] * out["today_price"].fillna(0.0)
    out.loc[out["ticker"] == "CASH", "market_value"] = out.loc[out["ticker"] == "CASH", "cost_basis"].fillna(0.0)
    out["pnl_dollar"] = out["market_value"] - out["cost_basis"].fillna(0.0)
    out["pnl_pct"] = out["pnl_dollar"] / out["cost_basis"].replace(0, pd.NA)
    total_mv = float(out["market_value"].sum())
    out["weight"] = out["market_value"] / total_mv if total_mv != 0 else 0.0
    out = out.sort_values("market_value", ascending=False)

    # Recompute IRR after live price overlay so IRR and PnL use the same terminal value.
    trades_df, _ = load_transactions(str(repo_path / "Transactions.csv"))
    trades_df["trade_date"] = pd.to_datetime(trades_df["trade_date"]).dt.normalize()
    trades_df = trades_df[trades_df["trade_date"] <= as_of].copy()
    for t in live_universe:
        sym_trades = trades_df[trades_df["symbol"] == t].copy()
        mv = float(out.loc[out["ticker"] == t, "market_value"].iloc[0]) if (out["ticker"] == t).any() else 0.0
        irr = _compute_symbol_irr(sym_trades, as_of, mv)
        out.loc[out["ticker"] == t, "irr"] = irr if irr is not None else pd.NA

    totals = {
        "market_value": float(out["market_value"].sum()),
        "cost_basis": float(out["cost_basis"].sum()),
        "pnl_dollar": float(out["pnl_dollar"].sum()),
        "pnl_pct": None,
    }
    if totals["cost_basis"] != 0:
        totals["pnl_pct"] = totals["pnl_dollar"] / totals["cost_basis"]

    changed = 0
    merged_old = out["ticker"].map(old_px).astype("float64")
    merged_new = pd.to_numeric(out["today_price"], errors="coerce")
    delta = (merged_new - merged_old).abs()
    changed = int((delta > 1e-9).sum())
    return out, totals, changed


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

    # Prefer cached mappings/dates so this works in hosted/offline mode.
    trading_dates = pd.DatetimeIndex([])
    cache_dir = repo_path / "outputs" / "cache"
    bench_cache = _load_if_exists(cache_dir / "benchmark_px.csv")
    if bench_cache is not None and not bench_cache.empty and "date" in bench_cache.columns:
        bc = bench_cache.copy()
        bc["date"] = pd.to_datetime(bc["date"], errors="coerce").dt.normalize()
        if "benchmark_return" in bc.columns:
            trading_dates = pd.DatetimeIndex(
                bc.loc[bc["benchmark_return"].notna(), "date"].dropna().unique()
            )
        elif "tri" in bc.columns:
            bc = bc.sort_values("date")
            bc["ret"] = pd.to_numeric(bc["tri"], errors="coerce").pct_change()
            trading_dates = pd.DatetimeIndex(bc.loc[bc["ret"].notna(), "date"].dropna().unique())
        elif "price" in bc.columns:
            bc = bc.sort_values("date")
            bc["ret"] = pd.to_numeric(bc["price"], errors="coerce").pct_change()
            trading_dates = pd.DatetimeIndex(bc.loc[bc["ret"].notna(), "date"].dropna().unique())
    if len(trading_dates) == 0:
        trading_dates = pd.DatetimeIndex(sorted(prices_df["date"].dropna().unique()))
    trading_dates = pd.DatetimeIndex(trading_dates)
    trading_dates = trading_dates[(trading_dates >= start) & (trading_dates <= end)]

    sector_by_symbol: dict[str, str] = {}
    sector_cache = _load_if_exists(cache_dir / "portfolio_sector_map.csv")
    if sector_cache is not None and not sector_cache.empty and {"symbol", "sector"}.issubset(set(sector_cache.columns)):
        sc = sector_cache.copy()
        sc["symbol"] = sc["symbol"].astype(str).str.strip()
        sc["sector"] = sc["sector"].astype(str).str.strip()
        sector_by_symbol = sc.drop_duplicates("symbol").set_index("symbol")["sector"].to_dict()

    missing = [s for s in symbols if s not in sector_by_symbol or not str(sector_by_symbol.get(s, "")).strip()]
    if missing:
        if "sector" in security.columns:
            sec_map = security[["symbol", "sector"]].dropna().drop_duplicates("symbol")
            for _, row in sec_map.iterrows():
                sym = str(row["symbol"]).strip()
                if sym in missing:
                    sector_by_symbol[sym] = str(row["sector"]).strip() or "Unknown"
        missing = [s for s in symbols if s not in sector_by_symbol or not str(sector_by_symbol.get(s, "")).strip()]

    if missing:
        try:
            with BloombergClient(BloombergConfig()) as bbg:
                alias_map = _load_alias_map(alias_path)
                resolved = _resolve_sector_by_symbol(bbg, missing, alias_map=alias_map)
            for sym, sec in resolved.items():
                if str(sec).strip():
                    sector_by_symbol[sym] = str(sec).strip()
        except Exception:
            pass
    for s in symbols:
        if not str(sector_by_symbol.get(s, "")).strip():
            sector_by_symbol[s] = "Unknown"
    split_events = (
        trades_df[trades_df["txn_type"].isin(POSITION_ADDITION_TYPES)][["trade_date", "symbol"]]
        .rename(columns={"trade_date": "date"})
        .copy()
    )
    panel = _portfolio_security_panel(
        positions_df,
        prices_df,
        nav_df,
        sector_by_symbol,
        trading_dates=trading_dates,
        split_events_df=split_events,
    )
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
            margin-top: -6px;
            margin-bottom: 8px;
        }
        .block-container {
            padding-top: 1.35rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            justify-content: flex-end;
            gap: 0.5rem;
        }
        .ima-title-row {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            margin-top: 6px;
        }
        .ima-inline-logo {
            width: 70px;
            height: auto;
            margin-top: 14px;
            margin-left: -14px;
            pointer-events: none;
        }
        .stDataFrame [data-testid="stDataFrameToolbar"] {
            display: none !important;
        }
        .stDataFrame button[aria-label*="column menu"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_path = repo / "logo.png"
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        st.markdown(
            (
                '<div class="ima-title-row">'
                '<h1 class="ima-header">IMA Terminal</h1>'
                f'<img class="ima-inline-logo" src="data:image/png;base64,{logo_b64}" alt="IMA logo" />'
                '</div>'
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<h1 class="ima-header">IMA Terminal</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="ima-subheader">UIUC Investment Management Academy</h3>', unsafe_allow_html=True)
    st.markdown(
        "- Live/intraday holdings monitor with auto-refresh\n"
        "- Upcoming earnings calendar for current positions\n"
        "- Historical performance vs benchmark\n"
        "- Brinson-style attribution and security decomposition"
    )
    bbg_ok, bbg_msg = _check_bloomberg_status()
    if bbg_ok:
        st.success(bbg_msg)
    else:
        st.warning(bbg_msg)
        st.info("Offline mode: live Bloomberg pulls are disabled. Displaying saved outputs and local fallback data where available.")
    last_updated = _outputs_last_updated(outputs)
    if last_updated is not None:
        st.caption(f"Loaded saved outputs last updated (America/Chicago): {last_updated.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    inception_date, today_date = _portfolio_date_bounds(repo)
    has_cache = _has_attribution_cache(repo)
    saved_start, saved_end = _outputs_date_window(outputs)
    default_start = max(pd.Timestamp("2025-01-01"), inception_date)
    if saved_start is not None:
        default_start = min(max(saved_start, inception_date), today_date)
    default_end = today_date
    if saved_end is not None:
        default_end = min(max(saved_end, inception_date), today_date)
    if default_start > default_end:
        default_start = inception_date

    start_key = "attr_start_date_input"
    end_key = "attr_end_date_input"
    range_key = "attr_quick_range"
    range_applied_key = "attr_quick_range_applied"
    if start_key not in st.session_state:
        st.session_state[start_key] = default_start.date()
    if end_key not in st.session_state:
        st.session_state[end_key] = default_end.date()
    if range_key not in st.session_state:
        st.session_state[range_key] = "YTD"

    st.session_state[start_key] = _clamp_date_to_bounds(
        st.session_state[start_key], inception_date, today_date, default_start
    ).date()
    st.session_state[end_key] = _clamp_date_to_bounds(
        st.session_state[end_key], inception_date, today_date, default_end
    ).date()

    tab_live, tab_performance, tab_attribution = st.tabs(["Live/Intraday", "Performance", "Attribution"])

    with tab_live:
        try:
            holdings_as_of_default = get_latest_market_date() if bbg_ok else None
        except Exception:
            holdings_as_of_default = None
        holdings_as_of_default = holdings_as_of_default or today_date
        holdings_as_of_default = min(max(pd.to_datetime(holdings_as_of_default).normalize(), inception_date), today_date)
        st.subheader("Live Data")
        holdings_as_of = st.date_input(
            "Holdings As Of",
            value=holdings_as_of_default.date(),
            min_value=inception_date.date(),
            max_value=today_date.date(),
            format="MM/DD/YYYY",
            key="holdings_asof_input",
        )
        holdings, _ = _compute_live_holdings_base(str(repo), str(holdings_as_of))
        if holdings is not None and not holdings.empty:
            _render_live_holdings_fragment(str(repo), str(holdings_as_of))

            st.subheader("Upcoming Events")
            earnings_tickers = tuple(
                sorted(
                    holdings.loc[
                        (holdings["ticker"] != "CASH") & (holdings["shares"].fillna(0) > 0),
                        "ticker",
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            )
            earnings = _compute_upcoming_earnings(str(repo), str(holdings_as_of), earnings_tickers)
            if earnings.empty:
                if bbg_ok:
                    st.caption("No upcoming earnings found for current holdings.")
                else:
                    st.caption("Upcoming earnings require Bloomberg connectivity in this runtime.")
            else:
                st.dataframe(
                    earnings,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "ticker": "Ticker",
                        "company": "Company",
                        "report_date": st.column_config.DateColumn("Earnings Date"),
                        "days_until": st.column_config.NumberColumn("Days Until", format="%d"),
                    },
                )

    with tab_performance:
        _render_performance_section(repo, outputs)

    with tab_attribution:
        st.subheader("Attribution")
        quick_options = ["YTD", "1Y", "2Y", "3Y", "5Y", "Since Inception"]
        quick = st.radio("Date Preset", options=quick_options, horizontal=True, key=range_key)
        if st.session_state.get(range_applied_key) != quick:
            preset_start = _period_start(today_date, quick, inception_date)
            st.session_state[start_key] = preset_start.date()
            st.session_state[end_key] = today_date.date()
            st.session_state[range_applied_key] = quick

        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input(
                "Start Date",
                min_value=inception_date.date(),
                max_value=today_date.date(),
                key=start_key,
                format="MM/DD/YYYY",
            )
        with c2:
            end = st.date_input(
                "End Date",
                min_value=inception_date.date(),
                max_value=today_date.date(),
                key=end_key,
                format="MM/DD/YYYY",
            )
        if pd.Timestamp(start) > pd.Timestamp(end):
            st.session_state[end_key] = start
            end = start

        unit = st.radio("Display Unit", options=["%", "bps"], horizontal=True, index=0)

        rc1, rc2 = st.columns([1, 2])
        with rc1:
            if st.button("Refresh Bloomberg Cache", disabled=not bbg_ok):
                cache_bar = st.progress(0, text="Starting cache refresh...")
                cache_step = st.empty()

                def _on_cache_progress(pct: int, msg: str) -> None:
                    pct = max(0, min(100, int(pct)))
                    cache_bar.progress(pct, text=msg)
                    cache_step.caption(f"Step: {msg}")

                refresh_bloomberg_cache(
                    repo,
                    start_date=inception_date,
                    end_date=today_date,
                    progress_callback=_on_cache_progress,
                )
                cache_bar.progress(100, text="Cache refresh complete")
                st.success("Bloomberg cache refreshed.")
                has_cache = True
        with rc2:
            st.caption("Use this first to fetch/update benchmark, sector, and portfolio market data. Date-range runs will reuse this cache.")

        run_disabled = not (bbg_ok or has_cache)
        if run_disabled:
            st.caption("Run Attribution is disabled: no Bloomberg connection and no local attribution cache found. Refresh cache on a Bloomberg-enabled machine first.")
        if st.button("Run Attribution", type="primary", disabled=run_disabled):
            progress_bar = st.progress(0, text="Starting attribution...")
            step_text = st.empty()

            def _on_progress(pct: int, msg: str) -> None:
                pct = max(0, min(100, int(pct)))
                progress_bar.progress(pct, text=msg)
                step_text.caption(f"Step: {msg}")

            try:
                run_attribution_for_window(
                    repo,
                    start_date=pd.Timestamp(start),
                    end_date=pd.Timestamp(end),
                    lookback_days=None,
                    progress_callback=_on_progress,
                )
                progress_bar.progress(100, text="Attribution complete")
                st.success("Attribution run complete.")
            except Exception:
                progress_bar.progress(100, text="Attribution failed")
                raise

        summary = _load_if_exists(outputs / "attribution_summary.csv")
        daily = _load_if_exists(outputs / "attribution_daily.csv")
        sector = _load_if_exists(outputs / "attribution_by_sector.csv")
        security = _load_if_exists(outputs / "attribution_by_security.csv")
        sec_decomp = None
        if security is not None:
            try:
                if daily is not None and not daily.empty and "date" in daily.columns:
                    d0 = pd.to_datetime(daily["date"]).min().strftime("%Y-%m-%d")
                    d1 = pd.to_datetime(daily["date"]).max().strftime("%Y-%m-%d")
                    sec_decomp = _compute_security_decomp(str(repo), d0, d1)
                else:
                    sec_decomp = _compute_security_decomp(str(repo), str(start), str(end))
            except Exception:
                sec_decomp = security.copy()
            if sec_decomp is not None and not sec_decomp.empty and "selection_effect" in sec_decomp.columns:
                sec_decomp = sec_decomp[sec_decomp["selection_effect"].abs() > 1e-12]
            if sec_decomp is not None and not sec_decomp.empty and "holding_days" in sec_decomp.columns:
                sec_decomp = sec_decomp[sec_decomp["holding_days"] > 0]

        if summary is not None:
            st.subheader("Summary")
            summary_scaled = summary.copy()
            summary_scaled["metric"] = (
                summary_scaled["metric"]
                .astype(str)
                .str.replace("_", " ", regex=False)
                .str.strip()
                .str.title()
            )
            if "value" in summary_scaled.columns:
                scale = 100.0 if unit == "%" else 10000.0
                summary_scaled["value"] = pd.to_numeric(summary_scaled["value"], errors="coerce") * scale
            st.dataframe(
                summary_scaled,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "metric": st.column_config.TextColumn("Metric"),
                    "value": st.column_config.NumberColumn("Value", format=_percent_unit_format(unit)),
                },
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
            sv = sv.sort_values("total_effect", ascending=False)
            sv_view = sv[["sector", "status", "alloc_effect", "select_effect", "total_effect"]]
            sv_scaled = _scaled_percent_like(sv_view, unit, skip_cols={"sector", "status"})
            st.dataframe(
                sv_scaled,
                use_container_width=True,
                hide_index=True,
                column_config=_effect_number_config(
                    sv_scaled,
                    unit,
                    skip_cols={"sector", "status"},
                ),
            )
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

        p2 = outputs / "attribution_cum_effects.png"
        if p2.exists():
            st.subheader("Cumulative Effects")
            st.image(str(p2))


if __name__ == "__main__":
    main()
