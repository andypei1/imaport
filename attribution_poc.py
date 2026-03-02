"""
Proof-of-concept attribution script.

What it does:
- Rebuilds portfolio NAV and flow-adjusted daily returns from Transactions.csv
- Pulls S&P 600 benchmark returns from Bloomberg (SML Index, PX_LAST)
- Builds a basic Brinson-style active return decomposition into:
  - allocation effect
  - selection effect
- Writes CSV outputs and simple PNG charts (no UI)

Notes:
- This is intentionally a POC. Benchmark sector weights/returns are approximated from
  current S&P 600 members with equal-weight aggregation by sector.
- Any gap between true benchmark active return and modeled active return is reported as residual.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

try:
    import blpapi  # type: ignore
except Exception:  # pragma: no cover - optional dependency in cloud/offline mode
    blpapi = None

from src.portfolio.cash import build_cash_ledger
from src.portfolio.load_transactions import load_transactions
from src.portfolio.nav import compute_daily_nav
from src.portfolio.positions import build_positions
from src.portfolio.prices import get_prices


@dataclass
class BloombergConfig:
    host: str = "localhost"
    port: int = 8194
    benchmark: str = "SML Index"


SML_SECTOR_INDEX_MAP: dict[str, str] = {
    "Communication Services": "S6TELS Index",
    "Consumer Discretionary": "S6COND Index",
    "Consumer Staples": "S6CONS Index",
    "Energy": "S6ENRS Index",
    "Financials": "S6FINL Index",
    "Health Care": "S6HLTH Index",
    "Industrials": "S6INDU Index",
    "Information Technology": "S6INFT Index",
    "Materials": "S6MATR Index",
    "Real Estate": "S6RLST Index",
    "Utilities": "S6UTIL Index",
}


def _safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}.new{path.suffix}")
        df.to_csv(alt, index=False)
        return alt


def _safe_savefig(fig, path: Path, dpi: int = 140) -> Path:
    try:
        fig.savefig(path, dpi=dpi)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}.new{path.suffix}")
        fig.savefig(alt, dpi=dpi)
        return alt


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
    bbg: "BloombergClient",
    symbols: list[str],
    field: str,
    alias_map: dict[str, str] | None = None,
) -> dict[str, str]:
    alias_map = alias_map or {}
    symbol_to_candidates = {s: _bbg_security_candidates(s, alias_map) for s in symbols}
    securities: list[str] = []
    for cands in symbol_to_candidates.values():
        for sec in cands:
            if sec not in securities:
                securities.append(sec)
    ref = bbg.get_reference(securities, [field]) if securities else {}
    out: dict[str, str] = {}
    for s in symbols:
        val = "Unknown"
        for sec in symbol_to_candidates.get(s, []):
            raw = ref.get(sec, {}).get(field)
            txt = str(raw).strip() if raw is not None else ""
            if txt and txt.lower() not in {"nan", "none"}:
                val = txt
                break
        out[s] = val
    return out


def _resolve_sector_by_symbol_with_debug(
    bbg: "BloombergClient",
    symbols: list[str],
    alias_map: dict[str, str] | None = None,
) -> tuple[dict[str, str], pd.DataFrame]:
    alias_map = alias_map or {}
    debug_rows: list[dict] = []
    out: dict[str, str] = {}
    for sym in symbols:
        cands = _bbg_security_candidates(sym, alias_map)
        ref = bbg.get_reference(cands, ["NAME", "GICS_SECTOR_NAME", "INDUSTRY_SECTOR", "ID_BB_GLOBAL"])
        chosen_sector = "Unknown"
        chosen_sec = ""
        chosen_field = ""
        for sec in cands:
            rec = ref.get(sec, {})
            gics = str(rec.get("GICS_SECTOR_NAME", "")).strip()
            ind = str(rec.get("INDUSTRY_SECTOR", "")).strip()
            name = str(rec.get("NAME", "")).strip()
            bbid = str(rec.get("ID_BB_GLOBAL", "")).strip()
            debug_rows.append(
                {
                    "symbol": sym,
                    "candidate_security": sec,
                    "name": name,
                    "gics_sector_name": gics,
                    "industry_sector": ind,
                    "id_bb_global": bbid,
                    "candidate_returned": bool(rec),
                }
            )
            if gics and gics.lower() not in {"nan", "none"}:
                chosen_sector = gics
                chosen_sec = sec
                chosen_field = "GICS_SECTOR_NAME"
                break
            if ind and ind.lower() not in {"nan", "none"}:
                chosen_sector = ind
                chosen_sec = sec
                chosen_field = "INDUSTRY_SECTOR"
                break
        out[sym] = chosen_sector
        debug_rows.append(
            {
                "symbol": sym,
                "candidate_security": "__chosen__",
                "name": chosen_sec,
                "gics_sector_name": chosen_sector,
                "industry_sector": chosen_field,
                "id_bb_global": "",
                "candidate_returned": chosen_sector != "Unknown",
            }
        )
    dbg = pd.DataFrame(debug_rows) if debug_rows else pd.DataFrame(
        columns=["symbol", "candidate_security", "name", "gics_sector_name", "industry_sector", "id_bb_global", "candidate_returned"]
    )
    return out, dbg


def _extend_positions_and_cash_to_date(
    positions_df: pd.DataFrame,
    cash_df: pd.DataFrame,
    target_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extend positions and cash through target_date by carrying latest values forward.
    External flow is set to 0 on added dates.
    """
    pos = positions_df.copy()
    cash = cash_df.copy()
    pos["date"] = pd.to_datetime(pos["date"]).dt.normalize()
    cash["date"] = pd.to_datetime(cash["date"]).dt.normalize()
    target_date = pd.to_datetime(target_date).normalize()

    # Extend positions.
    if not pos.empty:
        pos_max = pos["date"].max()
        if target_date > pos_max:
            extra_dates = pd.date_range(pos_max + pd.Timedelta(days=1), target_date, freq="D")
            last_by_symbol = pos.sort_values("date").groupby("symbol", as_index=False).tail(1)
            rows: list[dict] = []
            for d in extra_dates:
                for _, r in last_by_symbol.iterrows():
                    rows.append({"date": d, "symbol": r["symbol"], "shares": float(r["shares"])})
            if rows:
                pos = pd.concat([pos, pd.DataFrame(rows)], ignore_index=True)

    # Extend cash.
    if not cash.empty:
        cash_max = cash["date"].max()
        if target_date > cash_max:
            extra_dates = pd.date_range(cash_max + pd.Timedelta(days=1), target_date, freq="D")
            last_cash = float(cash.sort_values("date")["cash_balance"].iloc[-1])
            extra = pd.DataFrame({"date": extra_dates, "cash_balance": last_cash, "external_flow": 0.0})
            cash = pd.concat([cash, extra], ignore_index=True)

    return pos, cash


class BloombergClient:
    def __init__(self, cfg: BloombergConfig):
        self.cfg = cfg
        self._session: blpapi.Session | None = None

    def __enter__(self) -> "BloombergClient":
        if blpapi is None:
            raise RuntimeError("blpapi is not installed in this Python environment.")
        opts = blpapi.SessionOptions()
        opts.setServerHost(self.cfg.host)
        opts.setServerPort(self.cfg.port)
        session = blpapi.Session(opts)
        if not session.start():
            raise RuntimeError(f"Failed to start Bloomberg session at {self.cfg.host}:{self.cfg.port}")
        if not session.openService("//blp/refdata"):
            session.stop()
            raise RuntimeError("Failed to open Bloomberg service //blp/refdata")
        self._session = session
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._session is not None:
            self._session.stop()
            self._session = None

    @property
    def _svc(self):
        if self._session is None:
            raise RuntimeError("Bloomberg session not initialized")
        return self._session.getService("//blp/refdata")

    def _iter_response_messages(self, request) -> Iterable[blpapi.Message]:
        self._session.sendRequest(request)
        done = False
        while not done:
            ev = self._session.nextEvent()
            for msg in ev:
                yield msg
            if ev.eventType() == blpapi.Event.RESPONSE:
                done = True

    def get_historical_px_last(self, security_map: dict[str, str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Return DataFrame(date, symbol, price) for given symbol->Bloomberg security map."""
        if not security_map:
            return pd.DataFrame(columns=["date", "symbol", "price"])

        sec_to_symbols: dict[str, list[str]] = {}
        for sym, sec in security_map.items():
            sec_to_symbols.setdefault(sec, []).append(sym)

        rows: list[dict] = []
        # Chunk securities to keep response size predictable.
        securities = list(dict.fromkeys(security_map.values()))
        chunk_size = 150
        for i in range(0, len(securities), chunk_size):
            chunk = securities[i : i + chunk_size]
            req = self._svc.createRequest("HistoricalDataRequest")
            for sec in chunk:
                req.getElement("securities").appendValue(sec)
            req.getElement("fields").appendValue("PX_LAST")
            req.set("startDate", start.strftime("%Y%m%d"))
            req.set("endDate", end.strftime("%Y%m%d"))
            req.set("periodicitySelection", "DAILY")

            for msg in self._iter_response_messages(req):
                if msg.messageType() != blpapi.Name("HistoricalDataResponse"):
                    continue
                sec_data = msg.getElement("securityData")
                sec = sec_data.getElementAsString("security")
                if sec_data.hasElement("securityError"):
                    continue
                if not sec_data.hasElement("fieldData"):
                    continue
                field_data = sec_data.getElement("fieldData")
                for j in range(field_data.numValues()):
                    item = field_data.getValueAsElement(j)
                    if not item.hasElement("date") or not item.hasElement("PX_LAST"):
                        continue
                    d = pd.to_datetime(item.getElementAsDatetime("date")).normalize()
                    p = float(item.getElementAsFloat("PX_LAST"))
                    for sym in sec_to_symbols.get(sec, [sec]):
                        rows.append({"date": d, "symbol": sym, "price": p})

        if not rows:
            return pd.DataFrame(columns=["date", "symbol", "price"])
        out = pd.DataFrame(rows).drop_duplicates(subset=["date", "symbol"], keep="last")
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        return out.sort_values(["date", "symbol"]).reset_index(drop=True)

    def get_historical_field(
        self,
        security_map: dict[str, str],
        field: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return DataFrame(date, symbol, value) for any historical field."""
        if not security_map:
            return pd.DataFrame(columns=["date", "symbol", "value"])

        sec_to_symbols: dict[str, list[str]] = {}
        for sym, sec in security_map.items():
            sec_to_symbols.setdefault(sec, []).append(sym)

        rows: list[dict] = []
        securities = list(dict.fromkeys(security_map.values()))
        chunk_size = 150
        for i in range(0, len(securities), chunk_size):
            chunk = securities[i : i + chunk_size]
            req = self._svc.createRequest("HistoricalDataRequest")
            for sec in chunk:
                req.getElement("securities").appendValue(sec)
            req.getElement("fields").appendValue(field)
            req.set("startDate", start.strftime("%Y%m%d"))
            req.set("endDate", end.strftime("%Y%m%d"))
            req.set("periodicitySelection", "DAILY")

            for msg in self._iter_response_messages(req):
                if msg.messageType() != blpapi.Name("HistoricalDataResponse"):
                    continue
                sec_data = msg.getElement("securityData")
                sec = sec_data.getElementAsString("security")
                if sec_data.hasElement("securityError"):
                    continue
                if not sec_data.hasElement("fieldData"):
                    continue
                field_data = sec_data.getElement("fieldData")
                for j in range(field_data.numValues()):
                    item = field_data.getValueAsElement(j)
                    if not item.hasElement("date") or not item.hasElement(field):
                        continue
                    d = pd.to_datetime(item.getElementAsDatetime("date")).normalize()
                    try:
                        v = float(item.getElementAsFloat(field))
                    except Exception:
                        continue
                    for sym in sec_to_symbols.get(sec, [sec]):
                        rows.append({"date": d, "symbol": sym, "value": v})

        if not rows:
            return pd.DataFrame(columns=["date", "symbol", "value"])
        out = pd.DataFrame(rows).drop_duplicates(subset=["date", "symbol"], keep="last")
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        return out.sort_values(["date", "symbol"]).reset_index(drop=True)

    def get_reference(self, securities: list[str], fields: list[str]) -> dict[str, dict[str, object]]:
        """ReferenceDataRequest helper. Returns dict[security][field] = scalar or raw element string."""
        if not securities:
            return {}
        req = self._svc.createRequest("ReferenceDataRequest")
        for sec in securities:
            req.append("securities", sec)
        for fld in fields:
            req.append("fields", fld)

        out: dict[str, dict[str, object]] = {}
        for msg in self._iter_response_messages(req):
            if msg.messageType() != blpapi.Name("ReferenceDataResponse"):
                continue
            sec_array = msg.getElement("securityData")
            for i in range(sec_array.numValues()):
                sec_data = sec_array.getValueAsElement(i)
                sec = sec_data.getElementAsString("security")
                if sec_data.hasElement("securityError"):
                    continue
                field_data = sec_data.getElement("fieldData")
                rec: dict[str, object] = {}
                for fld in fields:
                    if not field_data.hasElement(fld):
                        continue
                    el = field_data.getElement(fld)
                    if el.isArray() or el.isComplexType():
                        rec[fld] = str(el)
                    else:
                        rec[fld] = el.getValue()
                out[sec] = rec
        return out

    def get_index_members(self, index_security: str) -> list[str]:
        req = self._svc.createRequest("ReferenceDataRequest")
        req.append("securities", index_security)
        req.append("fields", "INDX_MEMBERS")

        members: list[str] = []
        for msg in self._iter_response_messages(req):
            if msg.messageType() != blpapi.Name("ReferenceDataResponse"):
                continue
            sec_array = msg.getElement("securityData")
            for i in range(sec_array.numValues()):
                sec_data = sec_array.getValueAsElement(i)
                if sec_data.hasElement("securityError"):
                    continue
                fd = sec_data.getElement("fieldData")
                if not fd.hasElement("INDX_MEMBERS"):
                    continue
                arr = fd.getElement("INDX_MEMBERS")
                for j in range(arr.numValues()):
                    row = arr.getValueAsElement(j)
                    # Field label in Bloomberg output: "Member Ticker and Exchange Code"
                    for k in range(row.numElements()):
                        candidate = row.getElement(k)
                        if "Ticker" in str(candidate.name()) and candidate.isNull() is False:
                            members.append(str(candidate.getValue()))
                            break
        return sorted(set(members))


def _portfolio_security_panel(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    sector_by_symbol: dict[str, str],
    trading_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    pos = positions_df.copy()
    px = prices_df.copy()
    nav = nav_df.copy()

    pos["date"] = pd.to_datetime(pos["date"]).dt.normalize()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    nav["date"] = pd.to_datetime(nav["date"]).dt.normalize()

    px_wide = px.pivot_table(index="date", columns="symbol", values="price")
    px_wide = px_wide.sort_index().ffill()
    if trading_dates is not None:
        # Restrict to target trading dates before return calc to avoid holiday/weekend artifacts.
        px_wide = px_wide.reindex(pd.DatetimeIndex(sorted(pd.to_datetime(trading_dates).unique()))).ffill()
    ret_wide = px_wide.pct_change()

    pos_wide = pos.pivot_table(index="date", columns="symbol", values="shares", aggfunc="last")
    pos_wide = pos_wide.sort_index().ffill().fillna(0)

    common_dates = pos_wide.index.intersection(px_wide.index)
    if trading_dates is not None:
        common_dates = common_dates.intersection(trading_dates)
    pos_wide = pos_wide.reindex(common_dates)
    px_wide = px_wide.reindex(common_dates)
    ret_wide = ret_wide.reindex(common_dates)

    mv_wide = pos_wide.clip(lower=0) * px_wide
    mv_prev = mv_wide.shift(1)

    nav = nav.set_index("date").reindex(common_dates)
    nav_prev = nav["nav"].shift(1)

    panel = mv_prev.stack().rename("mv_prev").reset_index()
    panel.columns = ["date", "symbol", "mv_prev"]
    panel = panel.merge(
        ret_wide.stack().rename("sec_ret").reset_index().rename(columns={"level_0": "date", "level_1": "symbol"}),
        on=["date", "symbol"],
        how="left",
    )
    panel["sector"] = panel["symbol"].map(sector_by_symbol).fillna("Unknown")
    panel = panel.merge(
        nav_prev.rename_axis("date").reset_index(name="nav_prev"),
        on="date",
        how="left",
    )
    panel["w_prev"] = panel["mv_prev"] / panel["nav_prev"]
    panel["w_prev"] = panel["w_prev"].fillna(0.0)
    panel["sec_ret"] = panel["sec_ret"].fillna(0.0)

    # Drop first date (no lag return) and rows with no prior capital.
    panel = panel[panel["nav_prev"].notna() & (panel["nav_prev"] != 0)].copy()
    return panel


def _aggregate_portfolio_sector(panel: pd.DataFrame, nav_df: pd.DataFrame) -> pd.DataFrame:
    # Sector weights are sums of security weights.
    w = panel.groupby(["date", "sector"], as_index=False)["w_prev"].sum().rename(columns={"w_prev": "w_p"})

    # Sector return is weighted avg of security returns within sector.
    contrib = panel.copy()
    contrib["num"] = contrib["w_prev"] * contrib["sec_ret"]
    numer = contrib.groupby(["date", "sector"], as_index=False)["num"].sum()
    out = w.merge(numer, on=["date", "sector"], how="left")
    out["r_p"] = out["num"] / out["w_p"].replace(0, pd.NA)
    out["r_p"] = out["r_p"].fillna(0.0)
    out = out.drop(columns=["num"])

    # Add cash as a sector with zero return to capture allocation drag/benefit.
    nav = nav_df.copy()
    nav["date"] = pd.to_datetime(nav["date"]).dt.normalize()
    nav = (
        nav.sort_values("date")
        .set_index("date")
        .reindex(sorted(panel["date"].unique()))
        .ffill()
        .reset_index()
        .rename(columns={"index": "date"})
    )
    nav["nav_prev"] = nav["nav"].shift(1)
    nav["cash_prev"] = nav["cash_balance"].shift(1)
    cash = nav[["date", "cash_prev", "nav_prev"]].copy()
    cash["sector"] = "Cash"
    cash["w_p"] = (cash["cash_prev"] / cash["nav_prev"]).fillna(0.0)
    cash["r_p"] = 0.0
    cash = cash[["date", "sector", "w_p", "r_p"]]

    out = pd.concat([out, cash], ignore_index=True)
    out = out.groupby(["date", "sector"], as_index=False).agg({"w_p": "sum", "r_p": "last"})
    return out


def _compute_benchmark_sector_model(
    bench_members: list[str],
    bench_member_sectors: dict[str, str],
    bench_member_prices: pd.DataFrame,
) -> pd.DataFrame:
    if bench_member_prices.empty:
        return pd.DataFrame(columns=["date", "sector", "w_b", "r_b_s", "r_b_model"])

    px = bench_member_prices.copy()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px = px.sort_values(["symbol", "date"])
    px["ret"] = px.groupby("symbol")["price"].pct_change()
    px["sector"] = px["symbol"].map(bench_member_sectors).fillna("Unknown")
    px = px.dropna(subset=["ret"])

    # Equal-weight benchmark model from available members each day.
    cnt_sector = px.groupby(["date", "sector"], as_index=False)["symbol"].nunique().rename(columns={"symbol": "n_sec"})
    cnt_total = cnt_sector.groupby("date", as_index=False)["n_sec"].sum().rename(columns={"n_sec": "n_total"})
    model = cnt_sector.merge(cnt_total, on="date", how="left")
    model["w_b"] = model["n_sec"] / model["n_total"].replace(0, pd.NA)
    model["w_b"] = model["w_b"].fillna(0.0)

    r_sector = px.groupby(["date", "sector"], as_index=False)["ret"].mean().rename(columns={"ret": "r_b_s"})
    model = model.merge(r_sector, on=["date", "sector"], how="left")
    model["r_b_s"] = model["r_b_s"].fillna(0.0)

    rb = (
        model.assign(_x=model["w_b"] * model["r_b_s"])
        .groupby("date", as_index=False)["_x"]
        .sum()
        .rename(columns={"_x": "r_b_model"})
    )
    model = model.merge(rb, on="date", how="left")
    return model[["date", "sector", "w_b", "r_b_s", "r_b_model"]]


def _compute_benchmark_sector_model_from_sector_indices(
    bbg: BloombergClient,
    benchmark_security: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build benchmark sector weights/returns from Bloomberg S&P 600 sector indices.
    - Sector return from sector index PX_LAST.
    - Sector weight from CUR_MKT_CAP / benchmark CUR_MKT_CAP.
    """
    sec_map = dict(SML_SECTOR_INDEX_MAP)
    px = bbg.get_historical_field(sec_map, "PX_LAST", start, end).rename(columns={"value": "px"})
    cap = bbg.get_historical_field(sec_map, "CUR_MKT_CAP", start, end).rename(columns={"value": "cap"})
    total_cap = bbg.get_historical_field({"_total_": benchmark_security}, "CUR_MKT_CAP", start, end).rename(
        columns={"value": "total_cap"}
    )

    if px.empty or cap.empty or total_cap.empty:
        return pd.DataFrame(columns=["date", "sector", "w_b", "r_b_s", "r_b_model"])

    td = pd.DatetimeIndex(sorted(pd.to_datetime(trading_dates).unique()))
    rows: list[pd.DataFrame] = []
    for sector in sec_map.keys():
        p = px[px["symbol"] == sector][["date", "px"]].set_index("date").reindex(td).ffill()
        c = cap[cap["symbol"] == sector][["date", "cap"]].set_index("date").reindex(td).ffill()
        g = p.join(c, how="left")
        g["sector"] = sector
        g = g.reset_index().rename(columns={"index": "date"})
        rows.append(g)
    sector_df = pd.concat(rows, ignore_index=True)
    sector_df["r_b_s"] = sector_df.groupby("sector")["px"].pct_change().fillna(0.0)

    total = total_cap[["date", "total_cap"]].set_index("date").reindex(td).ffill().reset_index().rename(
        columns={"index": "date"}
    )
    sector_df = sector_df.merge(total, on="date", how="left")
    sector_df["w_b"] = sector_df["cap"] / sector_df["total_cap"].replace(0, pd.NA)
    sector_df["w_b"] = sector_df["w_b"].fillna(0.0)

    rb = (
        sector_df.assign(_x=sector_df["w_b"] * sector_df["r_b_s"])
        .groupby("date", as_index=False)["_x"]
        .sum()
        .rename(columns={"_x": "r_b_model"})
    )
    out = sector_df.merge(rb, on="date", how="left")
    return out[["date", "sector", "w_b", "r_b_s", "r_b_model"]]


def run_attribution_poc(
    repo: Path,
    lookback_days: int | None = None,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> None:
    transactions_path = repo / "Transactions.csv"
    if not transactions_path.exists():
        raise FileNotFoundError(f"Transactions file not found: {transactions_path}")

    trades_df, cashflows_df = load_transactions(str(transactions_path))
    positions_df = build_positions(trades_df)
    cash_df = build_cash_ledger(trades_df, cashflows_df)

    raw_min = pd.to_datetime(cash_df["date"]).min().normalize()
    raw_max = pd.to_datetime(cash_df["date"]).max().normalize()
    date_min = raw_min
    date_max = pd.to_datetime(end_date).normalize() if end_date is not None else raw_max
    if lookback_days is not None and lookback_days > 0:
        date_min = max(date_min, date_max - pd.Timedelta(days=lookback_days))
    if start_date is not None:
        date_min = max(raw_min, pd.to_datetime(start_date).normalize())
    if date_min > date_max:
        raise ValueError(f"Invalid window after clipping to transaction range: {date_min.date()} to {date_max.date()}")

    # Extend analysis window to latest benchmark trading date available in Bloomberg.
    cfg = BloombergConfig()
    with BloombergClient(cfg) as bbg:
        bench_probe = bbg.get_historical_px_last(
            {cfg.benchmark: cfg.benchmark},
            date_max - pd.Timedelta(days=14),
            pd.Timestamp.today().normalize(),
        )
    if not bench_probe.empty:
        bench_max = pd.to_datetime(bench_probe["date"]).max().normalize()
        if end_date is None and bench_max > date_max:
            date_max = bench_max
        if end_date is not None:
            date_max = min(date_max, bench_max)

    symbols = sorted(positions_df["symbol"].unique().tolist())
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

    # Limit working set to window and extend to latest benchmark date.
    positions_df = positions_df[pd.to_datetime(positions_df["date"]).dt.normalize().between(date_min, date_max)].copy()
    cash_df = cash_df[pd.to_datetime(cash_df["date"]).dt.normalize().between(date_min, date_max)].copy()
    positions_df, cash_df = _extend_positions_and_cash_to_date(positions_df, cash_df, date_max)
    nav_df = compute_daily_nav(positions_df, cash_df, prices_df)

    out_dir = repo / "outputs"
    out_dir.mkdir(exist_ok=True)

    fetch_start = date_min - pd.Timedelta(days=10)
    with BloombergClient(cfg) as bbg:
        # Benchmark (true S&P 600 index return).
        bench_px = bbg.get_historical_px_last({cfg.benchmark: cfg.benchmark}, fetch_start, date_max)
        bench_px = bench_px.rename(columns={"symbol": "benchmark"}).sort_values("date")
        bench_px["benchmark_return"] = bench_px["price"].pct_change()
        benchmark_dates = pd.DatetimeIndex(
            bench_px.loc[bench_px["benchmark_return"].notna(), "date"].sort_values().unique()
        )
        benchmark_dates = benchmark_dates[(benchmark_dates >= date_min) & (benchmark_dates <= date_max)]

        # Portfolio sectors.
        alias_map = _load_alias_map(alias_path)
        sector_by_symbol, sector_debug = _resolve_sector_by_symbol_with_debug(
            bbg,
            symbols,
            alias_map=alias_map,
        )

        panel = _portfolio_security_panel(
            positions_df,
            prices_df,
            nav_df,
            sector_by_symbol,
            trading_dates=benchmark_dates,
        )
        p_sec = _aggregate_portfolio_sector(panel, nav_df)

        # Benchmark sector model from Bloomberg S&P 600 sector indices (returns + cap weights).
        b_sec = _compute_benchmark_sector_model_from_sector_indices(
            bbg,
            cfg.benchmark,
            fetch_start,
            date_max,
            benchmark_dates,
        )

    _safe_to_csv(sector_debug, out_dir / "sector_resolution_debug.csv")

    # Build portfolio return on benchmark trading calendar to avoid holiday mismatches.
    nav_ts = nav_df.copy()
    nav_ts["date"] = pd.to_datetime(nav_ts["date"]).dt.normalize()
    nav_ts = nav_ts.sort_values("date")
    nav_ts["external_cum"] = nav_ts["external_flow"].fillna(0.0).cumsum()
    nav_idx = nav_ts.set_index("date")

    nav_at_t = nav_idx["nav"].reindex(benchmark_dates, method="ffill")
    ext_at_t = nav_idx["external_cum"].reindex(benchmark_dates, method="ffill").fillna(0.0)
    nav_prev = nav_at_t.shift(1)
    ext_prev = ext_at_t.shift(1).fillna(0.0)
    flow_interval = ext_at_t - ext_prev
    port_return = (nav_at_t - nav_prev - flow_interval) / nav_prev

    daily = pd.DataFrame(
        {
            "date": benchmark_dates,
            "daily_return": port_return.values,
            "external_flow": flow_interval.values,
        }
    )
    daily = daily.merge(bench_px[["date", "benchmark_return"]], on="date", how="left")
    daily = daily[daily["benchmark_return"].notna()].copy()

    # Brinson-like decomposition against sector model benchmark.
    decomp = p_sec.merge(b_sec, on=["date", "sector"], how="outer")
    decomp[["w_p", "r_p", "w_b", "r_b_s"]] = decomp[["w_p", "r_p", "w_b", "r_b_s"]].fillna(0.0)
    # Brinson-Fachler daily decomposition.
    rb = (
        decomp.assign(_x=decomp["w_b"] * decomp["r_b_s"])
        .groupby("date", as_index=False)["_x"]
        .sum()
        .rename(columns={"_x": "r_b"})
    )
    decomp = decomp.merge(rb, on="date", how="left")
    decomp["alloc_effect"] = (decomp["w_p"] - decomp["w_b"]) * (decomp["r_b_s"] - decomp["r_b"])
    decomp["select_pure_effect"] = decomp["w_b"] * (decomp["r_p"] - decomp["r_b_s"])
    decomp["interaction_effect"] = (decomp["w_p"] - decomp["w_b"]) * (decomp["r_p"] - decomp["r_b_s"])
    # Fold interaction into selection so Allocation + Selection = Active (Bloomberg-style 2-way view).
    decomp["select_effect"] = decomp["select_pure_effect"] + decomp["interaction_effect"]

    by_day = decomp.groupby("date", as_index=False)[["alloc_effect", "select_effect"]].sum()
    by_day = by_day.merge(
        b_sec.groupby("date", as_index=False)["r_b_model"].first(), on="date", how="left"
    )
    p_model = (
        p_sec.assign(_x=p_sec["w_p"] * p_sec["r_p"])
        .groupby("date", as_index=False)["_x"]
        .sum()
        .rename(columns={"_x": "r_p_model"})
    )
    by_day = by_day.merge(p_model, on="date", how="left")
    daily = daily.merge(by_day, on="date", how="left")

    daily[["alloc_effect", "select_effect", "r_b_model", "r_p_model"]] = daily[
        ["alloc_effect", "select_effect", "r_b_model", "r_p_model"]
    ].fillna(0.0)
    daily["active_return"] = daily["daily_return"] - daily["benchmark_return"]
    daily["active_model"] = daily["alloc_effect"] + daily["select_effect"]
    daily["model_gap"] = daily["active_return"] - daily["active_model"]

    # Cumulative compounding.
    daily["cum_portfolio"] = (1.0 + daily["daily_return"].fillna(0.0)).cumprod() - 1.0
    daily["cum_benchmark"] = (1.0 + daily["benchmark_return"].fillna(0.0)).cumprod() - 1.0
    daily["cum_active"] = (1.0 + daily["cum_portfolio"]) / (1.0 + daily["cum_benchmark"]) - 1.0
    daily["cum_model_portfolio"] = (1.0 + daily["r_p_model"].fillna(0.0)).cumprod() - 1.0
    daily["cum_model_benchmark"] = (1.0 + daily["r_b_model"].fillna(0.0)).cumprod() - 1.0
    daily["cum_active_model"] = (1.0 + daily["cum_model_portfolio"]) / (1.0 + daily["cum_model_benchmark"]) - 1.0
    daily["cum_alloc"] = daily["alloc_effect"].fillna(0.0).cumsum()
    daily["cum_select"] = daily["select_effect"].fillna(0.0).cumsum()
    daily["cum_model_gap"] = daily["model_gap"].fillna(0.0).cumsum()

    alloc_total = float(daily["alloc_effect"].fillna(0.0).sum())
    select_total = float(daily["select_effect"].fillna(0.0).sum())
    interaction_total = float(decomp["interaction_effect"].fillna(0.0).sum())
    selection_pure_total = float(decomp["select_pure_effect"].fillna(0.0).sum())
    model_active_total = float(daily["active_model"].fillna(0.0).sum())
    model_gap_total = float(daily["model_gap"].fillna(0.0).sum())
    summary = pd.DataFrame(
        {
            "metric": [
                "total_portfolio_return",
                "total_benchmark_return",
                "total_active_return",
                "model_active_contribution_sum",
                "allocation_effect",
                "selection_effect",
                "selection_pure_effect",
                "interaction_effect",
                "model_gap_sum",
                "model_active_return",
                "true_vs_model_active_return_gap",
            ],
            "value": [
                float(daily["cum_portfolio"].iloc[-1]),
                float(daily["cum_benchmark"].iloc[-1]),
                float(daily["cum_active"].iloc[-1]),
                model_active_total,
                alloc_total,
                select_total,
                selection_pure_total,
                interaction_total,
                model_gap_total,
                float(daily["cum_active_model"].iloc[-1]),
                float(daily["cum_active"].iloc[-1] - daily["cum_active_model"].iloc[-1]),
            ],
        }
    )

    sector_totals = decomp.groupby("sector", as_index=False)[["alloc_effect", "select_effect"]].sum()
    sector_totals["total_effect"] = sector_totals["alloc_effect"] + sector_totals["select_effect"]
    sector_totals = sector_totals.sort_values("total_effect", ascending=False)
    # Security-level selection effect (sums to sector selection totals).
    sec_sel = panel.merge(
        b_sec[["date", "sector", "r_b_s"]],
        on=["date", "sector"],
        how="left",
    )
    sec_sel["r_b_s"] = sec_sel["r_b_s"].fillna(0.0)
    sec_sel["selection_effect"] = sec_sel["w_prev"] * (sec_sel["sec_ret"] - sec_sel["r_b_s"])
    security_totals = (
        sec_sel.groupby(["symbol", "sector"], as_index=False)["selection_effect"]
        .sum()
        .sort_values("selection_effect", ascending=False)
    )

    daily_path = out_dir / "attribution_daily.csv"
    summary_path = out_dir / "attribution_summary.csv"
    sector_path = out_dir / "attribution_by_sector.csv"
    security_path = out_dir / "attribution_by_security.csv"
    daily_written = _safe_to_csv(daily, daily_path)
    summary_written = _safe_to_csv(summary, summary_path)
    sector_written = _safe_to_csv(sector_totals, sector_path)
    security_written = _safe_to_csv(security_totals, security_path)

    # Charts.
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(daily["date"], daily["cum_portfolio"], label="Portfolio")
    plt.plot(daily["date"], daily["cum_benchmark"], label="S&P 600 (SML Index)")
    plt.plot(daily["date"], daily["cum_active"], label="Active")
    plt.title("Cumulative Return (Flow-Adjusted Portfolio vs S&P 600)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig1_written = _safe_savefig(fig1, out_dir / "attribution_cum_returns.png", dpi=140)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(daily["date"], daily["cum_alloc"], label="Allocation")
    plt.plot(daily["date"], daily["cum_select"], label="Selection")
    plt.plot(daily["date"], daily["cum_model_gap"], label="Gap to True Active")
    plt.title("Cumulative Active Attribution (POC, contribution sum)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig2_written = _safe_savefig(fig2, out_dir / "attribution_cum_effects.png", dpi=140)
    plt.close(fig2)

    print(f"Wrote {daily_written} ({len(daily)} rows)")
    print(f"Wrote {summary_written} ({len(summary)} rows)")
    print(f"Wrote {sector_written} ({len(sector_totals)} rows)")
    print(f"Wrote {security_written} ({len(security_totals)} rows)")
    print(f"Wrote {fig1_written}")
    print(f"Wrote {fig2_written}")
    print("Summary:")
    for _, r in summary.iterrows():
        print(f"  {r['metric']}: {r['value']:.4%}")


def run_attribution_for_window(
    repo: Path,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    lookback_days: int | None = None,
) -> None:
    """
    Run attribution with an optional explicit date window.
    If start/end are omitted, defaults to full history up to latest benchmark date.
    """
    transactions_path = repo / "Transactions.csv"
    if not transactions_path.exists():
        raise FileNotFoundError(f"Transactions file not found: {transactions_path}")

    trades_df, cashflows_df = load_transactions(str(transactions_path))
    if trades_df.empty and cashflows_df.empty:
        raise RuntimeError("No transactions loaded.")

    cash_df = build_cash_ledger(trades_df, cashflows_df)
    raw_min = pd.to_datetime(cash_df["date"]).min().normalize()
    raw_max = pd.to_datetime(cash_df["date"]).max().normalize()

    s = pd.to_datetime(start_date).normalize() if start_date is not None else None
    e = pd.to_datetime(end_date).normalize() if end_date is not None else None
    if s is not None and e is not None and s > e:
        raise ValueError(f"start_date must be <= end_date (got {s.date()} > {e.date()})")

    # Determine benchmark latest date so windows remain market-aligned.
    cfg = BloombergConfig()
    with BloombergClient(cfg) as bbg:
        probe = bbg.get_historical_px_last(
            {cfg.benchmark: cfg.benchmark},
            raw_max - pd.Timedelta(days=14),
            pd.Timestamp.today().normalize(),
        )
    bench_max = pd.to_datetime(probe["date"]).max().normalize() if not probe.empty else raw_max

    date_min = raw_min
    date_max = max(raw_max, bench_max)
    if lookback_days is not None and lookback_days > 0:
        date_min = max(raw_min, date_max - pd.Timedelta(days=lookback_days))
    if s is not None:
        date_min = max(raw_min, s)
    if e is not None:
        date_max = min(date_max, e, bench_max)

    if date_min > date_max:
        raise ValueError(
            f"Requested window has no data after capping to available dates: {date_min.date()} to {date_max.date()}"
        )

    # Delegate by temporarily constraining lookback behavior through explicit dates.
    # We run the same pipeline, but with a pre-trimmed view using date_min/date_max.
    run_attribution_poc(repo, lookback_days=None, start_date=date_min, end_date=date_max)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run portfolio attribution POC.")
    parser.add_argument("--lookback-days", type=int, default=None, help="Trailing window in days.")
    parser.add_argument("--start-date", type=str, default=None, help="Window start (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="Window end (YYYY-MM-DD).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    start_date = pd.to_datetime(args.start_date).normalize() if args.start_date else None
    end_date = pd.to_datetime(args.end_date).normalize() if args.end_date else None
    run_attribution_for_window(
        repo_root,
        start_date=start_date,
        end_date=end_date,
        lookback_days=args.lookback_days,
    )
