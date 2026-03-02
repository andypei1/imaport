"""
Fetch close prices via Bloomberg Desktop API (blpapi) for NAV valuation.
Supports symbol aliases and manual fallback prices for exceptional cases.
When Bloomberg/manual still has gaps, uses last trade price fallback.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import blpapi  # type: ignore
except Exception:  # pragma: no cover - optional dependency in cloud/offline mode
    blpapi = None


def _load_symbol_aliases(path: Optional[Path]) -> dict[str, str]:
    """
    Load symbol -> Bloomberg security map from CSV.
    Preferred columns: symbol, bloomberg_ticker
    Backward compatible columns: symbol, yfinance_ticker
    """
    if path is None or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "symbol" not in df.columns:
        return {}
    ticker_col = None
    if "bloomberg_ticker" in df.columns:
        ticker_col = "bloomberg_ticker"
    elif "yfinance_ticker" in df.columns:
        ticker_col = "yfinance_ticker"
    if ticker_col is None:
        return {}
    return df.set_index("symbol")[ticker_col].astype(str).to_dict()


def _build_last_trade_price_fallback(
    trades_df: pd.DataFrame,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    For each (date, symbol) in range, compute price = last trade (buy or sale) on or before that date.
    Used when Bloomberg/manual have no data (e.g. delisted). Sales update the price for NAV.
    """
    if trades_df.empty or "trade_date" not in trades_df.columns or "symbol" not in trades_df.columns or "price" not in trades_df.columns:
        return pd.DataFrame(columns=["date", "symbol", "price"])

    trades = trades_df[["trade_date", "symbol", "price"]].copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"]).dt.normalize()
    trades = trades.dropna(subset=["trade_date", "symbol", "price"])
    trades = trades[trades["symbol"].isin(symbols)]

    all_dates = pd.date_range(start, end, freq="D")
    rows: list[dict] = []

    for sym in symbols:
        t = trades[trades["symbol"] == sym].sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        if t.empty:
            continue
        s = t.set_index("trade_date")["price"]
        s = s.reindex(all_dates).ffill()
        s = s.dropna()
        if s.empty:
            continue
        for d, p in s.items():
            rows.append({"date": d, "symbol": sym, "price": float(p)})

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "price"])
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out


def _fetch_bloomberg_prices(
    symbol_to_security: dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    host: str = "localhost",
    port: int = 8194,
) -> pd.DataFrame:
    """
    Fetch Bloomberg daily PX_LAST using HistoricalDataRequest.
    Returns DataFrame(date, symbol, price). Symbols with no data are omitted.
    """
    if not symbol_to_security:
        return pd.DataFrame(columns=["date", "symbol", "price"])
    if blpapi is None:
        return pd.DataFrame(columns=["date", "symbol", "price"])

    session_options = blpapi.SessionOptions()
    session_options.setServerHost(host)
    session_options.setServerPort(port)
    session = blpapi.Session(session_options)

    if not session.start():
        raise RuntimeError(f"Failed to start Bloomberg session at {host}:{port}")
    if not session.openService("//blp/refdata"):
        session.stop()
        raise RuntimeError("Failed to open Bloomberg service //blp/refdata")

    service = session.getService("//blp/refdata")
    request = service.createRequest("HistoricalDataRequest")
    for sec in dict.fromkeys(symbol_to_security.values()):
        request.getElement("securities").appendValue(sec)
    request.getElement("fields").appendValue("PX_LAST")
    request.set("startDate", start.strftime("%Y%m%d"))
    request.set("endDate", end.strftime("%Y%m%d"))
    request.set("periodicitySelection", "DAILY")

    session.sendRequest(request)

    sec_to_symbols: dict[str, list[str]] = {}
    for sym, sec in symbol_to_security.items():
        sec_to_symbols.setdefault(sec, []).append(sym)

    rows: list[dict] = []
    done = False
    while not done:
        event = session.nextEvent()
        for message in event:
            if message.messageType() != blpapi.Name("HistoricalDataResponse"):
                continue
            security_data = message.getElement("securityData")
            security = security_data.getElementAsString("security")
            if security_data.hasElement("securityError"):
                continue
            if not security_data.hasElement("fieldData"):
                continue
            field_data = security_data.getElement("fieldData")
            for i in range(field_data.numValues()):
                item = field_data.getValueAsElement(i)
                if not item.hasElement("date") or not item.hasElement("PX_LAST"):
                    continue
                date_value = pd.to_datetime(item.getElementAsDatetime("date")).normalize()
                price_value = float(item.getElementAsFloat("PX_LAST"))
                for symbol in sec_to_symbols.get(security, [security]):
                    rows.append({"date": date_value, "symbol": symbol, "price": price_value})
        if event.eventType() == blpapi.Event.RESPONSE:
            done = True

    session.stop()
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "price"])
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def get_latest_market_date(
    probe_security: str = "SML Index",
    lookback_days: int = 14,
    host: str = "localhost",
    port: int = 8194,
) -> Optional[pd.Timestamp]:
    """
    Return latest available trading date from Bloomberg for a probe security.
    Returns None if no data is available.
    """
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=lookback_days)
    probe = _fetch_bloomberg_prices({"_probe_": probe_security}, start, end, host=host, port=port)
    if probe.empty:
        return None
    return pd.to_datetime(probe["date"]).max().normalize()


def get_prices(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    alias_path: Optional[Path] = None,
    manual_prices_path: Optional[Path] = None,
    trades_for_fallback: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Download close prices for the given symbols and date range from Bloomberg.
    Uses alias_path to map BlackDiamond symbols to Bloomberg securities; output keeps original symbols.
    Fills missing (date, symbol) from manual_prices_path if provided.
    When still missing, uses last trade price fallback so price is unchanged until a sale updates it.

    Returns:
        DataFrame with columns: date, symbol, price.
    """
    if not symbols:
        return pd.DataFrame(columns=["date", "symbol", "price"])

    aliases = _load_symbol_aliases(alias_path)
    symbol_to_security = {s: aliases.get(s, f"{s} US Equity") for s in symbols}
    out = _fetch_bloomberg_prices(symbol_to_security, start, end)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out = out.dropna(subset=["price"])

    # Merge manual prices (delisted / user-provided): fill or overwrite (date, symbol).
    if manual_prices_path is not None and Path(manual_prices_path).exists():
        manual = pd.read_csv(manual_prices_path)
        manual.columns = manual.columns.str.strip().str.lower().str.replace(" ", "_")
        if "date" in manual.columns and "symbol" in manual.columns and "price" in manual.columns:
            manual["date"] = pd.to_datetime(manual["date"]).dt.normalize()
            manual = manual.dropna(subset=["date", "symbol", "price"])
            manual = manual[
                manual["symbol"].isin(symbols) & (manual["date"] >= start) & (manual["date"] <= end)
            ]
            if not manual.empty:
                out = pd.concat([out, manual[["date", "symbol", "price"]]], ignore_index=True)
                out = out.drop_duplicates(subset=["date", "symbol"], keep="last")
                out = out.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Fallback: where we still have no price, use last trade price so price doesn't change until we sell.
    if trades_for_fallback is not None and not trades_for_fallback.empty:
        fallback = _build_last_trade_price_fallback(trades_for_fallback, symbols, start, end)
        if not fallback.empty:
            # Prefer existing out; fill gaps with fallback (drop_duplicates keep="first" keeps out rows).
            out = pd.concat([out, fallback], ignore_index=True)
            out = out.drop_duplicates(subset=["date", "symbol"], keep="first")
            out = out.sort_values(["date", "symbol"]).reset_index(drop=True)

    return out if out.empty else out.sort_values(["date", "symbol"]).reset_index(drop=True)
