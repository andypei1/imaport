"""
Fetch raw (unadjusted) close prices via yfinance for NAV valuation.
Using unadjusted close with actual share counts gives correct value across stock splits.
Supports symbol aliases (e.g. MOG A -> MOG.A) and manual/delisted price file.
When no market price exists (e.g. delisted), uses last trade price (entry or sale) as fallback.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def _load_symbol_aliases(path: Optional[Path]) -> dict[str, str]:
    """Load symbol -> yfinance_ticker map from CSV. Columns: symbol, yfinance_ticker."""
    if path is None or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "symbol" not in df.columns or "yfinance_ticker" not in df.columns:
        return {}
    return df.set_index("symbol")["yfinance_ticker"].astype(str).to_dict()


def _build_last_trade_price_fallback(
    trades_df: pd.DataFrame,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    For each (date, symbol) in range, compute price = last trade (buy or sale) on or before that date.
    Used when yfinance/manual have no data (e.g. delisted). Sales update the price for NAV.
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


def get_prices(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    alias_path: Optional[Path] = None,
    manual_prices_path: Optional[Path] = None,
    trades_for_fallback: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Download raw (unadjusted) close prices for the given symbols and date range.
    Unadjusted close ensures NAV = actual_shares x actual_price on each date (correct across splits).
    Uses alias_path to map BlackDiamond symbols to yfinance tickers; output keeps original symbols.
    Fills missing (date, symbol) from manual_prices_path if provided.
    When still missing (e.g. delisted), uses last trade price (entry or sale) as fallback so
    price is unchanged until we have a sale; sales update the price for NAV.

    Returns:
        DataFrame with columns: date, symbol, price.
    """
    if not symbols:
        return pd.DataFrame(columns=["date", "symbol", "price"])

    aliases = _load_symbol_aliases(alias_path)
    symbol_to_yf = {s: aliases.get(s, s) for s in symbols}
    yf_tickers = list(dict.fromkeys(symbol_to_yf.values()))

    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    raw = yf.download(
        yf_tickers,
        start=start_str,
        end=end_str,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=False,
    )
    if raw.empty:
        out = pd.DataFrame(columns=["date", "symbol", "price"])
    else:
        if len(yf_tickers) == 1:
            yf_sym = yf_tickers[0]
            close = raw["Close"].copy() if "Close" in raw.columns else raw.iloc[:, 3]
            out = close.reset_index()
            out.columns = ["date", "price"]
            out["symbol"] = [s for s in symbols if symbol_to_yf[s] == yf_sym][0]
        else:
            if isinstance(raw.columns, pd.MultiIndex):
                close_df = raw.xs("Close", axis=1, level=1)
            else:
                close_df = raw[[c for c in raw.columns if "Close" in str(c)]].copy()
                close_df.columns = [c.replace(".Close", "") for c in close_df.columns]
            close_df = close_df.reset_index()
            close_df = close_df.rename(columns={close_df.columns[0]: "date"})
            long = close_df.melt(id_vars=["date"], var_name="yf_ticker", value_name="price")
            long = long.dropna(subset=["price"])
            yf_to_originals = {}
            for orig, yf_tick in symbol_to_yf.items():
                yf_to_originals.setdefault(yf_tick, []).append(orig)
            rows = []
            for _, r in long.iterrows():
                for orig in yf_to_originals.get(r["yf_ticker"], [r["yf_ticker"]]):
                    rows.append({"date": r["date"], "symbol": orig, "price": r["price"]})
            out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "symbol", "price"])

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

    # Fallback: where we still have no price, use last trade price (entry or sale) so price doesn't change until we sell.
    if trades_for_fallback is not None and not trades_for_fallback.empty:
        fallback = _build_last_trade_price_fallback(trades_for_fallback, symbols, start, end)
        if not fallback.empty:
            # Prefer existing out; fill gaps with fallback (drop_duplicates keep="first" keeps out rows).
            out = pd.concat([out, fallback], ignore_index=True)
            out = out.drop_duplicates(subset=["date", "symbol"], keep="first")
            out = out.sort_values(["date", "symbol"]).reset_index(drop=True)

    return out if out.empty else out.sort_values(["date", "symbol"]).reset_index(drop=True)
