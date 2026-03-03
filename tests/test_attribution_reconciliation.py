from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from attribution_poc import (
    _aggregate_portfolio_sector,
    _extend_positions_and_cash_to_date,
    _portfolio_security_panel,
    compute_brinson_bhb_daily,
)
from src.portfolio.cash import build_cash_ledger
from src.portfolio.load_transactions import POSITION_ADDITION_TYPES, load_transactions
from src.portfolio.nav import compute_daily_nav
from src.portfolio.positions import build_positions


def test_bhb_synthetic_two_sector_expected_values() -> None:
    date = pd.Timestamp("2025-01-02")
    p_sec = pd.DataFrame(
        {
            "date": [date, date],
            "sector": ["A", "B"],
            "w_p": [0.6, 0.4],
            "r_p": [0.10, 0.05],
        }
    )
    b_sec = pd.DataFrame(
        {
            "date": [date, date],
            "sector": ["A", "B"],
            "w_b": [0.5, 0.5],
            "r_b_s": [0.08, 0.04],
        }
    )

    decomp, by_day = compute_brinson_bhb_daily(p_sec, b_sec)
    _ = decomp
    day = by_day.iloc[0]

    assert day["alloc_effect"] == pytest.approx(0.004)
    assert day["select_pure_effect"] == pytest.approx(0.015)
    assert day["interaction_effect"] == pytest.approx(0.001)
    assert day["active_return_model"] == pytest.approx(0.02)
    assert day["active_model"] == pytest.approx(0.02)
    assert abs(float(day["recon_error"])) < 1e-12


def test_real_pipeline_reconciles_on_cached_window() -> None:
    repo = Path(__file__).resolve().parents[1]
    tx_path = repo / "Transactions.csv"
    px_path = repo / "outputs" / "cache" / "portfolio_prices.csv"
    bsec_path = repo / "outputs" / "cache" / "benchmark_sector_model.csv"
    smap_path = repo / "outputs" / "cache" / "portfolio_sector_map.csv"

    if not (tx_path.exists() and px_path.exists() and bsec_path.exists() and smap_path.exists()):
        pytest.skip("Required local data/cache files are not available")

    trades_df, cashflows_df = load_transactions(str(tx_path))
    positions_df = build_positions(trades_df)
    cash_df = build_cash_ledger(trades_df, cashflows_df)

    price_cache = pd.read_csv(px_path)
    price_cache["date"] = pd.to_datetime(price_cache["date"]).dt.normalize()
    b_sec = pd.read_csv(bsec_path)
    b_sec["date"] = pd.to_datetime(b_sec["date"]).dt.normalize()

    if b_sec.empty:
        pytest.skip("benchmark sector cache is empty")

    end = pd.to_datetime(b_sec["date"]).max().normalize()
    start = max(pd.to_datetime(cash_df["date"]).min().normalize(), end - pd.Timedelta(days=31))

    symbols = sorted(positions_df["symbol"].unique().tolist())
    prices_df = price_cache[
        price_cache["symbol"].isin(symbols)
        & (price_cache["date"] >= (start - pd.Timedelta(days=40)))
        & (price_cache["date"] <= end)
    ][["date", "symbol", "price"]].copy()

    positions_df = positions_df[pd.to_datetime(positions_df["date"]).dt.normalize().between(start, end)].copy()
    cash_df = cash_df[pd.to_datetime(cash_df["date"]).dt.normalize().between(start, end)].copy()
    positions_df, cash_df = _extend_positions_and_cash_to_date(positions_df, cash_df, end)

    nav_df = compute_daily_nav(positions_df, cash_df, prices_df)

    benchmark_dates = pd.DatetimeIndex(
        sorted(b_sec[b_sec["date"].between(start, end)]["date"].drop_duplicates().tolist())
    )
    if benchmark_dates.empty:
        pytest.skip("No overlapping benchmark dates in selected test window")

    smap_cache = pd.read_csv(smap_path)
    sector_by_symbol = (
        smap_cache[smap_cache["symbol"].isin(symbols)]
        .drop_duplicates("symbol", keep="last")
        .set_index("symbol")["sector"]
        .astype(str)
        .to_dict()
    )

    split_events = trades_df[trades_df["txn_type"].isin(POSITION_ADDITION_TYPES)][["trade_date", "symbol"]].rename(
        columns={"trade_date": "date"}
    )

    panel = _portfolio_security_panel(
        positions_df,
        prices_df,
        nav_df,
        sector_by_symbol,
        trading_dates=benchmark_dates,
        split_events_df=split_events,
    )
    assert not panel.empty

    p_sec = _aggregate_portfolio_sector(panel, nav_df)
    b_sec_window = b_sec[b_sec["date"].between(start, end)][["date", "sector", "w_b", "r_b_s"]].copy()

    _, by_day = compute_brinson_bhb_daily(p_sec, b_sec_window)
    assert not by_day.empty
    assert float(by_day["recon_error"].abs().max()) < 1e-10


def test_unmapped_tickers_go_to_other_and_reconcile() -> None:
    dates = pd.to_datetime(["2025-02-03", "2025-02-04"])
    positions_df = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "symbol": ["AAA", "ZZZ", "AAA", "ZZZ"],
            "shares": [1.0, 1.0, 1.0, 1.0],
        }
    )
    prices_df = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "symbol": ["AAA", "ZZZ", "AAA", "ZZZ"],
            "price": [100.0, 100.0, 110.0, 110.0],
        }
    )
    nav_df = pd.DataFrame(
        {
            "date": dates,
            "nav": [200.0, 220.0],
            "securities_value": [200.0, 220.0],
            "cash_balance": [0.0, 0.0],
            "daily_return": [None, 0.10],
            "external_flow": [0.0, 0.0],
        }
    )

    panel = _portfolio_security_panel(
        positions_df=positions_df,
        prices_df=prices_df,
        nav_df=nav_df,
        sector_by_symbol={"AAA": "Tech"},
        trading_dates=pd.DatetimeIndex(dates),
        split_events_df=None,
    )

    assert "Other" in set(panel["sector"])

    p_sec = _aggregate_portfolio_sector(panel, nav_df)
    b_sec = pd.DataFrame(
        {
            "date": [dates[1], dates[1]],
            "sector": ["Tech", "Other"],
            "w_b": [0.6, 0.4],
            "r_b_s": [0.08, 0.12],
        }
    )

    _, by_day = compute_brinson_bhb_daily(p_sec, b_sec)
    assert float(by_day["recon_error"].abs().max()) < 1e-12
