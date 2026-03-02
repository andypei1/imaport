# Minimal Portfolio Accounting

Converts periodic **BlackDiamond** transaction exports into a correct daily portfolio NAV and return series, separating investment performance from external cash flows (deposits and withdrawals).

No web app, no AI, no database - just correct portfolio accounting.

## Requirements

- Python 3.12+
- `pandas`, `blpapi`
- Bloomberg Terminal/Desktop API session available on `localhost:8194`

```bash
pip install -r requirements.txt
```

## Usage

Place your BlackDiamond export CSV as `Transactions.csv` in the repo root, then:

```bash
python main.py
```

Output is written to `outputs/nav.csv`, `outputs/holdings.csv` (date, symbol, shares - only non-zero holdings), and `outputs/negative_shares.csv` when any negative positions exist.

The NAV pipeline extends to the latest Bloomberg market date (if newer than the last transaction date) by carrying latest holdings and cash forward with zero external flow on added dates.

**Re-running after fixing the trade log:** Edit `Transactions.csv` (e.g. remove duplicates or bad rows, fix negatives), save the file, then run `python main.py` again. The pipeline always reads from `Transactions.csv` and overwrites the outputs.

## Pipeline

1. **Load & classify** - Read CSV; normalize columns (`trade_date`, `symbol`, `txn_type`, `units`, `price`, `amount`). Split into:
   - **Trades**: Buy/Sale of investable securities (excludes cash-equivalent symbols FEDXX, SWEEP0206, CASH).
   - **Cash flows**: Dividends, interest, deposits, withdrawals, transfers; and Buy/Sale of cash-equivalent symbols.

2. **Positions** - Cumulative shares per symbol through time; one row per (date, symbol). Negative share warnings if data is inconsistent.

3. **Cash ledger** - Daily cash balance and external flow from trade impacts (Buy: cash down, Sale: cash up) plus dividends, interest, deposits, withdrawals, transfers.

4. **Prices** - Bloomberg `PX_LAST` historical prices for the symbols and date range.

5. **NAV** - For each day: `securities_value = sum(shares x price)`; `portfolio_nav = securities_value + cash_balance`.

6. **Returns** - Daily time-weighted return excluding external flows:
   `return_t = (NAV_t - NAV_{t-1} - external_flow_t) / NAV_{t-1}`.

## CSV format

Expected columns (names are normalized from common BlackDiamond exports):

- **trade_date** (e.g. Date)
- **symbol** (e.g. Symbol)
- **txn_type** (e.g. Type): Buy, Sale, Dividend, Interest, Cash Deposit, Cash Withdrawal, Transfer In, Transfer Out
- **units**, **price**, **amount** (numeric; commas stripped)

Cash-equivalent symbols `FEDXX`, `SWEEP0206`, `CASH` are treated as cash movements only, not positions.

## Optional inputs

You can add `inputs/symbol_aliases.csv` to override how symbols map to Bloomberg securities.

Expected columns:

- `symbol`
- `bloomberg_ticker` (preferred) or `yfinance_ticker` (legacy name, still supported)

If no alias is provided, each symbol defaults to `<symbol> US Equity`.

## Attribution POC

Run:

```bash
py -3.12 attribution_poc.py
```

Default behavior uses the full available portfolio history.

Optional date window controls:

```bash
py -3.12 attribution_poc.py --lookback-days 365
py -3.12 attribution_poc.py --start-date 2022-01-01 --end-date 2023-12-31
```

This creates:

- `outputs/attribution_daily.csv`
- `outputs/attribution_summary.csv`
- `outputs/attribution_by_sector.csv`
- `outputs/attribution_cum_returns.png`
- `outputs/attribution_cum_effects.png`

POC methodology:

- Portfolio daily return is flow-adjusted (`daily_return` from NAV pipeline).
- Benchmark is S&P 600 (`SML Index`) using Bloomberg `PX_LAST`.
- Allocation vs selection is a basic sector-level Brinson-style decomposition using:
  - portfolio sector mapping from `GICS_SECTOR_NAME`
  - benchmark sector model from current `SML Index` members (`INDX_MEMBERS`) with equal-weight aggregation
- Because benchmark sector history and trade-timing effects are simplified, a model gap is expected.

## Local UI (Optional)

You can run a local Streamlit UI to test custom windows interactively:

```bash
streamlit run attribution_ui.py
```
