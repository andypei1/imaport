# Minimal Portfolio Accounting

Converts periodic **BlackDiamond** transaction exports into a correct daily portfolio NAV and return series, separating investment performance from external cash flows (deposits and withdrawals).

No web app, no AI, no database—just correct portfolio accounting.

## Requirements

- Python 3.10+
- `pandas`, `yfinance`

```bash
pip install -r requirements.txt
```

## Usage

Place your BlackDiamond export CSV as `Transactions.csv` in the repo root, then:

```bash
python main.py
```

Output is written to `outputs/nav.csv`, `outputs/holdings.csv` (date, symbol, shares — only non-zero holdings), and `outputs/negative_shares.csv` when any negative positions exist.

**Re-running after fixing the trade log:** Edit `Transactions.csv` (e.g. remove duplicates or bad rows, fix negatives), save the file, then run `python main.py` again. The pipeline always reads from `Transactions.csv` and overwrites the outputs.

## Pipeline

1. **Load & classify** — Read CSV; normalize columns (`trade_date`, `symbol`, `txn_type`, `units`, `price`, `amount`). Split into:
   - **Trades**: Buy/Sale of investable securities (excludes cash-equivalent symbols FEDXX, SWEEP0206, CASH).
   - **Cash flows**: Dividends, interest, deposits, withdrawals, transfers; and Buy/Sale of cash-equivalent symbols.

2. **Positions** — Cumulative shares per symbol through time; one row per (date, symbol). Negative share warnings if data is inconsistent.

3. **Cash ledger** — Daily cash balance and external flow from trade impacts (Buy: cash down, Sale: cash up) plus dividends, interest, deposits, withdrawals, transfers.

4. **Prices** — yfinance adjusted close for the symbols and date range.

5. **NAV** — For each day: `securities_value = sum(shares × price)`; `portfolio_nav = securities_value + cash_balance`.

6. **Returns** — Daily time-weighted return excluding external flows:
   `return_t = (NAV_t − NAV_{t−1} − external_flow_t) / NAV_{t−1}`.

## CSV format

Expected columns (names are normalized from common BlackDiamond exports):

- **trade_date** (e.g. Date)
- **symbol** (e.g. Symbol)
- **txn_type** (e.g. Type): Buy, Sale, Dividend, Interest, Cash Deposit, Cash Withdrawal, Transfer In, Transfer Out
- **units**, **price**, **amount** (numeric; commas stripped)

Cash-equivalent symbols `FEDXX`, `SWEEP0206`, `CASH` are treated as cash movements only, not positions.
