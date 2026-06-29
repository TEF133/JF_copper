# JF_copper

Streamlit dashboard for **gold** and **copper** futures.

## Step 1 — OHLC chart (current)

Pick a contract and plot its daily candlestick chart:

- **Commodity** — Gold (GC, COMEX) or Copper (HG, COMEX).
- **Contract type**
  - **Rolling (continuous)** — stitched continuous tenor. `M0` = front month,
    `M+n` = n months out.
  - **Fixed (outright)** — a single dated contract by expiry, e.g. `2026-08`.
- Date-range slider, optional volume panel, 20-day MA, and log price axis.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

Daily OHLC parquet snapshots live under `data/<commodity>/`:

| File | Used for | Key |
|------|----------|-----|
| `<c>_tenor_ohlc.parquet`    | Rolling / continuous | `rank` (0 = front) |
| `<c>_outright_ohlc.parquet` | Fixed / outright     | `expiry_ym` (YYYY-MM) |
| `<c>_futures_ohlcv.parquet` | Continuous front (rank 0), reference | — |

History: ~2018-12 → 2026-06. Snapshots are sourced from the Databento ingest
in the sibling `Vol_dashboard_mock` project; no API key is needed to run this app.