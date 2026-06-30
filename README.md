# JF_copper

Streamlit dashboard for **gold (GC)** and **copper (HG)** futures & options.

Pick a **commodity**, a **contract** (rolling continuous tenor `M0…M+11`, or a
fixed outright by expiry), and a **bar timeframe** (Daily / Weekly / Monthly),
then explore four tabs:

| Tab | Shows |
|-----|-------|
| 📈 **Price (OHLC)** | Candlesticks + volume + 20-bar MA, log axis option. |
| 📊 **Volatility** | Realised vol (up to 3 windows), ATM implied vol, N-bar ATR, IV−RV spread. |
| 🔢 **Open Interest** | Options OI by strike (calls vs puts), OI across the curve, total OI over time. |
| 🙂 **Skew & Smile** | Per-strike IV smile (Black-76 inverted from settles) + 25Δ risk-reversal / butterfly over time. |

## Run

```bash
pip install -r requirements.txt          # or: uv venv && uv pip install -r requirements.txt
streamlit run app.py
```

## Data

Daily snapshots under `data/<commodity>/` (sourced from the Databento ingest in
the sibling `Vol_dashboard_mock`; no API key needed to run):

| File | Used for |
|------|----------|
| `<c>_tenor_ohlc.parquet`     | Rolling / continuous OHLC (`rank` 0=front) |
| `<c>_outright_ohlc.parquet`  | Fixed / outright OHLC (`expiry_ym`) |
| `<c>_atm_iv_ts.parquet`      | ATM implied vol per tenor / expiry |
| `<c>_skew_ts.parquet`        | 25Δ risk-reversal / butterfly per tenor |
| `<c>_frontmonth_oi.parquet`  | Front-month chain OI + settle + F (smile inversion) |
| `<c>_front3_oi.parquet`      | Front-3 expiries chain OI by strike |
| `<c>_hist_chain_oi.parquet`  | Historical options OI (date·expiry·strike·right) |

History ~2018-12 → 2026-06. Intraday (1m/5m) is intentionally **not** committed
(too large for GitHub); a Databento ingest script is the planned source.

### Roadmap
- Short-dated **weekly / daily option series** (needs a fresh Databento pull).
- Intraday timeframes (1m–1h) via an ingest/regenerate script.