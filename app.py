"""
JF_copper — Gold & Copper futures dashboard
===========================================
Step 1: pick a rolling (continuous) or fixed (outright) contract and plot its
daily OHLC candlestick chart.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data_loader as dl

st.set_page_config(page_title="JF Copper · Gold & Copper", page_icon="🟡", layout="wide")

UP, DOWN = "#1A6B3A", "#8B1A1A"  # candle colours

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("Contract")

commodity = st.sidebar.selectbox("Commodity", list(dl.COMMODITIES))
meta = dl.COMMODITIES[commodity]
code, unit = meta["code"], meta["unit"]

contract_type = st.sidebar.radio("Contract type", ["Rolling (continuous)", "Fixed (outright)"])

if contract_type.startswith("Rolling"):
    tenors = dl.available_tenors(code)
    rank = st.sidebar.selectbox(
        "Tenor", tenors, format_func=dl.tenor_label,
        help="Stitched continuous series. M0 = front month, M+n = n months out.",
    )
    contract_label = dl.tenor_label(rank)
    ohlc = dl.load_rolling(code, rank)
else:
    expiries = dl.available_expiries(code)
    default_ix = expiries.index(dl.default_expiry(code)) if expiries else 0
    expiry = st.sidebar.selectbox(
        "Expiry (YYYY-MM)", expiries, index=default_ix,
        help="A single dated contract — no rolling. e.g. the Aug-2026 future.",
    )
    contract_label = f"{expiry} outright"
    ohlc = dl.load_fixed(code, expiry)

st.sidebar.divider()
show_volume = st.sidebar.checkbox("Show volume", value=True)
show_ma = st.sidebar.checkbox("20-day moving average", value=True)
log_scale = st.sidebar.checkbox("Log price axis", value=False)

# ── DATE RANGE ────────────────────────────────────────────────────────────────
st.title(f"{commodity} — {contract_label}")
st.caption(f"{meta['symbol']} · daily OHLC · price in {unit}")

if ohlc.empty:
    st.warning("No data for this selection.")
    st.stop()

dmin, dmax = ohlc.index.min().date(), ohlc.index.max().date()
default_start = max(dmin, (pd.Timestamp(dmax) - pd.DateOffset(years=1)).date())
start, end = st.slider(
    "Date range",
    min_value=dmin, max_value=dmax,
    value=(default_start, dmax),
    format="YYYY-MM-DD",
)
df = ohlc.loc[str(start):str(end)]

if df.empty:
    st.warning("No bars in the selected date range.")
    st.stop()

# ── CHART ─────────────────────────────────────────────────────────────────────
rows = 2 if show_volume else 1
fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
    row_heights=[0.78, 0.22] if show_volume else [1.0],
)

fig.add_trace(
    go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLC", increasing_line_color=UP, decreasing_line_color=DOWN,
    ),
    row=1, col=1,
)

if show_ma and len(df) >= 20:
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["close"].rolling(20).mean(),
            name="20D MA", line=dict(color="#1B2A4A", width=1.4, dash="dot"),
        ),
        row=1, col=1,
    )

if show_volume:
    vol_colors = [UP if c >= o else DOWN for o, c in zip(df["open"], df["close"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=vol_colors,
               marker_line_width=0, opacity=0.55),
        row=2, col=1,
    )
    fig.update_yaxes(title_text="Volume", row=2, col=1)

fig.update_yaxes(title_text=f"Price ({unit})", type="log" if log_scale else "linear", row=1, col=1)
fig.update_xaxes(rangeslider_visible=False)
fig.update_layout(
    height=640, margin=dict(l=50, r=20, t=30, b=30),
    showlegend=True, legend=dict(orientation="h", y=1.02, x=0),
    plot_bgcolor="#F7F9FB", paper_bgcolor="white",
)
st.plotly_chart(fig, width="stretch")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
last, first = df["close"].iloc[-1], df["close"].iloc[0]
chg = (last / first - 1) * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last close", f"{last:,.2f} {unit}", f"{chg:+.1f}% over range")
c2.metric("Range high", f"{df['high'].max():,.2f}")
c3.metric("Range low", f"{df['low'].min():,.2f}")
c4.metric("Bars", f"{len(df):,}", f"{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")

with st.expander("Show raw data"):
    st.dataframe(df.iloc[::-1], width="stretch")