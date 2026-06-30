"""
JF_copper — Gold & Copper futures dashboard
===========================================
Pick a rolling (continuous) or fixed (outright) contract and a bar timeframe,
then explore four tabs:
  • Price (OHLC)   — candlesticks + volume / MA.
  • Volatility     — realised vol (≤3 windows), ATM implied vol, ATR.
  • Open Interest  — options OI by strike, OI across the curve, OI history.
  • Skew & Smile   — per-strike IV smile (Black-76) + 25Δ skew over time.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import numpy as np

import active_contracts as ac
import data_loader as dl
import overfitting as ov
import strategies as strat

st.set_page_config(page_title="JF Copper · Gold & Copper", page_icon="🟡", layout="wide")

UP, DOWN = "#1A6B3A", "#8B1A1A"
RV_COLORS = ["#1E6B7A", "#C8922A", "#4A1B7A"]
IV_COLOR, ATR_COLOR, NAVY = "#8B1A1A", "#6B7C93", "#1B2A4A"
RV_CHOICES = [5, 10, 20, 30, 60, 90, 120]
MA_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#17becf", "#bcbd22"]


# Cache the heavier reads so toggling widgets stays snappy.
@st.cache_data(show_spinner=False)
def _oi_history(code):       return dl.oi_history(code)
@st.cache_data(show_spinner=False)
def _iv_smile_front(code):   return dl.iv_smile_front(code)
@st.cache_data(show_spinner=False)
def _oi_by_strike(code, e):  return dl.oi_by_strike(code, e)
@st.cache_data(show_spinner=False)
def _oi_term(code):          return dl.oi_term_structure(code)
@st.cache_data(show_spinner=False)
def _run_strategy(code, is_frac, embargo, w_spread, use_peak, peak_drop, exit_buf, cot_actor, shock_k):
    base = strat.Params(w_spread=w_spread, use_oi_peak=use_peak, peak_drop=peak_drop,
                        exit_buffer_days=exit_buf, cot_actor=cot_actor, shock_k=shock_k)
    return strat.run_is_oos(code, is_frac=is_frac, embargo=embargo, base=base)
@st.cache_data(show_spinner=False)
def _grid_matrix(code, w_spread, use_peak, peak_drop, exit_buf, cot_actor, shock_k):
    base = strat.Params(w_spread=w_spread, use_oi_peak=use_peak, peak_drop=peak_drop,
                        exit_buffer_days=exit_buf, cot_actor=cot_actor, shock_k=shock_k)
    return strat.grid_pnl_matrix(code, base)
@st.cache_data(show_spinner=False)
def _cot(code):              return dl.load_cot(code)
@st.cache_data(show_spinner=False)
def _fut_curve(code):        return dl.futures_oi_curve(code)
@st.cache_data(show_spinner=False)
def _fut_series(code, exp):  return dl.futures_oi_series(code, exp)


# ── SIDEBAR · CONTRACT ────────────────────────────────────────────────────────
st.sidebar.title("Contract")
commodity = st.sidebar.selectbox("Commodity", list(dl.COMMODITIES))
meta = dl.COMMODITIES[commodity]
code, unit = meta["code"], meta["unit"]

contract_type = st.sidebar.radio("Contract type", ["Rolling (continuous)", "Fixed (outright)"])
if contract_type.startswith("Rolling"):
    rank = st.sidebar.selectbox("Tenor", dl.available_tenors(code), format_func=dl.tenor_label,
                                help="Continuous series. M0 = front month, M+n = n months out.")
    contract_label = dl.tenor_label(rank)
    ohlc = dl.load_rolling(code, rank)
    iv_full = dl.load_iv_rolling(code, rank)
    iv_label = f"ATM IV ({'M0' if rank == 0 else f'M+{rank}'})"
    skew_rank = rank
else:
    expiries = dl.available_expiries(code)
    default_ix = expiries.index(dl.default_expiry(code)) if expiries else 0
    expiry = st.sidebar.selectbox("Expiry (YYYY-MM)", expiries, index=default_ix,
                                  help="A single dated contract — no rolling.")
    contract_label = f"{expiry} outright"
    ohlc = dl.load_fixed(code, expiry)
    iv_full = dl.load_iv_fixed(code, expiry)
    iv_label = f"ATM IV ({expiry})"
    skew_rank = 0  # show the front-tenor skew alongside a fixed contract

timeframe = st.sidebar.selectbox("Bar timeframe", list(dl.TIMEFRAMES), index=0,
                                 help="Daily bars, or resampled to weekly / monthly.")
rule, ppy = dl.TIMEFRAMES[timeframe]

# ── SIDEBAR · CHART OPTIONS ───────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader("Price chart")
show_volume = st.sidebar.checkbox("Show volume", value=True)
log_scale = st.sidebar.checkbox("Log price axis", value=False)

with st.sidebar.expander("Technical analysis", expanded=False):
    st.caption("Price overlays")
    sma_periods = st.multiselect("SMA (days)", dl.MA_PERIODS, default=[20])
    ema_periods = st.multiselect("EMA (days)", dl.MA_PERIODS, default=[])
    show_bb = st.checkbox("Bollinger Bands (20, 2σ)")
    show_sar = st.checkbox("Parabolic SAR")
    st.caption("Oscillators (each adds a panel)")
    show_rsi = st.checkbox("RSI (14)")
    show_macd = st.checkbox("MACD (12, 26, 9)")
    show_stoch = st.checkbox("Stochastic (14, 3)")
    show_adx = st.checkbox("ADX (14)")

st.sidebar.divider()
st.sidebar.subheader("Volatility")
rv_windows = st.sidebar.multiselect("Realised-vol windows (bars)", RV_CHOICES,
                                    default=[10, 20, 60], max_selections=3)
show_iv = st.sidebar.checkbox("Implied vol (ATM)", value=True)
show_atr = st.sidebar.checkbox("Average True Range (ATR)", value=True)
atr_window = st.sidebar.slider("ATR window (bars)", 2, 60, 14)

# ── HEADER + SHARED DATE RANGE ────────────────────────────────────────────────
st.title(f"{commodity} — {contract_label}")
st.caption(f"{meta['symbol']} · {timeframe.lower()} bars · price in {unit}")
if ohlc.empty:
    st.warning("No data for this selection.")
    st.stop()

bars = dl.resample_ohlc(ohlc, rule)
dmin, dmax = ohlc.index.min().date(), ohlc.index.max().date()
default_start = max(dmin, (pd.Timestamp(dmax) - pd.DateOffset(years=1)).date())
start, end = st.slider("Date range", min_value=dmin, max_value=dmax,
                       value=(default_start, dmax), format="YYYY-MM-DD")
s, e = str(start), str(end)
df = bars.loc[s:e]
if df.empty:
    st.warning("No bars in the selected date range.")
    st.stop()

tab_price, tab_vol, tab_oi, tab_smile, tab_lab = st.tabs(
    ["📈 Price (OHLC)", "📊 Volatility", "🔢 Open Interest", "🙂 Skew & Smile", "🧪 Strategies Lab"])

# ── TAB 1 · PRICE ─────────────────────────────────────────────────────────────
with tab_price:
    def clip(series):                      # indicators computed on full `bars`, shown over [s:e]
        return series.loc[s:e]

    # Assemble stacked panels: price (+overlays) on top, then volume / oscillators.
    panels = [("price", 3.0)]
    if show_volume: panels.append(("volume", 1.0))
    if show_rsi:    panels.append(("rsi", 1.3))
    if show_macd:   panels.append(("macd", 1.4))
    if show_stoch:  panels.append(("stoch", 1.3))
    if show_adx:    panels.append(("adx", 1.3))
    weights = [w for _, w in panels]
    heights = [w / sum(weights) for w in weights]
    row_of = {name: i + 1 for i, (name, _) in enumerate(panels)}
    nrows = len(panels)

    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=heights)
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="OHLC",
                                 increasing_line_color=UP, decreasing_line_color=DOWN), row=1, col=1)
    # ── price overlays ──
    for i, n in enumerate(sma_periods):
        fig.add_trace(go.Scatter(x=df.index, y=clip(dl.sma(bars["close"], n)), name=f"SMA {n}",
                                 line=dict(color=MA_PALETTE[i % len(MA_PALETTE)], width=1.3)), row=1, col=1)
    for i, n in enumerate(ema_periods):
        fig.add_trace(go.Scatter(x=df.index, y=clip(dl.ema(bars["close"], n)), name=f"EMA {n}",
                                 line=dict(color=MA_PALETTE[i % len(MA_PALETTE)], width=1.3, dash="dash")), row=1, col=1)
    if show_bb:
        mid, up_b, lo_b = dl.bollinger(bars["close"], 20, 2)
        fig.add_trace(go.Scatter(x=df.index, y=clip(up_b), name="BB upper",
                                 line=dict(color="#999", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=clip(lo_b), name="BB lower", fill="tonexty",
                                 fillcolor="rgba(120,120,120,0.10)", line=dict(color="#999", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=clip(mid), name="BB mid",
                                 line=dict(color="#999", width=1, dash="dot")), row=1, col=1)
    if show_sar:
        sar = clip(dl.parabolic_sar(bars["high"], bars["low"]))
        fig.add_trace(go.Scatter(x=sar.index, y=sar, name="SAR", mode="markers",
                                 marker=dict(color=NAVY, size=3)), row=1, col=1)
    fig.update_yaxes(title_text=f"Price ({unit})", type="log" if log_scale else "linear", row=1, col=1)

    # ── volume ──
    if show_volume:
        r = row_of["volume"]
        vc = [UP if c >= o else DOWN for o, c in zip(df["open"], df["close"])]
        fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=vc,
                             marker_line_width=0, opacity=0.55), row=r, col=1)
        fig.update_yaxes(title_text="Vol", row=r, col=1)
    # ── RSI ──
    if show_rsi:
        r = row_of["rsi"]
        rv = clip(dl.rsi(bars["close"], 14))
        fig.add_trace(go.Scatter(x=rv.index, y=rv, name="RSI", line=dict(color="#4A1B7A", width=1.4)), row=r, col=1)
        fig.add_hline(y=70, line=dict(color="#bbb", dash="dot"), row=r, col=1)
        fig.add_hline(y=30, line=dict(color="#bbb", dash="dot"), row=r, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=r, col=1)
    # ── MACD ──
    if show_macd:
        r = row_of["macd"]
        line, sig, hist = (clip(x) for x in dl.macd(bars["close"]))
        hc = [UP if v >= 0 else DOWN for v in hist]
        fig.add_trace(go.Bar(x=hist.index, y=hist, name="MACD hist", marker_color=hc,
                             marker_line_width=0, opacity=0.5), row=r, col=1)
        fig.add_trace(go.Scatter(x=line.index, y=line, name="MACD", line=dict(color=NAVY, width=1.4)), row=r, col=1)
        fig.add_trace(go.Scatter(x=sig.index, y=sig, name="signal", line=dict(color="#C8922A", width=1.2)), row=r, col=1)
        fig.update_yaxes(title_text="MACD", row=r, col=1)
    # ── Stochastic ──
    if show_stoch:
        r = row_of["stoch"]
        k, d = (clip(x) for x in dl.stochastic(bars["high"], bars["low"], bars["close"], 14, 3))
        fig.add_trace(go.Scatter(x=k.index, y=k, name="%K", line=dict(color="#1E6B7A", width=1.3)), row=r, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d, name="%D", line=dict(color="#8B1A1A", width=1.1)), row=r, col=1)
        fig.add_hline(y=80, line=dict(color="#bbb", dash="dot"), row=r, col=1)
        fig.add_hline(y=20, line=dict(color="#bbb", dash="dot"), row=r, col=1)
        fig.update_yaxes(title_text="Stoch", range=[0, 100], row=r, col=1)
    # ── ADX ──
    if show_adx:
        r = row_of["adx"]
        adx_, pdi, mdi = (clip(x) for x in dl.adx(bars["high"], bars["low"], bars["close"], 14))
        fig.add_trace(go.Scatter(x=adx_.index, y=adx_, name="ADX", line=dict(color=NAVY, width=1.6)), row=r, col=1)
        fig.add_trace(go.Scatter(x=pdi.index, y=pdi, name="+DI", line=dict(color=UP, width=1)), row=r, col=1)
        fig.add_trace(go.Scatter(x=mdi.index, y=mdi, name="−DI", line=dict(color=DOWN, width=1)), row=r, col=1)
        fig.update_yaxes(title_text="ADX", row=r, col=1)

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(height=int(380 + 150 * (nrows - 1)), margin=dict(l=50, r=20, t=30, b=30),
                      legend=dict(orientation="h", y=1.02, x=0), barmode="overlay",
                      plot_bgcolor="#F7F9FB", paper_bgcolor="white")
    st.plotly_chart(fig, width="stretch")

    last, first = df["close"].iloc[-1], df["close"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"{last:,.2f} {unit}", f"{(last/first-1)*100:+.1f}% over range")
    c2.metric("Range high", f"{df['high'].max():,.2f}")
    c3.metric("Range low", f"{df['low'].min():,.2f}")
    c4.metric("Bars", f"{len(df):,}", f"{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")
    with st.expander("Show raw data"):
        st.dataframe(df.iloc[::-1], width="stretch")

# ── TAB 2 · VOLATILITY ────────────────────────────────────────────────────────
with tab_vol:
    volfig = make_subplots(specs=[[{"secondary_y": True}]])
    plotted = False
    for color, w in zip(RV_COLORS, rv_windows):
        rv = (dl.realised_vol(bars["close"], w, ppy) * 100).loc[s:e]
        volfig.add_trace(go.Scatter(x=rv.index, y=rv, name=f"RV {w}", line=dict(color=color, width=1.8)),
                         secondary_y=False)
        plotted = True
    if show_iv:
        iv_tf = dl.resample_last(iv_full, rule)
        iv_disp = (iv_tf.reindex(bars.index).ffill(limit=5) * 100).loc[s:e]
        if iv_disp.notna().any():
            volfig.add_trace(go.Scatter(x=iv_disp.index, y=iv_disp, name=iv_label,
                                        line=dict(color=IV_COLOR, width=2.4)), secondary_y=False)
            plotted = True
        else:
            st.caption("ℹ️ No implied-vol data for this contract/tenor.")
    if show_atr:
        atr_disp = dl.atr(bars, atr_window).loc[s:e]
        volfig.add_trace(go.Scatter(x=atr_disp.index, y=atr_disp, name=f"ATR {atr_window}",
                                    line=dict(color=ATR_COLOR, width=1.6, dash="dot")), secondary_y=True)
        plotted = True
    if not plotted:
        st.info("Pick at least one series in the sidebar — an RV window, implied vol, or ATR.")
    else:
        volfig.update_yaxes(title_text="Annualised vol (%)", secondary_y=False)
        volfig.update_yaxes(title_text=f"ATR ({unit})", secondary_y=True, showgrid=False)
        volfig.update_xaxes(rangeslider_visible=False)
        volfig.update_layout(height=560, margin=dict(l=55, r=55, t=30, b=30),
                             legend=dict(orientation="h", y=1.02, x=0), hovermode="x unified",
                             plot_bgcolor="#F7F9FB", paper_bgcolor="white")
        st.plotly_chart(volfig, width="stretch")

        def _last(x):
            x = x.dropna(); return float(x.iloc[-1]) if not x.empty else None
        short_w = min(rv_windows) if rv_windows else None
        rv_now = _last(dl.realised_vol(bars["close"], short_w, ppy) * 100) if short_w else None
        iv_now = _last(iv_full * 100)
        atr_now = _last(dl.atr(bars, atr_window))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Realised vol ({short_w})" if short_w else "Realised vol",
                  f"{rv_now:.1f}%" if rv_now is not None else "—")
        m2.metric("Implied vol (ATM)", f"{iv_now:.1f}%" if iv_now is not None else "—")
        m3.metric("IV − RV spread", f"{iv_now-rv_now:+.1f} pts" if (iv_now and rv_now) else "—",
                  help="Positive = options price more vol than recently realised (rich).")
        m4.metric(f"ATR ({atr_window})", f"{atr_now:,.2f} {unit}" if atr_now is not None else "—")
        st.caption(f"Realised vol annualised with √{ppy} ({timeframe.lower()} bars). "
                   "Implied = ATM IV for the matching tenor/expiry. ATR in price units (right axis).")

# ── TAB 3 · OPEN INTEREST ─────────────────────────────────────────────────────
with tab_oi:
    # ── Futures OI by contract (per-expiry exchange OI) ──────────────────────
    if dl.has_futures_oi(code):
        st.subheader("Futures — open interest by contract")
        curve, snap = _fut_curve(code)
        cc1, cc2 = st.columns(2)
        with cc1:
            top = curve.sort_values("open_interest", ascending=False).head(18).sort_values("expiry_ym")
            fcv = go.Figure(go.Bar(x=top["expiry_ym"], y=top["open_interest"],
                                   marker_color=NAVY, opacity=0.85))
            fcv.update_layout(height=330, title=f"OI by futures contract — {snap:%Y-%m-%d}",
                              margin=dict(l=50, r=20, t=40, b=40), yaxis_title="Open interest",
                              xaxis_title="Contract month", plot_bgcolor="#F7F9FB", paper_bgcolor="white")
            st.plotly_chart(fcv, width="stretch")
        with cc2:
            fx_exps = dl.futures_oi_expiries(code)
            fx_def = dl.futures_oi_front(code)
            fx_ix = fx_exps.index(fx_def) if fx_def in fx_exps else len(fx_exps) - 1
            sel_fx = st.selectbox("Contract — OI over its life", fx_exps, index=fx_ix, key="fut_oi_exp")
            fser = _fut_series(code, sel_fx).loc[s:e]
            ffs = go.Figure(go.Scatter(x=fser.index, y=fser, line=dict(color="#1E6B7A", width=1.9),
                                       fill="tozeroy", fillcolor="rgba(30,107,122,0.08)", name="OI"))
            ffs.update_layout(height=330, title=f"{sel_fx} futures — open interest",
                              margin=dict(l=50, r=20, t=40, b=30), yaxis_title="Open interest",
                              plot_bgcolor="#F7F9FB", paper_bgcolor="white")
            st.plotly_chart(ffs, width="stretch")
        tot = curve["open_interest"].sum()
        front = curve.sort_values("open_interest", ascending=False).iloc[0] if not curve.empty else None
        d1, d2 = st.columns(2)
        d1.metric("Total futures OI (all contracts)", f"{tot:,.0f}", help=f"Snapshot {snap:%Y-%m-%d}")
        if front is not None:
            d2.metric("Most-active contract", f"{front['expiry_ym']}", f"{front['open_interest']:,.0f} OI")
        st.divider()
    else:
        st.caption(f"Per-contract futures OI not available for {commodity} yet "
                   "(gold only). Market-level COT shown below.")

    # ── Futures market positioning (CFTC COT) ────────────────────────────────
    st.subheader("Futures — market positioning (CFTC COT)")
    cot = _cot(code).loc[s:e]
    if cot.empty:
        st.info("No COT data in the selected date range (try widening it).")
    else:
        fc = make_subplots(specs=[[{"secondary_y": True}]])
        fc.add_trace(go.Scatter(x=cot.index, y=cot["open_interest"], name="Total OI (futures)",
                                line=dict(color=NAVY, width=2.2), fill="tozeroy",
                                fillcolor="rgba(27,42,74,0.06)"), secondary_y=False)
        pos_colors = {"managed_money_net": "#1E6B7A", "producer_net": "#8B1A1A", "swap_net": "#C8922A"}
        for col, label in dl.COT_POSITIONS.items():
            if col in cot.columns:
                fc.add_trace(go.Scatter(x=cot.index, y=cot[col], name=label,
                                        line=dict(color=pos_colors.get(col), width=1.5)), secondary_y=True)
        fc.add_hline(y=0, line_color="#999", secondary_y=True)
        fc.update_yaxes(title_text="Total open interest (contracts)", secondary_y=False)
        fc.update_yaxes(title_text="Net position (contracts)", secondary_y=True, showgrid=False)
        fc.update_layout(height=380, margin=dict(l=55, r=55, t=20, b=30),
                         legend=dict(orientation="h", y=1.04, x=0), hovermode="x unified",
                         plot_bgcolor="#F7F9FB", paper_bgcolor="white")
        st.plotly_chart(fc, width="stretch")

        latest = cot.iloc[-1]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total futures OI", f"{latest['open_interest']:,.0f}",
                  help=f"As of {cot.index[-1]:%Y-%m-%d} (weekly CFTC report).")
        if "managed_money_net" in cot.columns:
            mm = latest["managed_money_net"]
            k2.metric("Managed money net", f"{mm:,.0f}", "net long" if mm >= 0 else "net short")
        if "producer_net" in cot.columns:
            k3.metric("Producers net", f"{latest['producer_net']:,.0f}")
        if "swap_net" in cot.columns:
            k4.metric("Swap dealers net", f"{latest['swap_net']:,.0f}")
        st.caption("Total OI = all open futures positions (left). Net positioning by trader "
                   "group on the right axis: managed money = specs, producers = commercial hedgers.")

    st.divider()
    # ── Options chain OI ─────────────────────────────────────────────────────
    st.subheader("Options chain — open interest")
    st.caption("COMEX monthly options (latest snapshot + history).")
    oi_exps = dl.oi_expiries(code)
    sel_exp = st.selectbox("Expiry for strike ladder", oi_exps, key="oi_exp")
    obs = _oi_by_strike(code, sel_exp)
    if obs.empty:
        st.info("No OI by strike for this expiry.")
    else:
        f1 = go.Figure()
        f1.add_trace(go.Bar(x=obs["strike"], y=obs["call_oi"], name="Calls", marker_color=UP, opacity=0.7))
        f1.add_trace(go.Bar(x=obs["strike"], y=obs["put_oi"], name="Puts", marker_color=DOWN, opacity=0.7))
        f1.update_layout(barmode="overlay", height=380, title=f"OI by strike — {sel_exp}",
                         margin=dict(l=50, r=20, t=40, b=30), legend=dict(orientation="h", y=1.05, x=0),
                         xaxis_title=f"Strike ({unit})", yaxis_title="Open interest (lots)",
                         plot_bgcolor="#F7F9FB", paper_bgcolor="white")
        st.plotly_chart(f1, width="stretch")
        tot_c, tot_p = obs["call_oi"].sum(), obs["put_oi"].sum()
        a, b, c = st.columns(3)
        a.metric("Total call OI", f"{tot_c:,.0f}")
        b.metric("Total put OI", f"{tot_p:,.0f}")
        c.metric("Put/Call OI", f"{(tot_p/tot_c):.2f}" if tot_c else "—")

    g1, g2 = st.columns(2)
    with g1:
        term = _oi_term(code)
        ft = go.Figure(go.Bar(x=term["expiry"], y=term["total_oi"], marker_color=NAVY, opacity=0.8))
        ft.update_layout(height=320, title="OI across the curve (total per expiry)",
                         margin=dict(l=50, r=20, t=40, b=30), yaxis_title="Open interest",
                         plot_bgcolor="#F7F9FB", paper_bgcolor="white")
        st.plotly_chart(ft, width="stretch")
    with g2:
        hist = _oi_history(code).loc[s:e] if rule is None else _oi_history(code)
        fh = go.Figure(go.Scatter(x=hist.index, y=hist, line=dict(color="#1E6B7A", width=1.8), name="Total OI"))
        fh.update_layout(height=320, title="Total options OI over time",
                         margin=dict(l=50, r=20, t=40, b=30), yaxis_title="Open interest",
                         plot_bgcolor="#F7F9FB", paper_bgcolor="white")
        st.plotly_chart(fh, width="stretch")

# ── TAB 4 · SKEW & SMILE ──────────────────────────────────────────────────────
with tab_smile:
    sm, minfo = _iv_smile_front(code)
    st.caption(f"Per-strike IV smile — front contract {minfo['expiry']}, "
               f"snapshot {minfo['snapshot']}, F = {minfo['F']:.2f} {unit}, "
               f"T = {minfo['T_years']:.3f}y (Black-76 inverted from settle prices).")
    if sm.empty:
        st.info("Could not invert a smile from the latest snapshot.")
    else:
        fsm = go.Figure()
        for right, name, col in [("C", "Calls", UP), ("P", "Puts", DOWN)]:
            d = sm[sm["right"] == right]
            fsm.add_trace(go.Scatter(x=d["strike"], y=d["iv"] * 100, mode="markers+lines", name=name,
                                     line=dict(color=col, width=1.2), marker=dict(size=5)))
        fsm.add_vline(x=minfo["F"], line_dash="dash", line_color=NAVY, annotation_text="F")
        fsm.update_layout(height=420, title="Implied-vol smile by strike",
                          margin=dict(l=50, r=20, t=40, b=30), legend=dict(orientation="h", y=1.05, x=0),
                          xaxis_title=f"Strike ({unit})", yaxis_title="Implied vol (%)",
                          plot_bgcolor="#F7F9FB", paper_bgcolor="white")
        st.plotly_chart(fsm, width="stretch")

    sk = dl.load_skew(code, skew_rank)
    if rule is not None:
        sk = sk.resample(rule).last()
    sk = sk.loc[s:e]
    if sk.empty:
        st.info("No skew history for this tenor.")
    else:
        fk = make_subplots(specs=[[{"secondary_y": True}]])
        fk.add_trace(go.Scatter(x=sk.index, y=sk["rr25"], name="25Δ risk reversal",
                                line=dict(color="#4A1B7A", width=2)), secondary_y=False)
        fk.add_trace(go.Scatter(x=sk.index, y=sk["fly25"], name="25Δ butterfly",
                                line=dict(color="#C8922A", width=1.6)), secondary_y=False)
        fk.add_trace(go.Scatter(x=sk.index, y=sk["atm"], name="ATM vol",
                                line=dict(color=NAVY, width=1.4, dash="dot")), secondary_y=True)
        fk.add_hline(y=0, line_color="#999", secondary_y=False)
        fk.update_yaxes(title_text="RR / Fly (vol pts)", secondary_y=False)
        fk.update_yaxes(title_text="ATM vol (%)", secondary_y=True, showgrid=False)
        lbl = dl.tenor_label(skew_rank)
        fk.update_layout(height=380, title=f"25Δ skew over time — {lbl}",
                         margin=dict(l=55, r=55, t=40, b=30), legend=dict(orientation="h", y=1.05, x=0),
                         hovermode="x unified", plot_bgcolor="#F7F9FB", paper_bgcolor="white")
        st.plotly_chart(fk, width="stretch")
        st.caption("RR>0 = calls bid over puts (upside skew). Butterfly = wing richness vs ATM.")

# ── TAB 5 · STRATEGIES LAB ────────────────────────────────────────────────────
with tab_lab:
    LAB_FG, LAB_GRID, GREEN, RED, GREY = "#1B2A4A", "#CBD5E1", "#1A6B3A", "#8B1A1A", "#5B6B7B"

    def _style(fig, height, title=None):
        """Readable, high-contrast styling for every lab figure."""
        fig.update_layout(height=height, title=title, font=dict(color=LAB_FG, size=13),
                          title_font=dict(color=LAB_FG, size=16),
                          legend=dict(orientation="h", y=1.04, x=0, font=dict(color=LAB_FG, size=12)),
                          hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=64, r=64, t=46, b=36))
        ax = dict(showgrid=True, gridcolor=LAB_GRID, zeroline=True, zerolinecolor="#9AA7B5",
                  linecolor="#9AA7B5", title_font=dict(color=LAB_FG, size=13),
                  tickfont=dict(color=LAB_FG, size=12))
        fig.update_xaxes(rangeslider_visible=False, **ax)
        fig.update_yaxes(**ax)
        return fig

    st.subheader("Active / next-active calendar spread")
    cyc = ac.CYCLES[code]
    codes_txt = ", ".join(f"{m:02d}{c}" for m, c in zip(cyc["months"], cyc["codes"]))
    st.caption(
        f"**Active-contract cycle ({commodity}, CME-verified):** {codes_txt}. The *active* "
        "leg is the nearest cycle month not yet in delivery; it rolls to the next on its "
        "**First Position Date** (day before First Notice Day). Trade = **long front-active / "
        "short next-active**. Signal blends **spec positioning** (COT), the **active-contract "
        "OI tilt**, and **spread** mean-reversion. A position opens only **after the active "
        "contract's OI has peaked** and is forced flat a few days **before the roll** — i.e. "
        "held at most until the last day before that contract becomes the front. Params fit "
        "**in-sample only**, evaluated **out-of-sample**."
    )

    cc1, cc2, cc3 = st.columns(3)
    is_frac = cc1.slider("In-sample fraction", 0.40, 0.80, 0.60, 0.05,
                         help="Chronological split: earliest fraction fits params; rest is OOS.")
    embargo = cc2.slider("Embargo (days at boundary)", 0, 504, 252, 21,
                         help="Dropped around the IS/OOS split so trailing z-scores don't leak.")
    w_spread = cc3.slider("Spread mean-reversion weight", 0.0, 2.0, 0.5, 0.25,
                          help="Weight on fading an extended spread (−z of front−next).")
    cc4, cc5, cc6, cc7 = st.columns(4)
    actor_keys = list(strat.COT_ACTORS)
    cot_actor = cc4.selectbox("COT trader group", actor_keys,
                              index=actor_keys.index("producer"),
                              format_func=lambda k: strat.COT_ACTORS[k][2],
                              help="Which CFTC group drives the positioning signal. "
                                   "Empirically producers (commercials) predict the spread best; "
                                   "managed money (specs) is weak/contrarian.")
    use_peak = cc5.checkbox("Enter only after the OI peak", value=True,
                            help="Open a position only once the active contract's OI has rolled off its peak.")
    peak_drop = cc6.slider("OI peak drop (%)", 0, 20, 0, 1,
                           help="How far OI must fall below its in-contract max to count as 'peaked'.") / 100
    exit_buf = cc7.slider("Exit buffer before roll (days)", 0, 20, 3, 1,
                          help="Force flat this many days before the contract's First Position Date.")
    shock_k = st.slider("Remove big shocks — cap daily Δspread at ×σ (0 = off)", 0.0, 6.0, 0.0, 0.5,
                        help="Caps each day's spread move at k×trailing-σ (past vol only, no look-ahead) "
                             "to trim tail shocks and lift the Sharpe. Lower = more aggressive.")

    try:
        res = _run_strategy(code, is_frac, embargo, w_spread, use_peak, peak_drop, exit_buf, cot_actor, shock_k)
    except ValueError as exc:
        st.warning(str(exc)); st.stop()

    legs, feat, ev = res.legs, res.feat, res.events
    cur = legs.iloc[-1]
    a, b, c, d = st.columns(4)
    a.metric("Front-active", cur["front_ym"])
    b.metric("Next-active", cur["next_ym"])
    c.metric("Spread now", f"{cur['spread']:+.4f} {unit}")
    d.metric("Position", {1: "Long spread", -1: "Short spread", 0: "Flat"}[int(res.position.iloc[-1])])

    if not dl.has_futures_oi(code):
        st.info(f"ℹ️ {commodity} has no per-contract **futures** OI — the OI tilt falls back "
                "to per-contract **volume**. (Fetch copper futures OI to use real OI.)")

    # entry/exit events → marker points on the spread
    def _pts(want):
        ds = [dt for dt, frm, to in ev
              if (to == want if want in (1, -1) else (to == 0 and frm != 0)) and dt in legs.index]
        return ds, [legs["spread"].loc[dt] for dt in ds]
    lx, ly = _pts(1); sx, sy = _pts(-1); xx, xy = _pts(0)

    # ── spread + entries/exits + position ────────────────────────────────────
    spf = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.72, 0.28])
    spf.add_trace(go.Scatter(x=legs.index, y=legs["spread"], name="Spread (front−next)",
                             line=dict(color=LAB_FG, width=1.4)), row=1, col=1)
    spf.add_trace(go.Scatter(x=lx, y=ly, name="Go long", mode="markers",
                             marker=dict(symbol="triangle-up", size=12, color=GREEN,
                                         line=dict(width=1, color="white"))), row=1, col=1)
    spf.add_trace(go.Scatter(x=sx, y=sy, name="Go short", mode="markers",
                             marker=dict(symbol="triangle-down", size=12, color=RED,
                                         line=dict(width=1, color="white"))), row=1, col=1)
    spf.add_trace(go.Scatter(x=xx, y=xy, name="Exit", mode="markers",
                             marker=dict(symbol="x", size=9, color=GREY,
                                         line=dict(width=1))), row=1, col=1)
    spf.add_trace(go.Scatter(x=res.position.index, y=res.position, name="Position",
                             line=dict(color="#1E6B7A", width=1.2, shape="hv"),
                             fill="tozeroy", fillcolor="rgba(30,107,122,0.18)"), row=2, col=1)
    _style(spf, 470, "Spread, entries / exits, and position")
    spf.update_yaxes(title_text=f"Spread ({unit})", row=1, col=1)
    spf.update_yaxes(title_text="Pos", row=2, col=1, tickvals=[-1, 0, 1])
    st.plotly_chart(spf, width="stretch")

    # ── active-contract OI life-cycle: peak + roll crossover ─────────────────
    peak_dates = feat["oi_front"].groupby(legs["front_ym"]).idxmax().dropna()
    oif = go.Figure()
    oif.add_trace(go.Scatter(x=feat.index, y=feat["oi_front"], name="Active (front) OI",
                             line=dict(color=GREEN, width=1.8)))
    oif.add_trace(go.Scatter(x=feat.index, y=feat["oi_next"], name="Next-active OI",
                             line=dict(color="#C8922A", width=1.5)))
    oif.add_trace(go.Scatter(x=list(peak_dates.values),
                             y=[feat["oi_front"].loc[d] for d in peak_dates.values],
                             name="OI peak", mode="markers",
                             marker=dict(symbol="circle", size=8, color=RED,
                                         line=dict(width=1, color="white"))))
    for rd in legs.index[legs["roll"]]:
        oif.add_vline(x=rd, line_width=1, line_dash="dot", line_color="rgba(139,26,26,0.30)")
    oi_unit = "contracts" if dl.has_futures_oi(code) else "lots (volume proxy)"
    _style(oif, 360, f"Active-contract OI life-cycle — peak ● and roll ⋮ ({res.oi_source})")
    oif.update_yaxes(title_text=f"Open interest ({oi_unit})")
    st.plotly_chart(oif, width="stretch")
    st.caption("OI builds then rolls off as delivery nears; the active leg rolls where "
               "next-active OI overtakes it (dotted lines). Entries are gated to **after** "
               "the peak ●; exits are forced before the roll.")

    # ── IS / OOS equity ──────────────────────────────────────────────────────
    eq = res.pnl.cumsum()
    idx = legs.index
    cut = int(idx.searchsorted(res.boundary))
    oos_start = idx[min(cut + res.embargo, len(idx) - 1)]
    eqf = go.Figure()
    eqf.add_trace(go.Scatter(x=eq[eq.index <= res.boundary].index,
                             y=eq[eq.index <= res.boundary], name="In-sample",
                             line=dict(color=GREY, width=1.8)))
    eqf.add_trace(go.Scatter(x=eq[eq.index >= oos_start].index,
                             y=eq[eq.index >= oos_start], name="Out-of-sample",
                             line=dict(color=GREEN, width=2.4)))
    eqf.add_vrect(x0=res.boundary, x1=oos_start, fillcolor="rgba(139,26,26,0.07)",
                  line_width=0, annotation_text="embargo", annotation_position="top left")
    eqf.add_vline(x=res.boundary, line_dash="dash", line_color=RED)
    _style(eqf, 380, "Cumulative P&L (price points) — IS vs OOS")
    eqf.update_yaxes(title_text=f"Cum. P&L ({unit} pts)")
    st.plotly_chart(eqf, width="stretch")

    # ── metrics IS vs OOS ────────────────────────────────────────────────────
    def _fmt(m):
        sh = m["sharpe"]
        return {"Sharpe (annualised)": f"{sh:.2f}" if sh == sh else "—",
                "Total P&L (pts)": f"{m['total']:+.3f}",
                "Hit rate": f"{m['hit']*100:.0f}%" if m["hit"] == m["hit"] else "—",
                "Max drawdown (pts)": f"{m['max_dd']:.3f}",
                "# trades": f"{m['n_trades']}",
                "% time flat": f"{m['pct_flat']*100:.0f}%"}
    st.table(pd.DataFrame({"In-sample": _fmt(res.is_metrics),
                           "Out-of-sample": _fmt(res.oos_metrics)}))
    si, so = res.is_metrics["sharpe"], res.oos_metrics["sharpe"]
    deg = (f" · OOS/IS Sharpe = {so/si:.2f} (≈1 good, ≪1 = overfit)"
           if si and so == so and si == si and si != 0 else "")
    st.caption(f"Best IS config: **{res.params.label()}** · OI signal: *{res.oi_source}* · "
               f"split {is_frac:.0%}/{1-is_frac:.0%} at {res.boundary:%Y-%m-%d}{deg}")

    # ── OVERFITTING DIAGNOSTICS ──────────────────────────────────────────────
    st.markdown("#### Overfitting diagnostics")
    mat = _grid_matrix(code, w_spread, use_peak, peak_drop, exit_buf, cot_actor, shock_k)
    idx = legs.index
    cut = int(idx.searchsorted(res.boundary))
    is_idx, oos_idx = idx[:cut], idx[min(cut + res.embargo, len(idx) - 1):]

    def _csh(col, ix):
        r = mat[col].reindex(ix).dropna()
        return r.mean() / r.std() * np.sqrt(252) if len(r) > 10 and r.std() > 0 else np.nan
    is_s = np.array([_csh(c, is_idx) for c in mat.columns])
    oos_s = np.array([_csh(c, oos_idx) for c in mat.columns])
    ok = ~np.isnan(is_s) & ~np.isnan(oos_s)
    corr = np.corrcoef(is_s[ok], oos_s[ok])[0, 1] if ok.sum() > 2 else np.nan

    pbo = ov.pbo_cscv(mat, s=10)
    trial_sr = np.array([(mat[c].dropna().mean() / mat[c].dropna().std())
                         if mat[c].dropna().std() > 0 else np.nan for c in mat.columns])
    dsr = ov.deflated_sharpe(res.pnl, trial_sr)
    em, words = ov.verdict(pbo["pbo"], dsr["dsr"])
    si, so = res.is_metrics["sharpe"], res.oos_metrics["sharpe"]
    degr = so / si if si and si == si and si != 0 and so == so else float("nan")

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("PBO", f"{pbo['pbo']*100:.0f}%" if pbo["pbo"] == pbo["pbo"] else "—",
              help="Probability of Backtest Overfitting (CSCV). Fraction of folds where the "
                   "in-sample-best config lands below median out-of-sample. <30% good, ~50% = chance.")
    o2.metric("Deflated Sharpe", f"{dsr['dsr']*100:.0f}%" if dsr["dsr"] == dsr["dsr"] else "—",
              help="Prob. the true Sharpe is >0 after deflating for the number of grid trials, "
                   "sample length, skew & kurtosis. >90% = confident.")
    o3.metric("OOS/IS Sharpe", f"{degr:.2f}" if degr == degr else "—",
              help="Out-of-sample / in-sample Sharpe. ≈1 generalises; ≪1 = degradation.")
    o4.metric("IS–OOS grid corr", f"{corr:+.2f}" if corr == corr else "—",
              help="Correlation of IS vs OOS Sharpe across all grid configs. Positive = robust "
                   "ranking; negative = configs that shine IS tend to fail OOS (overfit).")
    st.caption(f"{em} **{words}.** Deflated SR uses N={dsr['n_trials']} trials "
               f"(SR≈{dsr['sr_ann']:+.2f} ann. vs deflated bar {dsr['sr_star_ann']:.2f}; "
               f"skew {dsr.get('skew', float('nan')):+.2f}, kurtosis {dsr.get('kurt', float('nan')):.0f}). "
               "Fat tails (high kurtosis) inflate a raw Sharpe — read DSR alongside PBO and the scatter.")

    sel = list(mat.columns).index(res.params.label()) if res.params.label() in list(mat.columns) else None
    sca = go.Figure()
    sca.add_trace(go.Scatter(x=is_s, y=oos_s, mode="markers", name="grid configs",
                             marker=dict(size=8, color="#5B6B7B", opacity=0.75)))
    if sel is not None:
        sca.add_trace(go.Scatter(x=[is_s[sel]], y=[oos_s[sel]], mode="markers", name="selected",
                                 marker=dict(symbol="star", size=16, color="#1A6B3A",
                                             line=dict(width=1, color="white"))))
    sca.add_hline(y=0, line_color="#9AA7B5"); sca.add_vline(x=0, line_color="#9AA7B5")
    _style(sca, 330, f"IS vs OOS Sharpe across {len(mat.columns)} grid configs (corr {corr:+.2f})")
    sca.update_xaxes(title_text="IS Sharpe"); sca.update_yaxes(title_text="OOS Sharpe")
    st.plotly_chart(sca, width="stretch")

    # ── CHOOSING σ (shock cap) ───────────────────────────────────────────────
    st.markdown("#### Choosing the shock cap σ")
    st.caption("σ is set by the **“Remove big shocks — cap daily Δspread at ×σ”** slider above. "
               "This curve shows its effect on the **current** config. Pick σ by **IS stability / "
               "lower drawdown**, not by chasing the OOS max (that would snoop the test set).")
    ks = [0.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5]
    sweep = strat.shock_sweep(code, res.params, ks, is_frac, embargo).sort_values("k")
    xv = sweep["k"].to_numpy()
    xlab = ["off" if k == 0 else f"{k:g}σ" for k in xv]
    swf = make_subplots(specs=[[{"secondary_y": True}]])
    swf.add_trace(go.Bar(x=xv, y=sweep["oos_maxdd"], name="OOS max drawdown",
                         marker_color="rgba(139,26,26,0.30)"), secondary_y=True)
    swf.add_trace(go.Scatter(x=xv, y=sweep["is_sharpe"], name="IS Sharpe",
                             line=dict(color="#5B6B7B", width=2)), secondary_y=False)
    swf.add_trace(go.Scatter(x=xv, y=sweep["oos_sharpe"], name="OOS Sharpe",
                             line=dict(color="#1A6B3A", width=2.4)), secondary_y=False)
    _style(swf, 340, "Sharpe & drawdown vs shock cap (current config)")
    swf.update_xaxes(tickvals=xv, ticktext=xlab, title_text="shock cap (×σ)")
    swf.update_yaxes(title_text="Sharpe", secondary_y=False)
    swf.update_yaxes(title_text=f"OOS max DD ({unit} pts)", secondary_y=True, showgrid=False)
    swf.add_vline(x=float(shock_k), line_dash="dash", line_color="#1B2A4A",
                  annotation_text="current", annotation_position="top")
    st.plotly_chart(swf, width="stretch")

    with st.expander("Signal components over time"):
        cf = go.Figure()
        for col, nm, cl in [("z_cot", f"COT {strat.COT_ACTORS[cot_actor][2]} z", "#1E6B7A"),
                            ("z_oi", "OI tilt z", "#C8922A"),
                            ("z_spread", "Spread z", "#4A1B7A")]:
            cf.add_trace(go.Scatter(x=feat.index, y=feat[col], name=nm, line=dict(width=1.3, color=cl)))
        cf.add_trace(go.Scatter(x=res.score.index, y=res.score, name="Combined score",
                                line=dict(width=2, color=LAB_FG)))
        cf.add_hline(y=res.params.z_enter, line_dash="dot", line_color=GREEN)
        cf.add_hline(y=-res.params.z_enter, line_dash="dot", line_color=RED)
        _style(cf, 340, "Standardised signal components & combined score")
        st.plotly_chart(cf, width="stretch")

    with st.expander("In-sample parameter grid (Sharpe by config)"):
        st.dataframe(res.grid.sort_values("is_sharpe", ascending=False).reset_index(drop=True),
                     width="stretch")
    with st.expander("Caveats & data notes"):
        st.markdown(
            "- **OI-peak entry / roll exit:** a position opens only after the active "
            "contract's OI has peaked, and is closed before the roll (First Position Date).\n"
            "- **Roll discipline:** the spread jumps when the leg pair rolls; that jump is "
            "zeroed out of returns (a roll is a contract change, not P&L).\n"
            "- **No look-ahead:** COT release-lagged (Tue→Fri) then forward-filled; z-scores "
            "are trailing; the position is applied with a 1-session execution lag.\n"
            "- **Copper OI:** no per-contract futures OI → OI tilt uses per-contract volume; "
            "copper COT only starts 2022, so its IS window is short.\n"
            "- Units are quoted price points (not ×contract multiplier)."
        )