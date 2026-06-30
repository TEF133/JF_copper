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

import data_loader as dl

st.set_page_config(page_title="JF Copper · Gold & Copper", page_icon="🟡", layout="wide")

UP, DOWN = "#1A6B3A", "#8B1A1A"
RV_COLORS = ["#1E6B7A", "#C8922A", "#4A1B7A"]
IV_COLOR, ATR_COLOR, NAVY = "#8B1A1A", "#6B7C93", "#1B2A4A"
RV_CHOICES = [5, 10, 20, 30, 60, 90, 120]


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
def _cot(code):              return dl.load_cot(code)


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
show_ma = st.sidebar.checkbox("20-bar moving average", value=True)
log_scale = st.sidebar.checkbox("Log price axis", value=False)

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

tab_price, tab_vol, tab_oi, tab_smile = st.tabs(
    ["📈 Price (OHLC)", "📊 Volatility", "🔢 Open Interest", "🙂 Skew & Smile"])

# ── TAB 1 · PRICE ─────────────────────────────────────────────────────────────
with tab_price:
    rows = 2 if show_volume else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.78, 0.22] if show_volume else [1.0])
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="OHLC",
                                 increasing_line_color=UP, decreasing_line_color=DOWN), row=1, col=1)
    if show_ma and len(df) >= 20:
        fig.add_trace(go.Scatter(x=df.index, y=df["close"].rolling(20).mean(), name="20-bar MA",
                                 line=dict(color=NAVY, width=1.4, dash="dot")), row=1, col=1)
    if show_volume:
        vc = [UP if c >= o else DOWN for o, c in zip(df["open"], df["close"])]
        fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=vc,
                             marker_line_width=0, opacity=0.55), row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text=f"Price ({unit})", type="log" if log_scale else "linear", row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(height=640, margin=dict(l=50, r=20, t=30, b=30),
                      legend=dict(orientation="h", y=1.02, x=0),
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
    # ── Futures OI & positioning (CFTC COT) ──────────────────────────────────
    st.subheader("Futures — open interest & positioning (CFTC COT)")
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