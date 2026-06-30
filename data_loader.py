"""
data_loader.py
==============
Loads daily OHLC for gold and copper from local parquet snapshots.

Two contract conventions are supported, matching how the futures are quoted:

  Rolling (continuous)  ->  <code>_tenor_ohlc.parquet
        Stitched continuous tenors. rank 0 = front month (M0),
        rank 1 = M+1, ... up to rank 11. Each row is one (date, rank).

  Fixed (outright)      ->  <code>_outright_ohlc.parquet
        Individual dated contracts keyed by expiry_ym (e.g. "2026-08").
        Each row is one (date, expiry_ym).

Both return a tidy OHLC frame indexed by date with columns:
    open, high, low, close, volume
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

# Display name -> (folder/prefix code, price unit, exchange code)
COMMODITIES: dict[str, dict[str, str]] = {
    "Gold":   {"code": "gold",   "unit": "$/oz", "symbol": "GC (COMEX)"},
    "Copper": {"code": "copper", "unit": "$/lb", "symbol": "HG (COMEX)"},
}

_OHLC = ["open", "high", "low", "close", "volume"]


def _parquet(code: str, name: str) -> Path:
    return DATA_DIR / code / f"{code}_{name}.parquet"


def _read(code: str, name: str) -> pd.DataFrame:
    df = pd.read_parquet(_parquet(code, name))
    df["date"] = pd.to_datetime(df["date"])
    # futures_ohlcv stores the close as 'futures_close'; normalise to 'close'
    if "close" not in df.columns and "futures_close" in df.columns:
        df = df.rename(columns={"futures_close": "close"})
    return df.sort_values("date")


# ── ROLLING / CONTINUOUS ──────────────────────────────────────────────────────
def available_tenors(code: str) -> list[int]:
    """Continuous tenor ranks available for this commodity (0 = front)."""
    df = _read(code, "tenor_ohlc")
    return sorted(int(r) for r in df["rank"].dropna().unique())


def tenor_label(rank: int) -> str:
    return "M0 (front month)" if rank == 0 else f"M+{rank}"


def load_rolling(code: str, rank: int) -> pd.DataFrame:
    """OHLC for one continuous tenor, indexed by date."""
    df = _read(code, "tenor_ohlc")
    df = df[df["rank"] == rank]
    return df.set_index("date")[_OHLC].dropna(how="all")


# ── FIXED / OUTRIGHT ──────────────────────────────────────────────────────────
def available_expiries(code: str, min_obs: int = 30) -> list[str]:
    """Outright contract expiries (YYYY-MM), chronological (earliest first).

    Filtered to contracts with at least `min_obs` daily bars so the dropdown
    isn't cluttered with barely-traded far-dated months.
    """
    df = _read(code, "outright_ohlc")
    counts = df.groupby("expiry_ym")["date"].size()
    keep = counts[counts >= min_obs].index
    return sorted(str(e) for e in keep)


def default_expiry(code: str) -> str:
    """The current 'front' outright: the contract that most recently traded at
    a real (non-zero) price. The close>0 filter skips stale/garbage prints on
    expired far contracts; ties break to the nearest expiry."""
    options = set(available_expiries(code))
    df = _read(code, "outright_ohlc")
    valid = df[(df["close"] > 0) & (df["expiry_ym"].astype(str).isin(options))]
    if valid.empty:
        opts = available_expiries(code)
        return opts[-1] if opts else ""
    last_traded = valid.groupby("expiry_ym")["date"].max()
    target = last_traded.max()
    return min(str(e) for e, d in last_traded.items() if d == target)


def load_fixed(code: str, expiry_ym: str) -> pd.DataFrame:
    """OHLC for one fixed/outright contract, indexed by date."""
    df = _read(code, "outright_ohlc")
    df = df[df["expiry_ym"].astype(str) == expiry_ym]
    return df.set_index("date")[_OHLC].dropna(how="all")


# ── VOLATILITY MEASURES (computed from OHLC) ──────────────────────────────────
def realised_vol(close: pd.Series, window: int, periods_per_year: int = 252) -> pd.Series:
    """Annualised realised volatility (decimal) = rolling std of daily log
    returns × √252, over `window` days."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=max(2, window // 2)).std() * np.sqrt(periods_per_year)


def atr(ohlc: pd.DataFrame, window: int) -> pd.Series:
    """N-day Average True Range (in price units). TR = max(H−L, |H−Cprev|, |L−Cprev|)."""
    high, low, close = ohlc["high"], ohlc["low"], ohlc["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=max(2, window // 2)).mean()


# ── IMPLIED VOL (ATM IV term-structure snapshots) ─────────────────────────────
def _read_iv(code: str) -> pd.DataFrame:
    df = pd.read_parquet(_parquet(code, "atm_iv_ts"))
    df["date"] = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    # Drop non-physical IV (zeros / thin far-contract garbage up to 900%+).
    df = df[(df["atm_iv"] > 0.01) & (df["atm_iv"] <= 2.0)]
    return df.sort_values("date")


def load_iv_rolling(code: str, rank: int) -> pd.Series:
    """ATM implied vol (decimal) for a continuous tenor.
    Rolling rank r (0=front) maps to IV tenor r+1 (1=front)."""
    df = _read_iv(code)
    s = df[df["tenor"] == rank + 1].drop_duplicates("date").set_index("date")["atm_iv"]
    return s.sort_index()


def load_iv_fixed(code: str, expiry_ym: str) -> pd.Series:
    """ATM implied vol (decimal) for a specific outright, matched on the IV
    row's contract expiry month — so it tracks that one contract over time."""
    df = _read_iv(code)
    mask = df["expiry"].dt.strftime("%Y-%m") == expiry_ym
    s = df[mask].drop_duplicates("date").set_index("date")["atm_iv"]
    return s.sort_index()

# ── TIMEFRAME RESAMPLING ──────────────────────────────────────────────────────
# label -> (pandas resample rule or None for raw daily, periods-per-year)
TIMEFRAMES: dict[str, tuple[str | None, int]] = {
    "Daily":   (None, 252),
    "Weekly":  ("W",  52),
    "Monthly": ("ME", 12),
}


def resample_ohlc(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    """Resample a daily OHLC frame to a coarser bar. rule=None -> unchanged."""
    if rule is None or df.empty:
        return df
    out = df.resample(rule).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    )
    return out.dropna(subset=["open"])


def resample_last(s: pd.Series, rule: str | None) -> pd.Series:
    """Resample a daily series to period-end last value. rule=None -> unchanged."""
    if rule is None or s.empty:
        return s
    return s.resample(rule).last()


# ── OPEN INTEREST (options chain) ─────────────────────────────────────────────
def _read_oi(code: str, name: str) -> pd.DataFrame:
    df = pd.read_parquet(_parquet(code, name))
    for col in ("snapshot", "expiry", "date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def oi_expiries(code: str) -> list[str]:
    """The near expiries available in the latest OI snapshot (front 3), YYYY-MM-DD."""
    df = _read_oi(code, "front3_oi")
    return [d.strftime("%Y-%m-%d") for d in sorted(df["expiry"].unique())]


def oi_by_strike(code: str, expiry: str) -> pd.DataFrame:
    """Latest-snapshot open interest by strike for one expiry, split call/put.
    Returns columns: strike, call_oi, put_oi, settle_c, settle_p."""
    df = _read_oi(code, "front3_oi")
    df = df[df["expiry"].dt.strftime("%Y-%m-%d") == expiry]
    piv = df.pivot_table(index="strike", columns="right",
                         values="open_interest", aggfunc="sum").fillna(0.0)
    piv = piv.rename(columns={"C": "call_oi", "P": "put_oi"})
    for c in ("call_oi", "put_oi"):
        if c not in piv.columns:
            piv[c] = 0.0
    return piv[["call_oi", "put_oi"]].reset_index().sort_values("strike")


def oi_term_structure(code: str) -> pd.DataFrame:
    """Total open interest per expiry (latest snapshot) — the OI distribution
    across the curve. Columns: expiry, total_oi."""
    df = _read_oi(code, "front3_oi")
    g = df.groupby(df["expiry"].dt.strftime("%Y-%m-%d"))["open_interest"].sum()
    return g.rename("total_oi").reset_index().rename(columns={"expiry": "expiry"})


def oi_history(code: str) -> pd.Series:
    """Total options open interest over time (sum across all strikes/expiries
    per date), from the historical chain. Indexed by date."""
    df = _read_oi(code, "hist_chain_oi")
    return df.groupby("date")["open_interest"].sum().sort_index()


# ── FUTURES OPEN INTEREST (CFTC COT) ──────────────────────────────────────────
COT_POSITIONS = {
    "managed_money_net": "Managed money (specs)",
    "producer_net": "Producers (commercials)",
    "swap_net": "Swap dealers",
}


def load_cot(code: str) -> pd.DataFrame:
    """Weekly CFTC Commitments-of-Traders for the futures market, indexed by
    date. Columns: open_interest (total futures OI) + net positioning by group
    (managed_money_net, producer_net, swap_net)."""
    df = pd.read_parquet(_parquet(code, "cot"))
    df["date"] = pd.to_datetime(df["date"])
    cols = ["open_interest", *COT_POSITIONS]
    keep = [c for c in cols if c in df.columns]
    return df.sort_values("date").set_index("date")[keep]


# ── FUTURES OPEN INTEREST PER CONTRACT (exchange, by expiry) ──────────────────
def has_futures_oi(code: str) -> bool:
    """True if a per-contract futures-OI snapshot file exists for this code."""
    return _parquet(code, "futures_oi").exists()


def load_futures_oi(code: str) -> pd.DataFrame:
    """Per-futures-contract open interest. Columns: date, expiry_ym,
    open_interest, settle (one row per contract per date)."""
    df = pd.read_parquet(_parquet(code, "futures_oi"))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def futures_oi_curve(code: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Latest-snapshot OI per futures expiry (the OI term structure).
    Returns (df[expiry_ym, open_interest, settle] sorted by expiry, snapshot_date)."""
    df = load_futures_oi(code)
    snap = df["date"].max()
    cur = df[(df["date"] == snap) & (df["open_interest"] > 0)].copy()
    return cur.sort_values("expiry_ym"), snap


def futures_oi_expiries(code: str, min_obs: int = 20) -> list[str]:
    """Futures contract months (YYYY-MM) with enough OI history, chronological."""
    df = load_futures_oi(code)
    counts = df.groupby("expiry_ym")["date"].size()
    return sorted(str(e) for e in counts[counts >= min_obs].index)


def futures_oi_front(code: str) -> str:
    """The most-active futures contract in the latest snapshot (max OI)."""
    cur, _ = futures_oi_curve(code)
    if cur.empty:
        return ""
    return str(cur.sort_values("open_interest", ascending=False)["expiry_ym"].iloc[0])


def futures_oi_series(code: str, expiry_ym: str) -> pd.Series:
    """Open interest over time for one futures contract, indexed by date."""
    df = load_futures_oi(code)
    s = df[df["expiry_ym"].astype(str) == expiry_ym].set_index("date")["open_interest"]
    return s[s > 0].sort_index()


# ── 25-DELTA SKEW (per-tenor smile summary over time) ─────────────────────────
def load_skew(code: str, rank: int) -> pd.DataFrame:
    """25Δ skew time-series for a continuous tenor. skew_ts rank is 1-based
    (1=front), so rolling rank r maps to skew rank r+1.
    Columns (vol points, %): atm, rr25, fly25 + derived call25_iv / put25_iv."""
    df = pd.read_parquet(_parquet(code, "skew_ts"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["rank"] == rank + 1].sort_values("date").set_index("date")
    out = df[["atm", "rr25", "fly25"]].copy()
    # RR = callIV - putIV ; BF = (callIV+putIV)/2 - atm  =>  reconstruct wings
    out["call25_iv"] = out["atm"] + out["fly25"] + out["rr25"] / 2.0
    out["put25_iv"] = out["atm"] + out["fly25"] - out["rr25"] / 2.0
    return out


# ── BLACK-76 IMPLIED VOL (per-strike smile snapshot) ──────────────────────────
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _b76_price(F: float, K: float, T: float, sigma: float, right: str) -> float:
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0) if right == "C" else max(K - F, 0.0)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / sq
    d2 = d1 - sq
    if right == "C":
        return F * _norm_cdf(d1) - K * _norm_cdf(d2)
    return K * _norm_cdf(-d2) - F * _norm_cdf(-d1)


def implied_vol_b76(price: float, F: float, K: float, T: float, right: str) -> float:
    """Bisection-invert a Black-76 (option-on-future) price to implied vol.
    Discounting r≈0 (settles are ~undiscounted). NaN if unsolvable."""
    if not (price > 0 and F > 0 and K > 0 and T > 0):
        return np.nan
    intrinsic = max(F - K, 0.0) if right == "C" else max(K - F, 0.0)
    if price < intrinsic - 1e-6:
        return np.nan
    lo, hi = 1e-4, 5.0
    p_lo = _b76_price(F, K, T, lo, right) - price
    p_hi = _b76_price(F, K, T, hi, right) - price
    if p_lo * p_hi > 0:
        return np.nan
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        p_mid = _b76_price(F, K, T, mid, right) - price
        if abs(p_mid) < 1e-7:
            return mid
        if p_lo * p_mid <= 0:
            hi, p_hi = mid, p_mid
        else:
            lo, p_lo = mid, p_mid
    return 0.5 * (lo + hi)


def iv_smile_front(code: str) -> tuple[pd.DataFrame, dict]:
    """Per-strike implied-vol smile for the FRONT contract, inverted from the
    latest frontmonth OI snapshot (which carries the futures level F).
    Returns (df[strike, right, iv, open_interest], meta)."""
    df = _read_oi(code, "frontmonth_oi")
    snap = df["snapshot"].max()
    exp = df["expiry"].max()
    F = float(df["F"].dropna().iloc[0]) if "F" in df.columns and df["F"].notna().any() else np.nan
    T = max((exp - snap).days, 0) / 365.0
    rows = []
    for _, r in df.iterrows():
        iv = implied_vol_b76(float(r["settle"]), F, float(r["strike"]), T, str(r["right"]))
        rows.append({"strike": float(r["strike"]), "right": str(r["right"]),
                     "iv": iv, "open_interest": float(r["open_interest"])})
    out = pd.DataFrame(rows).dropna(subset=["iv"])
    # keep a sensible vol band; drop deep-wing garbage
    out = out[(out["iv"] > 0.01) & (out["iv"] <= 2.0)].sort_values("strike")
    meta = {"snapshot": snap.date().isoformat(), "expiry": exp.date().isoformat(),
            "F": F, "T_years": round(T, 4)}
    return out, meta


# ── TECHNICAL INDICATORS (computed from OHLC) ─────────────────────────────────
MA_PERIODS = [5, 8, 10, 12, 20, 30, 50, 100, 200]


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    """Returns (mid, upper, lower) Bollinger Bands."""
    mid = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std()
    return mid, mid + k * sd, mid - k * sd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI (0-100)."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    """Returns (%K, %D) stochastic oscillator (0-100)."""
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    pct_k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return pct_k, pct_k.rolling(d, min_periods=d).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    """Returns (ADX, +DI, -DI) — Wilder's directional system."""
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    prev_c = close.shift()
    tr = pd.concat([(high - low).abs(), (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), plus_di, minus_di


def parabolic_sar(high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """Wilder's Parabolic SAR (stop-and-reverse), returned as a price series."""
    h, l = high.to_numpy(dtype=float), low.to_numpy(dtype=float)
    n = len(h)
    out = np.full(n, np.nan)
    if n < 2:
        return pd.Series(out, index=high.index)
    up = h[1] >= h[0]
    af = step
    ep = h[0] if up else l[0]
    sar = l[0] if up else h[0]
    out[0] = sar
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < sar:                       # reverse to down
                up, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > sar:                       # reverse to up
                up, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, max_step)
        out[i] = sar
    return pd.Series(out, index=high.index)
