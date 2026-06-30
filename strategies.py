"""
strategies.py
=============
Strategies Lab — a leak-free **calendar-spread** strategy:

    LONG the front-active contract  /  SHORT the next-active contract
    spread_t = close(front_active_t) - close(next_active_t)      (price points)

Signal blends three reads, z-scored on trailing windows:
  • COT positioning (specs)  — managed money vs producers, release-lagged.
  • Active-contract OI tilt   — (OI_front − OI_next)/(OI_front + OI_next).
  • Spread mean-reversion     — fade an extended spread (−z_spread).

Roll-/OI-aware gating (the active contract's life-cycle):
  • A position opens only AFTER the active contract's open interest has peaked
    (OI builds then rolls off into the roll — that peak is the entry window).
  • The position is forced FLAT a few days before the **roll** (First Position
    Date), i.e. it is held at most until the last day before the active becomes
    the front/spot month. The roll itself is visible as the OI crossover where
    next-active OI overtakes the active.

Discipline (no look-ahead): point-in-time contract selection (calendar only),
COT release-lagged then forward-filled, trailing z-scores, 1-session execution
lag, roll-gap returns zeroed. IS/OOS: chronological split, params fit on IS
only, embargo at the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

import active_contracts as ac
import data_loader as dl

TRADING_DAYS = 252


# ── PARAMS ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Params:
    z_window: int = 252        # trailing window for z-scores / OI/COT standardisation
    z_enter: float = 1.0       # |score| above this opens a position
    z_exit: float = 0.25       # |score| below this closes it (hysteresis)
    w_cot: float = 1.0         # weight on the COT positioning (specs) component
    w_oi: float = 1.0          # weight on the OI/volume tilt component
    w_spread: float = 0.5      # weight on the spread mean-reversion component (−z_spread)
    cot_lag_days: int = 3      # COT report→publication lag (Tue→Fri)
    exec_lag: int = 1          # sessions between signal and the position being on
    min_periods: int = 60      # z-score warm-up
    # ── roll-/OI-aware gating ────────────────────────────────────────────────
    use_oi_peak: bool = True   # only open once the active contract's OI has peaked
    peak_drop: float = 0.0     # "peaked" = OI below (1−peak_drop)·running max in-contract
    exit_buffer_days: int = 3  # force flat this many days before the roll (First Position Date)

    def label(self) -> str:
        return (f"win={self.z_window} enter={self.z_enter} exit={self.z_exit} "
                f"w=(cot {self.w_cot}, oi {self.w_oi}, sprd {self.w_spread})"
                f"{' +peak' if self.use_oi_peak else ''} exitBuf={self.exit_buffer_days}d")


# ── DATA BUILDING ─────────────────────────────────────────────────────────────
def _outright_pivot(code: str, field: str) -> pd.DataFrame:
    """date × expiry_ym matrix of a field from outright_ohlc (close>0 filtered).
    expiry_ym is normalised to a 'YYYY-MM' string so columns match active_legs."""
    df = dl._read(code, "outright_ohlc")
    df = df[df["close"] > 0].assign(expiry_ym=lambda d: d["expiry_ym"].astype(str))
    piv = df.pivot_table(index="date", columns="expiry_ym", values=field, aggfunc="last")
    return piv.sort_index()


def _pick(piv: pd.DataFrame, dates: pd.DatetimeIndex, yms) -> np.ndarray:
    """Vector of piv[date, ym] for aligned (date, ym) pairs; NaN if absent."""
    cols = {c: i for i, c in enumerate(piv.columns)}
    arr = piv.to_numpy()
    row_of = {d: i for i, d in enumerate(piv.index)}
    out = np.full(len(dates), np.nan)
    for k, (d, ym) in enumerate(zip(dates, yms)):
        i, j = row_of.get(d), cols.get(ym)
        if i is not None and j is not None:
            out[k] = arr[i, j]
    return out


def build_legs(code: str) -> pd.DataFrame:
    """Per trading day: front/next contracts, their closes, the spread and a
    roll flag. Index = price dates where both legs are present."""
    px = _outright_pivot(code, "close")
    legs = ac.active_legs(code, px.index)
    legs["front_close"] = _pick(px, px.index, legs["front_ym"])
    legs["next_close"] = _pick(px, px.index, legs["next_ym"])
    legs["spread"] = legs["front_close"] - legs["next_close"]
    return legs.dropna(subset=["spread"])


# ── SIGNAL COMPONENTS ─────────────────────────────────────────────────────────
def _trailing_z(s: pd.Series, win: int, min_periods: int) -> pd.Series:
    mu = s.rolling(win, min_periods=min_periods).mean()
    sd = s.rolling(win, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0, np.nan)


def cot_norm(code: str, index: pd.DatetimeIndex, lag_days: int) -> pd.Series:
    """Net spec-vs-commercial positioning, normalised by total OI, release-lagged
    and forward-filled onto `index`. (managed_money_net - producer_net)/open_interest."""
    cot = dl.load_cot(code)
    raw = (cot["managed_money_net"] - cot["producer_net"]) / cot["open_interest"].replace(0, np.nan)
    raw = raw.dropna()
    raw.index = raw.index + pd.Timedelta(days=lag_days)        # shift to release date
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    return raw.reindex(index, method="ffill")


def active_oi(code: str, legs: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Per-day OI of the front-active and next-active legs (+ tilt). Real futures
    OI for gold; per-contract volume fallback for copper. (df[oi_front, oi_next,
    tilt], source label)."""
    if dl.has_futures_oi(code):
        oi = dl.load_futures_oi(code).assign(expiry_ym=lambda d: d["expiry_ym"].astype(str))
        piv = oi.pivot_table(index="date", columns="expiry_ym",
                             values="open_interest", aggfunc="last").sort_index()
        piv = piv.reindex(legs.index).ffill(limit=3)
        source = "futures OI (per contract)"
    else:
        piv = _outright_pivot(code, "volume").reindex(legs.index)
        piv = piv.rolling(5, min_periods=1).mean()           # damp single-day noise
        source = "per-contract volume (no futures OI)"
    f = _pick(piv, legs.index, legs["front_ym"])
    n = _pick(piv, legs.index, legs["next_ym"])
    out = pd.DataFrame({"oi_front": f, "oi_next": n}, index=legs.index)
    denom = out["oi_front"] + out["oi_next"]
    out["tilt"] = np.where(denom > 0, (out["oi_front"] - out["oi_next"]) / denom, np.nan)
    return out, source


def days_to_roll(code: str, legs: pd.DataFrame) -> pd.Series:
    """Calendar days from each date to the front contract's First Position Date
    (when it stops being active). Computed from the calendar only."""
    def _fpd(ym: str) -> pd.Timestamp:
        return ac.first_position_date(int(ym[:4]), int(ym[5:7]))
    fpd = pd.DatetimeIndex(legs["front_ym"].map(_fpd).values)
    return pd.Series((fpd - legs.index).days, index=legs.index)


def _oi_peaked(oi_front: pd.Series, front_ym: pd.Series, drop: float) -> pd.Series:
    """True once the active contract's OI has come off its in-contract peak, i.e.
    OI < (1−drop)·running-max within the current front contract (reset each roll)."""
    g = pd.DataFrame({"oi": oi_front.to_numpy(), "ym": front_ym.to_numpy()},
                     index=oi_front.index)
    runmax = g.groupby("ym")["oi"].cummax()
    return (oi_front < runmax * (1.0 - drop)).fillna(False)


def base_features(code: str, legs: pd.DataFrame, p: Params) -> tuple[pd.DataFrame, str]:
    """All weight-independent inputs (computed once; the grid only re-weights)."""
    cot = cot_norm(code, legs.index, p.cot_lag_days)
    oi, source = active_oi(code, legs)
    feat = pd.DataFrame(index=legs.index)
    feat["z_cot"] = _trailing_z(cot, p.z_window, p.min_periods)
    feat["z_oi"] = _trailing_z(oi["tilt"], p.z_window, p.min_periods)
    feat["z_spread"] = _trailing_z(legs["spread"], p.z_window, p.min_periods)
    feat["oi_front"] = oi["oi_front"]
    feat["oi_next"] = oi["oi_next"]
    feat["tilt"] = oi["tilt"]
    feat["days_to_roll"] = days_to_roll(code, legs)
    feat["oi_peaked"] = _oi_peaked(oi["oi_front"], legs["front_ym"], p.peak_drop)
    return feat, source


def score_from(feat: pd.DataFrame, p: Params) -> pd.Series:
    """Directional score (>0 → long spread). Spread enters as mean-reversion."""
    return (p.w_cot * feat["z_cot"].fillna(0)
            + p.w_oi * feat["z_oi"].fillna(0)
            - p.w_spread * feat["z_spread"].fillna(0))


def positions_from_features(feat: pd.DataFrame, score: pd.Series, p: Params):
    """Long/short/flat with hysteresis, OI-peak entry gate and a forced flat
    before the roll. Returns (position series, list of (date, from, to) events)."""
    sc = score.to_numpy()
    peaked = feat["oi_peaked"].to_numpy()
    dtr = feat["days_to_roll"].to_numpy()
    idx = feat.index
    pos = np.zeros(len(idx))
    events, cur = [], 0
    for i in range(len(idx)):
        v = sc[i]
        near_roll = (not np.isnan(dtr[i])) and dtr[i] <= p.exit_buffer_days
        if near_roll or np.isnan(v):
            new = 0
        elif cur == 0:
            can_open = bool(peaked[i]) if p.use_oi_peak else True
            new = 1 if (can_open and v > p.z_enter) else (-1 if (can_open and v < -p.z_enter) else 0)
        else:
            if abs(v) < p.z_exit:
                new = 0
            elif v > p.z_enter:
                new = 1
            elif v < -p.z_enter:
                new = -1
            else:
                new = cur
        if new != cur:
            events.append((idx[i], cur, new))
        pos[i] = new
        cur = new
    return pd.Series(pos, index=idx), events


# ── BACKTEST ──────────────────────────────────────────────────────────────────
def _daily_pnl(legs: pd.DataFrame, position: pd.Series, exec_lag: int) -> pd.Series:
    """position · Δspread, with roll-day changes zeroed and a 1-session exec lag."""
    d_spread = legs["spread"].diff()
    d_spread = d_spread.mask(legs["roll"], 0.0).fillna(0.0)   # roll gap ≠ return
    return position.shift(exec_lag).fillna(0.0) * d_spread


def metrics(pnl: pd.Series, position: pd.Series) -> dict:
    pnl = pnl.dropna()
    if pnl.empty:
        return {"sharpe": np.nan, "total": 0.0, "hit": np.nan,
                "max_dd": 0.0, "n_trades": 0, "pct_flat": np.nan}
    sd = pnl.std()
    sharpe = (pnl.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan
    equity = pnl.cumsum()
    max_dd = float((equity.cummax() - equity).max())
    traded = pnl[position.shift(1).reindex(pnl.index).fillna(0) != 0]
    hit = float((traded > 0).mean()) if len(traded) else np.nan
    changes = position.reindex(pnl.index).fillna(0).diff().abs()
    n_trades = int((changes > 0).sum())
    pct_flat = float((position.reindex(pnl.index).fillna(0) == 0).mean())
    return {"sharpe": float(sharpe), "total": float(equity.iloc[-1]), "hit": hit,
            "max_dd": max_dd, "n_trades": n_trades, "pct_flat": pct_flat}


def backtest(code: str, p: Params, legs: pd.DataFrame | None = None,
             feat: pd.DataFrame | None = None, source: str = "") -> dict:
    if legs is None:
        legs = build_legs(code)
    if feat is None:
        feat, source = base_features(code, legs, p)
    score = score_from(feat, p)
    position, events = positions_from_features(feat, score, p)
    pnl = _daily_pnl(legs, position, p.exec_lag)
    return {"legs": legs, "feat": feat, "score": score, "position": position,
            "events": events, "pnl": pnl, "equity": pnl.cumsum(), "source": source}


# ── IS / OOS ──────────────────────────────────────────────────────────────────
@dataclass
class Result:
    code: str
    params: Params
    legs: pd.DataFrame
    feat: pd.DataFrame
    score: pd.Series
    position: pd.Series
    events: list
    pnl: pd.Series
    boundary: pd.Timestamp
    is_metrics: dict
    oos_metrics: dict
    grid: pd.DataFrame
    oi_source: str = ""
    embargo: int = 0


def _grid(base: Params) -> list[Params]:
    out = []
    for z_enter in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        for w in ((1.0, 1.0), (1.0, 0.5), (0.5, 1.0), (1.0, 0.0), (0.0, 1.0)):
            out.append(replace(base, z_enter=z_enter, z_exit=min(0.25, z_enter / 2),
                               w_cot=float(w[0]), w_oi=float(w[1])))
    return out


def run_is_oos(code: str, is_frac: float = 0.60, embargo: int = 252,
               trade_floor: int = 5, base: Params | None = None) -> Result:
    """Fit the (z_enter, w_cot, w_oi) grid on the IS segment only, evaluate OOS
    once. The roll/OI-peak gating params come from `base` and are held fixed."""
    base = base or Params()
    legs = build_legs(code)
    idx = legs.index
    if len(idx) < 200:
        raise ValueError(f"{code}: not enough spread history ({len(idx)} rows) for IS/OOS.")
    feat, source = base_features(code, legs, base)      # features depend only on base
    cut = int(len(idx) * is_frac)
    boundary = idx[cut]
    is_idx = idx[:cut]
    oos_idx = idx[cut + embargo:]

    rows, best, best_sharpe = [], None, -np.inf
    for p in _grid(base):
        score = score_from(feat, p)
        position, _ = positions_from_features(feat, score, p)
        pnl = _daily_pnl(legs, position, p.exec_lag)
        m_is = metrics(pnl.reindex(is_idx), position)
        rows.append({"params": p.label(), "z_enter": p.z_enter, "w_cot": p.w_cot,
                     "w_oi": p.w_oi, "is_sharpe": m_is["sharpe"], "is_trades": m_is["n_trades"]})
        s = m_is["sharpe"]
        if (m_is["n_trades"] >= trade_floor and s is not None
                and not np.isnan(s) and s > best_sharpe):
            best_sharpe, best = s, p

    p = best if best is not None else replace(base)
    score = score_from(feat, p)
    position, events = positions_from_features(feat, score, p)
    pnl = _daily_pnl(legs, position, p.exec_lag)
    return Result(
        code=code, params=p, legs=legs, feat=feat, score=score, position=position,
        events=events, pnl=pnl, boundary=boundary,
        is_metrics=metrics(pnl.reindex(is_idx), position),
        oos_metrics=metrics(pnl.reindex(oos_idx), position),
        grid=pd.DataFrame(rows), oi_source=source, embargo=embargo,
    )
