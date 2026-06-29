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

from pathlib import Path

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