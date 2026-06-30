"""
active_contracts.py
===================
CME/COMEX "active month" definitions and point-in-time selection of the
front-active and next-active contracts, used by the Strategies Lab.

Definitions verified against CME notices (settlement-procedure "active month"
wording + delivery process):

  • The active (lead) contract is the nearest contract of the liquid delivery
    cycle that is NOT yet in its spot/delivery phase (ex-spot).
  • It stops being active on its **First Position Date** = the business day
    immediately preceding **First Notice Day**, where FND = the last business
    day of the month immediately PRIOR to the delivery month (COMEX metals).
  • Note: every calendar month is *listed*; the 5/6-month cycle below is the
    liquid desk convention (where volume/OI concentrate), which is what we
    trade — not the literal "nearest listed month" CME definition.

Active delivery cycles (month numbers + CME letter codes):
  Copper HG : Mar, May, Jul, Sep, Dec      = H, K, N, U, Z
  Gold   GC : Feb, Apr, Jun, Aug, Oct, Dec = G, J, M, Q, V, Z

Selection is computed from the business-day calendar only, so it is fully
point-in-time and never peeks at future prices or open interest.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

# code -> (cycle month numbers, CME month codes)
CYCLES: dict[str, dict] = {
    "copper": {"months": (3, 5, 7, 9, 12), "codes": ("H", "K", "N", "U", "Z")},
    "gold":   {"months": (2, 4, 6, 8, 10, 12), "codes": ("G", "J", "M", "Q", "V", "Z")},
}

MONTH_CODE = {3: "H", 5: "K", 7: "N", 9: "U", 12: "Z",
              2: "G", 4: "J", 6: "M", 8: "Q", 10: "V"}

# Year span covering the bundled data (2018-2026) with head/tail room.
_YEAR_MIN, _YEAR_MAX = 2015, 2031


def _ym(y: int, m: int) -> str:
    return f"{y:04d}-{m:02d}"


@lru_cache(maxsize=512)
def first_notice_day(year: int, month: int) -> pd.Timestamp:
    """COMEX metals First Notice Day for the (year, month) delivery contract:
    the last business day of the month immediately prior to the delivery month."""
    py, pm = (year - 1, 12) if month == 1 else (year, month - 1)
    return pd.Timestamp(py, pm, 1) + pd.offsets.BMonthEnd(0)


@lru_cache(maxsize=512)
def first_position_date(year: int, month: int) -> pd.Timestamp:
    """The day the contract stops being 'active' — one business day before FND."""
    return first_notice_day(year, month) - pd.offsets.BDay(1)


@lru_cache(maxsize=8)
def _cycle_table(code: str) -> tuple[list[str], np.ndarray]:
    """(ordered list of cycle-contract YYYY-MM, their roll dates as int64 ns)."""
    months = CYCLES[code]["months"]
    contracts = sorted(
        (y, m) for y in range(_YEAR_MIN, _YEAR_MAX + 1) for m in months
    )
    yms = [_ym(y, m) for y, m in contracts]
    rolls = np.array([first_position_date(y, m).value for y, m in contracts], dtype="int64")
    return yms, rolls


def active_pair(code: str, on: pd.Timestamp) -> tuple[str, str]:
    """(front_active_ym, next_active_ym) for `code` as of date `on`, point-in-time.

    front = nearest cycle contract whose First Position Date is still in the
    future (hasn't rolled into delivery); next = the following cycle contract.
    """
    yms, rolls = _cycle_table(code)
    t = pd.Timestamp(on).value
    idx = int(np.searchsorted(rolls, t, side="right"))  # first roll strictly after t
    idx = min(idx, len(yms) - 2)
    return yms[idx], yms[idx + 1]


def active_legs(code: str, dates) -> pd.DataFrame:
    """front_ym / next_ym for each date in `dates` (DatetimeIndex), plus a
    `roll` flag marking the first day a new front contract takes over."""
    dates = pd.DatetimeIndex(dates)
    yms, rolls = _cycle_table(code)
    # rolls are int64 nanoseconds (Timestamp.value); normalise dates to ns too,
    # since a datetime64[s] index's asi8 would be seconds and break the compare.
    dts_ns = dates.as_unit("ns").asi8
    pos = np.searchsorted(rolls, dts_ns, side="right")
    pos = np.clip(pos, 0, len(yms) - 2)
    front = [yms[i] for i in pos]
    nxt = [yms[i + 1] for i in pos]
    out = pd.DataFrame({"front_ym": front, "next_ym": nxt}, index=dates)
    out["roll"] = out["front_ym"].ne(out["front_ym"].shift())
    out.iloc[0, out.columns.get_loc("roll")] = False  # first row is not a roll
    return out
