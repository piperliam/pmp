# pmp/seasons/mars_seasons.py
"""
Mars arrival season utilities for PMP.

Refactored from MAGPIE's `season_calc.py` to be library-friendly.

Features:
- Compute Mars areocentric solar longitude Ls [deg] for a datetime.
- Map Ls to a northern hemisphere season string.
- Given a departure datetime + TOF (days), compute arrival datetime,
  Mars Ls, and season label.

Dependencies:
    pip install marstime
"""

from __future__ import annotations
import datetime
from dataclasses import dataclass
from typing import Tuple

import marstime  # type: ignore


# -------------------------------
# Low-level helpers (kept from MAGPIE)
# -------------------------------

def j2000_offset_days(dt: datetime.datetime) -> float:
    """Convert datetime (UTC) to days since J2000 epoch."""
    j2000_epoch = datetime.datetime(2000, 1, 1, 12, 0, 0,
                                    tzinfo=datetime.timezone.utc)
    delta = dt - j2000_epoch
    return delta.total_seconds() / 86400.0


def mars_ls(dt: datetime.datetime) -> float:
    """
    Compute Mars areocentric solar longitude Ls [deg] for given datetime.

    Parameters
    ----------
    dt : datetime (timezone-aware UTC recommended)

    Returns
    -------
    float
        Areocentric solar longitude Ls in degrees [0, 360).
    """
    # NOTE: same logic as MAGPIE season_calc.py, just wrapped nicely.
    return float(marstime.Mars_Ls(j2000_offset_days(dt)))


def mars_north_season(ls_deg: float) -> str:
    """
    Map Ls to northern hemisphere season string.

    This is intentionally still "northern hemisphere" because that's
    the standard, but your mission can always interpret it relative
    to landing latitude.

    Parameters
    ----------
    ls_deg : float
        Ls in degrees [0, 360).

    Returns
    -------
    str
        Human-readable season label.
    """
    ls = ls_deg % 360.0
    if 0 <= ls < 90:
        return "Northern Spring (Ls 0-90°)"
    elif 90 <= ls < 180:
        return "Northern Summer (Ls 90-180°)"
    elif 180 <= ls < 270:
        return "Northern Autumn (Ls 180-270°)"
    else:
        return "Northern Winter (Ls 270-360°)"


# -------------------------------
# Higher-level mission helper
# -------------------------------

@dataclass
class MarsArrivalSeason:
    departure_utc: datetime.datetime
    tof_days: float
    arrival_utc: datetime.datetime
    ls_deg: float
    season_north: str

    def as_dict(self) -> dict:
        return {
            "departure_utc": self.departure_utc.isoformat(),
            "tof_days": float(self.tof_days),
            "arrival_utc": self.arrival_utc.isoformat(),
            "ls_deg": float(self.ls_deg),
            "season_north": self.season_north,
        }


def get_arrival_ls_and_season(
    depart_dt_utc: datetime.datetime,
    tof_days: float,
) -> MarsArrivalSeason:
    """
    Compute arrival datetime, Mars Ls, and northern season.

    Parameters
    ----------
    depart_dt_utc : datetime.datetime
        Departure datetime in UTC (timezone-aware strongly recommended).
    tof_days : float
        Time of flight in days.

    Returns
    -------
    MarsArrivalSeason
        Structured result with arrival time, Ls, and season label.
    """
    arrival_dt = depart_dt_utc + datetime.timedelta(days=tof_days)
    ls_deg = mars_ls(arrival_dt)
    season = mars_north_season(ls_deg)

    return MarsArrivalSeason(
        departure_utc=depart_dt_utc,
        tof_days=tof_days,
        arrival_utc=arrival_dt,
        ls_deg=ls_deg,
        season_north=season,
    )
