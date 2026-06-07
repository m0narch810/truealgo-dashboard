"""
GEX Engine — computes Gamma Exposure profile from a QQQ options chain snapshot.

GEX convention (SpotGamma / standard dealer-flow):
  Dealers sell calls to customers → dealers SHORT those calls → SHORT gamma on calls
  Dealers buy puts from customers → dealers LONG those puts → SHORT gamma on puts
  (puts have positive gamma but dealers are short the contracts)

  GEX at strike = (call_OI - put_OI) × gamma × 100 × spot
    > 0  → net call dominance at strike → dealers short gamma there → DESTABILIZING
    < 0  → net put dominance at strike  → dealers long gamma there  → also short gamma

  For wall detection we use ABS(GEX) — concentration of gamma regardless of sign.
  Gamma flip = strike where cumulative GEX (summed from lowest to highest strike) crosses zero.

Usage:
  from gex_engine import GEXProfile
  profile = GEXProfile.from_file("path/to/snapshot.txt")
  print(profile.gamma_flip)
  print(profile.top_walls(n=5))
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np


MAX_DTE_FOR_GEX = 7     # only include expirations within this many days (0DTE dominates intraday)
MIN_OI = 1              # skip strikes with zero open interest on both sides


@dataclass
class StrikeGEX:
    strike: float
    dte: int
    expiry_key: str
    call_oi: int
    put_oi: int
    gamma: float           # per-share gamma (same for call and put at same strike/exp)
    gex: float             # signed net GEX in dollar-gamma units
    abs_gex: float = field(init=False)

    def __post_init__(self):
        self.abs_gex = abs(self.gex)


@dataclass
class GEXProfile:
    snapshot_file: str
    snapshot_time: str      # e.g. "T0936"
    spot: float             # QQQ underlying price at snapshot time
    walls: list[StrikeGEX]  # all strikes with nonzero GEX, sorted by strike
    net_gex: float          # sum of all GEX → positive = destabilizing regime
    gamma_flip: Optional[float]  # strike where cumulative GEX crosses zero
    max_dte_used: int
    dex: float = 0.0        # net dealer delta exposure; positive = dealers net long delta

    @classmethod
    def from_file(cls, path: str | Path, max_dte: int = MAX_DTE_FOR_GEX) -> "GEXProfile":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        spot: float = d["underlyingPrice"]
        cmap: dict = d.get("callExpDateMap", {})
        pmap: dict = d.get("putExpDateMap", {})

        # Aggregate gamma×OI across all strikes for expirations within max_dte
        # Key: (strike, dte, exp_key) → {call_oi, put_oi, gamma}
        agg: dict[tuple, dict] = {}

        for side, chain in (("call", cmap), ("put", pmap)):
            for exp_key, strikes_dict in chain.items():
                dte = int(exp_key.split(":")[1])
                if dte > max_dte:
                    continue
                for strike_str, contracts in strikes_dict.items():
                    strike = float(strike_str)
                    c = contracts[0]
                    gamma = abs(float(c.get("gamma") or 0))
                    delta = float(c.get("delta") or 0)
                    oi = int(c.get("openInterest") or 0)
                    key = (strike, dte, exp_key)
                    if key not in agg:
                        agg[key] = {"call_oi": 0, "put_oi": 0, "gamma": gamma,
                                    "call_delta": 0.0, "put_delta": 0.0}
                    if side == "call":
                        agg[key]["call_oi"] += oi
                        agg[key]["call_delta"] = delta   # positive value ~0-1
                    else:
                        agg[key]["put_oi"] += oi
                        agg[key]["put_delta"] = delta    # negative value ~-1 to 0
                        # gamma can differ slightly by side for far strikes; take max
                        agg[key]["gamma"] = max(agg[key]["gamma"], gamma)

        # Aggregate per-expiration entries into a single entry per strike
        # (sum GEX across all expirations at the same strike price)
        strike_agg: dict[float, dict] = {}
        for (strike, dte, exp_key), v in agg.items():
            call_oi = v["call_oi"]
            put_oi = v["put_oi"]
            gamma = v["gamma"]
            if call_oi + put_oi < MIN_OI:
                continue
            gex = (call_oi - put_oi) * gamma * 100 * spot
            if strike not in strike_agg:
                strike_agg[strike] = {
                    "call_oi": 0, "put_oi": 0,
                    "gamma": 0.0, "gex": 0.0,
                    "min_dte": dte, "dominant_exp": exp_key,
                }
            sa = strike_agg[strike]
            sa["call_oi"] += call_oi
            sa["put_oi"] += put_oi
            sa["gex"] += gex
            if gamma > sa["gamma"]:
                sa["gamma"] = gamma
            if dte < sa["min_dte"]:
                sa["min_dte"] = dte
                sa["dominant_exp"] = exp_key

        walls: list[StrikeGEX] = []
        for strike, sa in strike_agg.items():
            walls.append(StrikeGEX(
                strike=strike,
                dte=sa["min_dte"],
                expiry_key=sa["dominant_exp"],
                call_oi=sa["call_oi"],
                put_oi=sa["put_oi"],
                gamma=sa["gamma"],
                gex=sa["gex"],
            ))

        walls.sort(key=lambda w: w.strike)

        net_gex = sum(w.gex for w in walls)
        gamma_flip = _find_gamma_flip(walls)

        # DEX = net dealer delta exposure (positive = dealers need to buy as price rises)
        # sum_strikes [(call_OI × call_delta + put_OI × put_delta) × 100 × spot]
        net_dex = sum(
            (v["call_oi"] * v.get("call_delta", 0.0) + v["put_oi"] * v.get("put_delta", 0.0)) * 100 * spot
            for v in agg.values()
            if v["call_oi"] + v["put_oi"] >= MIN_OI
        )

        # Parse snapshot time from filename: qqq-YYYY-MM-DD-THHSS.txt
        snap_time = path.stem.split("-T")[-1] if "-T" in path.stem else "unknown"

        return cls(
            snapshot_file=path.name,
            snapshot_time=snap_time,
            spot=spot,
            walls=walls,
            net_gex=net_gex,
            gamma_flip=gamma_flip,
            max_dte_used=max_dte,
            dex=net_dex,
        )

    def top_walls(self, n: int = 8) -> list[StrikeGEX]:
        """Return the N strikes with the highest absolute GEX."""
        return sorted(self.walls, key=lambda w: w.abs_gex, reverse=True)[:n]

    def regime(self) -> str:
        return "NEGATIVE GEX (destabilizing / trending)" if self.net_gex < 0 else "POSITIVE GEX (stabilizing / mean-reverting)"

    def to_dict(self) -> dict:
        top = self.top_walls(8)
        return {
            "file": self.snapshot_file,
            "time": self.snapshot_time,
            "spot": self.spot,
            "net_gex": round(self.net_gex, 2),
            "dex": round(self.dex, 2),
            "regime": self.regime(),
            "gamma_flip": self.gamma_flip,
            "top_walls": [
                {
                    "strike": w.strike,
                    "dte": w.dte,
                    "abs_gex": round(w.abs_gex, 2),
                    "gex": round(w.gex, 2),
                    "call_oi": w.call_oi,
                    "put_oi": w.put_oi,
                }
                for w in top
            ],
        }


def _find_gamma_flip(walls: list[StrikeGEX]) -> Optional[float]:
    """
    Cumulative GEX sweep from lowest strike upward.
    Gamma flip = first strike where running sum crosses zero.
    """
    if not walls:
        return None
    cumulative = 0.0
    prev_strike = None
    prev_cum = None
    for w in walls:
        cumulative += w.gex
        if prev_cum is not None and prev_cum * cumulative < 0:
            # Linear interpolation between prev_strike and w.strike
            span = w.strike - prev_strike
            ratio = abs(prev_cum) / (abs(prev_cum) + abs(cumulative))
            return round(prev_strike + span * ratio, 2)
        prev_strike = w.strike
        prev_cum = cumulative
    return None
