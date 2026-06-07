"""
Gamma Wall Level Validator
--------------------------
For every trading day in 2022 (morning snapshot), computes GEX walls,
maps them from QQQ strikes -> NQ price equivalents, then checks NQ 1m data
to see whether price reversed at those levels and how far it ran.

Output: results/gex_level_validation.csv  + results/daily_reports/YYYY-MM-DD.txt

Usage:
  python backtest_levels.py
  python backtest_levels.py --date 2022-06-15          # single day
  python backtest_levels.py --top 6 --min-run 100      # tune params
"""

import argparse
import csv
import json
from datetime import datetime, time as dtime
from pathlib import Path

import pandas as pd
import numpy as np

from gex_engine import GEXProfile

# -- Paths ----------------------------------------------------------------------
ROOT = Path(__file__).parent
SNAP_DIR = ROOT / "historicaldata" / "optionsdata"
NQ_1M_PATH = ROOT / "historicaldata" / "NQ_1m_clean.csv"
NQ_DAILY_PATH = ROOT / "historicaldata" / "NQ_daily_clean.csv"
RESULTS_DIR = ROOT / "results"
DAILY_DIR = RESULTS_DIR / "daily_reports"

# -- Parameters -----------------------------------------------------------------
DEFAULT_TOP_WALLS = 6          # how many gamma walls to test per day
REVERSAL_PROXIMITY_QQQ = 0.30  # QQQ strike must come within this many $ of wall
ATR_PERIOD = 20                # rolling ATR window (used for near_gamma_flip signal only)
MIN_REVERSAL_PCT = 0.004       # min reversal = 0.4% of NQ spot price
STRONG_REVERSAL_PCT = 0.010    # strong reversal = 1.0% of NQ spot price
LOOKFORWARD_BARS = 120         # 1m bars to look forward after touch (2 hours)
SESSION_START = dtime(9, 30)
SESSION_END = dtime(16, 0)
VAP_BIN_SIZE = 20.0            # NQ points per volume-at-price bucket
LVN_THRESHOLD = 0.65           # path avg volume < this fraction of session mean = clean
PROXIMITY_NQ_PTS = 60.0        # wall within this many NQ pts of spot = proximity signal
TIGHT_PROXIMITY_NQ_PTS = 30.0  # tight proximity for S10
LOCAL_GRADIENT_RATIO = 4.0     # wall abs_gex >= this x mean of +-2 neighbor strikes = spike
FLIP_NEAR_ATR_MULT = 0.50      # spot within this x ATR of flip = "near flip" (used inverted)
ATR_REGIME_THRESHOLD = 350.0   # S16: 20-day ATR >= this = high-vol regime (36.6% vs 55.2% reversal rate)


def compute_vap(bars: pd.DataFrame, bin_size: float = VAP_BIN_SIZE) -> dict:
    """
    Build a volume-at-price profile from NQ 1m bars.
    Volume is distributed proportionally across the bar's high-low range into bins.
    Returns {"bins": array, "volumes": array} or {} if data unavailable.
    """
    if bars.empty or "volume" not in bars.columns:
        return {}
    lo = bars["low"].values.astype(float)
    hi = bars["high"].values.astype(float)
    vol = bars["volume"].values.astype(float)
    if vol.sum() == 0:
        return {}

    bin_lo = np.floor(lo.min() / bin_size) * bin_size
    bin_hi = np.ceil(hi.max() / bin_size) * bin_size + bin_size
    bins = np.arange(bin_lo, bin_hi, bin_size)
    n_bins = len(bins) - 1
    if n_bins == 0:
        return {}

    vol_profile = np.zeros(n_bins)
    bar_range = np.maximum(hi - lo, 0.01)
    for j in range(n_bins):
        overlap = np.minimum(hi, bins[j + 1]) - np.maximum(lo, bins[j])
        overlap = np.maximum(overlap, 0.0)
        vol_profile[j] = float(np.sum(vol * overlap / bar_range))

    return {"bins": bins, "volumes": vol_profile}


def is_clean_path(nq_spot: float, nq_wall: float, vap: dict,
                  threshold: float = LVN_THRESHOLD) -> bool:
    """
    Returns True if the price path from nq_spot to nq_wall is a low-volume zone.
    Uses prior-session VAP: path avg volume < threshold x session mean = clean air.
    """
    if not vap or "bins" not in vap:
        return False
    bins = vap["bins"]
    volumes = vap["volumes"]
    mean_vol = float(volumes[volumes > 0].mean()) if np.any(volumes > 0) else 0.0
    if mean_vol == 0:
        return False

    path_lo = min(nq_spot, nq_wall)
    path_hi = max(nq_spot, nq_wall)
    in_path = (bins[:-1] >= path_lo - VAP_BIN_SIZE) & (bins[1:] <= path_hi + VAP_BIN_SIZE)
    path_vols = volumes[in_path]
    if len(path_vols) == 0:
        return False

    return float(path_vols.mean()) < threshold * mean_vol


def compute_local_gradient(wall_strike: float, all_walls: list, n_neighbors: int = 2) -> float:
    """
    Ratio of wall's abs_gex to mean abs_gex of the n_neighbors strikes on each side.
    A high ratio means the wall is a true gamma spike vs its neighbors.
    Returns 0.0 if insufficient neighbors.
    """
    strikes_sorted = sorted(all_walls, key=lambda w: w.strike)
    strikes = [w.strike for w in strikes_sorted]
    gex_map = {w.strike: w.abs_gex for w in strikes_sorted}

    try:
        idx = strikes.index(wall_strike)
    except ValueError:
        return 0.0

    lo = max(0, idx - n_neighbors)
    hi = min(len(strikes), idx + n_neighbors + 1)
    neighbor_gex = [gex_map[s] for i, s in enumerate(strikes[lo:hi], start=lo) if i != idx]
    if not neighbor_gex:
        return 0.0
    mean_neighbor = float(np.mean(neighbor_gex))
    if mean_neighbor == 0:
        return 0.0
    return gex_map[wall_strike] / mean_neighbor


def load_nq_daily_atr() -> dict[str, float]:
    """
    Returns a dict of {date_str: atr_value} using 20-day rolling ATR
    from the NQ daily CSV. Each day's ATR represents the typical range
    for that session -- used to set percentage-based reversal thresholds
    instead of fixed point values that become meaningless across price regimes.
    """
    df = pd.read_csv(NQ_DAILY_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    return {str(row["date"].date()): row["atr"] for _, row in df.iterrows() if pd.notna(row["atr"])}


def load_nq_1m() -> pd.DataFrame:
    df = pd.read_csv(NQ_1M_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["date_only"] = df["date"].dt.date
    df["time_only"] = df["date"].dt.time
    df = df[(df["time_only"] >= SESSION_START) & (df["time_only"] <= SESSION_END)]
    return df


def get_nq_at_time(nq_df: pd.DataFrame, date_str: str, snap_time: str) -> float | None:
    """Get NQ close price at snapshot time (T0936 -> 09:36, T1551 -> 15:51)."""
    hour = int(snap_time[:2])
    minute = int(snap_time[2:])
    target_time = dtime(hour, minute)
    day_df = nq_df[nq_df["date_only"].astype(str) == date_str]
    if day_df.empty:
        return None
    # Find closest bar
    diffs = day_df["time_only"].apply(
        lambda t: abs((t.hour * 60 + t.minute) - (hour * 60 + minute))
    )
    idx = diffs.idxmin()
    return float(day_df.loc[idx, "close"])


def validate_wall(
    wall_strike: float,
    wall_abs_gex: float,
    wall_gex: float,
    qqq_spot: float,
    nq_spot: float,
    nq_day: pd.DataFrame,
    snap_bar_idx: int,
    min_run: float = 100.0,
    strong_run: float = 150.0,
) -> dict:
    """
    Core question: after price touches this gamma wall, does it reverse?

    Finds the first bar where price touches the wall, then measures the max
    move in the reversal direction within LOOKFORWARD_BARS (2 hours).
    Primary metric: reversal_pts >= min_run (ATR-based threshold).
    HOD/LOD is recorded as secondary context only.
    """
    ratio = nq_spot / qqq_spot
    nq_wall = round(wall_strike * ratio, 2)
    prox_nq = REVERSAL_PROXIMITY_QQQ * ratio

    session_bars = nq_day.iloc[snap_bar_idx:].copy().reset_index(drop=True)
    if session_bars.empty:
        return _null_result(wall_strike, nq_wall, wall_abs_gex, wall_gex, ratio)

    session_high = float(session_bars["high"].max())
    session_low  = float(session_bars["low"].min())
    day_range    = round(session_high - session_low, 1)

    if nq_wall > nq_spot:
        approach_direction = "from_below"
        run_direction = "short"
        touch_mask = session_bars["high"].values >= nq_wall - prox_nq
        if not np.any(touch_mask):
            return _null_result(wall_strike, nq_wall, wall_abs_gex, wall_gex, ratio, touched=False)
        touch_idx = int(np.argmax(touch_mask))
        touch_price = float(session_bars.loc[touch_idx, "high"])
        touch_time = session_bars.loc[touch_idx, "time_only"] if "time_only" in session_bars.columns else None
        forward = session_bars.iloc[touch_idx:touch_idx + LOOKFORWARD_BARS]
        reversal_pts = round(touch_price - float(forward["low"].min()), 1) if not forward.empty else 0.0
        was_hod_or_lod = abs(nq_wall - session_high) <= prox_nq

    elif nq_wall < nq_spot:
        approach_direction = "from_above"
        run_direction = "long"
        touch_mask = session_bars["low"].values <= nq_wall + prox_nq
        if not np.any(touch_mask):
            return _null_result(wall_strike, nq_wall, wall_abs_gex, wall_gex, ratio, touched=False)
        touch_idx = int(np.argmax(touch_mask))
        touch_price = float(session_bars.loc[touch_idx, "low"])
        touch_time = session_bars.loc[touch_idx, "time_only"] if "time_only" in session_bars.columns else None
        forward = session_bars.iloc[touch_idx:touch_idx + LOOKFORWARD_BARS]
        reversal_pts = round(float(forward["high"].max()) - touch_price, 1) if not forward.empty else 0.0
        was_hod_or_lod = abs(nq_wall - session_low) <= prox_nq

    else:
        return _null_result(wall_strike, nq_wall, wall_abs_gex, wall_gex, ratio, touched=False)

    touch_min = touch_time.hour * 60 + touch_time.minute if touch_time else None
    return {
        "qqq_strike": wall_strike,
        "nq_equivalent": nq_wall,
        "abs_gex": round(wall_abs_gex, 1),
        "net_gex": round(wall_gex, 1),
        "qqq_nq_ratio": round(ratio, 4),
        "touched": True,
        "approach_direction": approach_direction,
        "touch_price_nq": round(touch_price, 1),
        "touch_time_min": touch_min,
        "session_high": round(session_high, 1),
        "session_low": round(session_low, 1),
        "day_range": day_range,
        "reversal_pts": reversal_pts,
        "reversed": reversal_pts >= min_run,
        "reversed_strong": reversal_pts >= strong_run,
        "run_points": reversal_pts,
        "run_direction": run_direction,
        "was_hod_or_lod": was_hod_or_lod,
    }


def _null_result(strike, nq_wall, abs_gex, gex, ratio, touched=False):
    return {
        "qqq_strike": strike,
        "nq_equivalent": nq_wall,
        "abs_gex": round(abs_gex, 1),
        "net_gex": round(gex, 1),
        "qqq_nq_ratio": round(ratio, 4),
        "touched": touched,
        "approach_direction": None,
        "touch_price_nq": None,
        "touch_time_min": None,
        "session_high": None,
        "session_low": None,
        "day_range": None,
        "reversal_pts": 0.0,
        "reversed": False,
        "reversed_strong": False,
        "run_points": 0.0,
        "run_direction": None,
        "was_hod_or_lod": False,
    }


def process_snapshot(
    snap_path: Path,
    session_date: str,       # date whose NQ session to validate against
    nq_df: pd.DataFrame,
    top_n: int,
    atr_map: dict[str, float],
    snap_role: str,          # "same_day" or "next_day"
    vap_cache: dict[str, dict] | None = None,
    persistent_strikes: set | None = None,
) -> list[dict]:
    """Validate GEX walls from one snapshot against a specific NQ session."""
    try:
        profile = GEXProfile.from_file(snap_path)
    except Exception as e:
        print(f"  [ERROR] {snap_path.name}: {e}")
        return []

    snap_time = profile.snapshot_time

    # For same_day snaps, start from snapshot bar. For next_day snaps, start from open.
    day_nq = nq_df[nq_df["date_only"].astype(str) == session_date].copy().reset_index(drop=True)
    if day_nq.empty:
        return []

    if snap_role == "same_day":
        snap_hour = int(snap_time[:2])
        snap_min = int(snap_time[2:])
        snap_t = snap_hour * 60 + snap_min
        diffs = day_nq["time_only"].apply(lambda t: abs(t.hour * 60 + t.minute - snap_t))
        snap_bar_idx = int(diffs.idxmin())
        nq_spot = float(day_nq.loc[snap_bar_idx, "close"])
    else:
        # Next-day: use session open, validate full session
        snap_bar_idx = 0
        nq_spot = float(day_nq.iloc[0]["open"])

    atr = atr_map.get(session_date)
    min_run = nq_spot * MIN_REVERSAL_PCT
    strong_run = nq_spot * STRONG_REVERSAL_PCT

    top_walls = profile.top_walls(n=top_n)
    walls_by_gex = sorted(top_walls, key=lambda w: w.abs_gex, reverse=True)
    top_abs = walls_by_gex[0].abs_gex if walls_by_gex else 1.0
    second_abs = walls_by_gex[1].abs_gex if len(walls_by_gex) > 1 else 0.0
    dominant_strike = walls_by_gex[0].strike if walls_by_gex else None
    top_is_dominant = second_abs > 0 and top_abs >= 1.5 * second_abs

    # Build rank map: rank by abs_gex descending within this snapshot
    wall_rank_map = {w.strike: rank + 1 for rank, w in enumerate(walls_by_gex)}

    results = []
    for wall in top_walls:
        r = validate_wall(
            wall_strike=wall.strike,
            wall_abs_gex=wall.abs_gex,
            wall_gex=wall.gex,
            qqq_spot=profile.spot,
            nq_spot=nq_spot,
            nq_day=day_nq,
            snap_bar_idx=snap_bar_idx,
            min_run=min_run,
            strong_run=strong_run,
        )
        r["date"] = session_date
        r["snap_file"] = snap_path.name
        r["snap_role"] = snap_role
        r["qqq_spot"] = profile.spot
        r["nq_spot_at_snap"] = round(nq_spot, 1)
        snap_time_raw = snap_path.stem.split("-T")[-1]
        snap_h, snap_m = int(snap_time_raw[:2]), int(snap_time_raw[2:])
        r["snap_time_min"] = snap_h * 60 + snap_m
        r["regime"] = "negative" if profile.net_gex < 0 else "positive"
        r["gamma_flip"] = profile.gamma_flip
        r["atr"] = round(atr, 1) if atr else None
        r["min_run_threshold"] = round(min_run, 1)
        r["strong_run_threshold"] = round(strong_run, 1)

        ratio = nq_spot / profile.spot
        nq_wall_price = wall.strike * ratio

        # S1: Wall is close to spot at snapshot time (gamma is fresh and near)
        s1_proximity = abs(nq_wall_price - nq_spot) <= PROXIMITY_NQ_PTS

        # S2: DEX opposes approach direction (dealers hedge in mean-reverting direction)
        # from_below (wall above) -> DEX < 0 (dealers short delta, sell rallies)
        # from_above (wall below) -> DEX > 0 (dealers long delta, buy dips)
        # Use wall position vs spot (not touch result) so untouched walls are scored correctly
        inferred_approach = "from_below" if nq_wall_price > nq_spot else "from_above"
        s2_neg_dex = (inferred_approach == "from_below" and profile.dex < 0) or \
                     (inferred_approach == "from_above" and profile.dex > 0)

        # S3: Net GEX is negative (inverted from old signal -- neg GEX = better wall holds)
        s3_neg_regime = profile.net_gex < 0

        # S4: Wall is a local gamma spike vs neighboring strikes
        gradient = compute_local_gradient(wall.strike, profile.walls)
        s4_local_gradient = gradient >= LOCAL_GRADIENT_RATIO

        # S5: Spot is NOT near the gamma flip (near flip = unreliable positioning)
        s5_far_from_flip = True
        if profile.gamma_flip and atr and not np.isnan(atr):
            flip_nq = profile.gamma_flip * ratio
            s5_far_from_flip = abs(nq_spot - flip_nq) >= FLIP_NEAR_ATR_MULT * atr

        # S6: ITM put wall above spot (from_below + neg strike GEX)
        # Dealers short ITM puts hedge aggressively = strong resistance at this level
        s6_itm_options = (r.get("approach_direction") == "from_below" and wall.gex < 0)

        # S7: Premium gamma day -- Mon/Wed (0DTE Mon + mid-week positioning)
        # Fri excluded: 47.5% rev rate = below 49% base rate in backtest
        session_dow = datetime.strptime(session_date, "%Y-%m-%d").weekday()  # 0=Mon 4=Fri
        s7_good_day = session_dow in (0, 2)

        # S8: Dominant snapshot -- top wall's GEX is 1.5x+ the second wall (concentrated gamma)
        s8_dominant_snap = top_is_dominant

        # S9: Fresh wall -- strike appears in only 1 snapshot today (NOT persistent)
        # Persistent walls (repeated in AM+PM) show LOWER reversal rate (46.5% vs 52.6%)
        # Fresh/unexpected walls that only appeared once = stronger mean-reversion
        snap_date = snap_path.stem.split("-T")[0].replace("qqq-", "")  # YYYY-MM-DD
        is_persistent = (persistent_strikes is not None and
                         (snap_date, wall.strike) in persistent_strikes)
        s9_persistent = not is_persistent  # inverted: fresh wall = signal ON

        # S10: Tight proximity -- wall within 30 NQ pts of spot (likely tested in first 30 min)
        s10_tight_proximity = abs(nq_wall_price - nq_spot) <= TIGHT_PROXIMITY_NQ_PTS

        # S11: Next-day snapshot (end-of-day QQQ positioning predicts next session better)
        # Empirically: next_day = 52.5% base vs same_day = 44.8% base reversal rate
        s11_same_day = snap_role == "next_day"

        # S12: Top-1 wall by abs_gex in this snapshot (rank 2 falls to 45.5%, only rank 1 = 59.7%)
        wall_rank = wall_rank_map.get(wall.strike, 99)
        s12_top_wall = wall_rank == 1

        # S13: LVN clean path -- prior-session VAP shows low volume between spot and wall
        session_vap = (vap_cache or {}).get(session_date, {})
        lvn_value = is_clean_path(nq_spot, nq_wall_price, session_vap)
        s13_lvn_clean = lvn_value

        # S14: ITM put wall AND proximal -- the 70.5% combo gets a bonus point
        s14_itm_proximal = s1_proximity and s6_itm_options

        # S15: approach from_below (at score 8+: 69.2% from_below vs 47.1% from_above)
        s15_from_below = (r.get("approach_direction") == "from_below")

        # S16: ATR regime gate -- 20-day NQ ATR >= 350 = high-vol environment
        # NOT a confluence signal: ATR is date-level (same for all walls), not wall-specific.
        # Adding to score inflates 63% of walls uniformly and destroys elite tier (81%→63%).
        # Use as OPERATIONAL GATE: score 5+ AND ATR>=380 → 68.4% at 2.12/wk (volume tier)
        #                          score 6+ AND ATR>=380 → 73.4% at 1.08/wk (precision tier)
        #                          score 7+ no ATR gate  → 69.1% at 1.33/wk (pure confluence)
        s16_atr_regime = bool(atr and atr >= ATR_REGIME_THRESHOLD)

        # S2, S3, S8 dropped: +0.7pp, +1.8pp, +2.6pp respectively -- below noise floor
        confluence_score = sum([s1_proximity,
                                s4_local_gradient, s5_far_from_flip,
                                s6_itm_options, s7_good_day,
                                s9_persistent, s10_tight_proximity, s11_same_day,
                                s12_top_wall, s13_lvn_clean, s14_itm_proximal,
                                s15_from_below])

        # Keep old signal columns for reference
        r["dex_opposing"]        = profile.dex < 0 if nq_wall_price > nq_spot else profile.dex > 0
        r["regime_stabilizing"]  = profile.net_gex > 0
        r["near_gamma_flip"]     = not s5_far_from_flip
        r["lvn_clean_path"]      = lvn_value
        r["dist_from_spot"]      = round(abs(nq_wall_price - nq_spot), 1)
        r["confluence_score"]    = confluence_score
        r["net_dex"]             = round(profile.dex, 2)
        r["s1_proximity"]        = s1_proximity
        r["s2_neg_dex"]          = s2_neg_dex
        r["s3_neg_regime"]       = s3_neg_regime
        r["s4_local_gradient"]   = s4_local_gradient
        r["s5_far_from_flip"]    = s5_far_from_flip
        r["s6_itm_options"]      = s6_itm_options
        r["s7_good_day"]         = s7_good_day
        r["s8_dominant_snap"]    = s8_dominant_snap
        r["s9_persistent"]       = s9_persistent
        r["s10_tight_proximity"] = s10_tight_proximity
        r["s11_same_day"]        = s11_same_day
        r["s12_top_wall"]        = s12_top_wall
        r["s13_lvn_clean"]       = s13_lvn_clean
        r["s14_itm_proximal"]    = s14_itm_proximal
        r["s15_from_below"]      = s15_from_below
        r["s16_atr_regime"]      = s16_atr_regime
        r["wall_rank"]           = wall_rank
        r["local_gradient"]      = round(gradient, 3)
        r["dex_aligned"]         = s2_neg_dex
        r["regime_trending"]     = s3_neg_regime
        r["wall_dominant"]       = s8_dominant_snap
        r["good_distance"]       = s1_proximity

        results.append(r)

    return results


def process_day(
    date_str: str,
    next_date_str: str | None,
    nq_df: pd.DataFrame,
    top_n: int,
    atr_map: dict[str, float],
    vap_cache: dict[str, dict] | None = None,
    persistent_strikes: set | None = None,
) -> list[dict]:
    all_snaps = sorted(SNAP_DIR.glob(f"qqq-{date_str}-T*.txt"))
    if not all_snaps:
        return []

    results = []
    for snap_path in all_snaps:
        snap_time_raw = snap_path.stem.split("-T")[-1]
        hour = int(snap_time_raw[:2])

        if hour < 12:
            rows = process_snapshot(snap_path, date_str, nq_df, top_n, atr_map,
                                    "same_day", vap_cache, persistent_strikes)
        else:
            if next_date_str is None:
                continue
            rows = process_snapshot(snap_path, next_date_str, nq_df, top_n, atr_map,
                                    "next_day", vap_cache, persistent_strikes)

        results.extend(rows)

    return results


def write_daily_report(date_str: str, rows: list[dict]) -> None:
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    out = DAILY_DIR / f"{date_str}.txt"
    lines = [f"=== GEX Level Validation: {date_str} ===\n"]
    if not rows:
        lines.append("No data.\n")
    else:
        r0 = rows[0]
        lines.append(f"QQQ Spot:    {r0['qqq_spot']}")
        lines.append(f"NQ Spot:     {r0['nq_spot_at_snap']}")
        lines.append(f"GEX Regime:  {r0['regime'].upper()}")
        lines.append(f"Gamma Flip:  {r0['gamma_flip']}")
        lines.append(f"Snapshot:    {r0['snap_file']}\n")
        lines.append(f"{'Strike':>8}  {'NQ Eq':>8}  {'|GEX|':>12}  {'Touched':>7}  {'Reversed':>8}  {'RevPts':>7}  {'Dir':>6}  {'HOD/LOD':>7}")
        lines.append("-" * 80)
        for r in rows:
            lines.append(
                f"{r['qqq_strike']:>8.1f}  {r['nq_equivalent']:>8.1f}  "
                f"{r['abs_gex']:>12,.0f}  {str(r['touched']):>7}  "
                f"{str(r.get('reversed', False)):>8}  "
                f"{r.get('reversal_pts', 0.0):>7.1f}  {str(r['run_direction'] or ''):>6}  "
                f"{str(r.get('was_hod_or_lod', False)):>7}"
            )
        touched = sum(1 for r in rows if r["touched"])
        reversed_count = sum(1 for r in rows if r.get("reversed"))
        reversed_strong = sum(1 for r in rows if r.get("reversed_strong"))
        lines.append("")
        lines.append(f"Summary: {touched}/{len(rows)} touched  |  {reversed_count}/{touched} reversed (min)  |  {reversed_strong}/{touched} reversed (strong)  |  ATR:{rows[0].get('atr','?')}")
    with open(out, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="GEX Level Backtester")
    parser.add_argument("--date", help="Single date YYYY-MM-DD")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_WALLS, help="Top N gamma walls per day")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading NQ daily ATR...")
    atr_map = load_nq_daily_atr()
    print(f"  ATR computed for {len(atr_map)} trading days")

    print("Loading NQ 1m data...")
    nq_df = load_nq_1m()
    print(f"  {len(nq_df):,} bars loaded  ({nq_df['date'].min().date()} to {nq_df['date'].max().date()})")

    if args.date:
        dates = [args.date]
    else:
        # All unique dates with any snapshot
        snap_files = sorted(SNAP_DIR.glob("qqq-2022-*-T*.txt"))
        seen = set()
        dates = []
        for f in snap_files:
            parts = f.stem.split("-")   # qqq-YYYY-MM-DD-THHSS
            date_str = f"{parts[1]}-{parts[2]}-{parts[3]}"
            if date_str not in seen:
                seen.add(date_str)
                dates.append(date_str)

    # Precompute persistent strikes: (date, strike) pairs that appear in 2+ snapshots same day
    print("Building persistent strike cache...")
    persistent_strikes: set[tuple] = set()
    for date_str in dates:
        day_snaps = sorted(SNAP_DIR.glob(f"qqq-{date_str}-T*.txt"))
        if len(day_snaps) < 2:
            continue
        snap_strikes: list[set] = []
        for sp in day_snaps:
            try:
                p = GEXProfile.from_file(sp)
                snap_strikes.append({w.strike for w in p.top_walls(n=DEFAULT_TOP_WALLS)})
            except Exception:
                continue
        if len(snap_strikes) >= 2:
            for strike in snap_strikes[0].intersection(*snap_strikes[1:]):
                persistent_strikes.add((date_str, strike))
    print(f"  {len(persistent_strikes)} persistent (date, strike) pairs found")

    # Precompute prior-session volume-at-price profiles for LVN signal
    print("Building volume-at-price cache (prior session per trading day)...")
    vap_cache: dict[str, dict] = {}
    for i, date_str in enumerate(dates):
        if i == 0:
            continue
        prev_date = dates[i - 1]
        prev_bars = nq_df[nq_df["date_only"].astype(str) == prev_date]
        vap_cache[date_str] = compute_vap(prev_bars)
    print(f"  VAP profiles built for {len(vap_cache)} sessions")

    print(f"Processing {len(dates)} trading days (morning same-day + afternoon next-day)...")

    all_rows = []
    for i, date_str in enumerate(dates):
        next_date = dates[i + 1] if i + 1 < len(dates) else None
        rows = process_day(date_str, next_date, nq_df, args.top, atr_map, vap_cache, persistent_strikes)
        all_rows.extend(rows)
        if rows:
            touched = sum(1 for r in rows if r["touched"])
            atr_val = rows[0].get("atr", "?")
            reversed_count = sum(1 for r in rows if r.get("reversed"))
            reversed_strong = sum(1 for r in rows if r.get("reversed_strong"))
            snaps = len(set(r["snap_file"] for r in rows))
            print(f"  [{i+1}/{len(dates)}] {date_str}  ATR:{atr_val}  snaps:{snaps}  walls:{len(rows)}  touched:{touched}  reversed:{reversed_count}  strong:{reversed_strong}")
            write_daily_report(date_str, rows)

    if not all_rows:
        print("No results generated.")
        return

    # Write master CSV
    out_csv = RESULTS_DIR / "gex_level_validation.csv"
    fieldnames = [
        "date", "snap_role", "qqq_spot", "nq_spot_at_snap", "regime", "gamma_flip",
        "qqq_strike", "nq_equivalent", "abs_gex", "net_gex", "qqq_nq_ratio",
        "touched", "approach_direction", "touch_price_nq", "touch_time_min", "snap_time_min",
        "session_high", "session_low", "day_range",
        "reversal_pts", "reversed", "reversed_strong", "run_direction",
        "was_hod_or_lod", "run_points",
        "atr", "min_run_threshold", "strong_run_threshold", "snap_file",
        "confluence_score",
        "s1_proximity", "s2_neg_dex", "s3_neg_regime", "s4_local_gradient", "s5_far_from_flip",
        "s6_itm_options", "s7_good_day", "s8_dominant_snap", "s9_persistent", "s10_tight_proximity", "s11_same_day", "s12_top_wall", "s13_lvn_clean", "s14_itm_proximal", "s15_from_below", "s16_atr_regime",
        "wall_rank",
        "local_gradient", "net_dex",
        "dist_from_spot",
        "dex_opposing", "regime_stabilizing", "near_gamma_flip", "lvn_clean_path",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary stats
    touched_rows       = [r for r in all_rows if r["touched"]]
    reversed_rows      = [r for r in touched_rows if r.get("reversed")]
    reversed_strong_rows = [r for r in touched_rows if r.get("reversed_strong")]
    hod_lod_rows       = [r for r in touched_rows if r.get("was_hod_or_lod")]
    avg_rev_pts        = np.mean([r["reversal_pts"] for r in reversed_rows]) if reversed_rows else 0

    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS  ({len(dates)} days, top {args.top} walls/day)")
    print(f"{'='*60}")
    print(f"Total walls tested:          {len(all_rows)}")
    print(f"Touched price:               {len(touched_rows)} ({100*len(touched_rows)/len(all_rows):.1f}%)")
    if touched_rows:
        pct_rev = 100 * len(reversed_rows) / len(touched_rows)
        pct_strong = 100 * len(reversed_strong_rows) / len(touched_rows)
        pct_hod = 100 * len(hod_lod_rows) / len(touched_rows)
        freq_rev = len(reversed_rows) / len(dates)
        freq_strong = len(reversed_strong_rows) / len(dates)
        print(f"Reversed (>= min_run ATR):   {len(reversed_rows)} ({pct_rev:.1f}% of touched)  {freq_rev:.2f}/day")
        print(f"Reversed (>= strong_run):    {len(reversed_strong_rows)} ({pct_strong:.1f}% of touched)  {freq_strong:.2f}/day")
        print(f"Avg reversal pts (reversed): {avg_rev_pts:.1f}")
        print(f"Wall = HOD or LOD (ref):     {len(hod_lod_rows)} ({pct_hod:.1f}% of touched)")

    # Confluence score breakdown
    if touched_rows and any("confluence_score" in r for r in touched_rows):
        print(f"\n{'-'*60}")
        print(f"CONFLUENCE SCORE BREAKDOWN  (touched walls, reversed = primary metric)")
        print(f"{'-'*60}")
        print(f"  {'Score':>5}  {'Touched':>7}  {'Reversed':>9}  {'Prec%':>6}  {'AvgRev':>8}  {'Freq/day':>9}")
        print(f"  {'-'*5}  {'-'*7}  {'-'*9}  {'-'*6}  {'-'*8}  {'-'*9}")
        for min_score in range(10):
            subset = [r for r in touched_rows if r.get("confluence_score", 0) >= min_score]
            hits = [r for r in subset if r.get("reversed")]
            if subset:
                prec = 100.0 * len(hits) / len(subset)
                avg_run = np.mean([r["reversal_pts"] for r in hits]) if hits else 0.0
                freq = len(subset) / len(dates)
                print(f"  {min_score}+    {len(subset):>7}  {len(hits):>9}  {prec:>5.1f}%  {avg_run:>8.1f}  {freq:>8.2f}")

        print(f"\n  Signal contribution (of touched walls that reversed):")
        signals = [
            ("s1_proximity",      "S1 proximity (<= 60 NQ pts from spot)"),
            ("s4_local_gradient", "S4 local gradient spike (>= 4.0x)"),
            ("s5_far_from_flip",  "S5 far from gamma flip"),
            ("s6_itm_options",    "S6 ITM put wall (from_below + neg GEX)"),
            ("s7_good_day",       "S7 premium day (Mon/Wed only)"),
            ("s9_persistent",     "S9 fresh wall (NOT persistent -- only in 1 snap)"),
            ("s10_tight_proximity", "S10 tight proximity (<= 30 NQ pts from spot)"),
            ("s11_same_day",       "S11 next-day snap (EOD positioning predicts next session)"),
            ("s12_top_wall",       "S12 top-1 wall by abs GEX (rank 1 = 59.7%)"),
            ("s13_lvn_clean",      "S13 LVN clean path (prior-session low volume to wall)"),
            ("s14_itm_proximal",   "S14 ITM put wall + proximal (S1 AND S6 combo bonus)"),
            ("s15_from_below",     "S15 approach from below (directional bias)"),
        ]
        for key, label in signals:
            with_sig = [r for r in touched_rows if r.get(key)]
            hits_with = [r for r in with_sig if r.get("reversed")]
            without_sig = [r for r in touched_rows if not r.get(key)]
            hits_without = [r for r in without_sig if r.get("reversed")]
            pct_with = 100 * len(hits_with) / len(with_sig) if with_sig else 0
            pct_without = 100 * len(hits_without) / len(without_sig) if without_sig else 0
            print(f"    {label:<30}: {pct_with:5.1f}% (n={len(with_sig):4})  vs  {pct_without:5.1f}% without")

    print(f"\nResults saved -> {out_csv}")
    print(f"Daily reports  -> {DAILY_DIR}/")


if __name__ == "__main__":
    main()
