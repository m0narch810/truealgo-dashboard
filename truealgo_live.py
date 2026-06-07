"""
TrueAlgo Live Scanner
=====================
12-signal confluence model on live QQQ/NQ gamma walls.

Output: QQQ strike | score (0-12) | signals fired | ATR gate

Data sources:
  FreeFlow API  - QQQ options chain (GEX per strike), live QQQ + NQ spot
  yfinance NQ=F - 20-day ATR (regime gate)
  yfinance NQ=F - prior-session 1m bars (LVN clean path, S13)

Run:
  python truealgo_live.py             # score once and exit
  python truealgo_live.py --live      # auto-schedule during market hours only

Config (edit truealgo_config.json to update session cookie):
  {"ff_session": "your_cookie_here"}
"""

import os
import sys
import json
import time
import argparse
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta, time as dtime
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────────────────────────
# FREEFLOW AUTH
# Cookie priority: truealgo_config.json > FF_SESSION env var > hardcoded fallback
# To update: edit truealgo_config.json — no code changes needed.
# ─────────────────────────────────────────────────────────────
_CONFIG_FILE = Path(__file__).parent / "truealgo_config.json"

def _load_ff_session() -> str:
    # 1. Config file (easiest to update)
    if _CONFIG_FILE.exists():
        try:
            cfg = json.loads(_CONFIG_FILE.read_text(encoding='utf-8'))
            if cfg.get('ff_session'):
                return cfg['ff_session']
        except Exception:
            pass
    # 2. Environment variable (used by GitHub Actions)
    if os.environ.get('FF_SESSION'):
        return os.environ['FF_SESSION']
    raise RuntimeError(
        "FF_SESSION not configured. "
        "Add ff_session to truealgo_config.json or set FF_SESSION env var."
    )

FF_HEADERS = {
    'Accept': '*/*',
    'Connection': 'keep-alive',
    'Referer': 'https://www.free-flow.site/?auth=success',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'sec-ch-ua-platform': '"Windows"',
}
FF_BASE   = "https://www.free-flow.site/api"
SYMBOL    = "QQQ"
ET        = ZoneInfo("America/New_York")
CACHE_DIR = Path(__file__).parent / "truealgo_cache"
CACHE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# PARAMETERS  (match backtest_levels.py exactly)
# ─────────────────────────────────────────────────────────────
TOP_N                  = 6
PROXIMITY_NQ_PTS       = 60.0
TIGHT_PROXIMITY_NQ_PTS = 30.0
LOCAL_GRADIENT_RATIO   = 4.0
FLIP_NEAR_ATR_MULT     = 0.50
ATR_REGIME_THRESHOLD   = 350.0
LVN_THRESHOLD          = 0.65
VAP_BIN_SIZE           = 20.0
ATR_PERIOD             = 20

# ─────────────────────────────────────────────────────────────
# WALL DATACLASS  (lightweight StrikeGEX equivalent)
# ─────────────────────────────────────────────────────────────
@dataclass
class Wall:
    qqq_strike: float   # QQQ ETF strike  ← output to user
    nq_strike:  float   # NQ futures equivalent  ← used for proximity calcs
    gex:        float   # signed net GEX
    abs_gex:    float   # |gex|

    # keep .strike pointing at QQQ so compute_local_gradient works unchanged
    @property
    def strike(self) -> float:
        return self.qqq_strike


# ─────────────────────────────────────────────────────────────
# FREEFLOW FETCH + AGGREGATE
# ─────────────────────────────────────────────────────────────
def _ff_session() -> requests.Session:
    s = requests.Session()
    s.cookies.update({'ff_session': _load_ff_session()})
    s.headers.update(FF_HEADERS)
    return s


def fetch_freeflow(exp: str | None = None) -> tuple[pd.DataFrame, float, float]:
    """
    Returns (agg_df, nq_spot, qqq_spot).
    If exp is None, finds the next trading day with available data (handles weekends).
    """
    sess = _ff_session()

    if exp is None:
        # Try today first, then walk forward up to 5 trading days
        candidates = [date.today()]
        d = date.today()
        while len(candidates) < 6:
            d += timedelta(days=1)
            if d.weekday() < 5:
                candidates.append(d)
        for candidate in candidates:
            exp = candidate.strftime("%Y-%m-%d")
            r   = sess.get(f"{FF_BASE}/futures-levels?symbol={SYMBOL}&exp={exp}", timeout=15)
            r.raise_for_status()
            data = r.json()
            if data.get('rows'):
                print(f"  Using expiry {exp}")
                break
        else:
            raise ValueError("No FreeFlow data found for next 5 trading days")
    else:
        r    = sess.get(f"{FF_BASE}/futures-levels?symbol={SYMBOL}&exp={exp}", timeout=15)
        r.raise_for_status()
        data = r.json()

    rows = data.get('rows', [])
    if not rows:
        raise ValueError(f"FreeFlow returned no rows for {exp}")

    df = pd.DataFrame(rows)

    # Ensure strike_futures exists
    if 'strike_futures' not in df.columns:
        ratio = data.get('ratio', 41.14)
        df['strike_futures'] = (df['strike_etf'] * ratio).round(1)

    # Per-row OI split
    oi    = pd.to_numeric(df.get('oi', 0), errors='coerce').fillna(0)
    right = df['right'] if 'right' in df.columns else pd.Series([''] * len(df))
    df['_call_oi'] = oi.where(right == 'C', 0.0)
    df['_put_oi']  = oi.where(right == 'P', 0.0)

    agg = df.groupby('strike_futures').agg(
        strike_etf = ('strike_etf',  'first'),
        net_gex    = ('gex',         'sum'),
        abs_gex    = ('ag',          'sum'),
        net_dex    = ('dex',         'sum'),
        call_oi    = ('_call_oi',    'sum'),
        put_oi     = ('_put_oi',     'sum'),
        total_oi   = ('oi',          'sum'),
    ).reset_index()

    nq_spot  = float(data.get('futures_price', 0))
    qqq_spot = float(data.get('etf_spot', 0))
    return agg, nq_spot, qqq_spot


# ─────────────────────────────────────────────────────────────
# BUILD WALL LIST + GAMMA FLIP
# ─────────────────────────────────────────────────────────────
def build_walls(agg: pd.DataFrame) -> list[Wall]:
    walls = []
    for _, row in agg.iterrows():
        walls.append(Wall(
            qqq_strike = float(row['strike_etf']),
            nq_strike  = float(row['strike_futures']),
            gex        = float(row['net_gex']),
            abs_gex    = float(row['abs_gex']),
        ))
    walls.sort(key=lambda w: w.qqq_strike)
    return walls


def compute_gamma_flip(walls: list[Wall]) -> float | None:
    """Cumulative GEX sweep in QQQ strike space; returns QQQ strike of zero-cross."""
    cumulative = 0.0
    prev_strike = prev_cum = None
    for w in walls:
        cumulative += w.gex
        if prev_cum is not None and prev_cum * cumulative < 0:
            span  = w.qqq_strike - prev_strike
            ratio = abs(prev_cum) / (abs(prev_cum) + abs(cumulative))
            return round(prev_strike + span * ratio, 2)
        prev_strike = w.qqq_strike
        prev_cum    = cumulative
    return None


# ─────────────────────────────────────────────────────────────
# YFINANCE: ATR + PRIOR-SESSION VAP
# ─────────────────────────────────────────────────────────────
def fetch_nq_atr() -> float | None:
    """20-day True Range ATR from NQ daily bars."""
    try:
        df = yf.download('NQ=F', period='60d', interval='1d', progress=False, auto_adjust=True)
        if df.empty or len(df) < ATR_PERIOD + 1:
            return None
        hi = df['High'].squeeze()
        lo = df['Low'].squeeze()
        cl = df['Close'].squeeze()
        tr = pd.concat([
            hi - lo,
            (hi - cl.shift(1)).abs(),
            (lo - cl.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(ATR_PERIOD).mean().iloc[-1])
        return atr if not np.isnan(atr) else None
    except Exception as e:
        print(f"  [ATR] yfinance error: {e}")
        return None


def fetch_prior_vap() -> dict:
    """
    Build volume-at-price profile from the most recent completed NQ session.
    Returns VAP dict with 'bins' and 'volumes' arrays (20-pt NQ buckets).
    """
    try:
        df = yf.download('NQ=F', period='5d', interval='1m', progress=False, auto_adjust=True)
        if df.empty:
            return {}

        # Convert index to ET
        idx = df.index
        if hasattr(idx, 'tz') and idx.tz is not None:
            idx_et = idx.tz_convert('America/New_York')
        else:
            idx_et = idx.tz_localize('UTC').tz_convert('America/New_York')

        df.index = idx_et
        today_et = datetime.now(ET).date()

        # Find the most recent completed session (yesterday or earlier)
        target_date = today_et - timedelta(days=1)
        for _ in range(5):
            session = df[
                (df.index.date == target_date) &
                (df.index.time >= dtime(9, 30)) &
                (df.index.time <= dtime(16, 0))
            ]
            if not session.empty:
                break
            target_date -= timedelta(days=1)

        if session.empty:
            return {}

        hi  = session['High'].values.astype(float)
        lo  = session['Low'].values.astype(float)
        vol = session['Volume'].values.astype(float)

        if vol.sum() == 0:
            return {}

        bin_lo = np.floor(lo.min() / VAP_BIN_SIZE) * VAP_BIN_SIZE
        bin_hi = np.ceil(hi.max()  / VAP_BIN_SIZE) * VAP_BIN_SIZE + VAP_BIN_SIZE
        bins   = np.arange(bin_lo, bin_hi, VAP_BIN_SIZE)
        n_bins = len(bins) - 1
        if n_bins == 0:
            return {}

        vol_profile = np.zeros(n_bins)
        bar_range   = np.maximum(hi - lo, 0.01)
        for j in range(n_bins):
            overlap = np.minimum(hi, bins[j + 1]) - np.maximum(lo, bins[j])
            overlap = np.maximum(overlap, 0.0)
            vol_profile[j] = float(np.sum(vol * overlap / bar_range))

        return {'bins': bins, 'volumes': vol_profile}

    except Exception as e:
        print(f"  [VAP] yfinance error: {e}")
        return {}


def is_clean_path(nq_spot: float, nq_wall: float, vap: dict) -> bool:
    if not vap or 'bins' not in vap:
        return False
    bins, volumes = vap['bins'], vap['volumes']
    mean_vol = float(volumes[volumes > 0].mean()) if np.any(volumes > 0) else 0.0
    if mean_vol == 0:
        return False
    path_lo  = min(nq_spot, nq_wall)
    path_hi  = max(nq_spot, nq_wall)
    in_path  = (bins[:-1] >= path_lo - VAP_BIN_SIZE) & (bins[1:] <= path_hi + VAP_BIN_SIZE)
    path_vols = volumes[in_path]
    if len(path_vols) == 0:
        return False
    return float(path_vols.mean()) < LVN_THRESHOLD * mean_vol


def compute_local_gradient(wall_qqq: float, all_walls: list[Wall], n_neighbors: int = 2) -> float:
    strikes   = [w.qqq_strike for w in all_walls]
    gex_map   = {w.qqq_strike: w.abs_gex for w in all_walls}
    try:
        idx = strikes.index(wall_qqq)
    except ValueError:
        return 0.0
    lo = max(0, idx - n_neighbors)
    hi = min(len(strikes), idx + n_neighbors + 1)
    neighbors = [gex_map[strikes[i]] for i in range(lo, hi) if i != idx]
    if not neighbors:
        return 0.0
    return float(gex_map[wall_qqq] / np.mean(neighbors))


# ─────────────────────────────────────────────────────────────
# PERSISTENT STRIKES CACHE  (S9: fresh vs persistent wall)
# ─────────────────────────────────────────────────────────────
def _cache_path(today: date) -> Path:
    return CACHE_DIR / f"strikes_{today.isoformat()}.json"


def load_persistent_cache(today: date) -> set[float]:
    """Returns set of QQQ strikes already seen today in prior snapshots."""
    p = _cache_path(today)
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()))
    except Exception:
        return set()


def update_persistent_cache(today: date, current_strikes: set[float]) -> set[float]:
    """
    Merges current snapshot strikes into today's cache.
    Returns the PREVIOUS seen set (before this snapshot) for S9 computation.
    """
    prev = load_persistent_cache(today)
    merged = prev | current_strikes
    _cache_path(today).write_text(json.dumps(sorted(merged)))
    return prev  # strikes seen BEFORE this snapshot → persistent if in current too


# ─────────────────────────────────────────────────────────────
# SIGNAL SCORING
# ─────────────────────────────────────────────────────────────
def score_walls(
    walls:          list[Wall],
    nq_spot:        float,
    qqq_spot:       float,
    atr:            float | None,
    vap:            dict,
    prev_seen:      set[float],   # QQQ strikes seen in prior snaps today
    snap_role:      str,          # "same_day" or "next_day"
    dow:            int,          # 0=Mon 4=Fri
) -> list[dict]:
    """Score top-N walls by abs_gex. Returns list of result dicts, sorted by score desc."""
    top_walls = sorted(walls, key=lambda w: w.abs_gex, reverse=True)[:TOP_N]
    wall_rank  = {w.qqq_strike: i + 1 for i, w in enumerate(top_walls)}

    # Top-wall dominance (S8, tracked but not scored)
    top_is_dominant = (
        len(top_walls) >= 2 and
        top_walls[0].abs_gex >= 1.5 * top_walls[1].abs_gex
    )

    gamma_flip = compute_gamma_flip(walls)
    net_gex    = sum(w.gex for w in walls)

    results = []
    for wall in top_walls:
        nq_wall = wall.nq_strike
        dist_nq = abs(nq_wall - nq_spot)

        # Infer approach direction from current position
        approach = "from_below" if nq_wall > nq_spot else "from_above"

        # ── S1: proximity ────────────────────────────────────────
        s1 = dist_nq <= PROXIMITY_NQ_PTS

        # ── S4: local gradient spike ─────────────────────────────
        gradient = compute_local_gradient(wall.qqq_strike, walls)
        s4 = gradient >= LOCAL_GRADIENT_RATIO

        # ── S5: far from gamma flip ───────────────────────────────
        s5 = True
        if gamma_flip and atr:
            ratio_nq   = nq_spot / qqq_spot if qqq_spot else 1.0
            flip_nq    = gamma_flip * ratio_nq
            s5 = abs(nq_spot - flip_nq) >= FLIP_NEAR_ATR_MULT * atr

        # ── S6: ITM put wall (from_below + neg GEX) ──────────────
        s6 = (approach == "from_below" and wall.gex < 0)

        # ── S7: Mon or Wed ────────────────────────────────────────
        s7 = dow in (0, 2)

        # ── S9: fresh wall (not seen in prior snap today) ─────────
        s9 = wall.qqq_strike not in prev_seen

        # ── S10: tight proximity ─────────────────────────────────
        s10 = dist_nq <= TIGHT_PROXIMITY_NQ_PTS

        # ── S11: next-day snap ────────────────────────────────────
        s11 = snap_role == "next_day"

        # ── S12: rank-1 wall by abs_gex ───────────────────────────
        s12 = wall_rank.get(wall.qqq_strike, 99) == 1

        # ── S13: LVN clean path ───────────────────────────────────
        s13 = is_clean_path(nq_spot, nq_wall, vap)

        # ── S14: ITM proximal combo ───────────────────────────────
        s14 = s1 and s6

        # ── S15: from below ───────────────────────────────────────
        s15 = (approach == "from_below")

        # ── S16: ATR regime gate (NOT scored, operational filter) ─
        s16 = bool(atr and atr >= ATR_REGIME_THRESHOLD)

        score = sum([s1, s4, s5, s6, s7, s9, s10, s11, s12, s13, s14, s15])

        results.append({
            'qqq_strike':  wall.qqq_strike,
            'nq_strike':   wall.nq_strike,
            'dist_nq':     round(dist_nq, 1),
            'approach':    approach,
            'gex':         wall.gex,
            'abs_gex':     wall.abs_gex,
            'rank':        wall_rank.get(wall.qqq_strike, 99),
            'gradient':    round(gradient, 2),
            'score':       score,
            'atr_gate':    s16,
            # individual signals
            's1': s1, 's4': s4, 's5': s5, 's6': s6,
            's7': s7, 's9': s9, 's10': s10, 's11': s11,
            's12': s12, 's13': s13, 's14': s14, 's15': s15,
        })

    results.sort(key=lambda r: r['score'], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────
def _sig_str(r: dict) -> str:
    parts = []
    if r['s14']: parts.append('S14:ITM+PROX')
    if r['s6']:  parts.append('S6:ITMput')
    if r['s12']: parts.append('S12:rank1')
    if r['s1']:  parts.append('S1:prox60')
    if r['s5']:  parts.append('S5:flip')
    if r['s4']:  parts.append('S4:grad')
    if r['s7']:  parts.append('S7:Mon/Wed')
    if r['s11']: parts.append('S11:nextday')
    if r['s9']:  parts.append('S9:fresh')
    if r['s10']: parts.append('S10:tight')
    if r['s13']: parts.append('S13:LVN')
    if r['s15']: parts.append('S15:below')
    return ' '.join(parts) if parts else '--'


def print_results(
    results:   list[dict],
    nq_spot:   float,
    qqq_spot:  float,
    atr:       float | None,
    snap_role: str,
    now:       datetime,
) -> None:
    atr_str  = f"{atr:.0f}" if atr else "N/A"
    gate_str = "PASS" if (atr and atr >= ATR_REGIME_THRESHOLD) else "FAIL"

    print()
    print("=" * 78)
    print(f"  TRUEALGO  {now.strftime('%Y-%m-%d %H:%M')}  |  QQQ: ${qqq_spot:.2f}  |  NQ: {nq_spot:.0f}")
    print(f"  Snap: {snap_role.upper()}  |  ATR(20d): {atr_str}  |  Regime gate: {gate_str} (>={ATR_REGIME_THRESHOLD:.0f})")
    print("=" * 78)
    print(f"  {'QQQ':>6}  {'DIST':>6}  {'DIR':>10}  {'SCORE':>5}  {'GATE':>4}  SIGNALS")
    print(f"  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*5}  {'-'*4}  {'-'*40}")

    for r in results:
        dist_str = f"{r['dist_nq']:+.0f} NQ"
        gate     = "OK" if r['atr_gate'] else "--"
        sigs     = _sig_str(r)
        flag     = " <--" if r['score'] >= 5 and r['atr_gate'] else ""
        print(
            f"  {r['qqq_strike']:>6.1f}"
            f"  {dist_str:>6}"
            f"  {r['approach']:>10}"
            f"  {r['score']:>5}"
            f"  {gate:>4}"
            f"  {sigs}{flag}"
        )

    # Highlight actionable setups
    tier_vol  = [r for r in results if r['score'] >= 5 and r['atr_gate']]
    tier_prec = [r for r in results if r['score'] >= 6 and r['atr_gate']]

    print()
    if tier_prec:
        print(f"  PRECISION (score 6+ ATR-gated, ~73% hist): "
              + ", ".join(f"${r['qqq_strike']:.1f}" for r in tier_prec))
    if tier_vol:
        print(f"  VOLUME    (score 5+ ATR-gated, ~65% hist): "
              + ", ".join(f"${r['qqq_strike']:.1f}" for r in tier_vol))
    if not tier_vol:
        print("  No setups meet threshold this snapshot.")
    print("=" * 78)
    print()


# ─────────────────────────────────────────────────────────────
# JSON OUTPUT  (for web dashboard / GitHub Actions)
# ─────────────────────────────────────────────────────────────
def write_json(results: list[dict], nq_spot: float, qqq_spot: float,
               atr: float | None, snap_role: str, out_path: str) -> None:
    now_et   = datetime.now(ET)
    atr_ok   = bool(atr and atr >= ATR_REGIME_THRESHOLD)
    levels   = []
    for r in results:
        levels.append({
            "qqq":     r["qqq_strike"],
            "dist":    r["dist_nq"],
            "dir":     r["approach"],
            "score":   r["score"],
            "gate":    r["atr_gate"],
            "rank":    r["rank"],
            "signals": _sig_str(r),
        })

    payload = {
        "updated":  now_et.strftime("%Y-%m-%d %H:%M ET"),
        "updated_iso": now_et.isoformat(),
        "nq":       round(nq_spot, 1),
        "qqq":      round(qqq_spot, 2),
        "atr":      round(atr, 1) if atr else None,
        "atr_ok":   atr_ok,
        "snap":     snap_role,
        "precision": [r["qqq_strike"] for r in results if r["score"] >= 6 and r["atr_gate"]],
        "volume":    [r["qqq_strike"] for r in results if r["score"] >= 5 and r["atr_gate"]],
        "levels":   levels,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Saved {out_path}  ({len(levels)} levels)")


# ─────────────────────────────────────────────────────────────
# MAIN RUN
# ─────────────────────────────────────────────────────────────
def run_once(json_out: str | None = None) -> None:
    now_et    = datetime.now(ET)
    now_local = datetime.now()
    today     = now_et.date()
    dow       = now_et.weekday()
    snap_role = "next_day" if now_et.hour >= 12 else "same_day"

    print(f"\nFetching FreeFlow  ({SYMBOL})...")
    try:
        agg, nq_spot, qqq_spot = fetch_freeflow()
    except Exception as e:
        print(f"  FreeFlow error: {e}")
        if "401" in str(e) or "403" in str(e):
            print("  Cookie expired — update ff_session in truealgo_config.json")
        return

    print(f"  NQ: {nq_spot:.0f}  QQQ: ${qqq_spot:.2f}  |  {len(agg)} strikes loaded")

    print("Fetching ATR (yfinance NQ=F daily)...")
    atr = fetch_nq_atr()
    print(f"  ATR(20d): {atr:.1f}" if atr else "  ATR: unavailable")

    print("Fetching prior-session VAP (yfinance NQ=F 1m)...")
    vap = fetch_prior_vap()
    print(f"  VAP built: {len(vap.get('volumes', []))} bins" if vap else "  VAP: unavailable")

    walls     = build_walls(agg)
    current_q = {w.qqq_strike for w in sorted(walls, key=lambda w: w.abs_gex, reverse=True)[:TOP_N]}
    prev_seen = update_persistent_cache(today, current_q)

    results = score_walls(walls, nq_spot, qqq_spot, atr, vap, prev_seen, snap_role, dow)

    if json_out:
        write_json(results, nq_spot, qqq_spot, atr, snap_role, json_out)
    else:
        print_results(results, nq_spot, qqq_spot, atr, snap_role, now_local)


# ─────────────────────────────────────────────────────────────
# MARKET HOURS SCHEDULER
# Scans run at: 09:00, 10:00, 11:00, 12:00, 13:00, 14:00, 15:00, 15:30 ET
# weekdays only. Outside those windows the loop sleeps until the next slot.
# ─────────────────────────────────────────────────────────────
SCAN_TIMES_ET = [
    dtime(9,  0),   # pre-open: loads yesterday PM snap as next_day signal
    dtime(10, 0),
    dtime(11, 0),
    dtime(12, 0),
    dtime(13, 0),
    dtime(14, 0),
    dtime(15, 0),
    dtime(15, 30),  # final EOD snap — strongest next_day setups for tomorrow
]


def _next_scan_dt() -> datetime:
    """Return the next scheduled scan datetime in ET."""
    now_et = datetime.now(ET)
    today  = now_et.date()

    # Try today's remaining slots first, then advance days
    for offset in range(7):
        candidate_date = today + timedelta(days=offset)
        if candidate_date.weekday() >= 5:   # skip Sat/Sun
            continue
        for t in SCAN_TIMES_ET:
            dt = datetime(candidate_date.year, candidate_date.month, candidate_date.day,
                          t.hour, t.minute, tzinfo=ET)
            if dt > now_et:
                return dt

    # Fallback (should never reach here for normal weeks)
    return now_et + timedelta(hours=1)


def run_live() -> None:
    print("\nLIVE MODE — market hours only (Mon-Fri, 09:00-15:30 ET). Ctrl+C to stop.")
    print(f"Scan times: {', '.join(t.strftime('%H:%M') for t in SCAN_TIMES_ET)} ET\n")

    while True:
        next_dt = _next_scan_dt()
        now_et  = datetime.now(ET)
        wait_s  = max(0, int((next_dt - now_et).total_seconds()))

        if wait_s > 0:
            print(f"  Sleeping until {next_dt.strftime('%a %Y-%m-%d %H:%M ET')} "
                  f"({wait_s // 3600}h {(wait_s % 3600) // 60}m away)...")
            try:
                # Sleep in 30-second ticks so Ctrl+C is responsive
                for _ in range(wait_s // 30):
                    time.sleep(30)
                remaining = wait_s % 30
                if remaining:
                    time.sleep(remaining)
            except KeyboardInterrupt:
                print("\nStopped.")
                return

        try:
            run_once()
        except KeyboardInterrupt:
            print("\nStopped.")
            return
        except Exception as e:
            print(f"\nError: {e} — will retry at next scheduled slot.")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrueAlgo Live Scanner")
    parser.add_argument("--live",     action="store_true",
                        help="Auto-schedule scans at market hours (Mon-Fri 09:00-15:30 ET)")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Write output as JSON to this path instead of printing")
    args = parser.parse_args()

    if args.live:
        run_live()
    else:
        run_once(json_out=args.json_out)
