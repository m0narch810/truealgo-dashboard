"""
Deep analysis of backtest results CSV.
Run after backtest_levels.py to find patterns worth adding as signals.
"""
import sys
import pandas as pd
import numpy as np

CSV = "results/gex_level_validation.csv"

df = pd.read_csv(CSV)
touched = df[df["touched"] == True].copy()
n = len(touched)
base_rate = touched["reversed"].mean()

print(f"\n{'='*60}")
print(f"DEEP ANALYSIS  ({n} touched walls, {base_rate:.1%} base reversal rate)")
print(f"{'='*60}")

# -- 1. Time of day -------------------------------------------------------
if "touch_time_min" in touched.columns:
    t = touched.dropna(subset=["touch_time_min"]).copy()
    t["touch_hour"] = (t["touch_time_min"] / 60).astype(int)
    print("\n--- Touch hour breakdown ---")
    print(f"{'Hour':>6}  {'N':>5}  {'Rev%':>6}  {'AvgPts':>7}")
    for hr, g in t.groupby("touch_hour"):
        rate = g["reversed"].mean()
        avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
        print(f"  {hr:02d}:xx  {len(g):>5}  {rate:>5.1%}  {avg:>7.1f}")

    t["session_half"] = t["touch_time_min"].apply(lambda x: "morning" if x < 780 else "afternoon")
    print("\n--- Morning (< 13:00) vs Afternoon ---")
    for half, g in t.groupby("session_half"):
        rate = g["reversed"].mean()
        avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
        print(f"  {half:12s}: {rate:.1%}  (n={len(g)}, avg_pts={avg:.1f})")

# -- 2. Direction breakdown -----------------------------------------------
if "approach_direction" in touched.columns:
    print("\n--- Approach direction breakdown ---")
    for d, g in touched.groupby("approach_direction"):
        rate = g["reversed"].mean()
        avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
        print(f"  {d:15s}: {rate:.1%}  (n={len(g)}, avg_pts={avg:.1f})")

# -- 3. Snap role ---------------------------------------------------------
if "snap_role" in touched.columns:
    print("\n--- Snap role breakdown ---")
    for role, g in touched.groupby("snap_role"):
        rate = g["reversed"].mean()
        avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
        print(f"  {role:12s}: {rate:.1%}  (n={len(g)}, avg_pts={avg:.1f})")

# -- 4. Proximity bins ----------------------------------------------------
if "dist_from_spot" in touched.columns:
    print("\n--- Distance from spot bins (NQ pts) ---")
    bins = [0, 20, 40, 60, 100, 200, 500, 9999]
    labels = ["0-20", "20-40", "40-60", "60-100", "100-200", "200-500", "500+"]
    touched["dist_bin"] = pd.cut(touched["dist_from_spot"].abs(), bins=bins, labels=labels)
    for lbl, g in touched.groupby("dist_bin", observed=True):
        rate = g["reversed"].mean()
        avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
        print(f"  dist {lbl:8s}: {rate:.1%}  (n={len(g)}, avg_pts={avg:.1f})")

# -- 5. Score breakdown by direction --------------------------------------
if "confluence_score" in touched.columns and "approach_direction" in touched.columns:
    trading_days = touched["date"].nunique() if "date" in touched.columns else 252
    print("\n--- Score breakdown: from_below only ---")
    below = touched[touched["approach_direction"] == "from_below"]
    print(f"{'Score':>6}  {'N':>5}  {'Rev%':>6}  {'Freq/day':>9}")
    for thresh in range(0, 16):
        g = below[below["confluence_score"] >= thresh]
        if len(g) < 5:
            break
        rate = g["reversed"].mean()
        freq = len(g) / trading_days
        print(f"  {thresh}+   {len(g):>5}  {rate:>5.1%}  {freq:>8.2f}")

    print("\n--- Score breakdown: from_above only ---")
    above = touched[touched["approach_direction"] == "from_above"]
    for thresh in range(0, 16):
        g = above[above["confluence_score"] >= thresh]
        if len(g) < 5:
            break
        rate = g["reversed"].mean()
        freq = len(g) / trading_days
        print(f"  {thresh}+   {len(g):>5}  {rate:>5.1%}  {freq:>8.2f}")

# -- 6. Day-of-week granular ----------------------------------------------
if "date" in touched.columns:
    from datetime import datetime
    touched["dow"] = touched["date"].apply(
        lambda d: datetime.strptime(str(d)[:10], "%Y-%m-%d").strftime("%A")
    )
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    print("\n--- Day of week breakdown ---")
    for day in dow_order:
        g = touched[touched["dow"] == day]
        if len(g) == 0:
            continue
        rate = g["reversed"].mean()
        avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
        print(f"  {day:12s}: {rate:.1%}  (n={len(g)}, avg_pts={avg:.1f})")

# -- 7. S9 fresh/persistent analysis -------------------------------------
if "s9_persistent" in touched.columns:
    print("\n--- S9 fresh wall analysis (True=fresh/non-persistent, False=persistent) ---")
    for v, g in touched.groupby("s9_persistent"):
        label = "fresh (1 snap)" if v else "persistent (2+ snaps)"
        rate = g["reversed"].mean()
        avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
        print(f"  {label:25s}: {rate:.1%}  (n={len(g)}, avg_pts={avg:.1f})")

# -- 8. Signal combo matrix: S1 x S6 ------------------------------------
if all(c in touched.columns for c in ["s1_proximity", "s6_itm_options"]):
    print("\n--- S1 proximity x S6 ITM combo matrix ---")
    for s1 in [False, True]:
        for s6 in [False, True]:
            g = touched[(touched["s1_proximity"] == s1) & (touched["s6_itm_options"] == s6)]
            if len(g) < 5:
                continue
            rate = g["reversed"].mean()
            print(f"  S1={int(s1)} S6={int(s6)}: {rate:.1%}  (n={len(g)})")

# -- 9. Local gradient quantiles -----------------------------------------
if "local_gradient" in touched.columns:
    print("\n--- Local gradient quantile breakdown ---")
    t2 = touched.dropna(subset=["local_gradient"])
    quantiles = t2["local_gradient"].quantile([0.25, 0.5, 0.75, 0.9]).values
    thresholds = [0] + list(quantiles) + [9999]
    labels2 = ["bot25%", "25-50%", "50-75%", "75-90%", "top10%"]
    t2 = t2.copy()
    t2["grad_bin"] = pd.cut(t2["local_gradient"], bins=thresholds, labels=labels2)
    for lbl, g in t2.groupby("grad_bin", observed=True):
        rate = g["reversed"].mean()
        avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
        q_lo = thresholds[list(labels2).index(lbl)]
        q_hi = thresholds[list(labels2).index(lbl) + 1]
        print(f"  grad [{q_lo:.1f},{q_hi:.1f}): {rate:.1%}  (n={len(g)}, avg_pts={avg:.1f})")

# -- 10. LVN clean path --------------------------------------------------
if "lvn_clean_path" in touched.columns:
    print("\n--- LVN clean path ---")
    for v, g in touched.groupby("lvn_clean_path"):
        label = "clean_path" if v else "obstructed"
        rate = g["reversed"].mean()
        avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
        print(f"  {label:12s}: {rate:.1%}  (n={len(g)}, avg_pts={avg:.1f})")

# -- 11. GEX magnitude quartiles ----------------------------------------
if "abs_gex" in touched.columns:
    print("\n--- Absolute GEX magnitude quartile breakdown ---")
    q25, q50, q75, q90 = touched["abs_gex"].quantile([0.25, 0.5, 0.75, 0.9]).values
    gex_bins = [0, q25, q50, q75, q90, 1e15]
    gex_labels = ["bot25%", "25-50%", "50-75%", "75-90%", "top10%"]
    touched = touched.copy()
    touched["gex_bin"] = pd.cut(touched["abs_gex"], bins=gex_bins, labels=gex_labels)
    for lbl, g in touched.groupby("gex_bin", observed=True):
        rate = g["reversed"].mean()
        avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
        print(f"  gex {lbl:8s}: {rate:.1%}  (n={len(g)}, avg_pts={avg:.1f})")

# -- 12. Wall rank analysis ----------------------------------------------
if "wall_rank" in touched.columns:
    print("\n--- Wall rank breakdown ---")
    for rank in sorted(touched["wall_rank"].dropna().unique()):
        g = touched[touched["wall_rank"] == rank]
        rate = g["reversed"].mean()
        avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
        print(f"  rank {int(rank)}: {rate:.1%}  (n={len(g)}, avg_pts={avg:.1f})")

# -- 13. S10 tight proximity ---------------------------------------------
if "s10_tight_proximity" in touched.columns:
    print("\n--- S10 tight proximity breakdown ---")
    for v, g in touched.groupby("s10_tight_proximity"):
        label = "tight (<= 30)" if v else "far (> 30)"
        rate = g["reversed"].mean()
        avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
        print(f"  {label:15s}: {rate:.1%}  (n={len(g)}, avg_pts={avg:.1f})")

# -- 14. S14 ITM proximal combo breakdown --------------------------------
if "s14_itm_proximal" in touched.columns:
    print("\n--- S14 ITM proximal combo (S1 AND S6) ---")
    for v, g in touched.groupby("s14_itm_proximal"):
        label = "S1+S6 combo" if v else "no combo"
        rate = g["reversed"].mean()
        avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
        print(f"  {label:12s}: {rate:.1%}  (n={len(g)}, avg_pts={avg:.1f})")

# -- 15b. ATR regime gate analysis (operational filter, NOT a score signal) --------
# ATR is a date-level signal: adding to score inflates 63% of walls uniformly.
# Best use: as a pre-filter gate that unlocks lower score thresholds.
if "atr" in touched.columns and "confluence_score" in touched.columns:
    t16 = touched.dropna(subset=["atr"]).copy()
    td16 = 365
    print("\n--- ATR OPERATIONAL GATE: score + ATR combinations ---")
    print(f"  {'Score':>6}  {'ATR gate':>9}  {'N':>5}  {'Rev%':>6}  {'Freq/wk':>8}")
    for sthresh in [5, 6, 7]:
        for atr_gate in [None, 350, 380, 400, 420]:
            sub = t16[t16["confluence_score"] >= sthresh]
            if atr_gate:
                sub = sub[sub["atr"] >= atr_gate]
            if len(sub) < 5:
                continue
            rate = sub["reversed"].mean()
            freq = len(sub) / td16 * 5
            lbl = f">={atr_gate}" if atr_gate else "no gate"
            print(f"  {sthresh}+    {lbl:>9}  {len(sub):>5}  {rate:>5.1%}  {freq:>7.2f}")
        print()
    print("--- Pareto frontier: accuracy vs frequency (key operating tiers) ---")
    combos = [
        (9, None, "from_below",  "Tier 0: score9+ from_below"),
        (7, None, "from_below",  "Tier 1: score7+ from_below"),
        (6, 380,  None,          "Tier 2A: score6+ ATR>=380"),
        (6, 350,  None,          "Tier 2B: score6+ ATR>=350"),
        (5, 380,  None,          "Tier 3: score5+ ATR>=380 (2x/wk)"),
        (5, 350,  None,          "Tier 4: score5+ ATR>=350"),
    ]
    print(f"  {'Tier':38s}  {'N':>5}  {'Rev%':>6}  {'Freq/wk':>8}")
    for sthresh, atr_gate, direction, label in combos:
        sub = t16[t16["confluence_score"] >= sthresh]
        if atr_gate:
            sub = sub[sub["atr"] >= atr_gate]
        if direction and "approach_direction" in sub.columns:
            sub = sub[sub["approach_direction"] == direction]
        if len(sub) < 3:
            continue
        rate = sub["reversed"].mean()
        freq = len(sub) / td16 * 5
        print(f"  {label:38s}  {len(sub):>5}  {rate:>5.1%}  {freq:>7.2f}")

# -- 16. Full score sweep + from_below high-score drill ------------------
if "confluence_score" in touched.columns:
    td = touched["date"].nunique() if "date" in touched.columns else 252
    print("\n--- Full score sweep (all directions) ---")
    print(f"  {'Score':>5}  {'N':>5}  {'Rev%':>6}  {'Freq/day':>9}  {'AvgPts':>7}")
    for thresh in range(0, 16):
        g = touched[touched["confluence_score"] >= thresh]
        if len(g) < 3:
            break
        rate = g["reversed"].mean()
        freq = len(g) / td
        avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
        print(f"  {thresh}+   {len(g):>5}  {rate:>5.1%}  {freq:>8.2f}  {avg:>7.1f}")

    if "approach_direction" in touched.columns:
        below2 = touched[touched["approach_direction"] == "from_below"]
        print("\n--- High-score from_below only ---")
        print(f"  {'Score':>5}  {'N':>5}  {'Rev%':>6}  {'Freq/day':>9}  {'AvgPts':>7}")
        for thresh in range(6, 16):
            g = below2[below2["confluence_score"] >= thresh]
            if len(g) < 3:
                break
            rate = g["reversed"].mean()
            freq = len(g) / td
            avg = g.loc[g["reversed"], "reversal_pts"].mean() if g["reversed"].any() else 0
            print(f"  {thresh}+   {len(g):>5}  {rate:>5.1%}  {freq:>8.2f}  {avg:>7.1f}")

print()
