"""Phase 1 gate: does the log5 PA model beat simpler baselines on held-out PAs?

Trained on 2015-2024, validated on 2025 (~183k PAs) and 2026 (~164k). Validating here
rather than at game level is the point of the whole design: 183,000 rows of evidence
instead of 1,300 coin flips.
"""
import sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
from pa_sim import CLASSES
from pa_sim.rates import RateTable, league_rates, log5

pa = pd.read_parquet("data/backtest_2026/pa_2015_2026.parquet")
pa["ev"] = pa["ev"].astype(str)
tr = pa[pa.game_year <= 2024]
print(f"train {len(tr):,} PAs (2015-2024)")

bat = RateTable(tr, "batter")
pit = RateTable(tr, "pitcher")
L = bat.league
print("\nfitted shrinkage constants k (pseudo-PA pulled toward the prior):")
for c, kb, kp in zip(CLASSES, bat.k, pit.k):
    print(f"  {c:4s} batter k={kb:7.1f}   pitcher k={kp:7.1f}   league={L[CLASSES.index(c)]*100:6.3f}%")

yidx = {c: i for i, c in enumerate(CLASSES)}

def evaluate(df, tag):
    y = df["ev"].map(yidx).values
    B  = np.array([bat.get(b) for b in df.batter.values])
    P  = np.array([pit.get(p) for p in df.pitcher.values])
    Bh = np.array([bat.get(b, h) for b, h in zip(df.batter.values, df.p_throws.values)])
    Ph = np.array([pit.get(p, s) for p, s in zip(df.pitcher.values, df.stand.values)])
    models = {
        "league baseline":            np.tile(L, (len(df), 1)),
        "batter only":                B,
        "pitcher only":               P,
        "log5(batter, pitcher)":      log5(B, P, L),
        "log5 + handedness splits":   log5(Bh, Ph, L),
    }
    print(f"\n{'='*76}\n{tag}  (n={len(df):,} PAs)\n{'='*76}")
    print(f"{'model':30}{'log-loss':>11}{'vs league':>12}{'top1 acc':>10}")
    out = {}
    for k, M in models.items():
        M = np.clip(M, 1e-9, 1); M = M / M.sum(axis=1, keepdims=True)
        ll = -np.mean(np.log(M[np.arange(len(df)), y]))
        acc = (M.argmax(axis=1) == y).mean() * 100
        out[k] = ll
        base = out.get("league baseline", ll)
        print(f"{k:30}{ll:>11.5f}{ll-base:>+12.5f}{acc:>10.2f}")
    return out

r25 = evaluate(pa[pa.game_year == 2025], "HELD-OUT 2025")
r26 = evaluate(pa[pa.game_year == 2026], "HELD-OUT 2026")

print(f"\n{'='*76}\nPHASE 1 GATE\n{'='*76}")
ok = True
for tag, r in [("2025", r25), ("2026", r26)]:
    g_league = r["league baseline"] - r["log5 + handedness splits"]
    g_batter = r["batter only"] - r["log5 + handedness splits"]
    print(f"  {tag}: log5+hand beats league by {g_league:+.5f} nats, "
          f"beats batter-only by {g_batter:+.5f} nats")
    if not (g_league > 0 and g_batter > 0):
        ok = False
print(f"\n  -> PHASE 1 {'PASS' if ok else 'FAIL'}: "
      f"{'the PA model carries real, additive matchup signal' if ok else 'no additive signal'}")
np.save("data/backtest_2026/pa_league_rates.npy", L)
