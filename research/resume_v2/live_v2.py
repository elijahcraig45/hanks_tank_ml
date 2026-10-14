import sys
W = sys.argv[1]
sys.path.insert(0, W); sys.path.insert(0, W + "/mod"); sys.path.insert(0, "/opt/pipeline/src/hanks_tank_ml/src"); sys.path.insert(0, "/opt/pipeline/src/hanks_tank_ml/src/cfb")
import logging; logging.disable(logging.WARNING)
import numpy as np
import espn_data; espn_data.fetch_history(last_season=2026)
from rankings import core, sources, resume
import resume_v2 as v2
from rankings.sources import SPORTS
spec = SPORTS["cfb"]
g = sources.load("cfb"); cur = g[g.season == 2026]; prior = g[g.season == 2025]; wk = int(cur.week.max()) + 1
seen = cur[cur.week < wk]
divs = core.season_divisions(prior, cur)
fbs = sorted({t for t, d in divs.items() if d == "fbs"} & (set(seen.home_team_name) | set(seen.away_team_name)))
ys = v2.fit_yardstick(seen, prior, wk, divs, "fbs", alpha=spec.margin_alpha, w0=spec.prior_w0, tau=spec.prior_tau, n_draws=2, seed=1, bench_teams=fbs)
sides = v2.sides_of(seen, ys.index)
inv = {i: t for t, i in ys.index.items()}
rec = {}
for x in seen.itertuples(index=False):
    w, l = resume._winner_loser(x); rec.setdefault(w, [0, 0])[0] += 1; rec.setdefault(l, [0, 0])[1] += 1
R, SC, CR = {}, {}, {}
for name, kind, lam in (("add.5 (v1 credit)", "add", 0.5), ("addfloor.5", "addfloor", 0.5), ("mult.5", "mult", 0.5), ("mult.75", "mult", 0.75)):
    cm = v2.credits(ys, sides, kind, lam, use_mean=True)[0]
    var = v2.credit_variance(ys, sides, kind, lam)
    tot = np.bincount(sides.team, weights=cm, minlength=len(ys.teams)); v = np.bincount(sides.team, weights=var, minlength=len(ys.teams))
    sor = {t: float(tot[ys.index[t]]) for t in fbs}; vr = {t: float(v[ys.index[t]]) for t in fbs}
    o = resume.order_division(fbs, seen, sor, vr, 0.5)
    R[name] = {t: i for i, t in enumerate(o, 1)}; SC[name] = sor; CR[name] = (cm, sides)
print("through week", wk - 1)
names = list(R)
print(f"{'':3s}{'mult.5 top 20':26s} rec   " + " ".join(f"{n[:14]:>14s}" for n in names))
for t in sorted(R["mult.5"], key=R["mult.5"].get)[:20]:
    print(f"{R['mult.5'][t]:2d} {t[:24]:24s} {rec[t][0]}-{rec[t][1]}  " + " ".join(f"{R[n][t]:14d}" for n in names))
for t in ["USC Trojans", "James Madison Dukes", "Georgia Tech Yellow Jackets"]:
    print(f"   {t[:24]:24s} {rec[t][0]}-{rec[t][1]}  " + " ".join(f"{R[n][t]:14d}" for n in names))
gt = ys.index["Georgia Tech Yellow Jackets"]
print("\nGeorgia Tech per-game credit (add.5 / addfloor.5 / mult.5 / mult.75):")
idx = np.flatnonzero(sides.team == gt)
for k in idx:
    opp = inv[sides.opp[k]]; res = ("W" if sides.won[k] else "L")
    print(f"  {res} {sides.margin[k]:+5.0f} vs {opp[:22]:22s} " + "  ".join(f"{CR[n][0][k]:+.2f}" for n in names))
