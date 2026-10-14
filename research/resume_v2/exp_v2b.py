"""Round 2 (EXPLORATORY, designed after round 1 failed; tuned on tune, evaluated once on eval).

Separates the credit from the ordering rule, adds the standardised score z = sum(credit) / sqrt(sum(var)), and orders with
calibrated luck-based probabilities P_ij = Phi((s_i - s_j) / sqrt(v_i + v_j)) instead of yardstick draws.
"""
import sys, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.stats import norm

W = sys.argv[1]
sys.path.insert(0, W)
import exp_v2 as X   # round-1 helpers (reads the same argv: OUT, module dir, workers, R)
from exp_v2 import core, resume, v2, spec, fbs_games, contradictions, split_half, ev  # noqa: E402

OUT = W
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
WEEKS, HORIZON = X.WEEKS, X.HORIZON


def arm(kind, lam=0.5, agg="sum", order="v1", z=0.5, eta=0.0):
    return dict(kind=kind, lam=lam, agg=agg, order=order, z=z, eta=eta)


ARMS = {}
ARMS["R0 add .5 sum v1order"] = arm("add")
ARMS["R1 addfloor .5 sum v1order"] = arm("addfloor")
for lam in (0.25, 0.5, 0.75):
    ARMS[f"R2 mult {lam} sum v1order"] = arm("mult", lam)
ARMS["R2 win sum v1order"] = arm("win", 0.0)
for z in (0.35, 0.5, 0.7, 1.0):
    ARMS[f"R3 mult .5 z v1order z={z}"] = arm("mult", 0.5, "z", "v1", z)
ARMS["R3 win z v1order z=0.5"] = arm("win", 0.0, "z", "v1", 0.5)
ARMS["R3 add .5 z v1order z=0.5"] = arm("add", 0.5, "z", "v1", 0.5)
ARMS["R3 mult .5 z sort"] = arm("mult", 0.5, "z", "sort")
ARMS["R3 mult .5 sum sort"] = arm("mult", 0.5, "sum", "sort")
for agg in ("sum", "z"):
    for eta in (0.3, 0.6, 1.0, 2.0, 4.0):
        ARMS[f"R4 mult .5 {agg} cons eta={eta}"] = arm("mult", 0.5, agg, "cons", eta=eta)


def run_board(season, week, cur, prior, divs, seed):
    seen = cur[cur["week"] < week]
    nxt = fbs_games(cur[(cur["week"] >= week) & (cur["week"] < week + HORIZON)])
    if seen.empty:
        return [], {}
    fbs_played = sorted({t for t, d in divs.items() if d == "fbs"} & (set(seen.home_team_name) | set(seen.away_team_name)))
    ys = v2.fit_yardstick(seen, prior, week, divs, "fbs", alpha=spec.margin_alpha, w0=spec.prior_w0, tau=spec.prior_tau,
                          n_draws=2, seed=seed, bench_teams=fbs_played)
    members = np.array([ys.index[t] for t in fbs_played if t in ys.index])
    names = [ys.teams[i] for i in members]
    sides = v2.sides_of(seen, ys.index)
    h = v2.played_matrix(seen, ys.index, members)
    sg = fbs_games(seen)
    rows, payload = [], {}

    def record(name, order_idx, p, extra=None):
        pos = {names[i]: r for r, i in enumerate(order_idx)}
        n, c = contradictions(pos, sg)
        tn, tc = contradictions({names[i]: r for r, i in enumerate(order_idx[:25])}, sg)
        fn, fw = contradictions(pos, nxt)
        rows.append(dict(season=season, week=week, arm=name, n=n, c=c, tn=tn, tc=tc, fn=fn, fw=fw, **(extra or {})))
        payload[name] = (names, order_idx, p)

    # the shipped board, exactly as published
    fit_kw = dict(C=spec.ridge_C, w0=spec.prior_w0, tau=spec.prior_tau, major="fbs", divisions=divs, model="margin",
                  margin_alpha=spec.margin_alpha, margin_cap=None, margin_scale=spec.margin_scale)
    S1, ha1 = resume.opponent_ratings(seen, prior, week, fit_kw)
    bench1 = resume.top_benchmark(S1, fbs_played, resume.BENCH_N)
    sor1, var1 = resume.record_scores(seen, S1, ha1, bench1, resume.LAM, spec.margin_scale)
    teams1 = [t for t in fbs_played if t in sor1]
    order1 = resume.order_division(teams1, seen, sor1, var1, resume.Z)
    s_arr = np.array([sor1.get(t, 0.0) for t in names]); v_arr = np.array([var1.get(t, 0.0) for t in names])
    p1 = norm.cdf((s_arr[:, None] - s_arr[None, :]) / np.sqrt(v_arr[:, None] + v_arr[None, :] + 1e-12))
    record("v1 shipped", np.array([names.index(t) for t in order1]), p1, dict(rel=np.nan, width=np.nan))

    cache = {}
    for name, a in ARMS.items():
        key = (a["kind"], a["lam"])
        if key not in cache:
            cm = v2.credits(ys, sides, a["kind"], a["lam"], use_mean=True)[0]
            gv = v2.credit_variance(ys, sides, a["kind"], a["lam"])
            tot = np.bincount(sides.team, weights=cm, minlength=len(ys.teams))
            var = np.bincount(sides.team, weights=gv, minlength=len(ys.teams))
            cache[key] = (cm, tot, var)
        cm, tot, var = cache[key]
        if a["agg"] == "z":
            score = tot / np.sqrt(np.maximum(var, 1e-9)); v_use = np.ones_like(var)
        else:
            score, v_use = tot, var
        s_m, v_m = score[members], v_use[members]
        p = norm.cdf((s_m[:, None] - s_m[None, :]) / np.sqrt(v_m[:, None] + v_m[None, :] + 1e-12))
        start = np.array(sorted(range(len(members)), key=lambda i: (-s_m[i], names[i])))
        if a["order"] == "sort":
            order_idx = start
        elif a["order"] == "cons":
            order_idx = v2.consensus_order(p, h, a["eta"], start)
        else:
            sdict = {names[i]: float(s_m[i]) for i in range(len(names))}; vdict = {names[i]: float(v_m[i]) for i in range(len(names))}
            od = resume.order_division(names, seen, sdict, vdict, a["z"])
            order_idx = np.array([names.index(t) for t in od])
        lo, hi = v2.ranges(p, order_idx, 0.2)
        record(name, order_idx, p, dict(rel=split_half(sides, cm, members), width=float(np.mean((hi - lo + 1)[:25]))))
    return rows, payload


def run(season):
    cur = X.GAMES[X.GAMES.season == season]
    prior = X.GAMES[X.GAMES.season == season - 1]
    prior = None if prior.empty else prior
    divs = core.season_divisions(prior, cur)
    all_rows, pairs, boards = [], [], {}
    for w in WEEKS:
        try:
            rows, payload = run_board(season, w, cur, prior, divs, seed=season * 100 + w)
        except Exception as e:  # noqa: BLE001
            print("board failed", season, w, repr(e), flush=True); continue
        all_rows += rows; boards[w] = payload
    for w, payload in boards.items():
        later = boards.get(w + HORIZON)
        if not later: continue
        for arm_name, (names, order_idx, p) in payload.items():
            if arm_name not in later: continue
            n2, o2, _ = later[arm_name]
            pos2 = {n2[i]: r for r, i in enumerate(o2)}
            top = order_idx[:40]
            for a_i in range(len(top)):
                for b_i in range(a_i + 1, len(top)):
                    i, j = top[a_i], top[b_i]
                    if names[i] in pos2 and names[j] in pos2:
                        pairs.append((season, w, arm_name, float(p[i, j]), int(pos2[names[i]] < pos2[names[j]]), b_i - a_i))
    return pd.DataFrame(all_rows), pd.DataFrame(pairs, columns=["season", "week", "arm", "p", "persist", "gap"])


if __name__ == "__main__":
    t0 = time.time()
    games = ev.load_games("cfb")
    seasons = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
    with ProcessPoolExecutor(WORKERS, initializer=X.init, initargs=(games,)) as pool:
        results = list(pool.map(run, seasons))
    rows = pd.concat([r[0] for r in results], ignore_index=True); pairs = pd.concat([r[1] for r in results], ignore_index=True)
    tune = {2017, 2018, 2019, 2021}
    rows["part"] = np.where(rows.season.isin(tune), "tune", "eval"); pairs["part"] = np.where(pairs.season.isin(tune), "tune", "eval")
    rows.to_parquet(f"{OUT}/v2b_rows.parquet"); pairs.to_parquet(f"{OUT}/v2b_pairs.parquet")
    json.dump(ARMS, open(f"{OUT}/v2b_arms.json", "w"), indent=1)
    print("done", len(rows), "rows,", len(pairs), "pairs in", round(time.time() - t0), "s", flush=True)
