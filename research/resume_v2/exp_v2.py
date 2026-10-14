"""Resume v2 experiment harness (docs/RESUME_V2_REPORT.md section 4).

For each season and week in 4..12 a board is built from games before that week for every ARM below, on the same data.
Per board and arm we record M1 (next-3-weeks accuracy), M2 (split-half reliability), M3 (contradicted played games),
and the pairs needed for M4 (order persistence calibration). Analysis is in analyze_v2.py.
"""
import sys, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, "/opt/pipeline/src/hanks_tank_ml/src")
sys.path.append(sys.argv[2])                      # dir holding resume_v2.py
from rankings import core, resume, evaluate as ev  # noqa: E402
import resume_v2 as v2  # noqa: E402
from rankings.sources import SPORTS  # noqa: E402

OUT = sys.argv[1]
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
R = int(sys.argv[4]) if len(sys.argv) > 4 else 100
spec = SPORTS["cfb"]
WEEKS = list(range(4, 13))
HORIZON = 3
GAMES = None


def init(g):
    global GAMES
    import logging
    logging.disable(logging.WARNING)
    GAMES = g


# ---------------------------------------------------------------------------------------------------- arms
# kind, lam, k (None = sum), order ("sort" | "cons"), eta, source ("Y" yardstick draws | "YB" + game bootstrap), q (range)
def arm(kind, lam=0.5, k=None, order="sort", eta=0.0, source="Y", q=0.2):
    return dict(kind=kind, lam=lam, k=k, order=order, eta=eta, source=source, q=q)


ARMS = {}
# Stage A: the credit
ARMS["A win"] = arm("win", 0.0)
for lam in (0.25, 0.5, 0.75):
    ARMS[f"A add {lam}"] = arm("add", lam)
    ARMS[f"A addfloor {lam}"] = arm("addfloor", lam)
for lam in (0.25, 0.5, 0.75, 0.9):
    ARMS[f"A mult {lam}"] = arm("mult", lam)
# Stage B: aggregation, with the multiplicative credit at 0.5
for k in (0.0, 1.0, 3.0, 6.0):
    ARMS[f"B mult .5 k={k:g}"] = arm("mult", 0.5, k)
# Stage C: consensus ordering with a head-to-head penalty
for k in (None, 3.0):
    for eta in (0.0, 0.15, 0.3, 0.5, 0.8):
        ARMS[f"C mult .5 k={'sum' if k is None else int(k)} cons {eta}"] = arm("mult", 0.5, k, "cons", eta)
# Stage D: where the uncertainty comes from
ARMS["D mult .5 k=3 cons .3 +boot"] = arm("mult", 0.5, 3.0, "cons", 0.3, "YB")
ARMS["D mult .5 k=3 sort +boot"] = arm("mult", 0.5, 3.0, "sort", 0.0, "YB")
ARMS["D mult .5 k=3 sort Y"] = arm("mult", 0.5, 3.0, "sort", 0.0, "Y")


# ---------------------------------------------------------------------------------------------------- helpers
def fbs_games(df):
    return df[(df["home_division"] == "fbs") & (df["away_division"] == "fbs")]


def contradictions(pos, games):
    n = c = 0
    for g in games.itertuples(index=False):
        w, l = resume._winner_loser(g)
        if w in pos and l in pos:
            n += 1
            c += pos[l] < pos[w]
    return n, c


def split_half(sides, cm, members, min_games=4):
    """Spearman-Brown split-half reliability of per-game mean credit across members (alternate games)."""
    a, b = [], []
    for t in members:
        idx = np.flatnonzero(sides.team == t)
        if len(idx) >= min_games:
            a.append(cm[idx[0::2]].mean())
            b.append(cm[idx[1::2]].mean())
    if len(a) < 20:
        return np.nan
    r = spearmanr(a, b)[0]
    return 2 * r / (1 + r) if r == r else np.nan


def boot_scores(c, sides, n_teams, k, rng):
    """(R, T) scores where every replicate also resamples the season's games (with replacement)."""
    n_games = sides.game.max() + 1
    out = np.zeros((c.shape[0], n_teams))
    for r in range(c.shape[0]):
        counts = rng.multinomial(n_games, np.full(n_games, 1.0 / n_games)).astype(float)
        w = counts[sides.game]
        a = v2.team_matrix(sides, n_teams, w)
        tot = np.asarray(a.T @ c[r])
        n = np.asarray(a.sum(axis=0)).ravel()
        out[r] = tot / np.maximum(n + (k or 0.0), 1e-9) if k is not None else tot
    return out


def run_board(season, week, cur, prior, divs, seed):
    """Returns (rows, pair_payload) for one board."""
    seen = cur[cur["week"] < week]
    nxt = fbs_games(cur[(cur["week"] >= week) & (cur["week"] < week + HORIZON)])
    if seen.empty:
        return [], {}
    fbs_played = sorted({t for t, d in divs.items() if d == "fbs"} & (set(seen.home_team_name) | set(seen.away_team_name)))
    ys = v2.fit_yardstick(seen, prior, week, divs, "fbs", alpha=spec.margin_alpha, w0=spec.prior_w0, tau=spec.prior_tau,
                          n_draws=R, seed=seed, bench_teams=fbs_played)
    members = np.array([ys.index[t] for t in fbs_played if t in ys.index])
    names = [ys.teams[i] for i in members]
    sides = v2.sides_of(seen, ys.index)
    h = v2.played_matrix(seen, ys.index, members)
    sg = fbs_games(seen)
    rng = np.random.default_rng(seed + 1)
    rows, payload = [], {}

    def record(name, order_idx, p, extra=None):
        pos = {names[i]: r for r, i in enumerate(order_idx)}
        n, c = contradictions(pos, sg)
        top = {names[i]: r for r, i in enumerate(order_idx[:25])}
        tn, tc = contradictions(top, sg)
        fn, fw = contradictions(pos, nxt)
        rows.append(dict(season=season, week=week, arm=name, n=n, c=c, tn=tn, tc=tc, fn=fn, fw=fw, **(extra or {})))
        payload[name] = (names, order_idx, p)

    # --- the shipped board (v1), exactly as published
    fit_kw = dict(C=spec.ridge_C, w0=spec.prior_w0, tau=spec.prior_tau, major="fbs", divisions=divs, model="margin",
                  margin_alpha=spec.margin_alpha, margin_cap=None, margin_scale=spec.margin_scale)
    try:
        S1, ha1 = resume.opponent_ratings(seen, prior, week, fit_kw)
        bench1 = resume.top_benchmark(S1, fbs_played, resume.BENCH_N)
        sor1, var1 = resume.record_scores(seen, S1, ha1, bench1, resume.LAM, spec.margin_scale)
        teams1 = [t for t in fbs_played if t in sor1]
        order1 = resume.order_division(teams1, seen, sor1, var1, resume.Z)
        idx1 = np.array([names.index(t) for t in order1])
        s_arr = np.array([sor1.get(t, 0.0) for t in names]); v_arr = np.array([var1.get(t, 0.0) for t in names])
        p1 = norm.cdf((s_arr[:, None] - s_arr[None, :]) / np.sqrt(v_arr[:, None] + v_arr[None, :] + 1e-12))
        record("v1 shipped", idx1, p1, dict(rel=np.nan, width=np.nan))
    except Exception as e:  # noqa: BLE001
        print("v1 failed", season, week, e, flush=True)

    cache = {}
    for name, a in ARMS.items():
        key = (a["kind"], a["lam"])
        if key not in cache:
            cache[key] = (v2.credits(ys, sides, a["kind"], a["lam"]), v2.credits(ys, sides, a["kind"], a["lam"], use_mean=True)[0])
        c, cm = cache[key]
        if a["source"] == "YB":
            sc = boot_scores(c, sides, len(ys.teams), a["k"], rng)
        else:
            sc = v2.scores(c, sides, len(ys.teams), a["k"])
        mean = sc.mean(axis=0)[members]
        p = v2.pairwise(sc, members)
        start = np.array(sorted(range(len(members)), key=lambda i: (-mean[i], names[i])))
        order_idx = v2.consensus_order(p, h, a["eta"], start) if a["order"] == "cons" else start
        lo, hi = v2.ranges(p, order_idx, a["q"])
        rel = split_half(sides, cm, members)
        record(name, order_idx, p, dict(rel=rel, width=float(np.mean((hi - lo + 1)[:25]))))
    return rows, payload


def run(season):
    cur = GAMES[GAMES.season == season]
    prior = GAMES[GAMES.season == season - 1]
    prior = None if prior.empty else prior
    divs = core.season_divisions(prior, cur)
    all_rows, pairs, boards = [], [], {}
    for w in WEEKS:
        try:
            rows, payload = run_board(season, w, cur, prior, divs, seed=season * 100 + w)
        except Exception as e:  # noqa: BLE001
            print("board failed", season, w, repr(e), flush=True)
            continue
        all_rows += rows
        boards[w] = payload
    # M4: persistence of order three weeks later, for pairs in the top 40
    for w, payload in boards.items():
        later = boards.get(w + HORIZON)
        if not later:
            continue
        for arm_name, (names, order_idx, p) in payload.items():
            if arm_name not in later:
                continue
            n2, o2, _ = later[arm_name]
            pos2 = {n2[i]: r for r, i in enumerate(o2)}
            top = order_idx[:40]
            for a_i in range(len(top)):
                for b_i in range(a_i + 1, len(top)):
                    i, j = top[a_i], top[b_i]
                    ni, nj = names[i], names[j]
                    if ni in pos2 and nj in pos2:
                        pairs.append((season, w, arm_name, float(p[i, j]), int(pos2[ni] < pos2[nj]), b_i - a_i))
    return pd.DataFrame(all_rows), pd.DataFrame(pairs, columns=["season", "week", "arm", "p", "persist", "gap"])


if __name__ == "__main__":
    t0 = time.time()
    games = ev.load_games("cfb")
    seasons = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
    with ProcessPoolExecutor(WORKERS, initializer=init, initargs=(games,)) as pool:
        results = list(pool.map(run, seasons))
    rows = pd.concat([r[0] for r in results], ignore_index=True)
    pairs = pd.concat([r[1] for r in results], ignore_index=True)
    tune = {2017, 2018, 2019, 2021}
    rows["part"] = np.where(rows.season.isin(tune), "tune", "eval")
    pairs["part"] = np.where(pairs.season.isin(tune), "tune", "eval")
    rows.to_parquet(f"{OUT}/v2_rows.parquet")
    pairs.to_parquet(f"{OUT}/v2_pairs.parquet")
    json.dump({k: v for k, v in ARMS.items()}, open(f"{OUT}/v2_arms.json", "w"), indent=1)
    print("done", len(rows), "board-arm rows,", len(pairs), "pairs in", round(time.time() - t0), "s", flush=True)
