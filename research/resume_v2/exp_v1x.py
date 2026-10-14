"""Round 3: the exact shipping candidate on the PRODUCTION module (rankings.resume): floor and margin weight vs live."""
import sys, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr

W = sys.argv[1]
sys.path.insert(0, W)
import exp_v2 as X
from exp_v2 import core, resume, spec, fbs_games, contradictions, ev  # noqa: E402
sys.path.insert(0, W + "/prod")        # the production module under test (resume.py with floor)
import importlib.util
sp = importlib.util.spec_from_file_location("rankings_resume_prod", W + "/prod/resume.py")
RP = importlib.util.module_from_spec(sp); sys.modules["rankings_resume_prod"] = RP
sp.loader.exec_module(RP)

WEEKS, HORIZON = X.WEEKS, X.HORIZON
ARMS = {"live": dict(lam=0.5, floor=False), "F.5": dict(lam=0.5, floor=True), "F.75": dict(lam=0.75, floor=True),
        "noF.75": dict(lam=0.75, floor=False)}


def split_half(per_team, min_games=4):
    a, b = [], []
    for credits in per_team.values():
        if len(credits) >= min_games:
            a.append(np.mean(credits[0::2])); b.append(np.mean(credits[1::2]))
    if len(a) < 20: return np.nan
    r = spearmanr(a, b)[0]
    return 2 * r / (1 + r) if r == r else np.nan


def run_board(season, week, cur, prior, divs):
    seen = cur[cur["week"] < week]
    nxt = fbs_games(cur[(cur["week"] >= week) & (cur["week"] < week + HORIZON)])
    if seen.empty: return [], {}
    fbs_played = sorted({t for t, d in divs.items() if d == "fbs"} & (set(seen.home_team_name) | set(seen.away_team_name)))
    fit_kw = dict(C=spec.ridge_C, w0=spec.prior_w0, tau=spec.prior_tau, major="fbs", divisions=divs, model="margin",
                  margin_alpha=spec.margin_alpha, margin_cap=None, margin_scale=spec.margin_scale)
    S, ha = RP.opponent_ratings(seen, prior, week, fit_kw)
    bench = RP.top_benchmark(S, fbs_played, RP.BENCH_N)
    sg = fbs_games(seen)
    rows, payload = [], {}
    for name, a in ARMS.items():
        sor, var = RP.record_scores(seen, S, ha, bench, a["lam"], spec.margin_scale, a["floor"])
        teams = [t for t in fbs_played if t in sor]
        order = RP.order_division(teams, seen, sor, var, RP.Z)
        pos = {t: i for i, t in enumerate(order)}
        n, c = contradictions(pos, sg); tn, tc = contradictions({t: i for i, t in enumerate(order[:25])}, sg); fn, fw = contradictions(pos, nxt)
        per = {}
        for team, _, _, credit, _ in RP.iter_credits(seen, S, ha, bench, a["lam"], spec.margin_scale, a["floor"]):
            if team in pos: per.setdefault(team, []).append(credit)
        s_arr = np.array([sor[t] for t in order]); v_arr = np.array([var[t] for t in order])
        p = norm.cdf((s_arr[:, None] - s_arr[None, :]) / np.sqrt(v_arr[:, None] + v_arr[None, :] + 1e-12))
        rows.append(dict(season=season, week=week, arm=name, n=n, c=c, tn=tn, tc=tc, fn=fn, fw=fw, rel=split_half(per), width=np.nan))
        payload[name] = (list(order), np.arange(len(order)), p)
    return rows, payload


def run(season):
    cur = X.GAMES[X.GAMES.season == season]; prior = X.GAMES[X.GAMES.season == season - 1]
    prior = None if prior.empty else prior
    divs = core.season_divisions(prior, cur)
    all_rows, pairs, boards = [], [], {}
    for w in WEEKS:
        try: rows, payload = run_board(season, w, cur, prior, divs)
        except Exception as e: print("board failed", season, w, repr(e), flush=True); continue
        all_rows += rows; boards[w] = payload
    for w, payload in boards.items():
        later = boards.get(w + HORIZON)
        if not later: continue
        for arm_name, (names, order_idx, p) in payload.items():
            if arm_name not in later: continue
            n2, _, _ = later[arm_name]
            pos2 = {t: r for r, t in enumerate(n2)}
            top = list(range(min(40, len(names))))
            for a_i in range(len(top)):
                for b_i in range(a_i + 1, len(top)):
                    if names[a_i] in pos2 and names[b_i] in pos2:
                        pairs.append((season, w, arm_name, float(p[a_i, b_i]), int(pos2[names[a_i]] < pos2[names[b_i]]), b_i - a_i))
    return pd.DataFrame(all_rows), pd.DataFrame(pairs, columns=["season", "week", "arm", "p", "persist", "gap"])


if __name__ == "__main__":
    t0 = time.time()
    games = ev.load_games("cfb")
    with ProcessPoolExecutor(3, initializer=X.init, initargs=(games,)) as pool:
        results = list(pool.map(run, [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]))
    rows = pd.concat([r[0] for r in results], ignore_index=True); pairs = pd.concat([r[1] for r in results], ignore_index=True)
    tune = {2017, 2018, 2019, 2021}
    rows["part"] = np.where(rows.season.isin(tune), "tune", "eval"); pairs["part"] = np.where(pairs.season.isin(tune), "tune", "eval")
    rows.to_parquet(f"{W}/v1x_rows.parquet"); pairs.to_parquet(f"{W}/v1x_pairs.parquet")
    print("done", len(rows), len(pairs), round(time.time() - t0), "s", flush=True)
