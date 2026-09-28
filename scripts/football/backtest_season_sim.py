"""Backtest the rest-of-season simulator: calibration of season-level probabilities.

For each season and as-of week N, the sim sees only results through week N (plus the two
prior seasons for the ratings), simulates the rest, and its probabilities are scored
against what actually happened:

  NFL   P(playoffs), P(division), P(#1 seed/bye), P(reach Super Bowl), P(win Super Bowl),
        final regular-season wins (RPS, MAE of the mean, 80% interval coverage)
  CFB   P(conference title), P(CFP), P(bye; 12-team years), P(national title), final
        regular-season wins, and the rank of the real CFP teams

Game models compared (src/season_sim/engine.py VARIANTS):
  point        ridge point estimate, independent games              (no rating uncertainty)
  draw         ridge + posterior rating draw per season path        (rating uncertainty)
  update       ridge refit weekly on simulated results, no draw     ("hot hand" / 538 style)
  draw_update  both
  record       current record, log5 with a 1-1 prior                (naive baseline)
  coin         every game 50/50                                     (floor)
  market       NFL only: closing spread of every remaining game. LEAKY: the closing line
               of a week-15 game is not known in week 3. An upper bound, not a baseline.

Preseason market win totals are NOT in nflverse (schedules carry per-game lines only), so
that comparison is not run.

Selection protocol: the production variant is chosen on NFL 2010-2017 (tune) and then
scored once on NFL 2018-2025 and CFB 2022-2025 (eval). CIs are paired bootstraps over
team-seasons (all of a team's as-of weeks resampled together), 2,000 resamples.

Usage:
  python scripts/football/backtest_season_sim.py nfl --schedule games.csv --seasons 2018-2025
  python scripts/football/backtest_season_sim.py cfb --games cfb_games_bq.parquet --rankings-dir D --teams T
  python scripts/football/backtest_season_sim.py report --inputs out_nfl.parquet out_cfb.parquet
  python scripts/football/backtest_season_sim.py validate --schedule games.csv   # tiebreakers
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from season_sim import cfb as cfbm  # noqa: E402
from season_sim import engine as eng  # noqa: E402
from season_sim import nfl as nflm  # noqa: E402
from season_sim import run  # noqa: E402

WEEKS = (3, 6, 9)
EPS = 1e-3


def _seasons(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# ------------------------------------------------------------------ actual outcomes
def nfl_actual(sched: pd.DataFrame, season: int) -> pd.DataFrame:
    frame = nflm.build_frame(sched, season)
    ns = nflm.NflSeason(frame, season)
    M = frame["margin"].to_numpy()[ns.rows][None, :]
    n_wc = 3 if season >= 2020 else 2
    st = ns.standings(M, np.random.default_rng(0), n_wc=n_wc)
    g = nflm.normalise_schedule(sched)
    sb = g[(g["season"] == season) & (g["game_type"] == "SB")].iloc[0]
    sb_teams = {sb["home_team"], sb["away_team"]}
    champ = sb["home_team"] if sb["margin"] > 0 else sb["away_team"]
    seeds = st["seeds"][0]
    rows = []
    for i, t in enumerate(nflm.TEAMS):
        rows.append({"team": t, "wins": float(st["wins"][0, i]),
                     "playoffs": int(i in seeds.ravel()), "division": int(st["div_win"][0, i]),
                     "bye": int(i in seeds[:, : (1 if n_wc == 3 else 2)]), "conf_title": int(t in sb_teams),
                     "champion": int(t == champ)})
    return pd.DataFrame(rows)


def cfb_actual(games: pd.DataFrame, season: int, ranked: list[str]) -> pd.DataFrame:
    frame = cfbm.build_frame(games, season)
    cs = cfbm.CfbSeason(frame, season, games)
    M = frame["margin"].to_numpy()[cs.rows][None, :]
    wins, _, _ = cs.records(M)
    g = cfbm.normalise_games(games)
    ccg = g[(g["season"] == season) & g["is_ccg"] & g["margin"].notna()]
    champs = {}
    champ_set = set()
    for r in ccg.itertuples():
        w = r.home_team if r.margin > 0 else r.away_team
        champ_set.add(w)
        if w in cs.idx:
            champs[r.hc] = np.array([cs.idx[w]])
    post = g[(g["season"] == season) & (g["is_postseason"] == 1) & g["margin"].notna()]
    last = post.sort_values("game_date").iloc[-1]
    champion = last["home_team"] if last["margin"] > 0 else last["away_team"]
    rank = {t: k for k, t in enumerate(ranked)}
    conf_arr = np.array([cs.conf_of[t] for t in cs.fbs])
    field = cfbm.select_field(np.array([[-rank.get(t, 99) for t in cs.fbs]], float),
                              champs, conf_arr, season)[0]
    field_names = [cs.fbs[i] for i in field]
    rows = []
    for i, t in enumerate(cs.fbs):
        rows.append({"team": t, "wins": float(wins[0, i]), "conf_title": int(t in champ_set),
                     "playoffs": int(t in field_names),
                     "bye": int(len(field_names) == 12 and t in field_names[:4]),
                     "champion": int(t == champion),
                     "cfp_rank": rank.get(t, np.nan) + 1 if t in rank else np.nan})
    return pd.DataFrame(rows)


def load_rankings(dirp: Path, teams_json: Path) -> dict[int, list[str]]:
    teams = json.loads(teams_json.read_text())["sports"][0]["leagues"][0]["teams"]
    id_to_name = {t["team"]["id"]: t["team"]["displayName"] for t in teams}
    out = {}
    for p in sorted(dirp.glob("cfp_*_w*.json")):
        season, w = int(p.stem.split("_")[1]), int(p.stem.split("_w")[1])
        d = json.loads(p.read_text())
        if d.get("ranks") and (season not in out or w == 16):
            ids = [x["team"]["$ref"].split("/teams/")[1].split("?")[0] for x in d["ranks"]]
            out[season] = [id_to_name[i] for i in ids]
    return out


# ------------------------------------------------------------------ prediction rows
def outcome_rows(o: run.Outcome, variant: str) -> pd.DataFrame:
    S = o.wins.shape[0]
    fl = o.flags
    rows = []
    max_g = int(o.n_games.max())
    for i, t in enumerate(o.teams):
        w = o.wins[:, i]
        dist = np.bincount(np.floor(w).astype(int), minlength=max_g + 1)[: max_g + 1] / S
        rows.append({
            "team": t, "variant": variant, "season": o.season, "as_of_week": o.as_of_week,
            "p_playoffs": fl["playoffs"][:, i].mean(), "p_bye": fl["bye"][:, i].mean(),
            "p_division": fl["division"][:, i].mean() if "division" in fl else np.nan,
            "p_conf_title": fl["conf_title"][:, i].mean(), "p_champion": fl["champion"][:, i].mean(),
            "mean_wins": w.mean(), "wins_p10": np.percentile(w, 10), "wins_p90": np.percentile(w, 90),
            "wins_dist": json.dumps(dist.round(5).tolist()),
            "exp_rank": o.final_rank[:, i].mean(), "runtime_s": o.runtime["total_s"],
        })
    return pd.DataFrame(rows)


def run_sport(sport: str, args) -> pd.DataFrame:
    variants = args.variants.split(",")
    frames = []
    if sport == "nfl":
        sched = pd.read_csv(args.schedule)
    else:
        games = pd.read_parquet(args.games)
        ranked = load_rankings(Path(args.rankings_dir), Path(args.teams))
    for season in _seasons(args.seasons):
        act = nfl_actual(sched, season) if sport == "nfl" else cfb_actual(games, season, ranked[season])
        act = act.add_prefix("y_").rename(columns={"y_team": "team"})
        for wk in WEEKS:
            for v in variants:
                if v == "market" and sport != "nfl":
                    continue
                t0 = time.time()
                if sport == "nfl":
                    o = run.sim_nfl(sched, season, wk, n_sims=args.sims, variant=v, seed=season * 100 + wk)
                else:
                    o = run.sim_cfb(games, season, wk, n_sims=args.sims, variant=v, seed=season * 100 + wk)
                r = outcome_rows(o, v).merge(act, on="team", how="left")
                frames.append(r)
                print(f"{sport} {season} wk{wk} {v:12s} {time.time() - t0:5.1f}s", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(args.out)
    return out


# ------------------------------------------------------------------ scoring
def _ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _brier(p, y):
    return (p - y) ** 2


def _rps(dist_json, y):
    d = np.array(json.loads(dist_json))
    cdf = np.cumsum(d)
    obs = (np.arange(len(d)) >= np.floor(y)).astype(float)
    return float(np.mean((cdf - obs) ** 2))


EVENTS = {"nfl": ["playoffs", "division", "bye", "conf_title", "champion"],
          "cfb": ["conf_title", "playoffs", "bye", "champion"]}


def per_row_scores(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    s = df[["team", "variant", "season", "as_of_week"]].copy()
    for e in EVENTS[sport]:
        y = df[f"y_{e}"].to_numpy(float)
        p = df[f"p_{e}"].to_numpy(float)
        ok = ~np.isnan(y)
        if sport == "cfb" and e == "bye":
            ok &= df["season"].to_numpy() >= 2024
        s[f"ll_{e}"] = np.where(ok, _ll(p, y), np.nan)
        s[f"br_{e}"] = np.where(ok, _brier(p, y), np.nan)
    s["rps_wins"] = [_rps(d, y) for d, y in zip(df["wins_dist"], df["y_wins"])]
    s["ae_wins"] = (df["mean_wins"] - df["y_wins"]).abs()
    s["cover80"] = ((df["y_wins"] >= df["wins_p10"]) & (df["y_wins"] <= df["wins_p90"])).astype(float)
    return s


def boot_ci(diff: pd.DataFrame, col: str, n=2000, seed=0):
    """Paired bootstrap over team-seasons of a per-row difference column."""
    g = diff.groupby(["team", "season"])[col].agg(["sum", "count"])
    g = g[g["count"] > 0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    sums, cnts = g["sum"].to_numpy()[idx].sum(1), g["count"].to_numpy()[idx].sum(1)
    est = g["sum"].sum() / g["count"].sum()
    lo, hi = np.percentile(sums / cnts, [2.5, 97.5])
    return est, lo, hi


def reliability(df: pd.DataFrame, e: str, bins=(0, .05, .15, .3, .5, .7, .85, .95, 1.0001)):
    p, y = df[f"p_{e}"].to_numpy(float), df[f"y_{e}"].to_numpy(float)
    ok = ~np.isnan(y)
    p, y = p[ok], y[ok]
    b = np.digitize(p, bins) - 1
    out = []
    for k in range(len(bins) - 1):
        m = b == k
        if m.any():
            out.append({"bin": f"{bins[k]:.2f}-{min(bins[k + 1], 1):.2f}", "n": int(m.sum()),
                        "mean_p": float(p[m].mean()), "obs": float(y[m].mean())})
    return out


def champion_ll(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-class log loss of the actual champion, one per (variant, season, week)."""
    rows = []
    for (v, s, w), g in df.groupby(["variant", "season", "as_of_week"]):
        p = g.loc[g["y_champion"] == 1, "p_champion"]
        if len(p):
            rows.append({"variant": v, "season": s, "as_of_week": w,
                         "champ_nll": -np.log(max(float(p.iloc[0]), 1.0 / 20000))})
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, sport: str, ref: str = "record") -> dict:
    sc = per_row_scores(df, sport)
    cols = [c for c in sc.columns if c.startswith(("ll_", "br_", "rps", "ae_", "cover"))]
    summary = sc.groupby("variant")[cols].mean()
    out = {"summary": summary.round(4).to_dict(orient="index"), "diffs": {}, "reliability": {},
           "by_week": sc.groupby(["variant", "as_of_week"])[cols].mean().round(4)
           .reset_index().to_dict(orient="records")}
    base = sc[sc["variant"] == ref].set_index(["team", "season", "as_of_week"])
    for v in sc["variant"].unique():
        if v == ref:
            continue
        cur = sc[sc["variant"] == v].set_index(["team", "season", "as_of_week"])
        d = (cur[cols] - base[cols]).reset_index()
        out["diffs"][f"{v}-{ref}"] = {c: [round(x, 5) for x in boot_ci(d, c)] for c in cols}
    for v in sc["variant"].unique():
        sub = df[df["variant"] == v]
        out["reliability"][v] = {e: reliability(sub, e) for e in EVENTS[sport]}
    ch = champion_ll(df)
    out["champion_nll"] = ch.groupby("variant")["champ_nll"].agg(["mean", "count"]).round(4).to_dict(orient="index")
    out["runtime_s_mean"] = df.groupby("variant")["runtime_s"].mean().round(3).to_dict()
    return out


def validate_tiebreakers(sched: pd.DataFrame, seasons) -> None:
    """Seed every season from its real results and compare with the real bracket:
    the playoff field, the bye(s), and the wild-card pairings, per conference."""
    g = nflm.normalise_schedule(sched)
    exact = n = 0
    for season in seasons:
        frame = nflm.build_frame(sched, season)
        ns = nflm.NflSeason(frame, season)
        n_wc = 3 if season >= 2020 else 2
        nb = 1 if n_wc == 3 else 2
        st = ns.standings(frame["margin"].to_numpy()[ns.rows][None, :],
                          np.random.default_rng(0), n_wc=n_wc)
        wc = g[(g["season"] == season) & (g["game_type"] == "WC")]
        dv = g[(g["season"] == season) & (g["game_type"] == "DIV")]
        byes = set(dv.home_team) - set(wc.home_team) - set(wc.away_team)
        for ci, c in enumerate(nflm.CONFS):
            mine = [nflm.TEAMS[i] for i in st["seeds"][0, ci]]
            k = len(mine)
            actual = {t for t in set(wc.home_team) | set(wc.away_team) | set(dv.home_team)
                      if nflm.CONF_OF[t] == c}
            pairs = {frozenset(p) for p in zip(wc.home_team, wc.away_team) if nflm.CONF_OF[p[0]] == c}
            mypairs = {frozenset((mine[i], mine[k - 1 - (i - nb)])) for i in range(nb, 4)}
            ok = set(mine) == actual and set(mine[:nb]) == {t for t in byes if nflm.CONF_OF[t] == c} \
                and mypairs == pairs
            exact += ok
            n += 1
            if not ok:
                print("MISMATCH", season, c, mine, sorted(actual))
    print(f"tiebreaker validation: {exact}/{n} conference-seasons exact (field, byes, pairings)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sport", choices=["nfl", "cfb", "report", "validate"])
    ap.add_argument("--schedule")
    ap.add_argument("--games")
    ap.add_argument("--rankings-dir")
    ap.add_argument("--teams")
    ap.add_argument("--seasons", default="2018-2025")
    ap.add_argument("--variants", default="point,draw,update,draw_update,record,coin,market")
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--out", default="season_sim_backtest.parquet")
    ap.add_argument("--inputs", nargs="*")
    ap.add_argument("--ref", default="record")
    a = ap.parse_args()
    if a.sport == "validate":
        validate_tiebreakers(pd.read_csv(a.schedule), _seasons(a.seasons))
        return
    if a.sport == "report":
        res = {}
        for p in a.inputs:
            df = pd.read_parquet(p)
            sport = "cfb" if "y_cfp_rank" in df.columns else "nfl"
            res[Path(p).stem] = report(df, sport, a.ref)
        print(json.dumps(res, indent=1, default=float))
        return
    run_sport(a.sport, a)


if __name__ == "__main__":
    main()
