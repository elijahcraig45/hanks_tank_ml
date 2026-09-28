"""Fit the CFP committee proxy used by src/season_sim/cfb.py, on real final rankings.

    score = end-of-regular-season margin-ridge rating + a * losses + b * conference champion
            + c * Group of Six member

Target: the committee's final top 25 for 2021-2025 (ESPN core API, ranking id 21; the
last regular-season week). Objective: pairwise ordering accuracy over every pair that
involves at least one ranked team (ranked above unranked, and the order within the 25),
over all FBS teams. Also reported: how many actual playoff teams the proxy picks when
the season's real format is applied to the real conference champions.

Usage (local caches, read-only):
  python scripts/football/fit_cfp_committee.py --games cfb_games_bq.parquet \
      --rankings-dir DIR_WITH_cfp_YYYY_wNN.json --teams espn_teams_all.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from season_sim import cfb as cfbm  # noqa: E402
from season_sim import engine as eng  # noqa: E402


def end_state(games: pd.DataFrame, season: int):
    frame = cfbm.build_frame(games, season)
    ops = eng.RidgeOps(frame, eng.mr.CFB_RIDGE)
    cs = cfbm.CfbSeason(frame, season, games)
    t_end = eng.mr.time_index(season, int(frame.loc[frame["season"] == season, "week"].max()) + 1)
    c, *_ = ops.solve(t_end, include_sim=False)
    rating = c[ops.team_index(cs.fbs)]
    M = ops.margin[cs.rows][None, :]
    wins, losses, _ = cs.records(M)
    g = cfbm.normalise_games(games)
    ccg = g[(g["season"] == season) & g["is_ccg"] & g["margin"].notna()]
    champ = np.zeros(cs.T)
    champs = {}
    for r in ccg.itertuples():
        w, l_ = (r.home_team, r.away_team) if r.margin > 0 else (r.away_team, r.home_team)
        if w in cs.idx:
            champ[cs.idx[w]] = 1
            champs[r.hc] = cs.idx[w]
        if l_ in cs.idx:
            losses[0, cs.idx[l_]] += 1
    return cs, rating, losses[0], champ, champs


def load_ranking(dirp: Path, season: int, id_to_name: dict) -> list[str]:
    for w in (16, 15):
        p = dirp / f"cfp_{season}_w{w}.json"
        if p.exists():
            d = json.loads(p.read_text())
            if d.get("ranks"):
                ids = [x["team"]["$ref"].split("/teams/")[1].split("?")[0] for x in d["ranks"]]
                return [id_to_name[i] for i in ids]
    raise FileNotFoundError(season)


def pair_acc(score, cs, ranked):
    pos = {t: k for k, t in enumerate(ranked)}
    idx = [cs.idx[t] for t in ranked if t in cs.idx]
    others = [i for i in range(cs.T) if cs.fbs[i] not in pos]
    ok = n = 0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            ok += score[idx[a]] > score[idx[b]]
            n += 1
        ok += (score[idx[a]] > score[others]).sum()
        n += len(others)
    return ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--rankings-dir", required=True)
    ap.add_argument("--teams", required=True)
    ap.add_argument("--seasons", default="2021,2022,2023,2024,2025")
    a = ap.parse_args()
    games = pd.read_parquet(a.games)
    teams = json.loads(Path(a.teams).read_text())["sports"][0]["leagues"][0]["teams"]
    id_to_name = {t["team"]["id"]: t["team"]["displayName"] for t in teams}
    seasons = [int(s) for s in a.seasons.split(",")]
    states = {}
    for s in seasons:
        cs, rating, losses, champ, champs = end_state(games, s)
        ranked = load_ranking(Path(a.rankings_dir), s, id_to_name)
        missing = [t for t in ranked if t not in cs.idx]
        if missing:
            print(s, "ranked teams not found:", missing)
        states[s] = (cs, rating, losses, champ, champs, ranked)

    g6 = {s: cfbm.group6([st[0].conf_of[t] for t in st[0].fbs], s) for s, st in states.items()}

    def acc(p, keep=None):
        lw, cb, gp = p
        return np.mean([pair_acc(cfbm.committee_score(r, l_, c, g6[s], lw, cb, gp), cs, rk)
                        for s, (cs, r, l_, c, _, rk) in states.items()
                        if keep is None or s in keep])

    def pick(grid, keep=None):
        """Smallest-magnitude weights within 0.001 of the best: the surface is a flat
        ridge that otherwise drifts to ever larger weights."""
        scored = [(acc(p, keep), p) for p in grid]
        top = max(x for x, _ in scored)
        near = [p for x, p in scored if x >= top - 0.001]
        return min(near, key=lambda p: (abs(p[0]) + abs(p[2]), abs(p[1])))

    lw_grid = np.arange(-15, 0.01, 0.5)
    grids = {
        "rating only": [(0.0, 0.0, 0.0)],
        "+ losses, champion": [(a, b, 0.0) for a in lw_grid for b in (0, 0.5, 1, 2, 4)],
        "+ Group of Six": [(a, b, c) for a in lw_grid for b in (0, 0.5, 1, 2, 4)
                           for c in np.arange(-20, 0.01, 1)],
    }
    chosen = {}
    for name, grid in grids.items():
        p = pick(grid)
        chosen[name] = p
        print(f"{name:20s} weights {tuple(round(float(x), 2) for x in p)}  pairwise {acc(p):.4f}")
    # leave-one-season-out: is the gain stable out of sample?
    for s in seasons:
        keep = [x for x in seasons if x != s]
        row = []
        for name, grid in grids.items():
            p = pick(grid, keep)
            row.append(f"{name} {acc(p, [s]):.4f}")
        print(f"LOSO {s}: " + " | ".join(row))
    best = chosen["+ Group of Six"]
    # field reproduction with the real format and the real champions
    for s, (cs, r, l_, c, champs, rk) in states.items():
        for name, p in (("rating", chosen["rating only"]), ("proxy", best)):
            sc = cfbm.committee_score(r, l_, c, g6[s], *p)[None, :]
            conf_arr = np.array([cs.conf_of[t] for t in cs.fbs])
            seeds = cfbm.select_field(sc, {k: np.array([v]) for k, v in champs.items()}, conf_arr, s)[0]
            actual_rank = {t: k for k, t in enumerate(rk)}
            real = cfbm.select_field(
                np.array([[-actual_rank.get(t, 99) for t in cs.fbs]], float),
                {k: np.array([v]) for k, v in champs.items()}, conf_arr, s)[0]
            got = len(set(seeds) & set(real))
            print(f"{s} {name:6s} field overlap {got}/{len(real)}  seeds exact "
                  f"{sum(int(x == y) for x, y in zip(seeds, real))}/{len(real)}")


if __name__ == "__main__":
    main()
