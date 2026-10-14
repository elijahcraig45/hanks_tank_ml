"""The season-scale resume board: who has earned it, not who would win next week.

The headline rating is a forecast: margin-based, schedule-adjusted, last season carried in. It answers "who wins
next week" and, on a season scale, it can put a team above the team that beat it. This board answers a different
question, from this season's results (no scores for the team itself):

  1. Opponent quality comes from a win/loss-weighted rating that also knows last season (a 25% win/loss blend of the
     headline fit), because five or six games of win/loss alone cannot tell a weak conference from a strong one.
  2. Strength of record is a team's wins minus the wins a typical TOP-25 team would expect on the same schedule at
     the same venues. Against an average team a long run of wins over weak opponents still scores well; against a
     top-25 yardstick it earns almost nothing, a win over a strong team earns a lot and a loss to a weak team costs
     a lot. A bounded margin credit joins the win (LAM below): after five or six games the win alone cannot separate
     unbeaten teams. Its variance is the sum of p(1-p), so two records are "tied" only when the gap is within what one season
     of games cannot separate.
  3. Teams are ordered by strength of record. Head-to-head may then reorder two teams only if they are TIED with each
     other (the gap is under Z standard deviations of the difference): the order contradicts as few played games as
     it can without ever moving a team past one whose record is clearly different.

One win therefore never outweighs a clear gap in resume. Preseason, or a team with no game this season, gets no rank: there is no resume to grade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from rankings import core

# Each game scores (1-LAM)*win + LAM*Phi(margin/SIGMA): a bounded margin credit, so a 40-point win counts more than a
# 3-point win but cannot run away. SIGMA is about two touchdowns. LAM=0 is a pure win/loss resume.
LAM = 0.5
SIGMA = 14.0

# Opponents are rated by the headline fit with this much weight on the win/loss likelihood (rest: margins).
BLEND = 0.25
# The resume yardstick: the average of the best BENCH_N teams in the division.
BENCH_N = 25

# Passes of the head-to-head search; a full pass that moves nobody ends it (a season settles in a few).
MAX_PASSES = 50

# Teams are tied when the gap in strength of record is under Z standard deviations of the difference. Chosen by
# walk-forward backtest (see docs/RANKINGS_RESUME.md), not by eye.
Z = 0.5


def _winner_loser(g):
    return ((g.home_team_name, g.away_team_name) if g.home_won
            else (g.away_team_name, g.home_team_name))


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(40)
_GH_W = _GH_W / _GH_W.sum()


def _game_score(margin, lam: float):
    return (1.0 - lam) * (np.asarray(margin) > 0) + lam * norm.cdf(np.asarray(margin) / SIGMA)


def _expected_score(mu: float, sg: float, lam: float) -> tuple[float, float]:
    """E and Var of the game score when the margin is Normal(mu, sg), by quadrature."""
    c = _game_score(mu + sg * _GH_X, lam)
    e = float((_GH_W * c).sum())
    return e, float((_GH_W * c * c).sum()) - e * e


def iter_credits(current: pd.DataFrame, strengths: pd.Series, home_adv: float, benchmark: float = 0.0,
                 lam: float = 0.0, margin_scale: float = 10.62):
    """Yield (team, opponent, game row, credit, variance) once per team per game.

    A game's credit is its score minus the score a team rated `benchmark` would expect against the same opponent at
    the same venue. With `lam` = 0 the score is the win (1 or 0) and the expectation is the win probability, so this is
    wins minus expected wins. With lam > 0 a bounded margin credit joins it (a game without scores counts as a win/loss).
    """
    sg = margin_scale * np.pi / np.sqrt(3.0)       # game-margin sd implied by the logistic scale
    has_margin = "margin" in current.columns
    for g in current.itertuples(index=False):
        m = getattr(g, "margin", np.nan) if has_margin else np.nan
        for team, opp, is_home, won in (
            (g.home_team_name, g.away_team_name, True, g.home_won),
            (g.away_team_name, g.home_team_name, False, 1 - g.home_won),
        ):
            if opp not in strengths.index:
                continue
            venue = 0.0 if g.neutral_site else (home_adv if is_home else -home_adv)
            if lam <= 0 or pd.isna(m):
                p = core.win_prob(benchmark + venue, float(strengths[opp]))
                yield team, opp, g, float(won) - p, p * (1.0 - p)
                continue
            mu = (benchmark + venue - float(strengths[opp])) / core.ELO_SCALE * margin_scale
            e, v = _expected_score(mu, sg, lam)
            yield team, opp, g, float(_game_score(m if is_home else -m, lam)) - e, v


def record_scores(current: pd.DataFrame, strengths: pd.Series, home_adv: float, benchmark: float = 0.0,
                  lam: float = 0.0, margin_scale: float = 10.62) -> tuple[dict[str, float], dict[str, float]]:
    """(strength of record, its variance) per team: the sum of `iter_credits` over its games.

    A benchmark of 0 is an average team, which beats most opponents, so a long run of wins over weak teams still scores
    well. A benchmark near the top of the division (see `top_benchmark`) is the usual resume yardstick: beating a weak
    team earns almost nothing, beating a strong one a lot, and losing to a weak one costs a lot.
    """
    sor: dict[str, float] = {}
    var: dict[str, float] = {}
    for team, _, _, credit, v in iter_credits(current, strengths, home_adv, benchmark, lam, margin_scale):
        sor[team] = sor.get(team, 0.0) + credit
        var[team] = var.get(team, 0.0) + v
    return sor, var


def top_benchmark(strengths: pd.Series, teams: list[str], n: int = 25) -> float:
    """Mean rating of the best `n` of `teams`: the 'typical top-25 team' a resume is graded against."""
    vals = sorted((float(strengths[t]) for t in teams if t in strengths.index), reverse=True)[:n]
    return float(np.mean(vals)) if vals else 0.0


def tied(a: str, b: str, sor: dict[str, float], var: dict[str, float], z: float) -> bool:
    """Two records cannot be told apart: the gap is under z standard deviations of the difference."""
    return abs(sor[a] - sor[b]) < z * np.sqrt(var[a] + var[b])


def order_division(teams: list[str], current: pd.DataFrame, sor: dict[str, float], var: dict[str, float],
                   z: float = Z) -> list[str]:
    """Best-first order of `teams` (all with a game this season).

    Starts from strength of record, then lets head-to-head move a team only among teams it is TIED with: a team is
    never moved past one whose record is clearly different, however the games between them went. Each move strictly
    lowers the number of played games the order contradicts, so the search ends.
    """
    order = sorted(teams, key=lambda t: (-sor[t], t))
    members = set(order)
    beat: dict[str, list[str]] = {t: [] for t in order}
    lost_to: dict[str, list[str]] = {t: [] for t in order}
    for g in current.itertuples(index=False):
        w, l = _winner_loser(g)
        if w in members and l in members:
            beat[w].append(l)
            lost_to[l].append(w)
    for _ in range(MAX_PASSES):
        moved = False
        for team in list(order):
            cur = order.index(team)
            rest = order[:cur] + order[cur + 1:]
            lo, hi = 0, len(rest)
            for k in range(cur - 1, -1, -1):          # nearest clearly-better team above: cannot pass it
                if not tied(team, rest[k], sor, var, z):
                    lo = k + 1
                    break
            for k in range(cur, len(rest)):           # nearest clearly-worse team below: cannot pass it
                if not tied(team, rest[k], sor, var, z):
                    hi = k
                    break
            pos = {t: i for i, t in enumerate(rest)}
            above = np.sort([pos[t] for t in beat[team]])
            below = np.sort([pos[t] for t in lost_to[team]])
            slots = np.arange(lo, hi + 1)
            # Inserting before rest[k]: a beaten team above the slot, or a team that beat us below it, is a conflict.
            cost = (np.searchsorted(above, slots, side="left")
                    + (len(below) - np.searchsorted(below, slots, side="left")))
            stay = cost[cur - lo]
            if cost.min() < stay:
                cands = np.flatnonzero(cost == cost.min())
                slot = int(slots[cands[np.argmin(np.abs(slots[cands] - cur))]])
                order = rest[:slot] + [team] + rest[slot:]
                moved = True
        if not moved:
            break
    return order


def opponent_ratings(current: pd.DataFrame, prior: pd.DataFrame | None, week: int, fit_kw: dict
                     ) -> tuple[pd.Series, float]:
    """(ratings, home-field points) used to grade every opponent. If the fit fails (one outcome class, e.g. only
    home wins so far) there is no schedule information yet: every opponent counts the same."""
    try:
        s, home_adv, _ = core.fit_with_prior(current, prior, week, **{**fit_kw, "model": "blend", "blend": BLEND})
        return s, home_adv
    except Exception:
        teams = sorted(set(current["home_team_name"]) | set(current["away_team_name"]))
        return pd.Series(0.0, index=teams), 0.0


RESUME_COLUMNS = ("resume_rank", "resume_sor", "resume_sd", "resume_tie_lo", "resume_tie_hi",
                  "resume_summary", "resume_games_json")
TEXT_COLUMNS = ("resume_summary", "resume_games_json")


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def _game_line(team: str, opp: str, g, credit: float, rank_of: dict[str, int]) -> dict:
    home = g.home_team_name == team
    pf, pa = (g.home_score, g.away_score) if home else (g.away_score, g.home_score)
    won = bool(g.home_won) == home
    return {"opp": opp, "opp_rank": rank_of.get(opp), "date": str(pd.Timestamp(g.game_date).date()),
            "site": "N" if g.neutral_site else ("H" if home else "A"), "won": won,
            "pf": _num(pf), "pa": _num(pa), "credit": round(float(credit), 2)}


def _phrase(line: dict) -> str:
    verb = ("won" if line["won"] else "lost")
    prep = {"A": "at", "H": "vs", "N": "vs"}[line["site"]] if line["won"] else {"A": "at", "H": "to", "N": "to"}[line["site"]]
    rank = f" (#{line['opp_rank']})" if line["opp_rank"] else ""
    return f"{verb} {prep} {line['opp']}{rank}, {line['credit']:+.2f}"


def summary_line(lines: list[dict], sor: float, lo: int, hi: int) -> str:
    wins = sum(1 for x in lines if x["won"])
    tie = f", tied with #{lo}\u2013#{hi}" if hi > lo else ""
    out = f"{wins}-{len(lines) - wins}, strength of record {sor:+.2f}{tie}."
    if lines:
        out += f" Best result: {_phrase(max(lines, key=lambda x: x['credit']))}."
    if len(lines) > 1:
        out += f" Weakest: {_phrase(min(lines, key=lambda x: x['credit']))}."
    return out


def resume_table(table: pd.DataFrame, current: pd.DataFrame, strengths: pd.Series | None, home_adv: float,
                 z: float = Z, lam: float = LAM, margin_scale: float = 10.62) -> pd.DataFrame:
    """Every resume column for `table`'s rows, index-aligned; all null where there is no resume to grade.

    resume_rank / resume_sor / resume_sd: the order, strength of record and its noise. resume_tie_lo..hi: the best and
    worst rank among the teams this one's record is tied with (itself included), so the site can say "tied with
    #3-#7". resume_summary: one line; resume_games_json: per-game results and credits.
    """
    import json

    out = pd.DataFrame(index=table.index)
    out["resume_rank"] = pd.array([pd.NA] * len(table), dtype="Int64")
    out["resume_sor"] = np.nan
    out["resume_sd"] = np.nan
    out["resume_tie_lo"] = pd.array([pd.NA] * len(table), dtype="Int64")
    out["resume_tie_hi"] = pd.array([pd.NA] * len(table), dtype="Int64")
    out["resume_summary"] = pd.Series([None] * len(table), index=table.index, dtype="object")
    out["resume_games_json"] = pd.Series([None] * len(table), index=table.index, dtype="object")
    if current is None or current.empty or strengths is None:
        return out
    played = set(current["home_team_name"]) | set(current["away_team_name"])
    for _, idx in table.groupby("division", dropna=False).groups.items():
        teams = [t for t in table.loc[idx, "team"] if t in played]
        if not teams:
            continue
        bench = top_benchmark(strengths, teams, BENCH_N)
        sor: dict[str, float] = {}
        var: dict[str, float] = {}
        credits: dict[str, list] = {}
        for team, opp, g, credit, v in iter_credits(current, strengths, home_adv, bench, lam, margin_scale):
            if team not in teams:
                continue
            sor[team] = sor.get(team, 0.0) + credit
            var[team] = var.get(team, 0.0) + v
            credits.setdefault(team, []).append((opp, g, credit))
        teams = [t for t in teams if t in sor]
        order = order_division(teams, current, sor, var, z)
        rank = {t: r for r, t in enumerate(order, start=1)}
        for team in teams:
            ties = [rank[u] for u in teams if u == team or tied(team, u, sor, var, z)]
            lines = [_game_line(team, opp, g, c, rank) for opp, g, c in sorted(credits[team], key=lambda x: x[1].game_date)]
            lo, hi = min(ties), max(ties)
            row = table.index[(table["team"] == team) & table.index.isin(idx)][0]
            out.loc[row, "resume_rank"] = rank[team]
            out.loc[row, "resume_sor"] = round(sor[team], 2)
            out.loc[row, "resume_sd"] = round(float(np.sqrt(var[team])), 2)
            out.loc[row, "resume_tie_lo"] = lo
            out.loc[row, "resume_tie_hi"] = hi
            out.loc[row, "resume_summary"] = summary_line(lines, sor[team], lo, hi)
            out.loc[row, "resume_games_json"] = json.dumps(lines)
    return out


def rank_resume(table: pd.DataFrame, current: pd.DataFrame, strengths: pd.Series | None, home_adv: float,
                z: float = Z, lam: float = LAM, margin_scale: float = 10.62) -> pd.Series:
    """Integer `resume_rank` within each division of `table` (null where the team has no game this season)."""
    return resume_table(table, current, strengths, home_adv, z, lam, margin_scale)["resume_rank"]
