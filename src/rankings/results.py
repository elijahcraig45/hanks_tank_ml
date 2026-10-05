"""A results-only order: the ranking that contradicts the fewest games already played.

The headline rating answers "who would win next week", so it blends this season with
last, uses margins, and corrects for schedule. This answers a different question — "who
has beaten whom" — and nothing else enters: no scores, no last season, no schedule
model. It is the order of the teams that leaves the smallest number of results where a
lower team beat a higher one (a minimum feedback arc set, found by local search).

It is NOT a forecast. Measured walk-forward on 2017-2025 college seasons (board as of
weeks 4, 5 and 6, scored on the next three weeks of games), it agrees with 98.6% of the
games already played and predicts the next ones about 7 points worse than the headline
rating (63.9% against 70.7%), the same as any ranker that fits history that tightly.
That is why it is published beside the rating, labelled as a record of results, and not
in its place.

Ties — teams the results cannot separate — fall back to a plain win/loss fit of this
season's games, so they are ordered by results too and the output is deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rankings import core

# Local search stops when a full pass moves nobody; the cap is a backstop, not a tuning
# knob (a 300-team season settles in under ten passes).
MAX_PASSES = 50


def _edges(games: pd.DataFrame) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    teams = sorted(set(games["home_team_name"]) | set(games["away_team_name"]))
    beat: dict[str, list[str]] = {t: [] for t in teams}
    lost_to: dict[str, list[str]] = {t: [] for t in teams}
    for g in games.itertuples(index=False):
        winner, loser = (
            (g.home_team_name, g.away_team_name) if g.home_won
            else (g.away_team_name, g.home_team_name)
        )
        beat[winner].append(loser)
        lost_to[loser].append(winner)
    return teams, beat, lost_to


def count_conflicts(order: list[str], games: pd.DataFrame) -> int:
    """Games in which the team ranked lower beat the team ranked higher."""
    pos = {t: i for i, t in enumerate(order)}
    n = 0
    for g in games.itertuples(index=False):
        winner, loser = (
            (g.home_team_name, g.away_team_name) if g.home_won
            else (g.away_team_name, g.home_team_name)
        )
        n += pos[loser] < pos[winner]
    return n


def start_order(games: pd.DataFrame, divisions: dict[str, str] | None,
                major: str | None, C: float) -> list[str]:
    """Best-first starting order from a win/loss fit of these games alone."""
    teams, beat, _ = _edges(games)
    try:
        strengths, _, _ = core.fit_ratings(games, C=C, divisions=divisions, major=major)
        return [t for t in strengths.index]
    except Exception:
        # A degenerate frame (one class of result): order by wins, then name.
        return sorted(teams, key=lambda t: (-len(beat[t]), t))


def min_violation_order(games: pd.DataFrame, start: list[str]) -> list[str]:
    """Reorder `start` until no single team can move to a slot with fewer conflicts.

    Each move strictly lowers the conflict count, so the search terminates; among equally
    good slots a team stays as close as it can to where it already was, which is what
    keeps the starting order's tie-breaks intact.
    """
    _, beat, lost_to = _edges(games)
    order = [t for t in start if t in beat]
    for _ in range(MAX_PASSES):
        moved = False
        for team in list(order):
            cur = order.index(team)
            rest = order[:cur] + order[cur + 1:]
            pos = {t: i for i, t in enumerate(rest)}
            beaten = np.sort([pos[t] for t in beat[team]])
            beaters = np.sort([pos[t] for t in lost_to[team]])
            slots = np.arange(len(rest) + 1)
            # Inserting before rest[k]: a beaten team above the slot, or a team that
            # beat us below it, is a conflict.
            cost = (np.searchsorted(beaten, slots, side="left")
                    + (len(beaters) - np.searchsorted(beaters, slots, side="left")))
            best = cost.min()
            if best < cost[cur]:
                candidates = np.flatnonzero(cost == best)
                slot = int(candidates[np.argmin(np.abs(candidates - cur))])
                order = rest[:slot] + [team] + rest[slot:]
                moved = True
        if not moved:
            break
    return order


def attach(table: pd.DataFrame, current: pd.DataFrame, divisions: dict[str, str] | None,
           major: str | None, C: float) -> pd.DataFrame:
    """Add `results_rank` (within the team's board) and `results_conflicts`.

    A team with no game this season has no results to order and gets nulls, as does the
    whole board in preseason.
    """
    out = table.copy()
    out["results_rank"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["results_conflicts"] = pd.array([pd.NA] * len(out), dtype="Int64")
    if current is None or current.empty:
        return out

    order = min_violation_order(current, start_order(current, divisions, major, C))
    pos = {t: i for i, t in enumerate(order)}

    conflicts = dict.fromkeys(order, 0)
    for g in current.itertuples(index=False):
        winner, loser = (
            (g.home_team_name, g.away_team_name) if g.home_won
            else (g.away_team_name, g.home_team_name)
        )
        if pos[loser] < pos[winner]:
            conflicts[winner] += 1
            conflicts[loser] += 1

    for _, idx in out.groupby("division", dropna=False).groups.items():
        members = [t for t in out.loc[idx, "team"] if t in pos]
        members.sort(key=lambda t: pos[t])
        ranks = {team: rank for rank, team in enumerate(members, start=1)}
        out.loc[idx, "results_rank"] = out.loc[idx, "team"].map(ranks).astype("Int64")
    out["results_conflicts"] = out["team"].map(conflicts).astype("Int64")
    return out
