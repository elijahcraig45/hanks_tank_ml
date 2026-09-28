"""Why each team is where it is, computed from the fitted rating itself.

Everything here is derived from the same fit that produced the board: no polls, no
outside ratings, no adjectives. Three things are computed per team.

1. Where the rating comes from: last season's games vs this season's.

   Both likelihoods are (weighted) ridge fits, and at the fitted optimum both satisfy

       beta = (X'SX + lam I)^-1 X'S z = Q X'S z

   The margin model is literally this, with S the sample weights, z the (capped)
   margins and lam = margin_alpha. The W/L Bradley-Terry fit satisfies it at its
   optimum too, as the IRLS fixed point: S = w p(1-p), z = X beta + (y - p)/(p(1-p))
   and lam = 1/C (the stationarity condition beta = C X'W(y - p), rearranged). The
   rating is a linear map M of beta (Elo scaling, the division baseline, centring), so

       rating = M Q X'S z = sum over games g of A[:, g] z[g]

   which splits exactly into a last-season part and a this-season part. The parts sum
   to the rating to floating-point precision for the margin model, and to the
   optimizer's tolerance for Bradley-Terry.

2. What each game is worth: rating(with the game) - rating(without it).

   Leaving a block B of games out of a ridge fit has a closed form (Woodbury):

       beta - beta_{-B} = Q X_B' (S_B^-1 - X_B Q X_B')^-1 (z_B - X_B beta)

   i.e. the game's residual times its leverage. For the margin model this is exactly
   the refit without those games; for Bradley-Terry it is the one-Newton-step
   approximation, which the tests check against real refits. Football reports single
   games; MLB reports each season series (all games against one opponent), because a
   single baseball game moves a rating by a fraction of a point.

3. How the schedule compares: mean opponent rating, played and still to play.

A plain-English summary line is then assembled from those numbers by fixed templates.
It never uses a word the numbers do not support. How firmly a team sits above its
neighbour is always quoted as a number: the share of bootstrap resamples that keep the
published order, followed by a label from ORDER_BANDS (below).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

try:  # package layout locally, flat module tree in Cloud Functions
    from rankings import core
except ImportError:  # pragma: no cover
    import core  # type: ignore

logger = logging.getLogger(__name__)

# How firmly one team sits above another, from p = the share of bootstrap resamples that
# rate it higher. The percentage is always shown (rounded to a whole number, and the band
# is read off that rounded number); the label is a coarse reading of it:
#
#   under 60%      "a coin flip"    the order flips in more than 40% of resamples
#   60% to < 75%   "a slight edge"  flips in 25-40%
#   75% to < 90%   "a clear edge"   flips in 10-25%
#   90% and over   "separated"      flips in fewer than 1 in 10
#
# The bands are symmetric around 50%: the point rank comes from the full fit and the
# resamples can disagree with it, so p can fall below one half. At 40% and under the label
# is the band of 100 - p plus "the other way" (31% reads "a slight edge the other way");
# 41-59% is a coin flip either side of 50. The backend compare endpoint (hanks_tank_backend
# src/utils/rankings-compare.ts) carries the same table and must stay identical.
ORDER_BANDS = ((0.90, "separated"), (0.75, "a clear edge"), (0.60, "a slight edge"))
ORDER_COIN_FLIP = "a coin flip"

# The machine-readable `tied` flag on a pair: the order holds in under 75% of resamples,
# i.e. "a coin flip" or "a slight edge". Kept for API consumers; the text no longer uses
# the word.
TIE_ORDER_P = 0.75

# How many best wins / worst losses to keep per team.
TOP_GAMES = 3

# Pair explanations list common opponents one by one up to this many; beyond it only
# the totals are kept.
MAX_COMMON_LISTED = 6


# ── The linear system behind a fit ─────────────────────────────────────────


@dataclass
class System:
    """One fitted model written as a weighted ridge solve.

    `frame` is every game in the fit (last season first, then this season) with an
    `is_current` flag; `rows` maps each fitted row of X back to its position in
    `frame` (games without a score are not in a margin fit).
    """

    teams: list[str]
    frame: pd.DataFrame
    rows: np.ndarray
    x: object  # scipy csr, n_fit x p
    s: np.ndarray
    z: np.ndarray
    beta: np.ndarray
    q: np.ndarray  # (X'SX + lam I)^-1, p x p
    m: np.ndarray  # coefficient -> centred Elo-scale rating, k x p
    to_elo: float

    @property
    def ratings(self) -> pd.Series:
        return pd.Series(self.m @ self.beta, index=self.teams)

    @property
    def home_adv(self) -> float:
        return float(self.beta[len(self.teams)] * self.to_elo)

    def parts(self) -> tuple[pd.Series, pd.Series]:
        """(last-season part, this-season part) of every team's rating."""
        mq = self.m @ self.q
        xts = self.x.T.multiply(self.s).tocsr()
        a = (xts.T @ mq.T).T  # k x n_fit, dense
        a = np.asarray(a)
        current = self.frame["is_current"].to_numpy(dtype=bool)[self.rows]
        prior_part = a[:, ~current] @ self.z[~current]
        current_part = a[:, current] @ self.z[current]
        return (pd.Series(prior_part, index=self.teams),
                pd.Series(current_part, index=self.teams))

    def leave_out(self, frame_positions) -> pd.Series:
        """rating(full) - rating(without these games), for every team."""
        pos = {int(p): i for i, p in enumerate(self.rows)}
        fit_rows = [pos[int(p)] for p in frame_positions if int(p) in pos]
        if not fit_rows:
            return pd.Series(0.0, index=self.teams)
        xb = self.x[fit_rows].toarray()
        sb = self.s[fit_rows]
        eb = self.z[fit_rows] - xb @ self.beta
        qxb = self.q @ xb.T
        k_inv = np.diag(1.0 / sb) - xb @ qxb
        delta_beta = qxb @ np.linalg.solve(k_inv, eb)
        return pd.Series(self.m @ delta_beta, index=self.teams)


def _mapping(teams: list[str], p: int, to_elo: float, divisions: dict[str, str],
             major: str | None) -> np.ndarray:
    k = len(teams)
    m = np.zeros((k, p))
    m[np.arange(k), np.arange(k)] = to_elo
    if major is not None:
        for i, t in enumerate(teams):
            if divisions.get(t) == major:
                m[i, k + 1] = core.DIV_SCALE * to_elo
    return m - m.mean(axis=0, keepdims=True)


def _fit_frame(current: pd.DataFrame | None, prior: pd.DataFrame | None, week: int,
               w0: float, tau: float) -> tuple[pd.DataFrame, np.ndarray]:
    """The exact games and sample weights core.fit_with_prior fits."""
    has_prior = prior is not None and not prior.empty
    has_current = current is not None and not current.empty
    if not has_current:
        frame = prior.assign(is_current=False)
        return frame.reset_index(drop=True), np.ones(len(frame))
    w = core.prior_weight(week, w0, tau) if has_prior else 0.0
    if w <= 0:
        frame = current.assign(is_current=True)
        return frame.reset_index(drop=True), np.ones(len(frame))
    frame = pd.concat([prior.assign(is_current=False), current.assign(is_current=True)],
                      ignore_index=True)
    weights = np.concatenate([np.full(len(prior), w), np.ones(len(current))])
    return frame, weights


def build_system(current, prior, week: int, *, C: float, w0: float, tau: float,
                 major: str | None, divisions: dict[str, str], model: str = "bt",
                 margin_alpha: float = 10.0, margin_cap: float | None = None,
                 margin_scale: float = 10.0, **_ignored) -> System:
    """Rebuild the fit core.fit_with_prior performs, as an explicit linear system."""
    frame, weights = _fit_frame(current, prior, week, w0, tau)
    teams = sorted(set(frame["home_team_name"]) | set(frame["away_team_name"]))
    x, y = core._design(frame, teams, divisions, major)
    p = x.shape[1]

    if model == "margin":
        z = frame["margin"].to_numpy(dtype=float)
        ok = ~np.isnan(z)
        if margin_cap is not None:
            z = np.clip(z, -margin_cap, margin_cap)
        rows = np.flatnonzero(ok)
        x, z, s = x[rows], z[rows], weights[rows]
        lam = margin_alpha
        to_elo = core.ELO_SCALE / margin_scale
        xts = x.T.multiply(s).tocsr()
        gram = (xts @ x).toarray()
        gram[np.diag_indices_from(gram)] += lam
        q = np.linalg.inv(gram)
        beta = q @ (xts @ z)
    elif model == "bt":
        rows = np.arange(len(frame))
        # The same solve core._solve publishes, so beta is the published fit.
        beta = core.bt_coef(x, y, C, weights)
        lam = 1.0 / C
        prob = expit(x @ beta)
        v = np.clip(prob * (1.0 - prob), 1e-9, None)
        s = weights * v
        z = x @ beta + (y - prob) / v
        to_elo = core.ELO_SCALE
        xts = x.T.multiply(s).tocsr()
        gram = (xts @ x).toarray()
        gram[np.diag_indices_from(gram)] += lam
        q = np.linalg.inv(gram)
    else:
        raise ValueError(f"no linear decomposition for model {model!r}")

    m = _mapping(teams, p, to_elo, divisions, major)
    return System(teams=teams, frame=frame, rows=rows, x=x, s=s, z=z, beta=beta, q=q,
                  m=m, to_elo=to_elo)


class Explainer:
    """The decomposition for any core.MODELS entry; a blend combines its two parts."""

    def __init__(self, current, prior, week, *, model="bt", blend=0.5, **fit_kw):
        self.model = model
        if model == "blend":
            self.systems = [
                (blend, build_system(current, prior, week, model="bt", **fit_kw)),
                (1.0 - blend, build_system(current, prior, week, model="margin", **fit_kw)),
            ]
        else:
            self.systems = [(1.0, build_system(current, prior, week, model=model, **fit_kw))]
        self.frame = self.systems[0][1].frame
        self.teams = self.systems[0][1].teams

    def _combine(self, fn) -> pd.Series:
        out = None
        for weight, system in self.systems:
            part = fn(system) * weight
            out = part if out is None else out + part.reindex(out.index)
        return out

    @property
    def ratings(self) -> pd.Series:
        return self._combine(lambda s: s.ratings)

    def parts(self) -> tuple[pd.Series, pd.Series]:
        prior = self._combine(lambda s: s.parts()[0])
        current = self._combine(lambda s: s.parts()[1])
        return prior, current

    def leave_out(self, frame_positions) -> pd.Series:
        return self._combine(lambda s: s.leave_out(frame_positions))


# ── Per-team evidence ──────────────────────────────────────────────────────


def _num(v, digits=1):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, digits)


def _score(row, side: str):
    col = f"{side}_score"
    if col not in row or row[col] is None or pd.isna(row[col]):
        return None
    return int(row[col])


def team_games(frame: pd.DataFrame, positions: np.ndarray, team: str,
               strengths: pd.Series, home_adv: float, points_per_elo: float | None
               ) -> list[dict]:
    """This team's games in `positions` of the fit frame, from its own side."""
    out = []
    sub = frame.iloc[positions]
    mask = (sub["home_team_name"] == team) | (sub["away_team_name"] == team)
    for pos, g in sub[mask].iterrows():
        home = g["home_team_name"] == team
        opp = g["away_team_name"] if home else g["home_team_name"]
        neutral = bool(g["neutral_site"])
        site = "N" if neutral else ("H" if home else "A")
        won = bool(g["home_won"]) if home else not bool(g["home_won"])
        pf, pa = (_score(g, "home"), _score(g, "away")) if home else \
            (_score(g, "away"), _score(g, "home"))
        margin = g.get("margin")
        margin = None if margin is None or pd.isna(margin) else float(margin) * (1 if home else -1)
        venue = 0.0 if neutral else (home_adv if home else -home_adv)
        eta = float(strengths.get(team, 0.0) - strengths.get(opp, 0.0) + venue)
        out.append({
            "pos": int(pos), "opp": opp, "site": site, "won": won,
            "pf": pf, "pa": pa, "margin": margin, "eta": eta,
            "date": str(pd.Timestamp(g["game_date"]).date()) if pd.notna(g.get("game_date")) else None,
            "week": int(g["week"]),
            "exp_win": core.win_prob(eta, 0.0),
            "exp_margin": eta * points_per_elo if points_per_elo else None,
        })
    return out


def _pct(p: float) -> int:
    """Percent rounded half up, matching JavaScript's Math.round in the backend."""
    return int(math.floor(p * 100 + 0.5))


def order_label(p: float) -> str:
    """Plain-language band for a resample share (see ORDER_BANDS).

    Banded on the displayed whole percentage, so the label always agrees with the
    number printed next to it (59.6% prints as 60% and reads "a slight edge").
    """
    shown = _pct(p)
    stronger = max(shown, 100 - shown)
    for floor, label in ORDER_BANDS:
        if stronger >= round(floor * 100):
            return label if shown >= 50 else f"{label} the other way"
    return ORDER_COIN_FLIP


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _signed(v: float, digits: int = 1) -> str:
    return f"{v:+.{digits}f}".replace("-", "−")


def _score_text(e: dict) -> str:
    where = {"H": "vs", "A": "at", "N": "vs"}[e["site"]]
    if e.get("pf") is not None and e.get("pa") is not None:
        return f"{e['pf']}-{e['pa']} {where} {e['opp']}"
    return f"{'W' if e['won'] else 'L'} {where} {e['opp']}"


def _series_text(e: dict) -> str:
    return f"{e['w']}-{e['l']} vs {e['opp']}"


def prior_share(prior_part: float, current_part: float) -> float | None:
    """Fraction of the rating carried over, only where that fraction means something.

    A percentage is only meaningful when both parts pull the same way as the total;
    a team whose last season says +40 and this season says -30 is not "57% last
    season", so it gets no percentage and the summary quotes the two parts instead.
    """
    total = prior_part + current_part
    if abs(total) < 1e-9:
        return None
    if prior_part * total < 0 or current_part * total < 0:
        return None
    return prior_part / total


def summary_line(r: dict, *, sport: str, season: int, board_size: int,
                 neighbours: list[tuple[int, float]]) -> str:
    """One deterministic sentence per team, built only from numbers on its row."""
    rank = r["rank"]
    if r.get("games_played", 0) == 0:
        text = (f"#{rank}: no {season} games yet, so the rating is entirely "
                f"{r['record_season']} results ({r['record']} in {r['record_season']}).")
        return text

    clauses = []
    head = f"#{rank} because: {r['record']}"
    sched = (f"a schedule rated {_ordinal(int(r['sched_rank']))} of {board_size}"
             if r.get("sched_rank") is not None else None)
    if sport == "mlb" and r.get("sor") is not None:
        # The MLB rating is fitted on wins and losses only, so the summary quotes wins
        # against expectation, never run differential the model does not use.
        head += (f", {abs(r['sor']):.1f} wins {'more' if r['sor'] >= 0 else 'fewer'} "
                 f"than an average team would get against {sched or 'this schedule'}")
    elif sport != "mlb" and r.get("avg_margin") is not None:
        head += f", average margin {_signed(r['avg_margin'])}"
        if sched:
            head += f" vs {sched}"
    elif sched:
        head += f" vs {sched}"
    clauses.append(head)

    share = r.get("prior_share")
    prev = season - 1
    if share is not None:
        clauses.append(f"{_pct(share)}% of the rating is carried over from {prev}")
    else:
        clauses.append(f"{prev} games contribute {_signed(r['rating_from_prior'], 0)} "
                       f"and {season} games {_signed(r['rating_from_current'], 0)} "
                       "to the rating")

    best, worst = r.get("best_wins") or [], r.get("worst_losses") or []
    describe = _series_text if sport == "mlb" else _score_text
    best_label, worst_label = (("best series", "worst series") if sport == "mlb"
                               else ("best win", "worst loss"))
    if best:
        b = best[0]
        clauses.append(f"{best_label}: {describe(b)} ({_signed(b['contrib'])} to the rating)")
    if worst:
        w = worst[0]
        clauses.append(f"{worst_label}: {describe(w)} ({_signed(w['contrib'])})")

    text = "; ".join(clauses) + "."
    # How often the resamples keep this team's order against each adjacent team: always
    # the number, then its band. `neighbours` holds (their rank, share with the higher
    # of the two ranked higher).
    orders = []
    for other, p in sorted(neighbours):
        side = "behind" if other < rank else "ahead of"
        tail = " of resamples" if not orders else ""
        orders.append(f"{side} #{other} in {_pct(p)}%{tail} ({order_label(p)})")
    if orders:
        joined = "; ".join(orders)
        text += f" {joined[0].upper()}{joined[1:]}."
    return text


def _index_of(items: list, item) -> int:
    for i, x in enumerate(items):
        if x is item:
            return i
    return items.index(item)


def _series(entries: list[dict], contrib_of) -> list[dict]:
    """Collapse games to one season series per opponent (MLB)."""
    by_opp: dict[str, list[dict]] = {}
    for e in entries:
        by_opp.setdefault(e["opp"], []).append(e)
    out = []
    for opp, games in by_opp.items():
        w = sum(1 for g in games if g["won"])
        runs_for = [g["pf"] for g in games if g["pf"] is not None]
        runs_against = [g["pa"] for g in games if g["pa"] is not None]
        exp_w = sum(g["exp_win"] for g in games)
        out.append({
            "opp": opp, "g": len(games), "w": w, "l": len(games) - w,
            "rf": int(sum(runs_for)) if runs_for else None,
            "ra": int(sum(runs_against)) if runs_against else None,
            "exp_w": _num(exp_w, 2), "over": _num(w - exp_w, 2),
            "contrib": _num(contrib_of(opp, [g["pos"] for g in games]), 2),
        })
    return sorted(out, key=lambda e: -(e["contrib"] or 0))


def remaining_strength(remaining: pd.DataFrame | None, strengths: pd.Series
                       ) -> dict[str, tuple[float, int]]:
    """Team -> (mean opponent rating, games) over games still to play."""
    out: dict[str, list[float]] = {}
    if remaining is None or remaining.empty:
        return {}
    for g in remaining.itertuples(index=False):
        for team, opp in ((g.home_team_name, g.away_team_name),
                          (g.away_team_name, g.home_team_name)):
            if opp in strengths.index:
                out.setdefault(team, []).append(float(strengths[opp]))
    return {t: (float(np.mean(v)), len(v)) for t, v in out.items()}


def order_probability(draws: pd.DataFrame | None, a: str, b: str) -> float | None:
    """Share of bootstrap replicates rating `a` above `b`."""
    if draws is None or a not in draws.columns or b not in draws.columns:
        return None
    both = draws[[a, b]].dropna()
    if both.empty:
        return None
    return float((both[a] > both[b]).mean())


def opponent_summary(entries: list[dict], sport: str) -> dict:
    """Collapse one team's entries against one opponent to a compact record."""
    if sport == "mlb":
        w = sum(e["w"] for e in entries)
        l = sum(e["l"] for e in entries)
        return {"w": w, "l": l, "over": _num(sum(e["over"] or 0 for e in entries), 2),
                "games": [f"{w}-{l}"]}
    w = sum(1 for e in entries if e["won"])
    margins = [e["margin"] for e in entries if e.get("margin") is not None]
    return {
        "w": w, "l": len(entries) - w,
        "margin": _num(sum(margins)) if margins else None,
        "over": _num(sum(e["over"] or 0 for e in entries)),
        "games": [_game_brief(e, sport) for e in entries],
    }


def pair_explanation(a: dict, b: dict, *, sport: str, points_per_elo: float | None,
                     p_order: float | None) -> dict:
    """Why `a` sits above `b`: every field computed from the two rows' own numbers.

    The backend's any-two-teams compare endpoint mirrors this function field for field
    (hanks_tank_backend src/utils/rankings-compare.ts), so an adjacent pair and an
    arbitrary one read identically.
    """
    gap = a["rating"] - b["rating"]
    out = {
        "a": a["team"], "b": b["team"], "a_rank": a["rank"], "b_rank": b["rank"],
        "gap": _num(gap), "gap_points": _num(gap * points_per_elo) if points_per_elo else None,
        "p_a_wins_neutral": _num(core.win_prob(a["rating"], b["rating"]), 3),
        "p_order": _num(p_order, 3),
        "tied": bool(p_order is not None and p_order < TIE_ORDER_P),
        "order_label": order_label(p_order) if p_order is not None else None,
        "gap_from_prior": _num(a["rating_from_prior"] - b["rating_from_prior"]),
        "gap_from_current": _num(a["rating_from_current"] - b["rating_from_current"]),
        "a_band": [a.get("rank_p05"), a.get("rank_p95")],
        "b_band": [b.get("rank_p05"), b.get("rank_p95")],
        "a_sched_rank": a.get("sched_rank"), "b_sched_rank": b.get("sched_rank"),
    }
    by_opp = lambda games: {  # noqa: E731
        o: [g for g in games if g["opp"] == o] for o in dict.fromkeys(g["opp"] for g in games)
    }
    a_opp, b_opp = by_opp(a.get("games") or []), by_opp(b.get("games") or [])
    out["h2h"] = opponent_summary(a_opp[b["team"]], sport) if b["team"] in a_opp else None
    common = [
        {"opp": o, "a": opponent_summary(a_opp[o], sport), "b": opponent_summary(b_opp[o], sport)}
        for o in a_opp if o in b_opp and o not in (a["team"], b["team"])
    ]
    out["common_totals"] = {
        "n": len(common),
        "a_w": sum(c["a"]["w"] for c in common), "a_l": sum(c["a"]["l"] for c in common),
        "b_w": sum(c["b"]["w"] for c in common), "b_l": sum(c["b"]["l"] for c in common),
        "a_over": _num(sum(c["a"]["over"] or 0 for c in common), 2),
        "b_over": _num(sum(c["b"]["over"] or 0 for c in common), 2),
    }
    # Baseball teams share nearly every opponent, so a per-opponent list is 28 rows
    # of noise; the totals carry the comparison there.
    out["common"] = common if len(common) <= MAX_COMMON_LISTED else []
    out["text"] = pair_text(out, sport=sport)
    return out


def _game_brief(g: dict, sport: str) -> str:
    if sport == "mlb":
        return f"{g['w']}-{g['l']}"
    if g.get("pf") is not None and g.get("pa") is not None:
        return f"{'W' if g['won'] else 'L'} {g['pf']}-{g['pa']}"
    return "W" if g["won"] else "L"


def pair_text(p: dict, *, sport: str) -> str:
    """Deterministic sentence for one pair; quotes numbers, never adjectives."""
    unit = (f"{p['gap']:.1f} rating points ({p['gap_points']:.1f} points of expected margin)"
            if p.get("gap_points") is not None else f"{p['gap']:.1f} rating points")
    parts = [f"{p['a']} is {unit} above {p['b']}; "
             f"P({p['a']} wins at a neutral site) = {_pct(p['p_a_wins_neutral'])}%"]
    parts.append(f"{_signed(p['gap_from_prior'])} of the gap comes from last season's games "
                 f"and {_signed(p['gap_from_current'])} from this season's")
    h2h = p.get("h2h")
    if h2h:
        parts.append(f"head to head {p['a']} went {h2h['w']}-{h2h['l']}"
                     + (f" ({', '.join(h2h['games'])})" if sport != "mlb" else ""))
    else:
        parts.append("they have not played each other this season")
    common, totals = p.get("common") or [], p.get("common_totals") or {}
    if common and sport != "mlb" and len(common) <= 3:
        briefs = "; ".join(
            f"{c['opp']}: {p['a']} {', '.join(c['a']['games'])}, {p['b']} {', '.join(c['b']['games'])}"
            for c in common)
        parts.append(f"common opponents: {briefs}")
    elif totals.get("n"):
        parts.append(f"against {totals['n']} common opponents {p['a']} went "
                     f"{totals['a_w']}-{totals['a_l']} and {p['b']} "
                     f"{totals['b_w']}-{totals['b_l']}")
    text = "; ".join(parts) + "."
    if p.get("p_order") is not None:
        text += (f" {p['a']} ranks ahead of {p['b']} in {_pct(p['p_order'])}% of resamples "
                 f"({order_label(p['p_order'])}).")
    return text


# ── Attaching to a board ───────────────────────────────────────────────────

# New board columns. All additive; the JSON ones are STRING in BigQuery.
JSON_COLUMNS = ("why_json", "games_json", "vs_next_json")
TEXT_COLUMNS = ("summary",) + JSON_COLUMNS


def attach(board: pd.DataFrame, *, sport: str, season: int, current, prior,
           week: int, fit_kw: dict, draws: pd.DataFrame | None,
           remaining: pd.DataFrame | None, margin_scale: float | None,
           check_against: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
    """Add the rationale columns to a built board. Returns (board, diagnostics)."""
    exp = Explainer(current, prior, week, **fit_kw)
    ratings = exp.ratings
    home_adv = float(sum(w * s.home_adv for w, s in exp.systems))
    prior_part, current_part = exp.parts()

    diag = {"decomposition_max_abs_error": float(
        (prior_part + current_part - ratings).abs().max())}
    if check_against is not None:
        diag["system_vs_published_max_abs"] = float(
            (ratings - check_against.reindex(ratings.index)).abs().max())

    model = fit_kw.get("model", "bt")
    points_per_elo = (margin_scale / core.ELO_SCALE) if model in ("margin", "blend") \
        and margin_scale else None
    frame = exp.frame
    has_current = bool(frame["is_current"].any())
    evidence = np.flatnonzero(frame["is_current"].to_numpy(dtype=bool)) if has_current \
        else np.arange(len(frame))

    loo_cache: dict = {}

    def contrib(team: str, positions) -> float:
        key = tuple(sorted(int(p) for p in positions))
        if key not in loo_cache:
            loo_cache[key] = exp.leave_out(list(key))
        return float(loo_cache[key].get(team, 0.0))

    remaining_of = remaining_strength(remaining, ratings)

    per_team: dict[str, dict] = {}
    for r in board.itertuples(index=False):
        team = r.team
        games = team_games(frame, evidence, team, ratings, home_adv, points_per_elo)
        if sport == "mlb":
            entries = _series(games, lambda opp, pos, t=team: contrib(t, pos))
            best = [e for e in entries if e["w"] > e["l"] and (e["contrib"] or 0) > 0]
            worst = [e for e in reversed(entries) if e["l"] > e["w"] and (e["contrib"] or 0) < 0]
        else:
            entries = []
            for g in games:
                c = contrib(team, [g["pos"]])
                over = (g["margin"] - g["exp_margin"]) if (
                    g["margin"] is not None and g["exp_margin"] is not None) else \
                    (float(g["won"]) - g["exp_win"])
                entry = {
                    "opp": g["opp"], "date": g["date"], "site": g["site"],
                    "won": g["won"], "pf": g["pf"], "pa": g["pa"], "margin": g["margin"],
                    "over": _num(over), "contrib": _num(c, 2),
                }
                # Margin models expect a margin; the W/L fit expects a win probability.
                if g["exp_margin"] is not None:
                    entry["exp_margin"] = _num(g["exp_margin"])
                else:
                    entry["exp_win"] = _num(g["exp_win"], 3)
                entries.append(entry)
            best = sorted([e for e in entries if e["won"] and (e["contrib"] or 0) > 0],
                          key=lambda e: -e["contrib"])
            worst = sorted([e for e in entries if not e["won"] and (e["contrib"] or 0) < 0],
                           key=lambda e: e["contrib"])
        opp_ratings = [float(ratings[g["opp"]]) for g in games if g["opp"] in ratings.index]
        margins = [g["margin"] for g in games if g["margin"] is not None]
        overs = [(g["margin"] - g["exp_margin"]) for g in games
                 if g["margin"] is not None and g["exp_margin"] is not None]
        rem = remaining_of.get(team)
        per_team[team] = {
            "games": entries,
            "best_wins": best[:TOP_GAMES],
            "worst_losses": worst[:TOP_GAMES],
            "games_played": len(games) if has_current else 0,
            "rating_from_prior": float(prior_part.get(team, np.nan)),
            "rating_from_current": float(current_part.get(team, np.nan)),
            "sched_strength": float(np.mean(opp_ratings)) if opp_ratings else None,
            "avg_margin": float(np.mean(margins)) if margins else None,
            "avg_over_expected": float(np.mean(overs)) if overs else None,
            "sched_remaining": rem[0] if rem else None,
            "sched_remaining_games": rem[1] if rem else 0,
        }

    out = board.copy()
    col = lambda k: out["team"].map(lambda t: per_team[t][k])  # noqa: E731
    out["rating_from_prior"] = col("rating_from_prior").round(1)
    # Rounded so the two displayed parts add up to the displayed rating exactly; the
    # unrounded parts sum to the rating to floating-point precision (see diagnostics).
    out["rating_from_current"] = (out["rating"] - out["rating_from_prior"]).round(1)
    shares = [prior_share(per_team[t]["rating_from_prior"], per_team[t]["rating_from_current"])
              for t in out["team"]]
    out["prior_share"] = pd.Series(shares, index=out.index, dtype="float64").round(3)
    out["games_played"] = col("games_played").astype("Int64")
    out["avg_margin"] = pd.to_numeric(col("avg_margin"), errors="coerce").round(1)
    out["avg_over_expected"] = pd.to_numeric(col("avg_over_expected"), errors="coerce").round(1)
    out["sched_strength"] = pd.to_numeric(col("sched_strength"), errors="coerce").round(1)
    out["sched_remaining"] = pd.to_numeric(col("sched_remaining"), errors="coerce").round(1)
    out["sched_remaining_games"] = col("sched_remaining_games").astype("Int64")
    group = out.groupby("division", dropna=False)
    out["sched_rank"] = group["sched_strength"].rank(ascending=False, method="min").astype("Int64")
    out["sched_remaining_rank"] = (
        group["sched_remaining"].rank(ascending=False, method="min").astype("Int64")
    )

    # Adjacent-pair explanations and the resample shares the summary line quotes.
    records = {}
    for i, r in out.iterrows():
        d = r.to_dict()
        d.update({k: per_team[r["team"]][k] for k in ("games", "best_wins", "worst_losses",
                                                      "games_played")})
        # The displayed (rounded) parts, so a pair's split adds up to the displayed gap
        # and matches what the backend computes from the same stored fields.
        d["rating_from_prior"] = float(r["rating_from_prior"])
        d["rating_from_current"] = float(r["rating_from_current"])
        for key in ("rank_p05", "rank_p95", "sched_rank"):
            d[key] = None if pd.isna(d.get(key)) else int(d[key])
        records[i] = d

    vs_next: dict = {}
    tied_with: dict = {i: [] for i in out.index}
    neighbours: dict = {i: [] for i in out.index}
    for _, grp in out.groupby("division", dropna=False, sort=False):
        idx = list(grp.sort_values("rank").index)
        for i, j in zip(idx, idx[1:]):
            a, b = records[i], records[j]
            p_order = order_probability(draws, a["team"], b["team"])
            pair = pair_explanation(a, b, sport=sport, points_per_elo=points_per_elo,
                                    p_order=p_order)
            vs_next[i] = pair
            if p_order is not None:
                neighbours[i].append((int(b["rank"]), p_order))
                neighbours[j].append((int(a["rank"]), p_order))
            if pair["tied"]:
                tied_with[i].append((int(b["rank"]), p_order))
                tied_with[j].append((int(a["rank"]), p_order))

    board_size = out.groupby("division", dropna=False)["team"].transform("count")
    summaries, why, games_json, next_json = [], [], [], []
    for i, r in out.iterrows():
        d = records[i]
        d["prior_share"] = None if pd.isna(r["prior_share"]) else float(r["prior_share"])
        d["avg_margin"] = None if pd.isna(r["avg_margin"]) else float(r["avg_margin"])
        d["sor"] = None if "sor" not in r or pd.isna(r["sor"]) else float(r["sor"])
        d["rating_from_prior"] = float(r["rating_from_prior"])
        d["rating_from_current"] = float(r["rating_from_current"])
        summaries.append(summary_line(d, sport=sport, season=season,
                                      board_size=int(board_size[i]),
                                      neighbours=neighbours[i]))
        why.append(json.dumps({
            # Indices into games_json, so each game is stored once.
            "best": [_index_of(d["games"], e) for e in d["best_wins"]],
            "worst": [_index_of(d["games"], e) for e in d["worst_losses"]],
            "tied_with": [{"rank": t, "p_order": _num(p, 3)} for t, p in sorted(tied_with[i])],
            "unit": "series" if sport == "mlb" else "game",
            "points_per_rating": _num(points_per_elo, 5) if points_per_elo else None,
            "prior_season": season - 1,
        }))
        games_json.append(json.dumps(d["games"]))
        next_json.append(json.dumps(vs_next[i]) if i in vs_next else None)
    out["summary"] = summaries
    out["why_json"] = why
    out["games_json"] = games_json
    out["vs_next_json"] = next_json
    return out, diag
