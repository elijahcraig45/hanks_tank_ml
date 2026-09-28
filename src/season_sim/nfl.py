"""NFL season structure: divisions, standings with tiebreakers, and the 14-team playoff.

Tiebreakers implemented (NFL.com "Tie-breaking procedures", checked 2026-09-28):

  Division (2 clubs, and 3+ clubs):
    1 head-to-head (win pct in games among the tied clubs)
    2 division record              3 common games
    4 conference record            5 strength of victory     6 strength of schedule
    9 net points, conference games 10 net points, all games  12 coin toss
  Wild card (2 clubs): head-to-head if they met, conference record, common games (min 4),
    SOV, SOS, net points conference, net points all, coin toss.
  Wild card (3+ clubs): first keep only the highest-ranked club of each division (by the
    division procedure), then head-to-head SWEEP only, conference record, common games
    (min 4), SOV, SOS, net points conference, net points all, coin toss.

  NOT implemented: steps 7-8 (combined ranking in points scored and allowed, conference
  and league) and 11 (net touchdowns). We simulate margins, not the two scores, so those
  are skipped; the sequence goes straight to net points and then the coin. These steps
  decide a tie a few times a decade.

  Whenever a step eliminates at least one club and two or more remain, the survivors
  restart at step 1 (the "revert" rule), and after a club is placed the remaining tied
  clubs restart too. Ties in the standings count as half a win.

Playoff format (unchanged for 2026; the Lions' 2025 reseeding proposal was withdrawn):
  seeds 1-4 are the division winners ranked by record (wild-card tiebreakers), 5-7 the
  wild cards. #1 has a bye. Wild card: 2v7, 3v6, 4v5 at the higher seed. Divisional:
  reseeded, #1 hosts the lowest surviving seed. Conference title at the higher seed.
  Super Bowl at a neutral site.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import engine as eng

# Stable since the 2002 realignment. nflverse keeps the code a franchise used at the
# time, so relocations are collapsed to today's code (as src/rankings/sources.py does).
FRANCHISE_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LV", "LAC"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LA", "SF", "SEA"],
}
TEAMS = [t for d in DIVISIONS.values() for t in d]
TEAM_NAMES = {
    "BUF": "Buffalo Bills", "MIA": "Miami Dolphins", "NE": "New England Patriots",
    "NYJ": "New York Jets", "BAL": "Baltimore Ravens", "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns", "PIT": "Pittsburgh Steelers", "HOU": "Houston Texans",
    "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "TEN": "Tennessee Titans",
    "DEN": "Denver Broncos", "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers", "DAL": "Dallas Cowboys", "NYG": "New York Giants",
    "PHI": "Philadelphia Eagles", "WAS": "Washington Commanders", "CHI": "Chicago Bears",
    "DET": "Detroit Lions", "GB": "Green Bay Packers", "MIN": "Minnesota Vikings",
    "ATL": "Atlanta Falcons", "CAR": "Carolina Panthers", "NO": "New Orleans Saints",
    "TB": "Tampa Bay Buccaneers", "ARI": "Arizona Cardinals", "LA": "Los Angeles Rams",
    "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks",
}
DIV_OF = {t: d for d, ts in DIVISIONS.items() for t in ts}
CONF_OF = {t: d.split()[0] for t, d in DIV_OF.items()}
CONFS = ("AFC", "NFC")
PLAYOFF_TYPES = ("WC", "DIV", "CON", "SB")
EPS = 1e-9


def normalise_schedule(sched: pd.DataFrame) -> pd.DataFrame:
    g = sched.copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(FRANCHISE_ALIASES)
    g["margin"] = pd.to_numeric(g["result"], errors="coerce").astype("float64")
    g["neutral"] = (g["location"].fillna("Home").astype(str) != "Home").astype(float)
    g["market_margin"] = pd.to_numeric(g.get("spread_line"), errors="coerce")
    if "gameday" in g.columns:  # nflverse's local (stadium) date, already YYYY-MM-DD
        g["game_day"] = pd.to_datetime(g["gameday"], errors="coerce").dt.strftime("%Y-%m-%d")
    return g


def build_frame(sched: pd.DataFrame, season: int, as_of_week: int | None = None
                ) -> pd.DataFrame:
    """History (two prior seasons, all game types) + this season's regular season.

    With `as_of_week`, results after that week are hidden (the backtest); otherwise every
    unplayed regular-season game is simulated.
    """
    g = normalise_schedule(sched)
    hist = g[(g["season"] >= season - 2) & (g["season"] < season) & g["margin"].notna()]
    cur = g[(g["season"] == season) & (g["game_type"] == "REG")].copy()
    if as_of_week is not None:
        cur.loc[cur["week"] > as_of_week, "margin"] = np.nan
    cur["sim"] = cur["margin"].isna()
    hist = hist.assign(sim=False)
    return eng.ridge_frame_from(pd.concat([hist, cur], ignore_index=True))


class NflSeason:
    """Fixed schedule structure for one season: who plays whom, which games count where."""

    def __init__(self, frame: pd.DataFrame, season: int):
        self.season = season
        rows = np.flatnonzero((frame["season"].to_numpy() == season))
        self.rows = rows
        f = frame.iloc[rows]
        self.idx = {t: i for i, t in enumerate(TEAMS)}
        self.h = np.array([self.idx[t] for t in f["home_team"]])
        self.a = np.array([self.idx[t] for t in f["away_team"]])
        T, G = len(TEAMS), len(rows)
        self.T, self.G = T, G
        self.Hinc = np.zeros((G, T))
        self.Ainc = np.zeros((G, T))
        self.Hinc[np.arange(G), self.h] = 1
        self.Ainc[np.arange(G), self.a] = 1
        conf = np.array([CONF_OF[t] for t in TEAMS])
        div = np.array([DIV_OF[t] for t in TEAMS])
        self.conf, self.div = conf, div
        self.is_div = div[self.h] == div[self.a]
        self.is_conf = conf[self.h] == conf[self.a]
        self.games = self.Hinc.sum(0) + self.Ainc.sum(0)
        self.div_games = (self.Hinc[self.is_div] + self.Ainc[self.is_div]).sum(0)
        self.conf_games = (self.Hinc[self.is_conf] + self.Ainc[self.is_conf]).sum(0)
        self.Mcnt = np.zeros((T, T))
        np.add.at(self.Mcnt, (self.h, self.a), 1)
        np.add.at(self.Mcnt, (self.a, self.h), 1)
        self.opps = [set(np.flatnonzero(self.Mcnt[i] > 0).tolist()) for i in range(T)]
        self.Ml = self.Mcnt.tolist()
        self.pair = self.h * T + self.a
        self.pair_rev = self.a * T + self.h
        self.week = f["week"].to_numpy()

    # ------------------------------------------------------------------ standings
    def standings(self, M: np.ndarray, rng: np.random.Generator, n_wc: int = 3) -> dict:
        """Seeds per path from a (S, G) matrix of home margins for this season's games."""
        S, T = M.shape[0], self.T
        hw = (M > 0) + 0.5 * (M == 0)
        aw = 1.0 - hw
        wins = hw @ self.Hinc + aw @ self.Ainc
        ties = (M == 0).astype(float) @ (self.Hinc + self.Ainc)
        pct = wins / self.games
        dv = self.is_div
        cf = self.is_conf
        div_pct = (hw[:, dv] @ self.Hinc[dv] + aw[:, dv] @ self.Ainc[dv]) / np.maximum(self.div_games, 1)
        conf_pct = (hw[:, cf] @ self.Hinc[cf] + aw[:, cf] @ self.Ainc[cf]) / np.maximum(self.conf_games, 1)
        net_all = M @ self.Hinc - M @ self.Ainc
        net_conf = M[:, cf] @ self.Hinc[cf] - M[:, cf] @ self.Ainc[cf]
        h2h = np.zeros((S, T * T))
        for k in range(self.G):  # G columns, each a vector op over paths
            h2h[:, self.pair[k]] += hw[:, k]
            h2h[:, self.pair_rev[k]] += aw[:, k]
        h2h = h2h.reshape(S, T, T)
        opp_games = self.Mcnt @ self.games
        sos = (wins @ self.Mcnt.T) / opp_games
        beaten_g = np.einsum("sij,j->si", h2h, self.games)
        sov = np.where(beaten_g > 0, np.einsum("sij,sj->si", h2h, wins) / np.maximum(beaten_g, 1e-9), 0.0)

        seeds = np.zeros((S, 2, 4 + n_wc), dtype=int)
        div_win = np.zeros((S, T), dtype=bool)
        conf_teams = {c: [i for i in range(T) if self.conf[i] == c] for c in CONFS}
        div_teams = {}
        for i in range(T):
            div_teams.setdefault(self.div[i], []).append(i)
        for s in range(S):
            ctx = _Ctx(self, pct[s].tolist(), div_pct[s].tolist(), conf_pct[s].tolist(),
                       sov[s].tolist(), sos[s].tolist(), net_conf[s].tolist(),
                       net_all[s].tolist(), h2h[s].tolist(), rng)
            for ci, c in enumerate(CONFS):
                winners = []
                for d, members in div_teams.items():
                    if not d.startswith(c):
                        continue
                    winners.append(_top_k(ctx, members, ctx.div_top, 1)[0])
                for i in winners:
                    div_win[s, i] = True
                top4 = _top_k(ctx, winners, ctx.wc_top, 4)
                rest = [i for i in conf_teams[c] if i not in winners]
                wc = _top_k(ctx, rest, ctx.wc_top, n_wc)
                seeds[s, ci, :] = top4 + wc
        return {"wins": wins, "ties": ties, "pct": pct, "seeds": seeds, "div_win": div_win}

    # ------------------------------------------------------------------ playoffs
    def playoffs(self, seeds: np.ndarray, res: "eng.SimResult", rng) -> dict:
        """Simulate WC -> DIV (reseeded) -> CON -> SB. Returns per-round participants."""
        S = seeds.shape[0]
        col = res.ops.team_index(TEAMS)
        rows = np.arange(S)
        noise = res.noise_sigma
        out = {"wc": {}, "div": {}, "con": {}, "sb": None}

        def play(home, away, neutral):
            mean = res.game_mean(col[home], col[away], np.full(home.shape, float(neutral)))
            m = eng.draw_margin(mean, noise, rng)
            return m > 0

        champs = []
        n = seeds.shape[2]
        nb = 8 - n  # byes: 1 with 7 seeds (2020-), 2 with 6 seeds (1990-2019)
        for ci in range(2):
            sd = seeds[:, ci, :]  # team index at seed k (0-based)
            wc_w = []
            wc_games = []
            for hs in range(nb + 1, 5):
                as_ = n + nb + 1 - hs
                ht, at = sd[:, hs - 1], sd[:, as_ - 1]
                hw = play(ht, at, 0)
                wseed = np.where(hw, hs, as_)
                wc_w.append(wseed)
                wc_games.append((ht, at, np.where(hw, ht, at)))
            surv = np.sort(np.stack([np.full(S, k + 1) for k in range(nb)] + wc_w, 1), axis=1)
            # Divisional, reseeded: #1 hosts the lowest surviving seed; the other two meet.
            d1h, d1a = surv[:, 0], surv[:, 3]
            d2h, d2a = surv[:, 1], surv[:, 2]
            t = lambda seedno: sd[rows, seedno - 1]  # noqa: E731
            g1 = play(t(d1h), t(d1a), 0)
            g2 = play(t(d2h), t(d2a), 0)
            s1 = np.where(g1, d1h, d1a)
            s2 = np.where(g2, d2h, d2a)
            ch, ca = np.minimum(s1, s2), np.maximum(s1, s2)
            gc = play(t(ch), t(ca), 0)
            cs = np.where(gc, ch, ca)
            champs.append(t(cs))
            out["wc"][ci] = wc_games
            out["div"][ci] = [(t(d1h), t(d1a), np.where(g1, t(d1h), t(d1a))),
                              (t(d2h), t(d2a), np.where(g2, t(d2h), t(d2a)))]
            out["con"][ci] = (t(ch), t(ca), t(cs))
        sbw = play(champs[0], champs[1], 1)
        out["sb"] = (champs[0], champs[1], np.where(sbw, champs[0], champs[1]))
        return out


class _Ctx:
    """One path's standings numbers, as Python lists for fast scalar access."""

    __slots__ = ("s", "pct", "divp", "confp", "sov", "sos", "netc", "neta", "h2h", "rng")

    def __init__(self, s, pct, divp, confp, sov, sos, netc, neta, h2h, rng):
        self.s, self.pct, self.divp, self.confp = s, pct, divp, confp
        self.sov, self.sos, self.netc, self.neta, self.h2h, self.rng = sov, sos, netc, neta, h2h, rng

    # -- metrics over a group; None = step not applicable
    def _h2h_pct(self, group):
        M = self.s.Ml
        vals = {}
        for i in group:
            g = sum(M[i][j] for j in group if j != i)
            if g == 0:
                return None
            vals[i] = sum(self.h2h[i][j] for j in group if j != i) / g
        return vals

    def _h2h_if_met(self, group):
        i, j = group
        return self._h2h_pct(group) if self.s.Ml[i][j] > 0 else None

    def _sweep(self, group):
        M = self.s.Ml
        for i in group:
            others = [j for j in group if j != i]
            if all(M[i][j] > 0 and self.h2h[i][j] >= M[i][j] - EPS for j in others):
                return {k: (1.0 if k == i else 0.0) for k in group}
        for i in group:
            others = [j for j in group if j != i]
            if all(M[i][j] > 0 and self.h2h[i][j] <= EPS for j in others):
                return {k: (0.0 if k == i else 1.0) for k in group}
        return None

    def _common(self, group, minimum=0):
        common = set.intersection(*(self.s.opps[i] for i in group)) - set(group)
        if not common:
            return None
        M = self.s.Ml
        vals = {}
        for i in group:
            g = sum(M[i][c] for c in common)
            if g < minimum or g == 0:
                return None
            vals[i] = sum(self.h2h[i][c] for c in common) / g
        return vals

    def _of(self, arr):
        return lambda group: {i: arr[i] for i in group}

    def _run(self, group, steps):
        while len(group) > 1:
            for step in steps:
                vals = step(group)
                if vals is None:
                    continue
                m = max(vals.values())
                keep = [i for i in group if vals[i] >= m - EPS]
                if len(keep) < len(group):
                    group = keep
                    break
            else:  # nothing separated them: coin toss
                return group[int(self.rng.integers(len(group)))]
        return group[0]

    def div_top(self, group):
        steps = [self._h2h_pct, self._of(self.divp), self._common, self._of(self.confp),
                 self._of(self.sov), self._of(self.sos), self._of(self.netc), self._of(self.neta)]
        return self._run(list(group), steps)

    def wc_top(self, group):
        group = list(group)
        while True:
            by_div: dict[str, list[int]] = {}
            for i in group:
                by_div.setdefault(self.s.div[i], []).append(i)
            if any(len(v) > 1 for v in by_div.values()):
                group = [self.div_top(v) if len(v) > 1 else v[0] for v in by_div.values()]
            if len(group) == 1:
                return group[0]
            first = self._h2h_if_met if len(group) == 2 else self._sweep
            steps = [first, self._of(self.confp), lambda g: self._common(g, 4),
                     self._of(self.sov), self._of(self.sos), self._of(self.netc),
                     self._of(self.neta)]
            before = len(group)
            for step in steps:
                vals = step(group)
                if vals is None:
                    continue
                m = max(vals.values())
                keep = [i for i in group if vals[i] >= m - EPS]
                if len(keep) < len(group):
                    group = keep
                    break
            if len(group) == 1:
                return group[0]
            if len(group) == before:
                return group[int(self.rng.integers(len(group)))]


def _top_k(ctx: _Ctx, teams: list[int], top_fn, k: int) -> list[int]:
    """The first k clubs of `teams` by record, ties broken by `top_fn` one club at a time."""
    remaining = list(teams)
    order = []
    while remaining and len(order) < k:
        best = max(ctx.pct[i] for i in remaining)
        tied = [i for i in remaining if ctx.pct[i] >= best - EPS]
        pick = tied[0] if len(tied) == 1 else top_fn(tied)
        order.append(pick)
        remaining.remove(pick)
    return order
