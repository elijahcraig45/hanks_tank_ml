"""Resume v2: a season-scale ranking built from stated requirements (docs/RESUME_V2_REPORT.md).

Pieces, each with the reason it exists in the report:

  yardstick   An exact Gaussian posterior over every team's strength, from the same ridge margin model the headline
              uses (a ridge penalty is a Gaussian prior, so the posterior is closed-form). Draws give a distribution
              for every opponent AND for the benchmark ("a typical top-25 team"), so uncertainty about the yardstick
              reaches the score instead of being ignored.
  credits     What one game is worth to a resume, measured against the benchmark. Kinds: `win` (wins above expected),
              `add` (v1's additive margin credit), `addfloor`, and `mult` (surprise x decisiveness), which cannot
              score a win below zero or a loss above it.
  scores      A team's sum of credits, optionally shrunk toward zero as sum / (n + k) so schedules of different length
              compare fairly.
  consensus   The order that minimises  sum over pairs [ P(j above i) + eta * 1(j beat i) ]  where the first term is read
              off the yardstick draws and the second charges a ranking for contradicting a played game. It reverses a
              played result only if the opposite order has probability above (1 + eta) / 2.
  ranges      From the same pairwise probabilities, so a team is always inside its own range.

Nothing here is wired into the pipeline. See `results.attach_boards` for what is published today (v1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import norm

from rankings import core

SIG_ADD = 14.0     # points: the margin at which the additive credit saturates (about two touchdowns)
SIG_MULT = 14.0    # points: the margin at which a result counts as fully decisive in the multiplicative credit
BENCH_N = 25       # the benchmark is the mean of the division's best BENCH_N teams


# ------------------------------------------------------------------------------------------------ the yardstick
@dataclass
class Yardstick:
    teams: list[str]
    index: dict[str, int]
    mean: np.ndarray        # (T,) posterior-mean strength, points
    draws: np.ndarray       # (R, T) posterior draws of strength, points
    home: np.ndarray        # (R,) home-field points per draw
    sigma: float            # game-margin sd, points
    bench: np.ndarray       # (R,) benchmark strength per draw
    bench_mean: float       # benchmark under the posterior mean
    home_mean: float

    @property
    def n_draws(self) -> int:
        return self.draws.shape[0]


def fit_yardstick(current: pd.DataFrame, prior: pd.DataFrame | None, week: int, divisions: dict[str, str],
                  major: str | None, alpha: float = 0.3, w0: float = 0.12, tau: float = 8.0,
                  n_draws: int = 100, seed: int = 0, bench_teams: list[str] | None = None) -> Yardstick:
    """Ridge margin model with this season's games at full weight and last season's at a decayed weight.

    Posterior: beta ~ N(beta_hat, sigma^2 (X'WX + alpha I)^-1), the Gaussian-prior reading of the same ridge fit
    `core.fit_with_prior(model="margin")` solves, so the mean agrees with the headline's up to scale and centering.
    """
    has_prior = prior is not None and not prior.empty
    frames, weights = [], []
    if has_prior:
        w = core.prior_weight(week, w0, tau)
        if w > 0:
            frames.append(prior)
            weights.append(np.full(len(prior), w))
    frames.append(current)
    weights.append(np.ones(len(current)))
    games = pd.concat(frames, ignore_index=True)
    w = np.concatenate(weights)
    teams = sorted(set(games["home_team_name"]) | set(games["away_team_name"]))
    index = {t: i for i, t in enumerate(teams)}
    k = len(teams)

    x, _ = core._design(games, teams, divisions, major)
    y = games["margin"].to_numpy(dtype=float)
    ok = ~np.isnan(y)
    xs = x[ok]
    y, w = y[ok], w[ok]
    xtw = xs.T.multiply(w).tocsr()
    xtwx = (xtw @ xs).toarray()
    gram = xtwx + alpha * np.eye(xtwx.shape[0])
    gram_inv = np.linalg.inv(gram)
    beta = gram_inv @ (xtw @ y)
    resid = y - xs @ beta
    edf = float(np.trace(gram_inv @ xtwx))
    sigma = float(np.sqrt(max((w * resid ** 2).sum() / max(w.sum() - edf, 1.0), 64.0)))   # floor at 8 points

    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky((gram_inv + gram_inv.T) / 2 + 1e-12 * np.eye(gram.shape[0]))
    z = rng.standard_normal((n_draws, gram.shape[0]))
    betas = np.vstack([beta, beta + sigma * z @ chol.T])          # row 0 is the posterior mean

    def strengths(b: np.ndarray) -> np.ndarray:
        s = b[:, :k].copy()
        if major is not None:
            is_major = np.array([divisions.get(t) == major for t in teams], dtype=float)
            s += b[:, [k + 1]] * core.DIV_SCALE * is_major
        return s

    s_all = strengths(betas)
    home_all = betas[:, k]
    # Benchmark: mean of the best BENCH_N among the teams that are on the board, per draw.
    members = bench_teams if bench_teams is not None else [t for t in teams if divisions.get(t) == major]
    cols = [index[t] for t in members if t in index]
    top = np.sort(s_all[:, cols], axis=1)[:, ::-1][:, :BENCH_N].mean(axis=1) if cols else np.zeros(len(betas))
    return Yardstick(teams=teams, index=index, mean=s_all[0], draws=s_all[1:], home=home_all[1:], sigma=sigma,
                     bench=top[1:], bench_mean=float(top[0]), home_mean=float(home_all[0]))


# ------------------------------------------------------------------------------------------------ games as arrays
@dataclass
class Sides:
    """One row per (team, game): who, against whom, where, and what happened. Margins are from the team's side."""
    team: np.ndarray
    opp: np.ndarray
    venue: np.ndarray       # +1 home, -1 away, 0 neutral
    margin: np.ndarray      # NaN where the game has no score
    won: np.ndarray         # bool
    game: np.ndarray        # index of the underlying game row, for bootstrap resampling


def sides_of(current: pd.DataFrame, index: dict[str, int]) -> Sides:
    team, opp, venue, margin, won, game = [], [], [], [], [], []
    has_margin = "margin" in current.columns
    for gi, g in enumerate(current.itertuples(index=False)):
        m = getattr(g, "margin", np.nan) if has_margin else np.nan
        m = np.nan if m is None or (isinstance(m, float) and np.isnan(m)) else float(m)
        neutral = bool(g.neutral_site)
        for t, o, v, mm, w in ((g.home_team_name, g.away_team_name, 0 if neutral else 1, m, bool(g.home_won)),
                               (g.away_team_name, g.home_team_name, 0 if neutral else -1,
                                -m if m == m else np.nan, not bool(g.home_won))):
            if t in index and o in index:
                team.append(index[t]); opp.append(index[o]); venue.append(v); margin.append(mm); won.append(w); game.append(gi)
    return Sides(np.array(team), np.array(opp), np.array(venue, dtype=float), np.array(margin), np.array(won, dtype=bool),
                 np.array(game))


# ------------------------------------------------------------------------------------------------ credits
CREDIT_KINDS = ("win", "add", "addfloor", "mult")


def expected_margin(ys: Yardstick, sides: Sides, use_mean: bool = False) -> np.ndarray:
    """(R, G) expected margin of the benchmark against each side's opponent at that venue; row 0 only if use_mean."""
    if use_mean:
        return (ys.bench_mean + sides.venue * ys.home_mean - ys.mean[sides.opp])[None, :]
    return ys.bench[:, None] + ys.home[:, None] * sides.venue[None, :] - ys.draws[:, sides.opp]


def credits(ys: Yardstick, sides: Sides, kind: str = "mult", lam: float = 0.5, use_mean: bool = False,
            sig: float | None = None) -> np.ndarray:
    """(R, G) credit of every side-game under every yardstick draw (or (1, G) under the posterior mean).

    win       W - p, with p the benchmark's chance to win: v0, wins above expected.
    add       (1 - lam)(W - p) + lam (Phi(M / SIG) - E[Phi]): v1. Can score a narrow win below zero.
    addfloor  `add` with wins floored at 0 and losses capped at 0.
    mult      win: (1 - p) f, loss: -p f, with f = (1 - lam) + lam (2 Phi(|M| / SIG) - 1) in [1 - lam, 1).
              The size of a result is how surprising it was for the benchmark; margin only scales it, within
              [1 - lam, 1). A win is never negative and a loss never positive, by construction.
    A game without a score counts as a plain win or loss (f at its midpoint for `mult`).
    """
    if kind not in CREDIT_KINDS:
        raise ValueError(f"unknown credit kind {kind!r}")
    sig = SIG_MULT if (kind == "mult" and sig is None) else (SIG_ADD if sig is None else sig)
    mu = expected_margin(ys, sides, use_mean)
    s = ys.sigma
    p = norm.cdf(mu / s)
    won = sides.won[None, :].astype(float)
    m = sides.margin[None, :]
    has = ~np.isnan(m)
    if kind == "win":
        return won - p
    if kind in ("add", "addfloor"):
        actual = (1 - lam) * won + lam * np.where(has, norm.cdf(np.nan_to_num(m) / sig), won)
        expected = (1 - lam) * p + lam * np.where(has, norm.cdf(mu / np.sqrt(sig ** 2 + s ** 2)), p)
        c = actual - expected
        if kind == "addfloor":
            c = np.where(sides.won[None, :], np.maximum(c, 0.0), np.minimum(c, 0.0))
        return c
    decisive = np.where(has, 2 * norm.cdf(np.abs(np.nan_to_num(m)) / sig) - 1, 0.5)
    f = (1 - lam) + lam * decisive
    return np.where(sides.won[None, :], (1 - p) * f, -p * f)


# ------------------------------------------------------------------------------------------------ credit variance
_LEG_X, _LEG_W = np.polynomial.legendre.leggauss(24)
_LEG_X, _LEG_W = (_LEG_X + 1) / 2, _LEG_W / 2               # nodes and weights on (0, 1)


def _benchmark_moments(mu: np.ndarray, s: float, credit_of_margin) -> tuple[np.ndarray, np.ndarray]:
    """E and Var, per side-game, of credit_of_margin(M) with M ~ Normal(mu, s).

    The credit jumps at M = 0 (a win versus a loss), so the integral is split there and each half is done by
    Gauss-Legendre in probability space, where the integrand is smooth. A single Gauss-Hermite rule cannot resolve the
    step when it sits far in the tail (against a weak opponent) and overstates the variance several-fold.
    """
    u0 = norm.cdf(-mu / s)                                   # P(benchmark loses)
    m1 = np.zeros_like(mu)
    m2 = np.zeros_like(mu)
    for lo, hi in ((np.zeros_like(u0), u0), (u0, np.ones_like(u0))):
        u = lo[:, None] + (hi - lo)[:, None] * _LEG_X[None, :]
        x = mu[:, None] + s * norm.ppf(np.clip(u, 1e-15, 1 - 1e-15))
        c = credit_of_margin(x)
        w = (hi - lo)[:, None] * _LEG_W[None, :]
        m1 += (c * w).sum(axis=1)
        m2 += (c ** 2 * w).sum(axis=1)
    return m1, np.maximum(m2 - m1 ** 2, 1e-9)


def credit_variance(ys: Yardstick, sides: Sides, kind: str = "mult", lam: float = 0.5, sig: float | None = None) -> np.ndarray:
    """(G,) variance of each side-game's credit if the BENCHMARK played it, at the posterior-mean yardstick.

    The benchmark's margin is Normal(mu, sigma); the credit is the same function of margin that `credits` applies to the
    actual one, so a team's `sum(credit) / sqrt(sum(variance))` is a standardised strength of record: how many standard
    deviations better than a typical top-25 team's season against this schedule.
    """
    sig = SIG_MULT if (kind == "mult" and sig is None) else (SIG_ADD if sig is None else sig)
    mu = expected_margin(ys, sides, use_mean=True)[0]
    s = ys.sigma
    p = norm.cdf(mu / s)[:, None]
    mu_c = mu[:, None]
    exp_add = ((1 - lam) * p + lam * norm.cdf(mu_c / np.sqrt(sig ** 2 + s ** 2)))

    def credit(x):                                           # x: (G, nodes) margins of the benchmark
        win = (x > 0).astype(float)
        if kind == "win":
            return win - p
        if kind in ("add", "addfloor"):
            c = (1 - lam) * win + lam * norm.cdf(x / sig) - exp_add
            return np.where(x > 0, np.maximum(c, 0.0), np.minimum(c, 0.0)) if kind == "addfloor" else c
        if kind == "mult":
            f = (1 - lam) + lam * (2 * norm.cdf(np.abs(x) / sig) - 1)
            return np.where(x > 0, (1 - p) * f, -p * f)
        raise ValueError(f"unknown credit kind {kind!r}")

    return _benchmark_moments(mu, s, credit)[1]


# ------------------------------------------------------------------------------------------------ scores
def team_matrix(sides: Sides, n_teams: int, weights: np.ndarray | None = None) -> csr_matrix:
    """(G, T) 0/1 (or weighted) indicator: sum credits by team with `credits @ team_matrix`."""
    w = np.ones(len(sides.team)) if weights is None else weights
    return csr_matrix((w, (np.arange(len(sides.team)), sides.team)), shape=(len(sides.team), n_teams))


def scores(c: np.ndarray, sides: Sides, n_teams: int, k: float | None = None,
           weights: np.ndarray | None = None) -> np.ndarray:
    """(R, T) team scores: the sum of credits, or sum / (n + k) when k is given (k=0 is the plain mean)."""
    a = team_matrix(sides, n_teams, weights)
    total = np.asarray((a.T @ c.T).T)
    if k is None:
        return total
    n = np.asarray(a.sum(axis=0)).ravel()
    return total / np.maximum(n + k, 1e-9)


# ------------------------------------------------------------------------------------------------ pairwise order
def pairwise(score: np.ndarray, members: np.ndarray) -> np.ndarray:
    """P[i, j] = share of draws in which member i scores above member j (ties count half). Shape (m, m)."""
    s = score[:, members]
    gt = (s[:, :, None] > s[:, None, :]).mean(axis=0)
    eq = (s[:, :, None] == s[:, None, :]).mean(axis=0)
    return gt + 0.5 * eq


def played_matrix(current: pd.DataFrame, index: dict[str, int], members: np.ndarray) -> np.ndarray:
    """H[j, i] = 1 if member j beat member i this season (any number of meetings counts once)."""
    pos = {int(t): r for r, t in enumerate(members)}
    h = np.zeros((len(members), len(members)))
    for g in current.itertuples(index=False):
        w, l = (g.home_team_name, g.away_team_name) if g.home_won else (g.away_team_name, g.home_team_name)
        if w in index and l in index and index[w] in pos and index[l] in pos:
            h[pos[index[w]], pos[index[l]]] = 1.0
    return h


def consensus_order(p: np.ndarray, h: np.ndarray, eta: float, start: np.ndarray, max_passes: int = 50) -> np.ndarray:
    """Order of the members (indices into p) minimising  sum over i above j of  P[j, i] + eta * H[j, i].

    Placing i above j costs the probability that j should be above i, plus eta if j beat i. Each move strictly lowers
    the total, so the search ends. `start` is the initial best-first order (normally by mean score).
    """
    cost = p.T + eta * h.T                        # cost[i, j]: price of putting i above j = P[j above i] + eta * [j beat i]
    order = list(start)
    for _ in range(max_passes):
        moved = False
        for team in list(order):
            cur = order.index(team)
            rest = order[:cur] + order[cur + 1:]
            above_cost = cost[rest, team]          # j above team costs cost[j, team]
            below_cost = cost[team, rest]          # team above j costs cost[team, j]
            csum_a = np.concatenate([[0.0], np.cumsum(above_cost)])
            csum_b = np.concatenate([[0.0], np.cumsum(below_cost)])
            slot_cost = csum_a + (csum_b[-1] - csum_b)         # cost of inserting before rest[k], k = 0..len(rest)
            best = slot_cost.min()
            if best < slot_cost[cur] - 1e-12:
                cands = np.flatnonzero(slot_cost <= best + 1e-12)
                slot = int(cands[np.argmin(np.abs(cands - cur))])
                order = rest[:slot] + [team] + rest[slot:]
                moved = True
        if not moved:
            break
    return np.array(order)


def ranges(p: np.ndarray, order: np.ndarray, q: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """(best, worst) rank (1-based) each member could reasonably hold, by member position in `order`.

    A team above could be passed if P(this team above it) is at least q; a team below could pass this one if
    P(it above this team) is at least q. The displayed rank is always inside its own range.
    """
    m = len(order)
    lo = np.zeros(m, dtype=int)
    hi = np.zeros(m, dtype=int)
    for r, i in enumerate(order):
        best, worst = r, r
        for r2 in range(r - 1, -1, -1):
            if p[i, order[r2]] >= q:
                best = r2
        for r2 in range(r + 1, m):
            if p[order[r2], i] >= q:
                worst = r2
        lo[r], hi[r] = best + 1, worst + 1
    return lo, hi
