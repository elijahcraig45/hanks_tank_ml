"""Run a rest-of-season simulation and shape it into the two output tables.

    sim_nfl(sched, season, ...)            -> Outcome (arrays; used by the backtest)
    sim_cfb(games, season, ...)            -> Outcome
    tables(outcome, computed_at)           -> (team_df, bracket_df), the BigQuery rows
    games_table(outcome, computed_at)      -> games_df, one row per remaining game

The production default is the variant chosen by the backtest (DEFAULT_VARIANT below; the
evidence is in docs/EXPERIMENT_LOG.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import cfb as cfbm
from . import engine as eng
from . import nfl as nflm

DEFAULT_VARIANT = "draw"
DEFAULT_SIMS = 10_000


@dataclass
class Outcome:
    sport: str
    season: int
    as_of_week: int
    n_sims: int
    variant: str
    teams: list[str]                 # sport team keys, in index order
    team_name: dict[str, str]
    team_abbr: dict[str, str]
    conference: dict[str, str]
    division: dict[str, str | None]
    current: pd.DataFrame            # per team: wins, losses, ties, conf_wins, conf_losses
    rating: np.ndarray               # (T,) as-of rating
    rating_sd: np.ndarray
    remaining: np.ndarray            # (T,) remaining games
    remaining_sos: np.ndarray
    wins: np.ndarray                 # (S, T) final regular-season wins
    n_games: np.ndarray              # (T,)
    seeds: np.ndarray                # (S, [2,] n_seeds) team idx by seed
    rounds: dict                     # bracket games: {(bracket, round): [(h, a, w), ...]}
    final_rank: np.ndarray           # (S, T) end-of-season rank (1 = best)
    flags: dict[str, np.ndarray] = field(default_factory=dict)   # (S, T) booleans
    runtime: dict = field(default_factory=dict)
    # Remaining (unplayed) regular-season games involving a simulated team, in schedule
    # order, and their simulated home margins: the SAME draws the standings used, so
    # per-game win probabilities add up to each team's expected remaining wins.
    games: pd.DataFrame | None = None           # game_id, week, game_day, home, away, ...
    game_margins: np.ndarray | None = None      # (S, n_games)
    tie_half: bool = False                      # NFL: a 0 margin is half a win to each side


def _current_record(M_known: np.ndarray, known: np.ndarray, H, A, conf_mask=None):
    m = np.where(known, M_known, np.nan)
    hw = np.nan_to_num((m > 0).astype(float)) * known
    hl = np.nan_to_num((m < 0).astype(float)) * known
    ht = np.nan_to_num((m == 0).astype(float)) * known
    if conf_mask is not None:
        hw, hl, ht = hw * conf_mask, hl * conf_mask, ht * conf_mask
    wins = hw @ H + hl @ A
    losses = hl @ H + hw @ A
    ties = ht @ (H + A)
    return wins, losses, ties


def _remaining_games(frame: pd.DataFrame, rows: np.ndarray, home_key, away_key,
                     home_name, away_name) -> pd.DataFrame:
    """Schedule rows for `rows` (frame positions), in date/week order is left to the
    caller; team keys are the same keys the team table uses."""
    f = frame.iloc[rows]
    day = f["game_day"].astype(str) if "game_day" in f.columns else pd.Series([None] * len(f))
    return pd.DataFrame({
        "game_id": (f["game_id"].astype(str).to_numpy() if "game_id" in f.columns
                    else np.array([f"row{r}" for r in rows])),
        "week": f["week"].to_numpy(int),
        "game_day": [None if d in ("nan", "NaT", "None") else d for d in day],
        "home": list(home_key), "away": list(away_key),
        "home_name": list(home_name), "away_name": list(away_name),
        "neutral": f["neutral"].to_numpy(float) > 0,
    })


def _order(games: pd.DataFrame) -> np.ndarray:
    """Schedule order: week, then date, then game id (a stable tiebreak)."""
    key = games.assign(_d=games["game_day"].fillna("9999"))
    return np.lexsort((key["game_id"].to_numpy(), key["_d"].to_numpy(),
                       key["week"].to_numpy()))


# ---------------------------------------------------------------------------- NFL
def sim_nfl(sched: pd.DataFrame, season: int, as_of_week: int | None = None,
            n_sims: int = DEFAULT_SIMS, variant: str = DEFAULT_VARIANT, seed: int = 0,
            cfg=None) -> Outcome:
    import time

    t0 = time.time()
    rng = np.random.default_rng(seed)
    frame = nflm.build_frame(sched, season, as_of_week)
    ops = eng.RidgeOps(frame, cfg or eng.mr.NFL_RIDGE)
    res = eng.simulate(ops, season, eng.VARIANTS[variant], n_sims, rng)
    t1 = time.time()
    ns = nflm.NflSeason(frame, season)
    M = res.margins_for(ns.rows)
    n_wc = 3 if season >= 2020 else 2  # the 14-team format began in 2020
    st = ns.standings(M, rng, n_wc=n_wc)
    t2 = time.time()
    po = ns.playoffs(st["seeds"], res, rng)
    col = ops.team_index(nflm.TEAMS)
    T = ns.T
    S = n_sims

    known = ~ops.sim[ns.rows]
    mk = ops.margin[ns.rows]
    w, l, t = _current_record(mk, known, ns.Hinc, ns.Ainc)
    cw, cl, _ = _current_record(mk, known, ns.Hinc, ns.Ainc, conf_mask=ns.is_conf)
    current = pd.DataFrame({"wins": w, "losses": l, "ties": t, "conf_wins": cw,
                            "conf_losses": cl}, index=nflm.TEAMS)
    rem = (~known).astype(float) @ (ns.Hinc + ns.Ainc)
    b0 = res.b0[col]
    opp_sum = (~known * b0[ns.a]) @ ns.Hinc + (~known * b0[ns.h]) @ ns.Ainc
    sos = np.where(rem > 0, opp_sum / np.maximum(rem, 1), np.nan)

    flags = {}
    seeds = st["seeds"]
    rows = np.arange(S)[:, None]

    def mark(idx):
        z = np.zeros((S, T), bool)
        z[np.arange(S), idx] = True
        return z

    in_po = np.zeros((S, T), bool)
    in_po[rows, seeds.reshape(S, -1)] = True
    flags["playoffs"] = in_po
    flags["division"] = st["div_win"]
    flags["bye"] = np.zeros((S, T), bool)
    for k in range(1 if n_wc == 3 else 2):
        flags["bye"] |= mark(seeds[:, 0, k]) | mark(seeds[:, 1, k])
    rounds = {}
    quarters = flags["bye"].copy()
    semis = np.zeros((S, T), bool)
    for ci, c in enumerate(nflm.CONFS):
        rounds[(c, "wild_card")] = po["wc"][ci]
        rounds[(c, "divisional")] = po["div"][ci]
        rounds[(c, "conference")] = [po["con"][ci]]
        for h, a, wv in po["wc"][ci]:
            quarters |= mark(wv)
        h, a, wv = po["con"][ci]
        semis |= mark(h) | mark(a)
    rounds[("NFL", "super_bowl")] = [po["sb"]]
    flags["quarters"] = quarters
    flags["conf_game"] = semis
    flags["semis"] = semis
    h, a, wv = po["sb"]
    flags["final"] = mark(h) | mark(a)
    flags["conf_title"] = flags["final"]
    flags["champion"] = mark(wv)

    end = res.B_end[:, col]
    final_rank = (-end).argsort(1).argsort(1) + 1
    rem_pos = np.flatnonzero(~known)
    fr = frame.iloc[ns.rows[rem_pos]]
    rg = _remaining_games(frame, ns.rows[rem_pos], fr["home_team"], fr["away_team"],
                          [nflm.TEAM_NAMES.get(t, t) for t in fr["home_team"]],
                          [nflm.TEAM_NAMES.get(t, t) for t in fr["away_team"]])
    order = _order(rg)
    t3 = time.time()
    return Outcome(
        sport="nfl", season=season,
        as_of_week=int(as_of_week if as_of_week is not None else _nfl_as_of(frame, season)),
        n_sims=S, variant=variant, teams=list(nflm.TEAMS),
        team_name=dict(nflm.TEAM_NAMES), team_abbr={t: t for t in nflm.TEAMS},
        conference=dict(nflm.CONF_OF), division=dict(nflm.DIV_OF), current=current,
        rating=b0, rating_sd=res.sd0[col], remaining=rem, remaining_sos=sos,
        wins=st["wins"], n_games=ns.games, seeds=seeds, rounds=rounds,
        final_rank=final_rank, flags=flags,
        runtime={"sim_s": t1 - t0, "standings_s": t2 - t1, "playoffs_s": t3 - t2,
                 "total_s": t3 - t0, "noise_sigma": res.noise_sigma},
        games=rg.iloc[order].reset_index(drop=True), game_margins=M[:, rem_pos[order]],
        tie_half=True,
    )


def _nfl_as_of(frame: pd.DataFrame, season: int) -> int:
    cur = (frame["season"] == season) & ~frame["sim"]
    return int(frame.loc[cur, "week"].max()) if cur.any() else 0


# ---------------------------------------------------------------------------- CFB
def sim_cfb(games: pd.DataFrame, season: int, as_of_week: int | None = None,
            n_sims: int = DEFAULT_SIMS, variant: str = DEFAULT_VARIANT, seed: int = 0,
            cfg=None, loss_weight: float | None = None, champ_bonus: float | None = None
            ) -> Outcome:
    import time

    t0 = time.time()
    rng = np.random.default_rng(seed)
    frame = cfbm.build_frame(games, season, as_of_week)
    ops = eng.RidgeOps(frame, cfg or eng.mr.CFB_RIDGE)
    res = eng.simulate(ops, season, eng.VARIANTS[variant], n_sims, rng)
    t1 = time.time()
    cs = cfbm.CfbSeason(frame, season, games)
    M = res.margins_for(cs.rows)
    col = ops.team_index(cs.fbs)
    S, T = n_sims, cs.T
    wins, losses, cw = cs.records(M)
    end_rating = res.B_end[:, col]
    tg = cs.title_games(M, cw, end_rating, rng)
    t2 = time.time()

    champ = np.zeros((S, T), bool)
    in_ccg = np.zeros((S, T), bool)
    champ_of = {}
    losses_all = losses.copy()
    wins_all = wins.copy()
    rows = np.arange(S)
    played_ccg = cfbm.played_title_games(games, season, cs, as_of_week)
    for c, pair in tg.items():
        if c in played_ccg:  # already played: the real pairing and the real result
            h0, a0, home_won = played_ccg[c]
            pair = np.tile([h0, a0], (S, 1))
            h, a = pair[:, 0], pair[:, 1]
            hw = np.full(S, bool(home_won))
        else:
            h, a = pair[:, 0], pair[:, 1]
            neutral = 0.0 if c in cfbm.HOSTED_TITLE_GAME else 1.0
            mean = res.game_mean(col[h], col[a], np.full(S, neutral), np.zeros((S, 1)))
            hw = eng.draw_margin(mean, res.noise_sigma, rng) > 0
        wv = np.where(hw, h, a)
        lv = np.where(hw, a, h)
        champ[rows, wv] = True
        in_ccg[rows, h] = True
        in_ccg[rows, a] = True
        wins_all[rows, wv] += 1
        losses_all[rows, lv] += 1
        champ_of[c] = wv
    conf_arr = np.array([cs.conf_of[t] for t in cs.fbs])
    g6 = cfbm.group6(conf_arr, season)[None, :]
    score = cfbm.committee_score(end_rating, losses_all, champ.astype(float), g6,
                                 loss_weight, champ_bonus)
    final_rank = (-score).argsort(1).argsort(1) + 1
    seeds = cfbm.select_field(score, champ_of, conf_arr, season)
    po = cfbm.play_cfp(seeds, res, col, rng)
    t3 = time.time()

    def mark(idx):
        z = np.zeros((S, T), bool)
        z[rows, idx] = True
        return z

    flags = {"conf_game": in_ccg, "conf_title": champ}
    in_po = np.zeros((S, T), bool)
    in_po[rows[:, None], seeds] = True
    flags["playoffs"] = in_po
    nb = 4 if seeds.shape[1] == 12 else 0
    bye = np.zeros((S, T), bool)
    if nb:
        bye[rows[:, None], seeds[:, :4]] = True
    flags["bye"] = bye
    rounds = {}
    quarters = bye.copy()
    for rname, games_ in po.items():
        rounds[("CFP", rname)] = games_
        for h, a, wv in games_:
            if rname == "first_round":
                quarters |= mark(wv)
    semis = np.zeros((S, T), bool)
    for h, a, _ in po["semifinal"]:
        semis |= mark(h) | mark(a)
    flags["quarters"] = quarters if nb else semis
    flags["semis"] = semis
    h, a, wv = po["final"][0]
    flags["final"] = mark(h) | mark(a)
    flags["champion"] = mark(wv)

    known = ~ops.sim[cs.rows]
    mk = ops.margin[cs.rows]
    w, l, t = _current_record(mk, known, cs.Hinc, cs.Ainc)
    confm = np.zeros(len(cs.rows))
    confm[cs.conf_rows] = 1
    cwn, cln, _ = _current_record(mk, known, cs.Hinc, cs.Ainc, conf_mask=confm)
    current = pd.DataFrame({"wins": w, "losses": l, "ties": t, "conf_wins": cwn,
                            "conf_losses": cln}, index=cs.fbs)
    rem = (~known).astype(float) @ (cs.Hinc + cs.Ainc)
    b_all = res.b0
    opp_h = b_all[ops.team_index(frame.iloc[cs.rows]["away_team"])]
    opp_a = b_all[ops.team_index(frame.iloc[cs.rows]["home_team"])]
    opp_sum = (~known * opp_h) @ cs.Hinc + (~known * opp_a) @ cs.Ainc
    sos = np.where(rem > 0, opp_sum / np.maximum(rem, 1), np.nan)
    div_of = {}
    for c, divs in cs.divisions.items():
        for d, members in divs.items():
            for i in members:
                div_of[cs.fbs[i]] = d
    as_of = as_of_week if as_of_week is not None else _nfl_as_of(frame, season)
    # Remaining games that touch an FBS team (unplayed FCS-vs-FCS games are not in the
    # frame). FCS opponents keep their ESPN abbreviation and name.
    rem_pos = np.flatnonzero(~known & ((cs.hi >= 0) | (cs.ai >= 0)))
    fr = frame.iloc[cs.rows[rem_pos]]
    rg = _remaining_games(frame, cs.rows[rem_pos], fr["home_abbr"], fr["away_abbr"],
                          fr["home_team"], fr["away_team"])
    order = _order(rg)
    return Outcome(
        sport="cfb", season=season, as_of_week=int(as_of), n_sims=S, variant=variant,
        teams=list(cs.fbs), team_name={t: t for t in cs.fbs},
        team_abbr={t: cs.abbr.get(t, t) for t in cs.fbs},
        conference={t: cfbm.CONF_NAMES.get(cs.conf_of[t], cs.conf_of[t]) for t in cs.fbs},
        division={t: div_of.get(t) for t in cs.fbs}, current=current,
        rating=res.b0[col], rating_sd=res.sd0[col], remaining=rem, remaining_sos=sos,
        wins=wins, n_games=cs.games, seeds=seeds, rounds=rounds, final_rank=final_rank,
        flags=flags,
        runtime={"sim_s": t1 - t0, "standings_s": t2 - t1, "postseason_s": t3 - t2,
                 "total_s": t3 - t0, "noise_sigma": res.noise_sigma},
        games=rg.iloc[order].reset_index(drop=True), game_margins=M[:, rem_pos[order]],
        tie_half=False,
    )


# ---------------------------------------------------------------------------- tables
ROUND_ORDER = {
    "nfl": {"wild_card": 1, "divisional": 2, "conference": 3, "super_bowl": 4},
    "cfb": {"first_round": 1, "quarterfinal": 2, "semifinal": 3, "final": 4},
}


def _slot_labels(sport: str, rnd: str, k: int, four: bool = False) -> str:
    if sport == "nfl":
        labels = {"wild_card": ["2 vs 7", "3 vs 6", "4 vs 5"],
                  "divisional": ["#1 seed vs lowest survivor", "other two survivors"],
                  "conference": ["conference championship"], "super_bowl": ["Super Bowl"]}
    elif four:
        labels = {"semifinal": ["1 vs 4", "2 vs 3"], "final": ["national championship"]}
    else:
        labels = {"first_round": ["8 vs 9", "7 vs 10", "6 vs 11", "5 vs 12"],
                  "quarterfinal": ["1 vs 8/9", "2 vs 7/10", "3 vs 6/11", "4 vs 5/12"],
                  "semifinal": ["QF1 vs QF4", "QF2 vs QF3"],
                  "final": ["national championship"]}
    return labels[rnd][k]


def _consensus_seeds(o: "Outcome", P: np.ndarray, members: list[int]) -> list[int]:
    """The most-likely bracket's seeds: a coherent field, not a per-seed argmax.

    NFL: each division's most likely winner, then the three most likely wild cards;
    CFB: the teams most likely to make the field. Within each group, teams are ordered by
    their expected seed given that they qualify. (A per-seed argmax can seat the same
    kind of team twice - it put two Group of Six teams in the 2026 field - because each
    seed's leader is chosen without regard to the others.)
    """
    n = P.shape[0]
    p_in = P.sum(0)
    exp_seed = (np.arange(1, n + 1)[:, None] * P).sum(0) / np.maximum(p_in, 1e-12)
    if o.sport == "nfl":
        divs: dict[str, list[int]] = {}
        for i in members:
            divs.setdefault(o.division[o.teams[i]], []).append(i)
        pdiv = o.flags["division"].mean(0)
        winners = [max(v, key=lambda i: pdiv[i]) for v in divs.values()]
        rest = sorted((i for i in members if i not in winners), key=lambda i: -p_in[i])
        wc = rest[: n - len(winners)]
        def cond(i, lo, hi):  # expected seed among seeds lo..hi (1-based)
            w = P[lo - 1:hi, i]
            return (np.arange(lo, hi + 1) * w).sum() / max(w.sum(), 1e-12)

        nd = len(winners)
        return (sorted(winners, key=lambda i: cond(i, 1, nd))
                + sorted(wc, key=lambda i: cond(i, nd + 1, n)))
    field = sorted(members, key=lambda i: -p_in[i])[:n]
    return sorted(field, key=lambda i: exp_seed[i])


def tables(o: Outcome, computed_at: pd.Timestamp | None = None,
           model_version: str = eng.MODEL_VERSION, min_prob: float = 0.001
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    computed_at = computed_at or pd.Timestamp.now(tz="UTC")
    S, T = o.wins.shape
    base = {"sport": o.sport, "season": o.season, "as_of_week": o.as_of_week,
            "computed_at": computed_at, "model_version": model_version, "n_sims": S}
    seeds_flat = o.seeds.reshape(S, -1, o.seeds.shape[-1]) if o.seeds.ndim == 3 else o.seeds[:, None, :]
    n_seed = seeds_flat.shape[-1]
    P_seed = np.zeros((T, n_seed))
    for b in range(seeds_flat.shape[1]):
        for k in range(n_seed):
            np.add.at(P_seed[:, k], seeds_flat[:, b, k], 1.0)
    P_seed /= S
    max_g = int(o.n_games.max())
    rows = []
    fl = o.flags
    per_team = team_game_fields(o)
    for i, t in enumerate(o.teams):
        w = o.wins[:, i]
        dist = np.bincount(np.floor(w).astype(int), minlength=max_g + 1)[: max_g + 1] / S
        cur = o.current.loc[t]
        rows.append({
            **base, "team": o.team_abbr[t], "team_name": o.team_name[t],
            "conference": o.conference[t], "division": o.division.get(t),
            "wins": int(cur["wins"]), "losses": int(cur["losses"]), "ties": int(cur["ties"]),
            "conf_wins": int(cur["conf_wins"]), "conf_losses": int(cur["conf_losses"]),
            "rating": float(o.rating[i]), "rating_sd": float(o.rating_sd[i]),
            "power_rank": int((o.rating > o.rating[i]).sum() + 1),
            "remaining_games": int(o.remaining[i]),
            "remaining_sos": None if np.isnan(o.remaining_sos[i]) else float(o.remaining_sos[i]),
            "mean_wins": float(w.mean()), "mean_losses": float(o.n_games[i] - w.mean()),
            "wins_p10": float(np.percentile(w, 10)), "wins_p50": float(np.percentile(w, 50)),
            "wins_p90": float(np.percentile(w, 90)),
            "wins_dist": json.dumps([round(float(x), 5) for x in dist]),
            "p_division": float(fl["division"][:, i].mean()) if "division" in fl else None,
            "p_conf_game": float(fl["conf_game"][:, i].mean()),
            "p_conf_title": float(fl["conf_title"][:, i].mean()),
            "p_playoffs": float(fl["playoffs"][:, i].mean()),
            "p_bye": float(fl["bye"][:, i].mean()),
            "p_seed": json.dumps([round(float(x), 5) for x in P_seed[i]]),
            "p_quarters": float(fl["quarters"][:, i].mean()),
            "p_semis": float(fl["semis"][:, i].mean()),
            "p_final": float(fl["final"][:, i].mean()),
            "p_champion": float(fl["champion"][:, i].mean()),
            "exp_final_rank": float(o.final_rank[:, i].mean()),
            "rank_p10": float(np.percentile(o.final_rank[:, i], 10)),
            "rank_p90": float(np.percentile(o.final_rank[:, i], 90)),
            **per_team[o.team_abbr[t]],
        })
    team_df = pd.DataFrame(rows)
    bracket_df = _bracket_rows(o, base, P_seed if o.sport == "cfb" else None, min_prob)
    return team_df, bracket_df


# ---------------------------------------------------------------------------- per game
def _home_win(o: Outcome) -> np.ndarray:
    """(S, n) home 'wins' per remaining game, counted exactly as the standings count
    them: NFL gives each side half a win on a 0 margin, CFB (no ties) needs margin > 0."""
    M = o.game_margins
    hw = (M > 0).astype(float)
    return hw + 0.5 * (M == 0) if o.tie_half else hw


def team_game_fields(o: Outcome) -> dict[str, dict]:
    """Per team (keyed like the team table's `team`): the remaining-games columns.

      rem_wins_mean          expected wins over the remaining games
      rem_wins_dist          JSON, P(remaining wins == k), k = 0..remaining games
      projected_wins_games   JSON list of game_ids: the k likeliest wins, k = round(rem_wins_mean)
      modal_sequence         the single most common W/L sequence, schedule order ('WLLW...')
      modal_sequence_freq    share of simulations that produced exactly that sequence
      modal_sequence_record_p  P(that W-L record, in any order)

    The win indicators are the simulated margins the standings used, so per-game P(win)
    summed over a team's games IS rem_wins_mean (same draws, a linear identity).
    """
    empty = {"rem_wins_mean": 0.0, "rem_wins_dist": json.dumps([1.0]),
             "projected_wins_games": json.dumps([]), "modal_sequence": "",
             "modal_sequence_freq": 1.0, "modal_sequence_record_p": 1.0}
    out = {o.team_abbr[t]: dict(empty) for t in o.teams}
    if o.games is None or not len(o.games):
        return out
    g = o.games
    hw = _home_win(o)
    p_home = hw.mean(0)
    S = hw.shape[0]
    home = g["home"].to_numpy()
    away = g["away"].to_numpy()
    gid = g["game_id"].to_numpy()
    for key in out:
        js = np.flatnonzero((home == key) | (away == key))
        if not len(js):
            continue
        is_home = home[js] == key
        W = np.where(is_home[None, :], hw[:, js], 1.0 - hw[:, js])     # (S, k), schedule order
        rem = W.sum(1)
        mean = float(rem.mean())
        dist = np.bincount(np.floor(rem + 1e-9).astype(int), minlength=len(js) + 1)[: len(js) + 1] / S
        p_win = np.where(is_home, p_home[js], 1.0 - p_home[js])
        k = int(np.floor(mean + 0.5))
        top = sorted(range(len(js)), key=lambda q: (-p_win[q], q))[:k]
        code = (W >= 1.0).astype(np.int64) @ (1 << np.arange(len(js), dtype=np.int64))
        vals, counts = np.unique(code, return_counts=True)
        best = int(vals[np.argmax(counts)])
        seq = "".join("W" if (best >> q) & 1 else "L" for q in range(len(js)))
        out[key] = {
            "rem_wins_mean": mean,
            "rem_wins_dist": json.dumps([round(float(x), 5) for x in dist]),
            "projected_wins_games": json.dumps([str(gid[js[q]]) for q in top]),
            "modal_sequence": seq,
            "modal_sequence_freq": float(counts.max() / S),
            "modal_sequence_record_p": float(dist[seq.count("W")]),
        }
    return out


def games_table(o: Outcome, computed_at: pd.Timestamp | None = None,
                model_version: str = eng.MODEL_VERSION) -> pd.DataFrame:
    """One row per remaining regular-season game: the marginal P(home win) over the
    simulated seasons (rating draws included) and the simulated margin's mean and 10th/90th
    percentiles. Conference title games and the postseason are not listed: their pairings
    are themselves simulated."""
    import datetime as dt

    computed_at = computed_at or pd.Timestamp.now(tz="UTC")
    cols = ["sport", "season", "as_of_week", "computed_at", "model_version", "n_sims",
            "game_id", "week", "game_date", "home", "away", "home_name", "away_name",
            "neutral", "p_home_win", "margin_mean", "margin_p10", "margin_p90"]
    if o.games is None or not len(o.games):
        return pd.DataFrame(columns=cols)
    g = o.games
    M = o.game_margins
    return pd.DataFrame({
        "sport": o.sport, "season": o.season, "as_of_week": o.as_of_week,
        "computed_at": computed_at, "model_version": model_version, "n_sims": M.shape[0],
        "game_id": g["game_id"].astype(str).to_numpy(), "week": g["week"].to_numpy(int),
        "game_date": [dt.date.fromisoformat(d) if isinstance(d, str) else None
                      for d in g["game_day"]],
        "home": g["home"].to_numpy(), "away": g["away"].to_numpy(),
        "home_name": g["home_name"].to_numpy(), "away_name": g["away_name"].to_numpy(),
        "neutral": g["neutral"].to_numpy(bool),
        "p_home_win": _home_win(o).mean(0), "margin_mean": M.mean(0),
        "margin_p10": np.percentile(M, 10, axis=0), "margin_p90": np.percentile(M, 90, axis=0),
    })[cols]


def _bracket_rows(o: Outcome, base: dict, _unused, min_prob: float) -> pd.DataFrame:
    S, T = o.wins.shape
    out = []
    four = o.sport == "cfb" and o.seeds.shape[-1] == 4
    brackets = list(nflm.CONFS) if o.sport == "nfl" else ["CFP"]
    modal_seeds = {}
    for bi, b in enumerate(brackets):
        sd = o.seeds[:, bi, :] if o.sport == "nfl" else o.seeds
        n = sd.shape[1]
        P = np.zeros((n, T))
        for k in range(n):
            P[k] = np.bincount(sd[:, k], minlength=T) / S
        members = [i for i, t in enumerate(o.teams)
                   if o.sport == "cfb" or o.conference[t] == b]
        ms = _consensus_seeds(o, P, members)
        modal_seeds[b] = ms
        for k in range(n):
            for i in np.flatnonzero(P[k] >= min_prob):
                t = o.teams[i]
                out.append({**base, "bracket": b, "round": "seed", "round_order": 0,
                            "slot": k + 1, "slot_label": f"#{k + 1} seed", "team": o.team_abbr[t],
                            "team_name": o.team_name[t], "p_slot": float(P[k, i]),
                            "p_win": float(P[k, i]), "is_modal": bool(ms[k] == i),
                            "modal_opponent": None})

    # Game rounds: p_slot / p_win per team, then the modal path.
    order = ROUND_ORDER[o.sport]
    modal_winner: dict[tuple, int] = {}
    for (b, rnd), games_ in sorted(o.rounds.items(), key=lambda kv: order[kv[0][1]]):
        for k, (h, a, w) in enumerate(games_):
            ps = (np.bincount(h, minlength=T) + np.bincount(a, minlength=T)) / S
            pw = np.bincount(w, minlength=T) / S
            pair = _modal_pair(o, b, rnd, k, modal_seeds, modal_winner)
            if pair is not None:
                x, y = pair
                win = x if pw[x] >= pw[y] else y
                modal_winner[(b, rnd, k)] = win
            label = _slot_labels(o.sport, rnd, k, four)
            for i in np.flatnonzero(ps >= min_prob):
                t = o.teams[i]
                is_m = pair is not None and i in pair
                opp = None
                if is_m:
                    j = pair[1] if i == pair[0] else pair[0]
                    opp = o.team_abbr[o.teams[j]]
                out.append({**base, "bracket": b, "round": rnd, "round_order": order[rnd],
                            "slot": k + 1, "slot_label": label, "team": o.team_abbr[t],
                            "team_name": o.team_name[t], "p_slot": float(ps[i]),
                            "p_win": float(pw[i]), "is_modal": bool(is_m),
                            "modal_opponent": opp})
    return pd.DataFrame(out)


def _modal_pair(o, b, rnd, k, modal_seeds, mw):
    if o.sport == "nfl":
        if rnd == "super_bowl":
            return (mw[("AFC", "conference", 0)], mw[("NFC", "conference", 0)])
        ms = modal_seeds[b]
        if rnd == "wild_card":
            return (ms[[1, 2, 3][k]], ms[[6, 5, 4][k]])
        if rnd == "divisional":
            surv = sorted(ms.index(mw[(b, "wild_card", j)]) for j in range(3))
            return (ms[0], ms[surv[2]]) if k == 0 else (ms[surv[0]], ms[surv[1]])
        if rnd == "conference":
            return (mw[(b, "divisional", 0)], mw[(b, "divisional", 1)])
        return None
    ms = modal_seeds["CFP"]
    if len(ms) == 4:
        if rnd == "semifinal":
            return (ms[0], ms[3]) if k == 0 else (ms[1], ms[2])
        return (mw[(b, "semifinal", 0)], mw[(b, "semifinal", 1)])
    if rnd == "first_round":  # slot k feeds quarterfinal k
        return (ms[7 - k], ms[8 + k])
    if rnd == "quarterfinal":
        return (ms[k], mw[(b, "first_round", k)])
    if rnd == "semifinal":
        return (mw[(b, "quarterfinal", 0)], mw[(b, "quarterfinal", 3)]) if k == 0 else \
            (mw[(b, "quarterfinal", 1)], mw[(b, "quarterfinal", 2)])
    return (mw[(b, "semifinal", 0)], mw[(b, "semifinal", 1)])
