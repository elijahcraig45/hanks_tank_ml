"""FBS season structure: conference standings, title games, committee proxy, CFP brackets.

What is modelled, and how honestly:

  * Conference standings use conference games only (ESPN's conferenceCompetition flag,
    both teams in the same FBS conference). Title games are between the top two, or the
    division winners where a conference still had divisions (Sun Belt every season; SEC,
    Big Ten, MAC in 2022-23; ACC and Mountain West in 2022; see DIVISIONS).
  * Tiebreakers are an APPROXIMATION. Every conference publishes its own ladder, and the
    SEC, ACC and Big 12 end theirs in analytics composites nobody outside can reproduce.
    We apply: conference win pct -> head-to-head (two teams that met, or a complete round
    robin among three or more) -> win pct against common conference opponents -> the
    end-of-season rating (standing in for the composites) -> coin toss, with the survivors
    restarting whenever a step eliminates someone.
  * Title-game site: hosted by the higher seed in the American, C-USA, Mountain West and
    Sun Belt (as the data shows), neutral elsewhere.
  * The CFP committee is not a formula. We rank teams by a committee proxy,
        score = end-of-season rating + a * losses + b * conference champion + c * Group of Six,
    with the weights fitted to the committee's real final rankings 2021-2025
    (scripts/football/fit_cfp_committee.py). It is an approximation: it reproduces the
    real 2024 field and 11 of 12 teams in 2025, and it has no notion of injuries,
    "eye test", or head-to-head arguments the committee makes in the room.

CFP formats (verified 2026-09-28 against collegefootballplayoff.com, 2026-01-23, and
ncaa.com): 2014-2023 four teams (1v4, 2v3); 2024 twelve teams, the five highest-ranked
conference champions plus seven at-large, the top four CONFERENCE CHAMPIONS seeded 1-4 with
byes; 2025 the same field, seeded straight by ranking; 2026 twelve teams with automatic bids
for the Big Ten, SEC, ACC and Big 12 champions plus the highest-ranked Group of Six team
(champion or not), seven at-large, seeded straight by ranking, top four get byes. First
round at the higher seed's campus; quarterfinals, semifinals and the final at neutral sites.
Quarterfinal bowl assignments and rematch avoidance are not modelled (they do not change
who advances).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import engine as eng

FBS_CONFS = {"151", "1", "4", "5", "12", "18", "15", "17", "9", "8", "37"}
INDEPENDENT = "18"
POWER4 = ("8", "5", "1", "4")          # SEC, Big Ten, ACC, Big 12
GROUP6 = {"151", "12", "15", "17", "9", "37"}
HOSTED_TITLE_GAME = {"151", "12", "17", "37"}
CONF_NAMES = {
    "151": "American", "1": "ACC", "4": "Big 12", "5": "Big Ten", "12": "Conference USA",
    "18": "FBS Independents", "15": "MAC", "17": "Mountain West", "9": "Pac-12",
    "8": "SEC", "37": "Sun Belt",
}
# Power conferences by season, for the pre-2026 "power vs group" split (Pac-12 was a
# power conference through 2023).
POWER_BY_SEASON = {y: set(POWER4) | ({"9"} if y <= 2023 else set()) for y in range(2014, 2031)}

# Divisions by season, as ESPN abbreviations (converted to names at build time).
_SBC_E = ["APP", "CCU", "GASO", "GAST", "JMU", "MRSH", "ODU"]
_SBC_W = ["ARST", "TROY", "TXST", "UL", "ULM", "USA", "USM"]
_SBC_W26 = ["ARST", "LT", "TROY", "UL", "ULM", "USA", "USM"]
_SEC = {"East": ["FLA", "UGA", "UK", "MIZ", "SC", "TENN", "VAN"],
        "West": ["ALA", "ARK", "AUB", "LSU", "MISS", "MSST", "TA&M"]}
_B1G = {"East": ["IU", "MD", "MICH", "MSU", "OSU", "PSU", "RUTG"],
        "West": ["ILL", "IOWA", "MINN", "NEB", "NU", "PUR", "WISC", "WIS"]}
_MAC = {"East": ["AKR", "BGSU", "BUFF", "KENT", "M-OH", "OHIO"],
        "West": ["BALL", "CMU", "EMU", "NIU", "TOL", "WMU"]}
DIVISIONS = {
    2022: {"37": {"East": _SBC_E, "West": _SBC_W}, "8": _SEC, "5": _B1G, "15": _MAC,
           "1": {"Atlantic": ["BC", "CLEM", "FSU", "LOU", "NCST", "NCSU", "SYR", "WAKE"],
                 "Coastal": ["DUKE", "GT", "MIA", "UNC", "PITT", "UVA", "VT"]},
           "17": {"Mountain": ["AFA", "BOIS", "CSU", "UNM", "USU", "WYO"],
                  "West": ["FRES", "HAW", "NEV", "SDSU", "SJSU", "UNLV"]}},
    2023: {"37": {"East": _SBC_E, "West": _SBC_W}, "8": _SEC, "5": _B1G, "15": _MAC},
    2024: {"37": {"East": _SBC_E, "West": _SBC_W}},
    2025: {"37": {"East": _SBC_E, "West": _SBC_W}},
    2026: {"37": {"East": _SBC_E, "West": _SBC_W26}},
}
TIE_EPS = 1e-9

# Committee proxy: score = rating + LOSS_WEIGHT * losses + CHAMP_BONUS * conference champion
#                          + G6_PENALTY * (Group of Six member)
# Fitted by scripts/football/fit_cfp_committee.py on the committee's final top 25,
# 2021-2025 (pairwise ordering accuracy over every pair involving a ranked team):
#   rating only 0.940; + losses/champion 0.972; + the Group of Six term 0.983.
# Leave-one-season-out, each added term raises held-out accuracy in all 5 seasons.
# Without the G6 term the proxy put one-loss James Madison 9th in 2025 (committee: 24th)
# and unbeaten Liberty 11th in 2023 (23rd). The objective is a flat ridge that drifts
# toward ever larger weights, so the rule is: the smallest weights within 0.001 of the
# best. With the real champions and format it picks 4/4, 4/4, 3/4 of the 2021-23 fields,
# 12/12 in 2024 and 10/12 in 2025.
LOSS_WEIGHT = -8.0
CHAMP_BONUS = 4.0
G6_PENALTY = -12.0


def group6(conf_ids, season: int) -> np.ndarray:
    """Group of Five/Six membership: the Pac-12 counts from 2024 (after the exodus)."""
    g = GROUP6 if season >= 2024 else GROUP6 - {"9"}
    return np.isin(np.asarray(conf_ids, dtype=str), list(g)).astype(float)


def cfp_format(season: int) -> str:
    if season <= 2023:
        return "four"
    if season == 2024:
        return "twelve_2024"
    if season == 2025:
        return "twelve_2025"
    return "twelve_2026"


def _et_date(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts) - pd.Timedelta(hours=5)


def normalise_games(games: pd.DataFrame) -> pd.DataFrame:
    """Plain dtypes, team key = display name, conference ids as strings, flags for
    conference games and title games."""
    g = games.copy()
    for c in ("season", "week", "is_postseason", "neutral_site", "conference_game"):
        g[c] = pd.to_numeric(g[c]).astype("float64").fillna(0).astype(int)
    g["margin"] = pd.to_numeric(g["result"]).astype("float64")
    # fillna first: pandas 3's str dtype keeps a missing id missing through astype(str)
    # (pandas 2 made it the string "None"), and an all-missing team then has no mode.
    g["hc"] = g["home_conference_id"].fillna("None").astype(str)
    g["ac"] = g["away_conference_id"].fillna("None").astype(str)
    g["home_abbr"], g["away_abbr"] = g["home_team"].astype(str), g["away_team"].astype(str)
    g["home_team"], g["away_team"] = g["home_team_name"].astype(str), g["away_team_name"].astype(str)
    tbd = g["home_team"].str.contains("TBD") | g["away_team"].str.contains("TBD")
    g = g[~tbd].copy()
    same = (g["hc"] == g["ac"]) & g["hc"].isin(FBS_CONFS - {INDEPENDENT})
    army_navy = {frozenset(("ARMY", "NAVY"))}
    an = [frozenset((h, a)) in army_navy for h, a in zip(g["home_abbr"], g["away_abbr"])]
    g["is_ccg"] = same & (g["conference_game"] == 0) & (g["week"] >= 13) & \
        (g["is_postseason"] == 0) & ~np.array(an) & (_et_date(g["game_date"]).dt.month == 12)
    g["is_conf"] = same & (g["conference_game"] == 1) & ~g["is_ccg"]
    g["neutral"] = g["neutral_site"].astype(float)
    g["game_day"] = _eastern_day(g["game_date"])
    return g


def _eastern_day(ts: pd.Series) -> pd.Series:
    """Calendar date in US Eastern time (ESPN kickoffs are UTC; a TBD kickoff is stored as
    local midnight, 04:00Z/05:00Z, which a flat -5h shift would move to the day before)."""
    t = pd.to_datetime(ts, errors="coerce", utc=True)
    return t.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")


def team_conferences(g: pd.DataFrame, season: int) -> dict[str, str]:
    s = g[g["season"] == season]
    sides = pd.concat([s[["home_team", "hc"]].set_axis(["t", "c"], axis=1),
                       s[["away_team", "ac"]].set_axis(["t", "c"], axis=1)])
    return sides.groupby("t")["c"].agg(lambda x: x.mode().iloc[0]).to_dict()


def build_frame(games: pd.DataFrame, season: int, as_of_week: int | None = None) -> pd.DataFrame:
    """Two prior seasons (every played game) plus this season's regular season, title
    games excluded (they are simulated from the standings). FCS-vs-FCS games still to be
    played are dropped: they cannot move an FBS standing."""
    g = normalise_games(games)
    conf_by_season = {y: team_conferences(g, y) for y in range(season - 2, season + 1)}

    # One division per team for the whole window: the one it plays in THIS season (or
    # the latest season it appears in). The division term is a level shift between two
    # populations, so a program moving up (North Dakota State and Sacramento State in
    # 2026) must keep one rating scale across its FCS history and its FBS present.
    # Tagging each game with that season's division instead credited NDSU with the whole
    # FBS-over-FCS gap on top of its FCS rating (19.7 points, #6 in FBS, after 4 games).
    latest: dict[str, str] = {}
    for y in sorted(conf_by_season):
        latest.update(conf_by_season[y])

    def fbs(team, y=None):
        return latest.get(team) in FBS_CONFS

    g["fbs_diff"] = [int(fbs(h)) - int(fbs(a)) for h, a in zip(g["home_team"], g["away_team"])]
    hist = g[(g["season"] >= season - 2) & (g["season"] < season) & g["margin"].notna()]
    cur = g[(g["season"] == season) & (g["is_postseason"] == 0) & ~g["is_ccg"]].copy()
    if as_of_week is not None:
        cur.loc[cur["week"] > as_of_week, "margin"] = np.nan
    touches_fbs = [fbs(h, season) or fbs(a, season) for h, a in zip(cur["home_team"], cur["away_team"])]
    cur = cur[cur["margin"].notna() | np.array(touches_fbs)]
    cur["sim"] = cur["margin"].isna()
    fr = pd.concat([hist.assign(sim=False), cur], ignore_index=True)
    out = eng.ridge_frame_from(fr, covariates=("fbs_diff",))
    out["is_conf"] = fr["is_conf"].to_numpy(bool)
    out["hc"] = fr["hc"].to_numpy()
    out["home_abbr"] = fr["home_abbr"].to_numpy()
    out["away_abbr"] = fr["away_abbr"].to_numpy()
    return out


class CfbSeason:
    """Fixed FBS structure for one season."""

    def __init__(self, frame: pd.DataFrame, season: int, games: pd.DataFrame | None = None):
        self.season = season
        cur = frame["season"].to_numpy() == season
        self.rows = np.flatnonzero(cur)
        f = frame.iloc[self.rows]
        sides = pd.concat([f[["home_team", "hc"]].set_axis(["t", "c"], axis=1)])
        conf = {}
        if games is not None:
            conf = team_conferences(normalise_games(games), season)
        else:  # from the frame: home side's conference on conference games is exact
            conf = sides.groupby("t")["c"].agg(lambda x: x.mode().iloc[0]).to_dict()
        teams_all = set(f["home_team"]) | set(f["away_team"])
        self.fbs = sorted(t for t in teams_all if conf.get(t) in FBS_CONFS)
        self.conf_of = {t: conf[t] for t in self.fbs}
        self.idx = {t: i for i, t in enumerate(self.fbs)}
        self.T = len(self.fbs)
        abbr = pd.concat([f[["home_team", "home_abbr"]].set_axis(["t", "a"], axis=1),
                          f[["away_team", "away_abbr"]].set_axis(["t", "a"], axis=1)])
        self.abbr = abbr.drop_duplicates("t").set_index("t")["a"].to_dict()
        self.hi = np.array([self.idx.get(t, -1) for t in f["home_team"]])
        self.ai = np.array([self.idx.get(t, -1) for t in f["away_team"]])
        self.is_conf = f["is_conf"].to_numpy(bool)
        G, T = len(self.rows), self.T
        self.Hinc = np.zeros((G, T))
        self.Ainc = np.zeros((G, T))
        hm, am = self.hi >= 0, self.ai >= 0
        self.Hinc[np.flatnonzero(hm), self.hi[hm]] = 1
        self.Ainc[np.flatnonzero(am), self.ai[am]] = 1
        self.games = self.Hinc.sum(0) + self.Ainc.sum(0)
        self.confs: dict[str, list[int]] = {}
        for t in self.fbs:
            c = self.conf_of[t]
            if c != INDEPENDENT:
                self.confs.setdefault(c, []).append(self.idx[t])
        # Conferences that crown a champion: at least four members with conference games.
        self.confs = {c: m for c, m in self.confs.items() if len(m) >= 4}
        name_of_abbr = {}
        for t, a in self.abbr.items():
            name_of_abbr.setdefault(a, t)
        self.divisions: dict[str, dict[str, list[int]]] = {}
        for c, divs in DIVISIONS.get(season, {}).items():
            if c not in self.confs:
                continue
            members = set(self.confs[c])
            conv = {d: [self.idx[name_of_abbr[a]] for a in ab
                        if a in name_of_abbr and name_of_abbr[a] in self.idx
                        and self.idx[name_of_abbr[a]] in members]
                    for d, ab in divs.items()}
            if sum(len(v) for v in conv.values()) == len(members):
                self.divisions[c] = conv
        ci = self.is_conf & hm & am
        self.conf_rows = np.flatnonzero(ci)
        self.cH, self.cA = self.hi[ci], self.ai[ci]
        self.cGames = np.zeros(T)
        np.add.at(self.cGames, self.cH, 1)
        np.add.at(self.cGames, self.cA, 1)

    def records(self, M: np.ndarray):
        hw = (M > 0).astype(float)
        wins = hw @ self.Hinc + (1 - hw) @ self.Ainc
        losses = self.games[None, :] - wins
        cm = M[:, self.conf_rows] > 0
        cw = np.zeros((M.shape[0], self.T))
        np.add.at(cw.T, self.cH, cm.T.astype(float))
        np.add.at(cw.T, self.cA, (~cm).T.astype(float))
        return wins, losses, cw

    def title_games(self, M: np.ndarray, cw: np.ndarray, rating: np.ndarray, rng) -> dict:
        """Per conference, the two title-game teams per path (team indices)."""
        S = M.shape[0]
        cm = M[:, self.conf_rows] > 0
        out = {}
        cpct = cw / np.maximum(self.cGames, 1)
        for c, members in self.confs.items():
            pair = np.zeros((S, 2), dtype=int)
            # conference games among members: (row positions, home, away)
            sel = np.isin(self.cH, members) & np.isin(self.cA, members)
            gh, ga, gm = self.cH[sel], self.cA[sel], cm[:, sel]
            divs = self.divisions.get(c)
            for s in range(S):
                ctx = _ConfCtx(cpct[s], rating[s], gh, ga, gm[s], rng)
                if divs:
                    pair[s] = [ctx.top(v) for v in divs.values()][:2]
                    if ctx.rank_key(pair[s, 1]) > ctx.rank_key(pair[s, 0]):
                        pair[s] = pair[s, ::-1]
                else:
                    first = ctx.top(members)
                    second = ctx.top([m for m in members if m != first])
                    pair[s] = [first, second]
            out[c] = pair
        return out


class _ConfCtx:
    """Approximate conference tiebreaker for one path."""

    def __init__(self, cpct, rating, gh, ga, gwin, rng):
        self.cpct, self.rating, self.rng = cpct, rating, rng
        self.gh, self.ga, self.gwin = gh, ga, gwin

    def rank_key(self, i):
        return (self.cpct[i], self.rating[i])

    def _h2h(self, group):
        gs = set(group)
        w = {i: 0.0 for i in group}
        n = {i: 0 for i in group}
        met = set()
        for h, a, hw in zip(self.gh, self.ga, self.gwin):
            if h in gs and a in gs:
                w[h] += hw
                w[a] += 1 - hw
                n[h] += 1
                n[a] += 1
                met.add(frozenset((h, a)))
        k = len(group)
        if len(met) < k * (k - 1) // 2:  # not a complete round robin
            return None
        return {i: w[i] / n[i] for i in group}

    def _common(self, group):
        opp = {i: set() for i in group}
        for h, a in zip(self.gh, self.ga):
            if h in opp:
                opp[h].add(a)
            if a in opp:
                opp[a].add(h)
        common = set.intersection(*opp.values()) - set(group)
        if not common:
            return None
        vals = {}
        for i in group:
            w = n = 0
            for h, a, hw in zip(self.gh, self.ga, self.gwin):
                if h == i and a in common:
                    w += hw
                    n += 1
                elif a == i and h in common:
                    w += 1 - hw
                    n += 1
            vals[i] = w / n if n else 0.0
        return vals

    def top(self, members):
        best = max(self.cpct[i] for i in members)
        group = [i for i in members if self.cpct[i] >= best - TIE_EPS]
        steps = [self._h2h, self._common, lambda g: {i: self.rating[i] for i in g}]
        while len(group) > 1:
            for step in steps:
                vals = step(group)
                if vals is None:
                    continue
                m = max(vals.values())
                keep = [i for i in group if vals[i] >= m - TIE_EPS]
                if len(keep) < len(group):
                    group = keep
                    break
            else:
                return group[int(self.rng.integers(len(group)))]
        return group[0]


def played_title_games(games: pd.DataFrame, season: int, cs: "CfbSeason",
                       as_of_week: int | None = None) -> dict[str, tuple[int, int, int]]:
    """Conference title games already played (and visible as of `as_of_week`):
    conference id -> (home idx, away idx, home_won)."""
    g = normalise_games(games)
    g = g[(g["season"] == season) & g["is_ccg"] & g["margin"].notna()]
    if as_of_week is not None:
        g = g[g["week"] <= as_of_week]
    out = {}
    for r in g.itertuples():
        if r.home_team in cs.idx and r.away_team in cs.idx:
            out[r.hc] = (cs.idx[r.home_team], cs.idx[r.away_team], int(r.margin > 0))
    return out


def committee_score(rating: np.ndarray, losses: np.ndarray, champ: np.ndarray,
                    g6: np.ndarray | float = 0.0, loss_weight: float | None = None,
                    champ_bonus: float | None = None, g6_penalty: float | None = None
                    ) -> np.ndarray:
    lw = LOSS_WEIGHT if loss_weight is None else loss_weight
    cb = CHAMP_BONUS if champ_bonus is None else champ_bonus
    gp = G6_PENALTY if g6_penalty is None else g6_penalty
    return rating + lw * losses + cb * champ + gp * g6


def select_field(score: np.ndarray, champ_of: dict[str, np.ndarray], conf_of_team: np.ndarray,
                 season: int) -> np.ndarray:
    """(S, n_seeds) team indices by seed, from committee scores (S, T).

    champ_of[c] = (S,) champion index per conference; conf_of_team = (T,) conference id.
    """
    S, T = score.shape
    fmt = cfp_format(season)
    order = np.argsort(-score, axis=1)                      # rank order per path
    rank = np.empty_like(order)
    rank[np.arange(S)[:, None], order] = np.arange(T)[None, :]
    if fmt == "four":
        return order[:, :4]
    champs = np.stack([champ_of[c] for c in champ_of], 1) if champ_of else np.zeros((S, 0), int)
    seeds = np.zeros((S, 12), dtype=int)
    g6 = np.isin(conf_of_team, list(GROUP6))
    for s in range(S):
        ch = list(dict.fromkeys(champs[s].tolist()))
        ch.sort(key=lambda i: rank[s, i])
        if fmt in ("twelve_2024", "twelve_2025"):
            auto = ch[:5]
        else:
            p4 = [champ_of[c][s] for c in POWER4 if c in champ_of]
            best_g6 = next(int(i) for i in order[s] if g6[i])
            auto = list(dict.fromkeys(p4 + [best_g6]))
        field = list(auto)
        for i in order[s]:
            if len(field) >= 12:
                break
            if i not in field:
                field.append(int(i))
        # If the automatic bids pushed the field past 12 (cannot happen with <=6 autos)
        field.sort(key=lambda i: rank[s, i])
        if fmt == "twelve_2024":
            top4 = sorted(ch[:4], key=lambda i: rank[s, i])
            rest = [i for i in field if i not in top4]
            field = top4 + rest
        seeds[s] = field[:12]
    return seeds


def play_cfp(seeds: np.ndarray, res: "eng.SimResult", col: np.ndarray, rng) -> dict:
    """Simulate the bracket. Returns per-round (home, away, winner) arrays of team idx."""
    S, n = seeds.shape
    rows = np.arange(S)
    noise = res.noise_sigma

    def play(h, a, neutral):
        mean = res.game_mean(col[h], col[a], np.full(h.shape, float(neutral)),
                             np.zeros(h.shape + (1,)))
        return eng.draw_margin(mean, noise, rng) > 0

    out = {}
    if n == 4:
        sf = []
        for hs, as_ in ((1, 4), (2, 3)):
            h, a = seeds[:, hs - 1], seeds[:, as_ - 1]
            w = play(h, a, 1)
            sf.append((h, a, np.where(w, h, a)))
        h, a = sf[0][2], sf[1][2]
        w = play(h, a, 1)
        out["semifinal"] = sf
        out["final"] = [(h, a, np.where(w, h, a))]
        return out
    fr = []
    # Listed in bracket order, so first-round game k feeds quarterfinal k.
    for hs, as_ in ((8, 9), (7, 10), (6, 11), (5, 12)):
        h, a = seeds[:, hs - 1], seeds[:, as_ - 1]
        w = play(h, a, 0)
        fr.append((h, a, np.where(w, h, a)))
    qf = []
    for top, feeder in ((1, 0), (2, 1), (3, 2), (4, 3)):  # 1 v 8/9, 2 v 7/10, 3 v 6/11, 4 v 5/12
        h, a = seeds[:, top - 1], fr[feeder][2]
        w = play(h, a, 1)
        qf.append((h, a, np.where(w, h, a)))
    sf = []
    for x, y in ((0, 3), (1, 2)):
        h, a = qf[x][2], qf[y][2]
        w = play(h, a, 1)
        sf.append((h, a, np.where(w, h, a)))
    h, a = sf[0][2], sf[1][2]
    w = play(h, a, 1)
    out["first_round"], out["quarterfinal"], out["semifinal"] = fr, qf, sf
    out["final"] = [(h, a, np.where(w, h, a))]
    _ = rows
    return out
