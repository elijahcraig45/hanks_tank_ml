"""Per-sport game loaders, each returning the same canonical frame.

The rankings engine wants one shape:

    season, week, game_date, home_team_name, away_team_name, home_won,
    neutral_site, division (optional)

Every source here reads a PUBLIC feed — ESPN for college, nflverse for the NFL, MLB
StatsAPI for baseball — so a rankings run needs no GCP credentials. That is deliberate:
the ratings are reproducible from the open record, and BigQuery is only where results
are published.

Each sport also carries its own fit parameters. They are not interchangeable: a
162-game baseball season identifies 30 teams far better than a 12-game college season
identifies 380, so the ridge strength and how fast last season decays genuinely differ.
Values here are measured by rankings.tune, not guessed.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:  # package layout locally, flat module tree in Cloud Functions
    from rankings.http import get_bytes as _get_bytes, cache_dir as _cache_dir
except ImportError:  # pragma: no cover
    from http_transport import get_bytes as _get_bytes, cache_dir as _cache_dir

logger = logging.getLogger(__name__)

# Writable wherever this runs; /workspace is read-only in Cloud Functions.
CACHE = _cache_dir("rankings")

CANONICAL = [
    "season", "week", "game_date", "home_team_name", "away_team_name",
    "home_won", "neutral_site", "division", "home_division", "away_division",
    # Home score minus away score. Optional for the W/L fit, required by the margin
    # model; a loader that cannot supply it leaves it null.
    "margin",
]

# How old a cache of an in-progress season may get before it is refetched. Every
# loader here used to read a cache file forever once it existed, so a warm container
# or a laptop kept rating a season as of the day it was first fetched.
MAX_CACHE_AGE_HOURS = 3.0


def _is_stale(path: Path, max_age_hours: float = MAX_CACHE_AGE_HOURS) -> bool:
    import time

    if not path.exists():
        return True
    return (time.time() - path.stat().st_mtime) > max_age_hours * 3600


@dataclass(frozen=True)
class SportSpec:
    key: str
    label: str
    dataset_env: str
    default_dataset: str
    # Name of the stronger division, or None for a single-division sport. Passing a
    # value switches on the division covariate in the design matrix.
    major_division: str | None
    # Only rank teams in this division (college splits the board; others show all).
    board_divisions: tuple[str, ...]
    ridge_C: float
    prior_w0: float
    prior_tau: float
    # Regular-season length in "weeks", used only to scale the prior decay sensibly.
    season_weeks: int
    # Which likelihood the rating uses (see core.MODELS) and the margin model's knobs.
    # Defaults reproduce the original W/L-only engine.
    model: str = "bt"
    margin_alpha: float = 10.0
    margin_cap: float | None = None
    margin_scale: float = 10.0
    blend: float = 0.5

    def model_kw(self) -> dict:
        return {
            "model": self.model, "margin_alpha": self.margin_alpha,
            "margin_cap": self.margin_cap, "margin_scale": self.margin_scale,
            "blend": self.blend,
        }


SPORTS: dict[str, SportSpec] = {
    "cfb": SportSpec(
        key="cfb", label="College Football",
        dataset_env="CFB_DATASET", default_dataset="cfb_season",
        major_division="fbs", board_divisions=("fbs", "fcs"),
        # Tuned by 5-fold CV and walk-forward log loss across 2022-2025 in the
        # original college implementation; carried over unchanged.
        ridge_C=2.0, prior_w0=1.0, prior_tau=8.0, season_weeks=16,
    ),
    "nfl": SportSpec(
        key="nfl", label="NFL",
        dataset_env="NFL_DATASET", default_dataset="nfl_season",
        major_division=None, board_divisions=(),
        # Walk-forward over 2021-2025 (1,355 games): C=0.5 minimizes log loss at
        # 0.6451 against a 0.6931 coin flip. tau is flat across 6-10 (0.6451-0.6459,
        # i.e. noise), so 8 is taken as the middle of the plateau rather than the
        # nominal argmin. Heavier penalties are clearly worse: C=8 scores 0.6742.
        ridge_C=0.5, prior_w0=1.0, prior_tau=8.0, season_weeks=18,
    ),
    "mlb": SportSpec(
        key="mlb", label="MLB",
        dataset_env="MLB_RANKINGS_DATASET", default_dataset="mlb_2026_season",
        major_division=None, board_divisions=(),
        # Walk-forward over 2023-2025 (7,281 games): C=0.06, and the entire surface
        # spans only 0.6808-0.6909. Read that honestly — a coin flip is 0.6931, so
        # team strength buys about 0.012 nats in baseball against roughly 0.048 in
        # the NFL. Ratings here are a fair summary of who has been best; they are
        # NOT a useful game predictor, and the wide bootstrap bands say so.
        #
        # The heavy penalty is a consequence: with so little true separation, the
        # likelihood is nearly flat and shrinkage is what keeps a 100-win team from
        # being credited with more than the schedule can support. tau barely matters
        # (8 and 20 tie), which is itself evidence there is little signal to carry.
        ridge_C=0.06, prior_w0=1.0, prior_tau=20.0, season_weeks=27,
    ),
}


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to the canonical shape and drop undecided games."""
    if df.empty:
        return pd.DataFrame(columns=CANONICAL)
    for col in ("division", "home_division", "away_division", "margin"):
        if col not in df.columns:
            df[col] = None
    out = df[CANONICAL].copy()
    out["margin"] = pd.to_numeric(out["margin"], errors="coerce")
    out = out[out["home_won"].notna()]
    out["home_won"] = out["home_won"].astype(int)
    out["neutral_site"] = out["neutral_site"].fillna(0).astype(int)
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["game_date"] = pd.to_datetime(out["game_date"])
    return out.sort_values(["season", "week", "game_date"]).reset_index(drop=True)


def _curl(url: str, timeout: int = 90) -> bytes:
    """Delegates to the shared transport; see rankings.http for why."""
    return _get_bytes(url, timeout)


# ── College football ────────────────────────────────────────────────────────
CFB_CORE = ("https://sports.core.api.espn.com/v2/sports/football/leagues/"
            "college-football/seasons")
CFB_DIVISION_GROUPS = {"fbs": 80, "fcs": 81}


def cfb_division_conferences(season: int, refresh: bool = False) -> dict[str, str]:
    """ESPN conference id -> "fbs"/"fcs" for one season, from ESPN's group tree.

    This is the membership source of truth. Reading a team's division off the feed its
    games came from is not: an FCS feed lists every D2/NAIA opponent an FCS team
    plays, which put 29 non-D1 schools on the 2026 FCS board, and a game ESPN listed
    only in the FBS feed (Delaware v Delaware State, 2025 week 1) tagged an FCS team
    FBS. A conference belongs to exactly one division per season, and a team's
    conference is on every game row, so conference membership settles both.

    Empty on failure; the caller falls back to the feed-based inference.
    """
    import re

    cached = CACHE / f"cfb_division_conferences_{season}.json"
    if cached.exists() and not refresh:
        return json.loads(cached.read_text())

    out: dict[str, str] = {}
    for division, group in CFB_DIVISION_GROUPS.items():
        url = f"{CFB_CORE}/{season}/types/2/groups/{group}/children?limit=100"
        try:
            listing = json.loads(_get_bytes(url, 30).decode())
        except Exception as exc:
            logger.warning("CFB %s conference listing failed for %d: %s",
                           division, season, exc)
            return {}
        for item in listing.get("items", []):
            found = re.search(r"/groups/(\d+)", item.get("$ref", ""))
            if found:
                out[found.group(1)] = division
    if out:
        cached.write_text(json.dumps(out))
    return out


def cfb_team_divisions(chunk: pd.DataFrame, conf_division: dict[str, str]
                       ) -> dict[str, str | None]:
    """Team abbreviation -> its own division in this season's games, or None.

    None means "not Division I": the team's conference is in neither the FBS nor the
    FCS tree. Such teams stay in the fit — their games are real evidence about the
    D1 teams they played — but never on a board.
    """
    sides = pd.concat([
        chunk[["home_team", "home_conference_id"]].set_axis(["team", "conf"], axis=1),
        chunk[["away_team", "away_conference_id"]].set_axis(["team", "conf"], axis=1),
    ])
    sides["division"] = sides["conf"].map(
        lambda c: conf_division.get(str(c)) if c is not None and pd.notna(c) else None
    )
    out: dict[str, str | None] = {}
    for team, group in sides.groupby("team"):
        known = group["division"].dropna()
        # A team's conference is constant within a season; the mode guards against a
        # stray row rather than expressing any real ambiguity.
        out[team] = known.mode().iloc[0] if not known.empty else None
    return out


def load_cfb() -> pd.DataFrame:
    """FBS and FCS games from the college ingest's ESPN cache.

    Both divisions are loaded together on purpose: the schedule graph is weakly
    connected and cross-division games are the only edges tying the two populations to
    a common scale. Fitting them apart makes the ladders incomparable.
    """
    # Locally the college modules live in ../cfb; in the deployed Cloud Function the
    # whole tree is flattened, so espn_data is already importable at top level.
    import sys
    cfb_dir = Path(__file__).resolve().parents[1] / "cfb"
    if cfb_dir.is_dir():
        sys.path.insert(0, str(cfb_dir))
    from espn_data import load_games, resolve_team_divisions  # noqa: E402

    games = load_games().copy()

    # Divisions belong to a (team, season) pair, so resolve one season at a time:
    # Sacramento State and North Dakota State are FCS in 2025 and FBS in 2026. Keyed on
    # the team ABBREVIATION (`home_team`/`away_team`), not the display name.
    home_div, away_div = [], []
    for season, chunk in games.groupby("season"):
        conf_division = cfb_division_conferences(int(season))
        if conf_division:
            div_of = cfb_team_divisions(chunk, conf_division)
        else:
            # Offline fallback: infer from the feeds. Known to admit non-D1
            # opponents, so it is only used when ESPN's group tree is unreachable.
            logger.warning("CFB %s: no conference tree, inferring divisions from feeds",
                           season)
            div_of = resolve_team_divisions(chunk)
        home_div.append(chunk["home_team"].map(div_of))
        away_div.append(chunk["away_team"].map(div_of))

    games["home_division"] = pd.concat(home_div).reindex(games.index)
    games["away_division"] = pd.concat(away_div).reindex(games.index)
    games["margin"] = pd.to_numeric(games["home_score"], errors="coerce") - \
        pd.to_numeric(games["away_score"], errors="coerce")
    # `division` stays the feed tag for backward compatibility with the college
    # pipeline; the board uses the per-side columns.
    return _finalize(games)


# ── NFL ─────────────────────────────────────────────────────────────────────
NFL_GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
NFL_CACHE = CACHE / "nfl_games.csv"

# Relocated franchises. nflverse keeps the abbreviation a team used at the time, which
# is right for a game log and wrong for a rating: keyed on the raw code, the Raiders'
# 2019 season belongs to a different "team" than their 2020 one, so last season's games
# cannot inform this season's rating and a phantom OAK row appears on the board.
# Collapsing to the current code makes one franchise one entity.
NFL_FRANCHISE_ALIASES = {
    "OAK": "LV",    # Oakland -> Las Vegas, 2020
    "SD": "LAC",    # San Diego -> Los Angeles, 2017
    "STL": "LA",    # St. Louis -> Los Angeles, 2016
}


NFL_GAME_TYPES = ("REG", "WC", "DIV", "CON", "SB")


def load_nfl(refresh: bool = False) -> pd.DataFrame:
    """nflverse schedules, 1999 to date, cached to disk.

    Cached because nflverse serves from GitHub with no SLA and this runs inside a cron;
    a GitHub outage should degrade to last-good rather than fail the run.
    """
    # nflverse republishes games.csv as results land, so a cache that exists is not a
    # cache that is current. Refetch when it is older than MAX_CACHE_AGE_HOURS; a
    # failed refetch still degrades to the last good copy.
    if refresh or _is_stale(NFL_CACHE):
        try:
            NFL_CACHE.write_bytes(_curl(NFL_GAMES_URL))
        except Exception as exc:
            if not NFL_CACHE.exists():
                raise
            logger.warning("nflverse fetch failed (%s); using cached copy", exc)

    raw = pd.read_csv(NFL_CACHE)
    # nflverse labels playoff rounds individually (WC/DIV/CON/SB) and has no "POST"
    # value, so filtering on REG+POST silently dropped every playoff game — which is
    # why the 2025 board stopped at week 18.
    g = raw[raw["game_type"].isin(NFL_GAME_TYPES)].copy()
    g["margin"] = pd.to_numeric(g["result"], errors="coerce")
    g["home_won"] = pd.to_numeric(g["result"], errors="coerce").apply(
        lambda r: None if pd.isna(r) or r == 0 else int(r > 0)
    )
    # `location` is "Home" or "Neutral"; internationals and Super Bowls are neutral.
    g["neutral_site"] = (g["location"].astype(str) != "Home").astype(int)
    g = g.rename(columns={
        "home_team": "home_team_name",
        "away_team": "away_team_name",
        "gameday": "game_date",
    })
    for column in ("home_team_name", "away_team_name"):
        g[column] = g[column].replace(NFL_FRANCHISE_ALIASES)
    return _finalize(g)


# ── MLB ─────────────────────────────────────────────────────────────────────
MLB_SCHEDULE = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?sportId=1&season={season}&gameType=R"
    "&fields=dates,date,games,gamePk,gameType,teams,home,away,team,id,name,score,"
    "isWinner,status,detailedState,codedGameState"
)


# Official, decided games. "Completed Early" is a game called after it became official
# (rain, curfew): it counts in the standings exactly like a Final, so it counts here.
MLB_DECIDED_STATES = {"Final", "Completed Early"}

# Coded states that can never change again. A season whose every game sits in one of
# these, fetched after its last scheduled date, is frozen and never needs refetching.
MLB_TERMINAL_CODES = {"F", "D", "C", "O"}


def _mlb_cache_is_current(path: Path, payload: dict) -> bool:
    """True if a cached schedule can be used as-is.

    A finished season is immutable, so once it has been fetched complete it is kept
    forever. Anything else — the season in progress, or a past season whose cache was
    written mid-season — is refetched when older than MAX_CACHE_AGE_HOURS.
    """
    import datetime as dt

    dates = [d.get("date") for d in payload.get("dates", []) if d.get("date")]
    codes = {
        (g.get("status") or {}).get("codedGameState")
        for d in payload.get("dates", []) for g in d.get("games", [])
    }
    if dates and codes <= MLB_TERMINAL_CODES:
        fetched = dt.date.fromtimestamp(path.stat().st_mtime)
        if fetched > dt.date.fromisoformat(max(dates)):
            return True
    return not _is_stale(path)


def _mlb_payload(season: int, refresh: bool = False) -> dict:
    cached = CACHE / f"mlb_games_{season}.json"
    if cached.exists() and not refresh:
        payload = json.loads(cached.read_text())
        if _mlb_cache_is_current(cached, payload):
            return payload
    try:
        payload = json.loads(_curl(MLB_SCHEDULE.format(season=season)).decode())
    except Exception as exc:
        if not cached.exists():
            raise
        logger.warning("StatsAPI fetch failed for %d (%s); using cached copy", season, exc)
        return json.loads(cached.read_text())
    cached.write_text(json.dumps(payload))
    return payload


def mlb_rows(payload: dict, season: int) -> list[dict]:
    """One row per decided game, keyed on gamePk.

    A suspended game is listed on BOTH its original date and the date it was resumed,
    each time showing the final score — 824912 (2026) was counted twice, putting five
    clubs' records one game off the official standings. The same game must count
    once, so rows are keyed on gamePk and the latest listing (the resumption, when the
    result actually happened) wins.
    """
    by_pk: dict = {}
    for date in payload.get("dates", []):
        for game in date.get("games", []):
            home, away = game["teams"]["home"], game["teams"]["away"]
            hs, as_ = home.get("score"), away.get("score")
            state = (game.get("status") or {}).get("detailedState")
            if state not in MLB_DECIDED_STATES or hs is None or as_ is None or hs == as_:
                continue
            key = game.get("gamePk") or (date["date"], home["team"]["id"], away["team"]["id"])
            row = {
                "season": season,
                "game_pk": game.get("gamePk"),
                "game_date": date["date"],
                "home_team_id": home["team"]["id"],
                "away_team_id": away["team"]["id"],
                "home_team_name": home["team"]["name"],
                "away_team_name": away["team"]["name"],
                "home_won": int(hs > as_),
                "margin": hs - as_,
                "neutral_site": 0,
            }
            previous = by_pk.get(key)
            if previous is None or row["game_date"] >= previous["game_date"]:
                by_pk[key] = row
    return list(by_pk.values())


def load_mlb(seasons: tuple[int, ...], refresh: bool = False) -> pd.DataFrame:
    """Completed regular-season MLB games from the public StatsAPI.

    Baseball has no neutral sites worth modelling and no divisions in the
    Bradley-Terry sense (the AL/NL split is not a strength tier — interleague play is
    frequent enough to tie them to one scale), so both columns are constant.

    `week` is the index of the ISO week within the season, used only so the prior
    decay has something to count.
    """
    frames = []
    for season in seasons:
        rows = mlb_rows(_mlb_payload(season, refresh=refresh), season)
        if not rows:
            logger.warning("no completed MLB games for %d", season)
            continue

        df = pd.DataFrame(rows)
        df["game_date"] = pd.to_datetime(df["game_date"])
        opening = df["game_date"].min()
        df["week"] = ((df["game_date"] - opening).dt.days // 7) + 1
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=CANONICAL)

    combined = pd.concat(frames, ignore_index=True)

    # Franchises get renamed — Oakland Athletics became Athletics for 2025 — and keying
    # a rating on the display name splits one club into two entities, so its earlier
    # games never inform its current rating and a phantom 0-0 row lands on the board.
    # The StatsAPI team id is stable across renames, so resolve every id to the newest
    # name it has used and key on that.
    latest = (
        combined.sort_values("season")
        .groupby("home_team_id")["home_team_name"].last()
        .to_dict()
    )
    for side in ("home", "away"):
        combined[f"{side}_team_name"] = (
            combined[f"{side}_team_id"].map(latest)
            .fillna(combined[f"{side}_team_name"])
        )

    return _finalize(combined)


def load(sport: str, seasons: tuple[int, ...] | None = None) -> pd.DataFrame:
    if sport == "cfb":
        return load_cfb()
    if sport == "nfl":
        return load_nfl()
    if sport == "mlb":
        if not seasons:
            raise ValueError("MLB needs explicit seasons; StatsAPI is queried per year")
        return load_mlb(seasons)
    raise ValueError(f"unknown sport: {sport}")
