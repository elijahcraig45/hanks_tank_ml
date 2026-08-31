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
]


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
    for col in ("division", "home_division", "away_division"):
        if col not in df.columns:
            df[col] = None
    out = df[CANONICAL].copy()
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

    # A game's own `division` column names the ESPN feed it came from, not either
    # team's division; resolve_team_divisions is what actually knows. Resolve one
    # season at a time, because programs move up: Sacramento State and North Dakota
    # State are FCS in 2025 and FBS in 2026, and a single map over all seasons would
    # backdate that.
    # resolve_team_divisions keys on the team ABBREVIATION (its `home_team`/`away_team`
    # columns), while the rankings engine identifies teams by display name — map
    # through the abbreviation, not the name.
    home_div, away_div = [], []
    for _season, chunk in games.groupby("season"):
        div_of = resolve_team_divisions(chunk)
        home_div.append(chunk["home_team"].map(div_of))
        away_div.append(chunk["away_team"].map(div_of))

    games["home_division"] = pd.concat(home_div).reindex(games.index)
    games["away_division"] = pd.concat(away_div).reindex(games.index)
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


def load_nfl(refresh: bool = False) -> pd.DataFrame:
    """nflverse schedules, 1999 to date, cached to disk.

    Cached because nflverse serves from GitHub with no SLA and this runs inside a cron;
    a GitHub outage should degrade to last-good rather than fail the run.
    """
    if refresh or not NFL_CACHE.exists():
        try:
            NFL_CACHE.write_bytes(_curl(NFL_GAMES_URL))
        except Exception as exc:
            if not NFL_CACHE.exists():
                raise
            logger.warning("nflverse fetch failed (%s); using cached copy", exc)

    raw = pd.read_csv(NFL_CACHE)
    g = raw[raw["game_type"].isin(["REG", "POST"])].copy()
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


def load_mlb(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Completed regular-season MLB games from the public StatsAPI.

    Baseball has no neutral sites worth modelling and no divisions in the
    Bradley-Terry sense (the AL/NL split is not a strength tier — interleague play is
    frequent enough to tie them to one scale), so both columns are constant.

    `week` is the index of the ISO week within the season, used only so the prior
    decay has something to count.
    """
    frames = []
    for season in seasons:
        cached = CACHE / f"mlb_games_{season}.json"
        if cached.exists():
            payload = json.loads(cached.read_text())
        else:
            payload = json.loads(_curl(MLB_SCHEDULE.format(season=season)).decode())
            cached.write_text(json.dumps(payload))

        rows = []
        for date in payload.get("dates", []):
            for game in date.get("games", []):
                home, away = game["teams"]["home"], game["teams"]["away"]
                hs, as_ = home.get("score"), away.get("score")
                state = (game.get("status") or {}).get("detailedState")
                if state != "Final" or hs is None or as_ is None or hs == as_:
                    continue
                rows.append({
                    "season": season,
                    "game_date": date["date"],
                    "home_team_id": home["team"]["id"],
                    "away_team_id": away["team"]["id"],
                    "home_team_name": home["team"]["name"],
                    "away_team_name": away["team"]["name"],
                    "home_won": int(hs > as_),
                    "neutral_site": 0,
                })
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
