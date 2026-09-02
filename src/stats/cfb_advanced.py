"""College football advanced stats, players and betting lines from CollegeFootballData.

Fills three gaps ESPN could not. In each case the limitation was real and is documented
in the modules that hit it — this is not a rewrite of working code:

  advanced stats  the college pipeline carries no play-by-play, so it had no EPA
                  equivalent of nfl_historical.team_week_epa at all.
  player stats    ESPN's sortable athlete endpoint returns "-" for the very stat it
                  sorts on (see stats/build.py), so college was leaders-only.
  betting lines   ESPN's free feed carries none, so cfb/pipeline.py nulled spread_line
                  and the site's Vegas baseline was hardcoded NULL for college.

ESPN stays the source of record for games, polls, conference structure and FPI. Joins
are on `game_id`, never on team name: CFBD is internally inconsistent about names
(`/scoreboard` says "TCU Horned Frogs" and matches the BigQuery key, `/games` says
"TCU" and does not), while the game id is ESPN's in every one of them.
"""

from __future__ import annotations

import logging

import pandas as pd

try:
    from stats import cfbd
except ImportError:  # pragma: no cover - flat module tree in Cloud Functions
    import cfbd  # type: ignore

logger = logging.getLogger(__name__)

TEXT_TEAM_COLUMNS = ("team", "conference", "opponent", "opponent_conference",
                     "classification", "season_type")


def _current(season: int, ttl_hours: float = 12.0) -> float | None:
    """Completed seasons cache forever; the live one gets a TTL."""
    import datetime

    return ttl_hours if season >= datetime.date.today().year else None


# --------------------------------------------------------------------------- #
# Team advanced stats
# --------------------------------------------------------------------------- #

def fetch_team_season_advanced(season: int) -> pd.DataFrame:
    """Season-level advanced stats, both divisions.

    `classification` genuinely filters here, so it is two calls — unlike the per-game
    endpoint below, where the same parameter is ignored.
    """
    rows: list[dict] = []
    for classification in cfbd.SITE_CLASSIFICATIONS:
        payload = cfbd.get(
            "/stats/season/advanced",
            {"year": season, "classification": classification},
            ttl_hours=_current(season),
        )
        for record in payload or []:
            row = {
                "season": record.get("season", season),
                "team": record.get("team"),
                "conference": record.get("conference"),
                "classification": classification,
            }
            row.update(cfbd.split_off_def(record))
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = cfbd.as_frame(rows, text_columns=TEXT_TEAM_COLUMNS)
    logger.info("cfb team season advanced: %d teams, %d columns", len(df), len(df.columns))
    return df


def fetch_team_game_advanced(season: int) -> pd.DataFrame:
    """Per-team, per-game advanced stats — the week-grain table the site's stats page wants.

    One call covers the whole season. Note `classification` is accepted and IGNORED by
    this endpoint: asking for fbs and fcs returns byte-identical payloads, so calling it
    twice would double the spend for nothing. Verified 2026-09-02.
    """
    payload = cfbd.get(
        "/stats/game/advanced",
        {"year": season},
        ttl_hours=_current(season),
    )

    rows: list[dict] = []
    for record in payload or []:
        row = {
            "season": record.get("season", season),
            "week": record.get("week"),
            "season_type": record.get("seasonType"),
            "game_id": record.get("gameId"),
            "team": record.get("team"),
            "opponent": record.get("opponent"),
        }
        row.update(cfbd.split_off_def(record))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = cfbd.as_frame(rows, text_columns=TEXT_TEAM_COLUMNS + ("game_id",))
    # game_id is the join key to cfb_historical.games and cfb_season.game_predictions,
    # and must stay a string: those tables key on ESPN's id as text.
    df["game_id"] = df["game_id"].astype("string")
    logger.info("cfb team game advanced: %d team-games, %d columns",
                len(df), len(df.columns))
    return df


def fetch_team_season_epa(season: int) -> pd.DataFrame:
    """Opponent-adjusted season EPA.

    Named for its grain: this is SEASON-level, so it is not the analogue of
    nfl_historical.team_week_epa despite the similar field names. Calling it
    team_week_epa would invite feeding season-final numbers into a week-5 prediction,
    which is leakage. Requires a paid tier; the caller treats a 401 as a skipped step.
    """
    payload = cfbd.get("/wepa/team/season", {"year": season},
                       ttl_hours=_current(season))

    rename = {
        "epa": "epa", "epa_allowed": "opp_epa",
        "success_rate": "success_rate", "success_rate_allowed": "opp_success_rate",
        "rushing": "rushing_epa", "rushing_allowed": "opp_rushing_epa",
        "explosiveness": "explosiveness", "explosiveness_allowed": "opp_explosiveness",
    }

    rows = []
    for record in payload or []:
        flat = cfbd.flatten(record)
        row = {
            "season": flat.get("year", season),
            "team": flat.get("team"),
            "conference": flat.get("conference"),
        }
        # Normalised into the same own/opp_ convention as every other CFB table, so
        # "allowed" never reaches BigQuery in two different shapes.
        for src, dest in rename.items():
            if src in flat:
                row[dest] = flat[src]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return cfbd.as_frame(rows, text_columns=TEXT_TEAM_COLUMNS)


# --------------------------------------------------------------------------- #
# Player stats — the ESPN dead end
# --------------------------------------------------------------------------- #

def fetch_player_season(season: int) -> pd.DataFrame:
    """Full per-player season table, pivoted from CFBD's long form."""
    payload = cfbd.get("/stats/player/season", {"year": season},
                       ttl_hours=_current(season))
    df = cfbd.pivot_player_season(payload or [])
    if df.empty:
        return df

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("int64")
    for col in ("player_id", "player_name", "position", "team", "conference"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    logger.info("cfb player season: %d players, %d columns", len(df), len(df.columns))
    return df


# Categories worth a leaderboard, with the direction that counts as better and a
# volume floor. ESPN's leaders feed can express neither, which is why its version of
# stat_leaders lacks higher_is_better and why a "leader" in a rate stat there can be
# someone with three attempts.
LEADER_CATEGORIES: list[dict] = [
    {"key": "passing_yds", "label": "Passing Yards", "higher": True},
    {"key": "passing_td", "label": "Passing TD", "higher": True},
    {"key": "passing_ypa", "label": "Yards per Attempt", "higher": True,
     "min_col": "passing_att", "min_value": 200},
    # CFBD quotes completion percentage as a 0-1 rate where ESPN used 0-100, so it
    # needs scaling for display or the board reads "0.77".
    {"key": "passing_pct", "label": "Completion %", "higher": True,
     "min_col": "passing_att", "min_value": 200, "as_pct": True},
    # Ascending boards need a floor or the "leader" is whoever threw two passes.
    {"key": "passing_int", "label": "Fewest Interceptions", "higher": False,
     "min_col": "passing_att", "min_value": 200},
    {"key": "rushing_yds", "label": "Rushing Yards", "higher": True},
    {"key": "rushing_td", "label": "Rushing TD", "higher": True},
    {"key": "rushing_ypc", "label": "Yards per Carry", "higher": True,
     "min_col": "rushing_car", "min_value": 100},
    {"key": "receiving_yds", "label": "Receiving Yards", "higher": True},
    {"key": "receiving_rec", "label": "Receptions", "higher": True},
    {"key": "receiving_td", "label": "Receiving TD", "higher": True},
    {"key": "receiving_ypr", "label": "Yards per Reception", "higher": True,
     "min_col": "receiving_rec", "min_value": 40},
    {"key": "defensive_tot", "label": "Total Tackles", "higher": True},
    {"key": "defensive_sacks", "label": "Sacks", "higher": True},
    {"key": "defensive_tfl", "label": "Tackles for Loss", "higher": True},
    {"key": "interceptions_int", "label": "Interceptions", "higher": True},
    {"key": "kicking_fgm", "label": "Field Goals Made", "higher": True},
]


def leaders_from_players(players: pd.DataFrame, season: int,
                        limit: int = 25) -> pd.DataFrame:
    """Derive leaderboards from the player table rather than a separate feed.

    Deliberately the same source as the player lookup: a leaderboard that disagrees
    with the table beneath it is worse than no leaderboard. This is the principle
    stats/nfl_stats.py already states, now available for college too.
    """
    if players.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for spec in LEADER_CATEGORIES:
        col = spec["key"]
        if col not in players.columns:
            continue
        subset = players[players[col].notna()]

        # The volume qualifier. Without it a rate-stat board — and any ascending board
        # like fewest interceptions — is topped by whoever barely played, which is the
        # specific thing ESPN's leaders feed cannot express and the reason these are
        # derived here rather than taken from it.
        floor_col = spec.get("min_col")
        if floor_col and floor_col in subset.columns:
            subset = subset[subset[floor_col].fillna(0) >= spec["min_value"]]
        elif floor_col:
            logger.info("%s: qualifier column %s absent, board skipped",
                        col, floor_col)
            continue

        if subset.empty:
            continue
        ranked = subset.sort_values(col, ascending=not spec["higher"]).head(limit)
        for rank, (_, r) in enumerate(ranked.iterrows(), start=1):
            rows.append({
                "season": season,
                "sport": "cfb",
                "category": col,
                "category_label": spec["label"],
                "higher_is_better": spec["higher"],
                "qualifier": (f"min {spec['min_value']} {spec['min_col']}"
                              if spec.get("min_col") else None),
                "rank": rank,
                "value": float(r[col]),
                "display_value": (f"{r[col] * 100:.1f}%" if spec.get("as_pct")
                                  else f"{r[col]:g}"),
                "player_id": r.get("player_id"),
                "player_name": r.get("player_name"),
                "position": r.get("position"),
                "team": r.get("team"),
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ("sport", "category", "category_label", "display_value",
                "qualifier", "player_id", "player_name", "position", "team"):
        df[col] = df[col].astype("string")
    return df


# --------------------------------------------------------------------------- #
# Betting lines
# --------------------------------------------------------------------------- #

def fetch_lines(season: int) -> pd.DataFrame:
    """Consensus betting lines per game.

    Two things matter here and both are easy to get wrong.

    Sign: CFBD quotes a spread that is negative when the home side is favoured;
    nflverse — and therefore this stack's stored `spread_line`, and the backend
    expression that scores the Vegas baseline — is positive when the home side is
    favoured. So the sign is flipped once, here, at ingest.

    Fan-out: a game carries one row per sportsbook. Left-joining that to predictions
    would multiply rows and quietly corrupt every aggregate while still returning
    success, so providers are collapsed to a median before anything downstream sees it.
    """
    payload = cfbd.get("/lines", {"year": season}, ttl_hours=_current(season))

    rows: list[dict] = []
    for game in payload or []:
        quotes = game.get("lines") or []
        spreads = [q["spread"] for q in quotes if q.get("spread") is not None]
        totals = [q["overUnder"] for q in quotes if q.get("overUnder") is not None]
        opens = [q["spreadOpen"] for q in quotes if q.get("spreadOpen") is not None]

        if not spreads and not totals:
            continue

        rows.append({
            "game_id": str(game.get("id")),
            "season": game.get("season", season),
            "week": game.get("week"),
            "home_team": game.get("homeTeam"),
            "away_team": game.get("awayTeam"),
            # Flipped to the stack's convention: positive means home favoured.
            "spread_line": -pd.Series(spreads).median() if spreads else None,
            "spread_open": -pd.Series(opens).median() if opens else None,
            "total_line": pd.Series(totals).median() if totals else None,
            "provider_count": len(quotes),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ("game_id", "home_team", "away_team"):
        df[col] = df[col].astype("string")
    for col in ("season", "week", "provider_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("spread_line", "spread_open", "total_line"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("cfb lines: %d games", len(df))
    return df


def assert_spread_sign(lines: pd.DataFrame, games: pd.DataFrame) -> float:
    """Fail loudly if the spread sign convention is inverted.

    Without this check an inverted sign produces a Vegas baseline near 50% that looks
    like a weak-but-plausible model result rather than a bug. Correlating the stored
    line against the actual margin is the cheapest way to be certain, and it costs one
    query on a completed season.
    """
    merged = lines.merge(
        games[["game_id", "home_score", "away_score"]], on="game_id", how="inner"
    )
    merged = merged[merged["spread_line"].notna() & merged["home_score"].notna()]
    if len(merged) < 50:
        logger.info("spread sign check skipped: only %d scored games", len(merged))
        return float("nan")

    margin = merged["home_score"] - merged["away_score"]
    corr = float(margin.corr(merged["spread_line"]))
    if corr <= 0:
        raise ValueError(
            f"spread_line correlates {corr:.3f} with the home margin — the sign is "
            "inverted. Positive spread_line must mean the home side is favoured."
        )
    logger.info("spread sign check passed (corr %.3f with home margin)", corr)
    return corr
