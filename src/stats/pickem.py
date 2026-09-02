"""The unified pickable-game table behind the pick'em contest.

One table, both sports. The picker, the kickoff lock and the grading all read it, so
none of them has to know where a sport's schedule came from — which matters because the
two sources have nothing in common: the NFL schedule is an nflverse parquet keyed on
team abbreviations with Eastern kickoff times, and the college one is a
CollegeFootballData endpoint keyed on school names with UTC timestamps.

Lives in src/stats/ for the same reason cfbd.py does: both deploy scripts already stage
`stats/` as a package into both Cloud Function images, so this needs no deploy-script
change. Each sport's Cloud Function writes only its own slice.

Why BigQuery rather than a document store: grading a pick means joining it to the game's
result and its closing spread, and both already live here. Keeping picks in the same
warehouse turns scoring into a view — always current, with no scheduled job that can
fail — instead of an export, a join elsewhere, and a write back.
"""

from __future__ import annotations

import datetime as dt
import logging
import os

import pandas as pd

try:
    from stats import cfbd
except ImportError:  # pragma: no cover - flat module tree in Cloud Functions
    import cfbd  # type: ignore

logger = logging.getLogger(__name__)

PICKEM_DATASET = os.environ.get("PICKEM_DATASET", "pickem")

# The column contract. Every writer produces exactly these, in this order, so the table
# stays one shape regardless of which sport wrote last.
COLUMNS: list[tuple[str, str]] = [
    ("sport", "string"),
    ("division", "string"),
    ("season", "Int64"),
    ("week", "Int64"),
    ("game_id", "string"),
    ("kickoff", "datetime64[ns, UTC]"),
    ("start_time_tbd", "boolean"),
    ("home_team", "string"),
    ("away_team", "string"),
    ("home_display", "string"),
    ("away_display", "string"),
    ("home_conference", "string"),
    ("away_conference", "string"),
    ("neutral_site", "boolean"),
    ("spread_line", "float64"),
    ("total_line", "float64"),
    ("home_score", "Int64"),
    ("away_score", "Int64"),
    ("completed", "boolean"),
    # Per-side context for the pick sheet. Attached at ingest rather than joined per
    # request: the sheet is the most-requested endpoint, and the college join needs a
    # name reconstruction that is better done once in Python than in every query.
    ("home_rank", "Int64"),
    ("away_rank", "Int64"),
    ("home_rating", "float64"),
    ("away_rating", "float64"),
    ("home_record", "string"),
    ("away_record", "string"),
    # Which season the record is FROM. The preseason board carries last year's, and
    # "14-3" beside a week 1 game reads as this year's unless it says otherwise.
    ("home_record_season", "Int64"),
    ("away_record_season", "Int64"),
    ("home_ap_rank", "Int64"),
    ("away_ap_rank", "Int64"),
    ("home_coaches_rank", "Int64"),
    ("away_coaches_rank", "Int64"),
    ("home_fpi", "float64"),
    ("away_fpi", "float64"),
    ("home_fpi_rank", "Int64"),
    ("away_fpi_rank", "Int64"),
    ("home_streak", "string"),
    ("away_streak", "string"),
    # The site's own model, where it has published a pick for this game.
    ("model_home_win_prob", "float64"),
    ("model_pick", "string"),
    ("model_confidence", "string"),
]


def _conform(rows: list[dict]) -> pd.DataFrame:
    """Pin the frame to the column contract so BigQuery autodetect is deterministic."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for name, dtype in COLUMNS:
        if name not in df.columns:
            df[name] = None
        try:
            df[name] = df[name].astype(dtype)
        except (TypeError, ValueError):
            # A column that will not cast is more useful as nulls of the right type
            # than as a load failure at 7am; the contract is what downstream reads.
            logger.warning("pickem.games: could not cast %s to %s", name, dtype)
            df[name] = pd.Series([None] * len(df), dtype=dtype)
    return df[[c for c, _ in COLUMNS]]


# --------------------------------------------------------------------------- #
# NFL
# --------------------------------------------------------------------------- #

NFLVERSE_SCHEDULE = (
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
)

# nflverse quotes kickoff as a local date plus a wall-clock time, in US Eastern. Stored
# as UTC because the lock has to be an instant, not a time in somebody's timezone.
NFL_TZ = "America/New_York"


def fetch_nfl_games(season: int) -> pd.DataFrame:
    """Every NFL game of a season, played or not, with kickoff and the closing line."""
    import io as _io

    try:
        from rankings.http import get_bytes
    except ImportError:  # pragma: no cover
        from http_transport import get_bytes  # type: ignore

    raw = get_bytes(NFLVERSE_SCHEDULE, timeout=120)
    sched = pd.read_parquet(_io.BytesIO(raw))
    sched = sched[sched["season"] == season]
    if sched.empty:
        logger.warning("nflverse has no %d schedule yet", season)
        return pd.DataFrame()

    # Full names where we have them; the schedule itself only carries abbreviations.
    names = _nfl_team_names()

    rows = []
    for r in sched.to_dict("records"):
        rows.append({
            "sport": "nfl",
            "division": None,
            "season": r.get("season"),
            "week": r.get("week"),
            "game_id": str(r.get("game_id")),
            "kickoff": _nfl_kickoff(r.get("gameday"), r.get("gametime")),
            "start_time_tbd": pd.isna(r.get("gametime")),
            "home_team": r.get("home_team"),
            "away_team": r.get("away_team"),
            "home_display": names.get(r.get("home_team"), r.get("home_team")),
            "away_display": names.get(r.get("away_team"), r.get("away_team")),
            "home_conference": None,
            "away_conference": None,
            "neutral_site": str(r.get("location", "")).lower() == "neutral",
            # Already this stack's convention: positive means the home side is favoured.
            "spread_line": r.get("spread_line"),
            "total_line": r.get("total_line"),
            "home_score": r.get("home_score"),
            "away_score": r.get("away_score"),
            "completed": pd.notna(r.get("result")),
        })

    df = _conform(rows)
    logger.info("pickem nfl %d: %d games (%d unplayed)",
                season, len(df), int((~df["completed"].fillna(False)).sum()))
    return df


def _nfl_team_names() -> dict:
    """Abbreviation -> full name, from the teams table the pipeline already writes."""
    try:
        from google.cloud import bigquery

        project = os.environ.get("GCP_PROJECT", "hankstank")
        dataset = os.environ.get("NFL_HIST_DATASET", "nfl_historical")
        client = bigquery.Client(project=project)
        rows = client.query(
            f"SELECT team_abbr, team_name FROM `{project}.{dataset}.teams`"
        ).result()
        return {r["team_abbr"]: r["team_name"] for r in rows}
    except Exception as exc:
        # Cosmetic only — the abbreviation is still a usable label.
        logger.info("could not load NFL team names (%s)", str(exc)[:120])
        return {}


def _nfl_kickoff(gameday, gametime) -> pd.Timestamp | None:
    if gameday is None or pd.isna(gameday):
        return None
    day = str(gameday)[:10]
    clock = "13:00" if gametime is None or pd.isna(gametime) else str(gametime)[:5]
    try:
        naive = pd.Timestamp(f"{day} {clock}")
    except ValueError:
        return None
    return naive.tz_localize(NFL_TZ, nonexistent="shift_forward",
                             ambiguous=True).tz_convert("UTC")


# --------------------------------------------------------------------------- #
# College
# --------------------------------------------------------------------------- #

def fetch_cfb_games(season: int, divisions: tuple[str, ...] = ("fbs",)) -> pd.DataFrame:
    """College games for the divisions the contest covers.

    FBS only by default. A pick sheet of several hundred games is not a pick sheet, and
    the college slate runs past 300 a week once the lower divisions are included.
    """
    rows: list[dict] = []
    for division in divisions:
        payload = cfbd.get(
            "/games",
            {"year": season, "classification": division},
            # The current season's schedule still moves — kickoff times get set, games
            # get flexed — so it is refetched rather than cached forever.
            ttl_hours=6.0 if season >= dt.date.today().year else None,
        )
        for g in payload or []:
            # A game is in scope if either side is in a covered division; a
            # cross-division opponent is still a pickable game.
            if g.get("homeClassification") not in divisions and \
               g.get("awayClassification") not in divisions:
                continue
            rows.append({
                "sport": "cfb",
                "division": division,
                "season": g.get("season", season),
                "week": g.get("week"),
                "game_id": str(g.get("id")),
                "kickoff": pd.Timestamp(g["startDate"]).tz_convert("UTC")
                if g.get("startDate") else None,
                "start_time_tbd": bool(g.get("startTimeTBD")),
                "home_team": g.get("homeTeam"),
                "away_team": g.get("awayTeam"),
                "home_display": g.get("homeTeam"),
                "away_display": g.get("awayTeam"),
                "home_conference": g.get("homeConference"),
                "away_conference": g.get("awayConference"),
                "neutral_site": bool(g.get("neutralSite")),
                # Filled from the lines table below: /games carries no spread.
                "spread_line": None,
                "total_line": None,
                "home_score": g.get("homePoints"),
                "away_score": g.get("awayPoints"),
                "completed": bool(g.get("completed")),
            })

    if not rows:
        return pd.DataFrame()

    df = _conform(rows).drop_duplicates(subset=["game_id"], keep="first")
    df = _attach_cfb_lines(df, season)
    logger.info("pickem cfb %d: %d games (%d unplayed, %d with a spread)",
                season, len(df), int((~df["completed"].fillna(False)).sum()),
                int(df["spread_line"].notna().sum()))
    return df


def _attach_cfb_lines(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Join the consensus spread already ingested by cfb_advanced.fetch_lines.

    Read from BigQuery rather than refetched: the sign has already been flipped to this
    stack's convention there and verified against outcomes, and doing it twice invites
    the two copies to disagree.
    """
    try:
        from google.cloud import bigquery

        project = os.environ.get("GCP_PROJECT", "hankstank")
        dataset = os.environ.get("CFB_DATASET", "cfb_season")
        client = bigquery.Client(project=project)
        lines = client.query(f"""
            SELECT game_id, spread_line, total_line
            FROM `{project}.{dataset}.betting_lines`
            WHERE season = {int(season)}
        """).to_dataframe()
    except Exception as exc:
        logger.info("no CFB lines to attach (%s)", str(exc)[:120])
        return df

    if lines.empty:
        return df

    lines["game_id"] = lines["game_id"].astype("string")
    merged = df.drop(columns=["spread_line", "total_line"]).merge(
        lines, on="game_id", how="left"
    )
    return _conform(merged.to_dict("records"))


# --------------------------------------------------------------------------- #
# Per-side context: rankings, polls, FPI, form
# --------------------------------------------------------------------------- #

def _bq_client():
    from google.cloud import bigquery

    return bigquery.Client(project=os.environ.get("GCP_PROJECT", "hankstank"))


def _cfb_name_map(season: int) -> dict:
    """CFBD school name -> the ESPN display name the rankings board keys on.

    The two feeds disagree: the schedule says "TCU", the board says "TCU Horned Frogs".
    Reconstructing `school + " " + mascot` from /teams resolves all 138 schools on the
    2026 sheet exactly, with no override list — and exactly is the point, because a
    prefix match would tie "Miami" to both "Miami Hurricanes" and "Miami (OH)
    RedHawks".
    """
    payload = cfbd.get("/teams", {"year": season}, ttl_hours=None)
    out = {}
    for t in payload or []:
        if t.get("classification") not in cfbd.SITE_CLASSIFICATIONS:
            continue
        school, mascot = t.get("school"), t.get("mascot")
        if not school:
            continue
        out[school] = f"{school} {mascot}".strip() if mascot else school
    return out


def _rankings(sport: str, season: int) -> dict:
    """Team key -> the board row, keyed the way pickem.games names teams.

    NFL abbreviations match the board directly; college needs the name map above.
    """
    project = os.environ.get("GCP_PROJECT", "hankstank")
    dataset = os.environ.get(
        "NFL_DATASET" if sport == "nfl" else "CFB_DATASET",
        "nfl_season" if sport == "nfl" else "cfb_season",
    )
    # SELECT * because the boards differ by sport: there is no AP or coaches poll in
    # the NFL, so naming those columns fails the whole query for that sport rather than
    # returning nulls. The tables are a few hundred rows, so the cost is nothing.
    try:
        rows = list(_bq_client().query(f"""
            SELECT *
            FROM `{project}.{dataset}.power_rankings`
            WHERE season = {int(season)}
              AND as_of_week = (
                SELECT MAX(as_of_week) FROM `{project}.{dataset}.power_rankings`
                WHERE season = {int(season)}
              )
        """).result())
    except Exception as exc:
        logger.info("no rankings to attach for %s (%s)", sport, str(exc)[:120])
        return {}

    # dict(r) keeps whatever the board actually has; missing fields read as None
    # through .get() downstream rather than failing.
    board = {r["team"]: dict(r) for r in rows}
    if sport == "nfl":
        return board

    # Translate the board onto the schedule's own naming.
    name_map = _cfb_name_map(season)
    resolved, unresolved = {}, []
    for school, display in name_map.items():
        if display in board:
            resolved[school] = board[display]
        else:
            unresolved.append(school)
    if unresolved:
        # Logged rather than raised: a team with no board row still has a pickable
        # game, it just shows without context.
        logger.info("%d schools have no rankings row (e.g. %s)",
                    len(unresolved), unresolved[:5])
    return resolved


def _streaks(df: pd.DataFrame) -> dict:
    """Team -> current form, like "W3" or "L2".

    Computed from the completed games in this very frame rather than from another
    table, so it needs no name translation and cannot disagree with the schedule it is
    displayed beside. A season's own results are also what a streak means — carrying one
    across an offseason would be misleading.
    """
    played = df[df["completed"].fillna(False)].copy()
    if played.empty:
        return {}
    played = played.sort_values("kickoff")

    results: dict[str, list[str]] = {}
    for r in played.to_dict("records"):
        hs, as_ = r.get("home_score"), r.get("away_score")
        if pd.isna(hs) or pd.isna(as_):
            continue
        if hs == as_:
            outcome_home = outcome_away = "T"
        else:
            outcome_home = "W" if hs > as_ else "L"
            outcome_away = "L" if hs > as_ else "W"
        results.setdefault(r["home_team"], []).append(outcome_home)
        results.setdefault(r["away_team"], []).append(outcome_away)

    out = {}
    for team, seq in results.items():
        last = seq[-1]
        run = 0
        for o in reversed(seq):
            if o != last:
                break
            run += 1
        out[team] = f"{last}{run}"
    return out


def _model_picks(sport: str, season: int) -> dict:
    """game_id -> the site's own prediction, where one has been published."""
    project = os.environ.get("GCP_PROJECT", "hankstank")
    dataset = os.environ.get(
        "NFL_DATASET" if sport == "nfl" else "CFB_DATASET",
        "nfl_season" if sport == "nfl" else "cfb_season",
    )
    try:
        rows = list(_bq_client().query(f"""
            SELECT game_id, home_win_probability, predicted_winner, confidence_tier
            FROM `{project}.{dataset}.game_predictions`
            WHERE season = {int(season)}
        """).result())
    except Exception as exc:
        logger.info("no model picks to attach for %s (%s)", sport, str(exc)[:120])
        return {}
    return {str(r["game_id"]): dict(r) for r in rows}


def attach_context(sport: str, season: int, df: pd.DataFrame) -> pd.DataFrame:
    """Add per-side rankings, polls, FPI and form to a games frame."""
    if df.empty:
        return df

    board = _rankings(sport, season)
    streaks = _streaks(df)
    models = _model_picks(sport, season)

    def side(key: str, field: str):
        return [(board.get(t) or {}).get(field) for t in df[key]]

    for prefix, key in (("home", "home_team"), ("away", "away_team")):
        df[f"{prefix}_rank"] = side(key, "rank")
        df[f"{prefix}_rating"] = side(key, "rating")
        df[f"{prefix}_record"] = side(key, "record")
        df[f"{prefix}_record_season"] = side(key, "record_season")
        df[f"{prefix}_ap_rank"] = side(key, "ap_rank")
        df[f"{prefix}_coaches_rank"] = side(key, "coaches_rank")
        df[f"{prefix}_fpi"] = side(key, "fpi")
        df[f"{prefix}_fpi_rank"] = side(key, "fpi_rank")
        df[f"{prefix}_streak"] = [streaks.get(t) for t in df[key]]

    df["model_home_win_prob"] = [
        (models.get(g) or {}).get("home_win_probability") for g in df["game_id"]
    ]
    df["model_pick"] = [
        (models.get(g) or {}).get("predicted_winner") for g in df["game_id"]
    ]
    df["model_confidence"] = [
        (models.get(g) or {}).get("confidence_tier") for g in df["game_id"]
    ]

    attached = int(df["home_rank"].notna().sum())
    logger.info("pickem %s: context on %d/%d games, %d with a model pick",
                sport, attached, len(df), int(df["model_pick"].notna().sum()))
    return _conform(df.to_dict("records"))


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #

def write_games(sport: str, season: int, df: pd.DataFrame) -> dict:
    """Replace this sport's slice of a season, leaving every other slice alone.

    Scoped by (sport, season) because the two sports are written by two different Cloud
    Functions on different days. A truncating write would mean whichever ran last
    deleted the other's games — the same failure the CFB games table already suffered
    once from a WRITE_TRUNCATE default.
    """
    from google.cloud import bigquery

    if df.empty:
        return {"sport": sport, "season": season, "rows": 0, "skipped": "no rows"}

    project = os.environ.get("GCP_PROJECT", "hankstank")
    client = bigquery.Client(project=project)

    ref = bigquery.Dataset(f"{project}.{PICKEM_DATASET}")
    ref.location = "US"
    try:
        client.get_dataset(ref)
    except Exception:
        client.create_dataset(ref)
        logger.info("created dataset %s", PICKEM_DATASET)

    table_id = f"{project}.{PICKEM_DATASET}.games"
    try:
        client.query(
            f"DELETE FROM `{table_id}` WHERE sport = @sport AND season = @season",
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("sport", "STRING", sport),
                bigquery.ScalarQueryParameter("season", "INT64", int(season)),
            ]),
        ).result()
    except Exception as exc:
        logger.info("%s: pre-delete skipped (%s)", table_id, str(exc)[:120])

    client.load_table_from_dataframe(
        df, table_id,
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            schema_update_options=[
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
            ],
        ),
    ).result()
    logger.info("loaded %d %s rows -> %s", len(df), sport, table_id)
    return {"sport": sport, "season": season, "rows": len(df), "table": table_id}


def refresh(sport: str, season: int) -> dict:
    """Fetch, enrich and write one sport's slice. The Cloud Function entry point."""
    if sport == "nfl":
        games = fetch_nfl_games(season)
    elif sport == "cfb":
        games = fetch_cfb_games(season)
    else:
        raise ValueError(f"unknown sport: {sport}")

    # Never fatal: a sheet without rankings is still a usable sheet, and losing the
    # schedule because a board was missing would be the wrong trade.
    try:
        games = attach_context(sport, season, games)
    except Exception as exc:
        logger.warning("could not attach context for %s: %s", sport, str(exc)[:200])

    return write_games(sport, season, games)
