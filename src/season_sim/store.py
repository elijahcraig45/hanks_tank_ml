"""BigQuery writes for the season sim, and the `season_sim` mode both functions call.

Safety rules (the football functions have overwritten production data before):
  * dry_run computes everything and returns a summary; no BigQuery client is created.
  * Writes are scoped to exactly one (season, as_of_week): DELETE that slice, then append.
    Other weeks and seasons are never touched, so a rerun replaces only itself.
  * Tables must already exist (CREATE_NEVER), with the schema from
    scripts/gcp/football/create_season_sim_tables.sql; the load uses that schema rather
    than autodetecting one.
  * Every table's schema is checked BEFORE anything is deleted: a frame column the table
    lacks (the ALTER in scripts/gcp/football/alter_season_sim_per_game.sql not yet run)
    fails the whole write with nothing touched, rather than silently dropping columns.
  * Only the three season_sim tables are ever written. Nothing reads them but the
    backend's /api/season-sim route.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from . import run

logger = logging.getLogger(__name__)

TEAM_TABLE = "season_sim_team"
BRACKET_TABLE = "season_sim_bracket"
GAMES_TABLE = "season_sim_games"
MAX_SIMS = 20_000


def _check_scope(df: pd.DataFrame, season: int, as_of_week: int) -> None:
    if df.empty:
        return
    if set(df["season"].unique()) != {season} or set(df["as_of_week"].unique()) != {as_of_week}:
        raise ValueError("season_sim rows must all belong to one (season, as_of_week)")


def write(team_df: pd.DataFrame, bracket_df: pd.DataFrame, project: str, dataset: str,
          season: int, as_of_week: int, client=None, games_df: pd.DataFrame | None = None
          ) -> dict:
    """Replace the (season, as_of_week) slice of each table."""
    from google.cloud import bigquery

    frames = [(TEAM_TABLE, team_df), (BRACKET_TABLE, bracket_df)]
    if games_df is not None:
        frames.append((GAMES_TABLE, games_df))
    for _, df in frames:
        _check_scope(df, season, as_of_week)
    client = client or bigquery.Client(project=project)
    schemas = {}
    for table, df in frames:  # all checks first: nothing is deleted if any table is wrong
        table_id = f"{project}.{dataset}.{table}"
        schema = client.get_table(table_id).schema  # raises if the DDL was never run
        missing = [c for c in df.columns if c not in {f.name for f in schema}]
        if missing:
            raise ValueError(f"{table_id} lacks columns {missing}: run "
                             "scripts/gcp/football/alter_season_sim_per_game.sql")
        schemas[table] = schema
    written = {}
    for table, df in frames:
        table_id = f"{project}.{dataset}.{table}"
        schema = schemas[table]
        client.query(
            f"DELETE FROM `{table_id}` WHERE season = @season AND as_of_week = @week",
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("season", "INT64", season),
                bigquery.ScalarQueryParameter("week", "INT64", as_of_week),
            ]),
        ).result()
        if df.empty:
            written[table] = 0
            continue
        cols = [f.name for f in schema]
        job = client.load_table_from_dataframe(
            df[cols], table_id,
            job_config=bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND", create_disposition="CREATE_NEVER",
                schema=schema),
        )
        job.result()
        written[table] = len(df)
        logger.info("season_sim: %d rows -> %s (%d wk%d)", len(df), table_id, season, as_of_week)
    return written


def summary(team_df: pd.DataFrame, k: int = 10) -> list[dict]:
    top = team_df.sort_values("p_champion", ascending=False).head(k)
    return [{"team": r.team, "record": f"{r.wins}-{r.losses}" + (f"-{r.ties}" if r.ties else ""),
             "mean_wins": round(r.mean_wins, 2), "p_playoffs": round(r.p_playoffs, 3),
             "p_champion": round(r.p_champion, 4)} for r in top.itertuples()]


def run_mode(sport: str, games: pd.DataFrame, season: int, project: str, dataset: str,
             req: dict, dry_run: bool, client=None) -> dict:
    """Simulate, then write (or not). `games` is the sport's schedule/games frame."""
    n_sims = min(int(req.get("n_sims", run.DEFAULT_SIMS)), MAX_SIMS)
    variant = str(req.get("variant", run.DEFAULT_VARIANT))
    week = req.get("week")
    as_of = int(week) if week is not None else None
    started = postseason_started(sport, games, season)
    if started:
        # The bracket model would re-simulate playoff games that have been played.
        return {"season": season, "skipped": f"{sport} postseason under way ({started})",
                "written": 0, "dry_run": dry_run}
    t0 = time.time()
    if sport == "nfl":
        o = run.sim_nfl(games, season, as_of, n_sims=n_sims, variant=variant,
                        seed=int(req.get("seed", season * 100 + (as_of or 0))))
    else:
        o = run.sim_cfb(games, season, as_of, n_sims=n_sims, variant=variant,
                        seed=int(req.get("seed", season * 100 + (as_of or 0))))
    computed_at = pd.Timestamp.now(tz="UTC")
    team_df, bracket_df = run.tables(o, computed_at=computed_at)
    games_df = run.games_table(o, computed_at=computed_at)
    info = {"season": season, "as_of_week": o.as_of_week, "n_sims": n_sims,
            "variant": variant, "teams": len(team_df), "bracket_rows": len(bracket_df),
            "game_rows": len(games_df),
            "seconds": round(time.time() - t0, 2),
            "runtime": {k: round(float(v), 3) for k, v in o.runtime.items()},
            "top": summary(team_df)}
    if dry_run:
        return {**info, "dry_run": True, "written": 0}
    info["written"] = write(team_df, bracket_df, project, dataset, season, o.as_of_week,
                            client=client, games_df=games_df)
    return info


def postseason_started(sport: str, games: pd.DataFrame, season: int) -> str | None:
    """Name of the first played postseason game type, or None."""
    cur = games[pd.to_numeric(games["season"]) == season]
    if sport == "nfl":
        played = pd.to_numeric(cur["result"], errors="coerce").notna()
        post = cur[played & (cur["game_type"] != "REG")]
        return str(post["game_type"].iloc[0]) if len(post) else None
    played = pd.to_numeric(cur["result"], errors="coerce").notna()
    post = cur[played & (pd.to_numeric(cur["is_postseason"]).fillna(0) == 1)]
    return "bowl or CFP game" if len(post) else None


def as_of_week_from(games: pd.DataFrame, season: int, week_col: str = "week",
                    played_col: str = "margin") -> int | None:
    cur = games[(pd.to_numeric(games["season"]) == season)]
    played = pd.to_numeric(cur[played_col], errors="coerce").notna()
    return int(pd.to_numeric(cur.loc[played, week_col]).max()) if played.any() else None

