#!/usr/bin/env python3
"""
Create the mlb_2026_season BigQuery dataset and all tables.

Tables mirror the historical schema but are scoped to 2026 only.
Includes partitioning and clustering optimized for daily refreshes.

Usage:
    python create_dataset.py
    python create_dataset.py --dry-run
"""

import argparse
import logging
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "hankstank"
DATASET = "mlb_2026_season"
LOCATION = "US"

# ---------------------------------------------------------------------------
# Table schemas
# ---------------------------------------------------------------------------

TEAMS = [
    bigquery.SchemaField("team_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("team_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("team_code", "STRING"),
    bigquery.SchemaField("location_name", "STRING"),
    bigquery.SchemaField("team_name_full", "STRING"),
    bigquery.SchemaField("league_id", "INT64"),
    bigquery.SchemaField("league_name", "STRING"),
    bigquery.SchemaField("division_id", "INT64"),
    bigquery.SchemaField("division_name", "STRING"),
    bigquery.SchemaField("venue_id", "INT64"),
    bigquery.SchemaField("venue_name", "STRING"),
    bigquery.SchemaField("first_year_of_play", "INT64"),
    bigquery.SchemaField("active", "BOOL"),
    bigquery.SchemaField("synced_at", "TIMESTAMP"),
]

TEAM_STATS = [
    bigquery.SchemaField("team_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("team_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("stat_type", "STRING", mode="REQUIRED"),  # batting | pitching
    bigquery.SchemaField("games_played", "INT64"),
    # batting
    bigquery.SchemaField("at_bats", "FLOAT64"),
    bigquery.SchemaField("runs", "FLOAT64"),
    bigquery.SchemaField("hits", "FLOAT64"),
    bigquery.SchemaField("doubles", "FLOAT64"),
    bigquery.SchemaField("triples", "FLOAT64"),
    bigquery.SchemaField("home_runs", "FLOAT64"),
    bigquery.SchemaField("rbi", "FLOAT64"),
    bigquery.SchemaField("stolen_bases", "FLOAT64"),
    bigquery.SchemaField("caught_stealing", "FLOAT64"),
    bigquery.SchemaField("walks", "FLOAT64"),
    bigquery.SchemaField("strikeouts", "INT64"),
    bigquery.SchemaField("batting_avg", "FLOAT64"),
    bigquery.SchemaField("obp", "FLOAT64"),
    bigquery.SchemaField("slg", "FLOAT64"),
    bigquery.SchemaField("ops", "FLOAT64"),
    bigquery.SchemaField("total_bases", "FLOAT64"),
    bigquery.SchemaField("hit_by_pitch", "FLOAT64"),
    bigquery.SchemaField("sac_flies", "FLOAT64"),
    bigquery.SchemaField("sac_bunts", "FLOAT64"),
    bigquery.SchemaField("left_on_base", "FLOAT64"),
    # pitching
    bigquery.SchemaField("wins", "FLOAT64"),
    bigquery.SchemaField("losses", "FLOAT64"),
    bigquery.SchemaField("win_percentage", "FLOAT64"),
    bigquery.SchemaField("era", "FLOAT64"),
    bigquery.SchemaField("games_started", "FLOAT64"),
    bigquery.SchemaField("games_finished", "FLOAT64"),
    bigquery.SchemaField("complete_games", "FLOAT64"),
    bigquery.SchemaField("shutouts", "FLOAT64"),
    bigquery.SchemaField("saves", "FLOAT64"),
    bigquery.SchemaField("save_opportunities", "FLOAT64"),
    bigquery.SchemaField("holds", "FLOAT64"),
    bigquery.SchemaField("blown_saves", "FLOAT64"),
    bigquery.SchemaField("innings_pitched", "FLOAT64"),
    bigquery.SchemaField("hits_allowed", "FLOAT64"),
    bigquery.SchemaField("runs_allowed", "FLOAT64"),
    bigquery.SchemaField("earned_runs", "FLOAT64"),
    bigquery.SchemaField("home_runs_allowed", "FLOAT64"),
    bigquery.SchemaField("walks_allowed", "FLOAT64"),
    bigquery.SchemaField("pitching_strikeouts", "FLOAT64"),
    bigquery.SchemaField("whip", "FLOAT64"),
    bigquery.SchemaField("batters_faced", "FLOAT64"),
    bigquery.SchemaField("wild_pitches", "FLOAT64"),
    bigquery.SchemaField("hit_batsmen", "FLOAT64"),
    bigquery.SchemaField("balks", "FLOAT64"),
    bigquery.SchemaField("snapshot_date", "DATE"),
    bigquery.SchemaField("synced_at", "TIMESTAMP"),
]

PLAYER_STATS = [
    bigquery.SchemaField("player_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("player_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("team_id", "INT64"),
    bigquery.SchemaField("team_name", "STRING"),
    bigquery.SchemaField("stat_type", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("games_played", "INT64"),
    # batting
    bigquery.SchemaField("at_bats", "FLOAT64"),
    bigquery.SchemaField("runs", "FLOAT64"),
    bigquery.SchemaField("hits", "FLOAT64"),
    bigquery.SchemaField("doubles", "FLOAT64"),
    bigquery.SchemaField("triples", "FLOAT64"),
    bigquery.SchemaField("home_runs", "FLOAT64"),
    bigquery.SchemaField("rbi", "FLOAT64"),
    bigquery.SchemaField("stolen_bases", "FLOAT64"),
    bigquery.SchemaField("caught_stealing", "FLOAT64"),
    bigquery.SchemaField("walks", "FLOAT64"),
    bigquery.SchemaField("strikeouts", "INT64"),
    bigquery.SchemaField("batting_avg", "FLOAT64"),
    bigquery.SchemaField("obp", "FLOAT64"),
    bigquery.SchemaField("slg", "FLOAT64"),
    bigquery.SchemaField("ops", "FLOAT64"),
    bigquery.SchemaField("total_bases", "FLOAT64"),
    bigquery.SchemaField("hit_by_pitch", "FLOAT64"),
    bigquery.SchemaField("sac_flies", "FLOAT64"),
    bigquery.SchemaField("sac_bunts", "FLOAT64"),
    bigquery.SchemaField("left_on_base", "FLOAT64"),
    # pitching
    bigquery.SchemaField("wins", "FLOAT64"),
    bigquery.SchemaField("losses", "FLOAT64"),
    bigquery.SchemaField("win_percentage", "FLOAT64"),
    bigquery.SchemaField("era", "FLOAT64"),
    bigquery.SchemaField("games_started", "FLOAT64"),
    bigquery.SchemaField("games_finished", "FLOAT64"),
    bigquery.SchemaField("complete_games", "FLOAT64"),
    bigquery.SchemaField("shutouts", "FLOAT64"),
    bigquery.SchemaField("saves", "FLOAT64"),
    bigquery.SchemaField("save_opportunities", "FLOAT64"),
    bigquery.SchemaField("holds", "FLOAT64"),
    bigquery.SchemaField("blown_saves", "FLOAT64"),
    bigquery.SchemaField("innings_pitched", "FLOAT64"),
    bigquery.SchemaField("hits_allowed", "FLOAT64"),
    bigquery.SchemaField("runs_allowed", "FLOAT64"),
    bigquery.SchemaField("earned_runs", "FLOAT64"),
    bigquery.SchemaField("home_runs_allowed", "FLOAT64"),
    bigquery.SchemaField("walks_allowed", "FLOAT64"),
    bigquery.SchemaField("pitching_strikeouts", "FLOAT64"),
    bigquery.SchemaField("whip", "FLOAT64"),
    bigquery.SchemaField("batters_faced", "FLOAT64"),
    bigquery.SchemaField("wild_pitches", "FLOAT64"),
    bigquery.SchemaField("hit_batsmen", "FLOAT64"),
    bigquery.SchemaField("balks", "FLOAT64"),
    bigquery.SchemaField("snapshot_date", "DATE"),
    bigquery.SchemaField("synced_at", "TIMESTAMP"),
]

STANDINGS = [
    bigquery.SchemaField("team_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("team_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("league_id", "INT64"),
    bigquery.SchemaField("league_name", "STRING"),
    bigquery.SchemaField("division_id", "INT64"),
    bigquery.SchemaField("division_name", "STRING"),
    bigquery.SchemaField("wins", "INT64"),
    bigquery.SchemaField("losses", "INT64"),
    bigquery.SchemaField("win_percentage", "FLOAT64"),
    bigquery.SchemaField("games_back", "FLOAT64"),
    bigquery.SchemaField("wildcard_games_back", "FLOAT64"),
    bigquery.SchemaField("division_rank", "STRING"),
    bigquery.SchemaField("league_rank", "STRING"),
    bigquery.SchemaField("runs_scored", "INT64"),
    bigquery.SchemaField("runs_allowed", "INT64"),
    bigquery.SchemaField("run_differential", "INT64"),
    bigquery.SchemaField("home_wins", "INT64"),
    bigquery.SchemaField("home_losses", "INT64"),
    bigquery.SchemaField("away_wins", "INT64"),
    bigquery.SchemaField("away_losses", "INT64"),
    bigquery.SchemaField("streak", "STRING"),
    bigquery.SchemaField("last_ten", "STRING"),
    bigquery.SchemaField("snapshot_date", "DATE"),
    bigquery.SchemaField("synced_at", "TIMESTAMP"),
]

GAMES = [
    bigquery.SchemaField("game_pk", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("game_date", "DATE"),
    bigquery.SchemaField("game_type", "STRING"),  # S=Spring, R=Regular, P=Playoff
    bigquery.SchemaField("home_team_id", "INT64"),
    bigquery.SchemaField("home_team_name", "STRING"),
    bigquery.SchemaField("away_team_id", "INT64"),
    bigquery.SchemaField("away_team_name", "STRING"),
    bigquery.SchemaField("home_score", "INT64"),
    bigquery.SchemaField("away_score", "INT64"),
    bigquery.SchemaField("venue_id", "INT64"),
    bigquery.SchemaField("venue_name", "STRING"),
    bigquery.SchemaField("status", "STRING"),
    bigquery.SchemaField("status_code", "STRING"),
    bigquery.SchemaField("innings", "INT64"),
    bigquery.SchemaField("winning_pitcher_id", "INT64"),
    bigquery.SchemaField("winning_pitcher_name", "STRING"),
    bigquery.SchemaField("losing_pitcher_id", "INT64"),
    bigquery.SchemaField("losing_pitcher_name", "STRING"),
    bigquery.SchemaField("save_pitcher_id", "INT64"),
    bigquery.SchemaField("save_pitcher_name", "STRING"),
    bigquery.SchemaField("synced_at", "TIMESTAMP"),
]

ROSTERS = [
    bigquery.SchemaField("team_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("team_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("player_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("player_name", "STRING"),
    bigquery.SchemaField("jersey_number", "STRING"),
    bigquery.SchemaField("position_code", "STRING"),
    bigquery.SchemaField("position_name", "STRING"),
    bigquery.SchemaField("position_type", "STRING"),
    bigquery.SchemaField("status", "STRING"),
    bigquery.SchemaField("snapshot_date", "DATE"),
    bigquery.SchemaField("synced_at", "TIMESTAMP"),
]

TRANSACTIONS = [
    bigquery.SchemaField("transaction_id", "INT64"),
    bigquery.SchemaField("date", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("type_code", "STRING"),
    bigquery.SchemaField("type_desc", "STRING"),
    bigquery.SchemaField("description", "STRING"),
    bigquery.SchemaField("from_team_id", "INT64"),
    bigquery.SchemaField("from_team_name", "STRING"),
    bigquery.SchemaField("from_team_abbreviation", "STRING"),
    bigquery.SchemaField("to_team_id", "INT64"),
    bigquery.SchemaField("to_team_name", "STRING"),
    bigquery.SchemaField("to_team_abbreviation", "STRING"),
    bigquery.SchemaField("person_id", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("person_full_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("person_link", "STRING"),
    bigquery.SchemaField("resolution", "STRING"),
    bigquery.SchemaField("notes", "STRING"),
    bigquery.SchemaField("synced_at", "TIMESTAMP"),
]

STATCAST_PITCHES = [
    bigquery.SchemaField("pitch_type", "STRING"),
    bigquery.SchemaField("game_date", "DATE"),
    bigquery.SchemaField("game_year", "INT64"),
    bigquery.SchemaField("game_pk", "INT64"),
    bigquery.SchemaField("pitcher", "INT64"),
    bigquery.SchemaField("batter", "INT64"),
    bigquery.SchemaField("player_name", "STRING"),
    bigquery.SchemaField("events", "STRING"),
    bigquery.SchemaField("description", "STRING"),
    bigquery.SchemaField("release_speed", "FLOAT64"),
    bigquery.SchemaField("release_pos_x", "FLOAT64"),
    bigquery.SchemaField("release_pos_z", "FLOAT64"),
    bigquery.SchemaField("release_spin_rate", "FLOAT64"),
    bigquery.SchemaField("spin_axis", "FLOAT64"),
    bigquery.SchemaField("release_extension", "FLOAT64"),
    bigquery.SchemaField("effective_speed", "FLOAT64"),
    bigquery.SchemaField("zone", "INT64"),
    bigquery.SchemaField("plate_x", "FLOAT64"),
    bigquery.SchemaField("plate_z", "FLOAT64"),
    bigquery.SchemaField("pfx_x", "FLOAT64"),
    bigquery.SchemaField("pfx_z", "FLOAT64"),
    bigquery.SchemaField("game_type", "STRING"),
    bigquery.SchemaField("stand", "STRING"),
    bigquery.SchemaField("p_throws", "STRING"),
    bigquery.SchemaField("home_team", "STRING"),
    bigquery.SchemaField("away_team", "STRING"),
    bigquery.SchemaField("balls", "INT64"),
    bigquery.SchemaField("strikes", "INT64"),
    bigquery.SchemaField("inning", "INT64"),
    bigquery.SchemaField("inning_topbot", "STRING"),
    bigquery.SchemaField("outs_when_up", "INT64"),
    bigquery.SchemaField("hit_distance_sc", "FLOAT64"),
    bigquery.SchemaField("launch_speed", "FLOAT64"),
    bigquery.SchemaField("launch_angle", "FLOAT64"),
    bigquery.SchemaField("estimated_ba_using_speedangle", "FLOAT64"),
    bigquery.SchemaField("estimated_woba_using_speedangle", "FLOAT64"),
    bigquery.SchemaField("estimated_slg_using_speedangle", "FLOAT64"),
    bigquery.SchemaField("woba_value", "FLOAT64"),
    bigquery.SchemaField("woba_denom", "FLOAT64"),
    bigquery.SchemaField("babip_value", "FLOAT64"),
    bigquery.SchemaField("iso_value", "FLOAT64"),
    bigquery.SchemaField("vx0", "FLOAT64"),
    bigquery.SchemaField("vy0", "FLOAT64"),
    bigquery.SchemaField("vz0", "FLOAT64"),
    bigquery.SchemaField("ax", "FLOAT64"),
    bigquery.SchemaField("ay", "FLOAT64"),
    bigquery.SchemaField("az", "FLOAT64"),
    bigquery.SchemaField("sz_top", "FLOAT64"),
    bigquery.SchemaField("sz_bot", "FLOAT64"),
    bigquery.SchemaField("bat_speed", "FLOAT64"),
    bigquery.SchemaField("swing_length", "FLOAT64"),
    bigquery.SchemaField("attack_angle", "FLOAT64"),
    bigquery.SchemaField("delta_home_win_exp", "FLOAT64"),
    bigquery.SchemaField("delta_run_exp", "FLOAT64"),
    bigquery.SchemaField("pitch_name", "STRING"),
    bigquery.SchemaField("home_score", "INT64"),
    bigquery.SchemaField("away_score", "INT64"),
    bigquery.SchemaField("synced_at", "TIMESTAMP"),
]


# ---------------------------------------------------------------------------
# Table definitions with partitioning / clustering
# ---------------------------------------------------------------------------

TABLE_DEFS = {
    "teams": {
        "schema": TEAMS,
        "description": "2026 MLB team reference data",
        "clustering": ["team_id"],
    },
    "team_stats": {
        "schema": TEAM_STATS,
        "description": "2026 team batting & pitching stats (daily snapshots)",
        "partition_field": "snapshot_date",
        "partition_type": "DAY",
        "clustering": ["team_id", "stat_type"],
    },
    "player_stats": {
        "schema": PLAYER_STATS,
        "description": "2026 individual player stats (daily snapshots)",
        "partition_field": "snapshot_date",
        "partition_type": "DAY",
        "clustering": ["team_id", "player_id", "stat_type"],
    },
    "standings": {
        "schema": STANDINGS,
        "description": "2026 daily standings snapshots",
        "partition_field": "snapshot_date",
        "partition_type": "DAY",
        "clustering": ["division_id", "team_id"],
    },
    "games": {
        "schema": GAMES,
        "description": "2026 game results (spring training + regular season)",
        "partition_field": "game_date",
        "partition_type": "DAY",
        "clustering": ["game_type", "home_team_id", "away_team_id"],
    },
    "rosters": {
        "schema": ROSTERS,
        "description": "2026 team rosters (periodic snapshots)",
        "partition_field": "snapshot_date",
        "partition_type": "DAY",
        "clustering": ["team_id"],
    },
    "transactions": {
        "schema": TRANSACTIONS,
        "description": "2026 MLB transactions",
        "clustering": ["date", "person_id"],
    },
    "statcast_pitches": {
        "schema": STATCAST_PITCHES,
        "description": "2026 pitch-by-pitch Statcast data",
        "partition_field": "game_date",
        "partition_type": "DAY",
        "clustering": ["game_pk", "pitcher", "batter"],
    },
}


def create_dataset(client: bigquery.Client, dry_run: bool = False):
    dataset_ref = bigquery.DatasetReference(PROJECT, DATASET)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = LOCATION
    dataset.description = (
        "Live 2026 MLB season data — refreshed daily via Cloud Scheduler. "
        "Includes spring training, regular season, and postseason."
    )
    dataset.labels = {"sport": "mlb", "season": "2026", "env": "production"}

    if dry_run:
        logger.info("[DRY RUN] Would create dataset %s.%s", PROJECT, DATASET)
        return

    dataset = client.create_dataset(dataset, exists_ok=True)
    logger.info("Dataset %s.%s ready", PROJECT, DATASET)


def create_tables(client: bigquery.Client, dry_run: bool = False):
    for name, defn in TABLE_DEFS.items():
        table_id = f"{PROJECT}.{DATASET}.{name}"
        table = bigquery.Table(table_id, schema=defn["schema"])
        table.description = defn.get("description", "")

        # Partitioning
        pf = defn.get("partition_field")
        if pf:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=pf,
            )

        # Clustering
        cl = defn.get("clustering")
        if cl:
            table.clustering_fields = cl

        if dry_run:
            logger.info("[DRY RUN] Would create table %s", table_id)
            continue

        table = client.create_table(table, exists_ok=True)
        logger.info("Table %s ready (partition=%s, cluster=%s)", table_id, pf, cl)


def main():
    parser = argparse.ArgumentParser(description="Create mlb_2026_season dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = parser.parse_args()

    client = bigquery.Client(project=PROJECT)
    create_dataset(client, dry_run=args.dry_run)
    create_tables(client, dry_run=args.dry_run)

    logger.info("Done — %d tables configured in %s.%s", len(TABLE_DEFS), PROJECT, DATASET)


if __name__ == "__main__":
    main()
