#!/usr/bin/env python3
"""Import pick'em picks from the Google Sheet exports into BigQuery.

A one-time migration, kept in the repo so the provenance of those rows is auditable and
so a correction can be re-run rather than hand-patched. It is idempotent: picks upsert on
a deterministic pick_id, so running it twice changes nothing.

    python scripts/import_pickem_sheet.py --dry-run ~/Downloads/*.csv
    python scripts/import_pickem_sheet.py --write ~/Downloads/*.csv

Sheet shape (as exported 2026-09-02):
    0  Game ID          ESPN's id, which is also pickem.games.game_id
    2  Away Team        the pick columns hold one of these two names verbatim
    4  Home Team
    6  Spread           free text, "TCU -8.5" / "N/A" — not parsed; the real, sign-
                        verified line is already in pickem.games
    13 Winner           unpopulated in the export ("-"); results come from pickem.games
    14 Elijah's Picks   header is blank in the week 2 export
    15 Jacob's Picks
    22 Vegas Picks      the favourite. Not imported: the grading view derives the
                        market's call from the closing line itself.

The picks are STRAIGHT UP, not against the spread. That is measured, not assumed: they
agree with the Vegas favourite 97% and 93% of the time in week 1, and nobody taking
spread value picks the favourite 97% of the time. Week 2 also has almost no spreads in
the sheet at all, which ATS picks could not have been made against.

Identity: these predate any Google sign-in, so there is no subject claim to key on. Rows
are written against a provisional user_id of "email:<address>" and the backend adopts
them on that person's first sign-in, matching on the verified email. See
pickem.controller.adoptImportedPicks.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = os.environ.get("GCP_PROJECT", "hankstank")
DATASET = os.environ.get("PICKEM_DATASET", "pickem")
SPORT = "cfb"
PICK_TYPE = "su"

# Sheet column index -> the person it belongs to. The week 2 export has a blank header
# over Elijah's column, so the mapping is positional rather than by header text.
PICKERS: dict[int, dict] = {
    14: {"name": "Elijah", "email": "elijahcraig45@gmail.com"},
    15: {"name": "Jacob", "email": "jacobstroud00@gmail.com"},
}

# Values that mean "no pick" rather than a team.
BLANKS = {"", "-", "n/a", "#value!", "#n/a", "tbd"}

COL_GAME_ID, COL_AWAY, COL_HOME = 0, 2, 4


def provisional_user_id(email: str) -> str:
    """The placeholder id, replaced by the Google subject on first sign-in."""
    return f"email:{email.strip().lower()}"


def week_from_filename(path: Path) -> int | None:
    m = re.search(r"week\s*(\d+)", path.name, re.I)
    return int(m.group(1)) if m else None


def season_from_filename(path: Path, default: int = 2026) -> int:
    m = re.search(r"(20\d\d)", path.name)
    return int(m.group(1)) if m else default


def parse_sheet(path: Path) -> tuple[list[dict], list[str]]:
    """Return (picks, problems). One pick row per person per game."""
    week = week_from_filename(path)
    season = season_from_filename(path)
    if week is None:
        return [], [f"{path.name}: could not read a week number from the filename"]

    rows = list(csv.reader(io.open(path, encoding="utf-8-sig")))
    if len(rows) < 2:
        return [], [f"{path.name}: no data rows"]

    picks: list[dict] = []
    problems: list[str] = []

    for line_no, r in enumerate(rows[1:], start=2):
        if len(r) <= max(PICKERS):
            continue
        game_id = r[COL_GAME_ID].strip()
        away, home = r[COL_AWAY].strip(), r[COL_HOME].strip()
        if not game_id.isdigit() or not away or not home:
            continue

        for idx, who in PICKERS.items():
            raw = r[idx].strip()
            if raw.lower() in BLANKS:
                continue

            # The sheet names one of the two teams on the row, so the side comes from
            # the row itself. No fuzzy matching against another table's spelling —
            # which is the whole reason picks are stored as sides.
            if raw == away:
                side = "away"
            elif raw == home:
                side = "home"
            else:
                problems.append(
                    f"{path.name}:{line_no} {who['name']} picked {raw!r}, "
                    f"which is neither {away!r} nor {home!r}"
                )
                continue

            picks.append({
                "user_id": provisional_user_id(who["email"]),
                "email": who["email"],
                "display_name": who["name"],
                "sport": SPORT,
                "season": season,
                "week": week,
                "game_id": game_id,
                "pick_type": PICK_TYPE,
                "selected": side,
            })

    return picks, problems


def verify_games(client, picks: list[dict]) -> tuple[set[str], set[str]]:
    """Which of the sheet's game ids actually exist in pickem.games."""
    from google.cloud import bigquery

    ids = sorted({p["game_id"] for p in picks})
    rows = client.query(
        f"SELECT game_id FROM `{PROJECT}.{DATASET}.games` "
        "WHERE game_id IN UNNEST(@ids)",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("ids", "STRING", ids)
        ]),
    ).result()
    known = {r["game_id"] for r in rows}
    return known, set(ids) - known


def write(client, picks: list[dict]) -> None:
    """Upsert users and picks. Idempotent on pick_id."""
    from google.cloud import bigquery

    people = {}
    for p in picks:
        people[p["user_id"]] = (p["email"], p["display_name"])

    for user_id, (email, name) in sorted(people.items()):
        client.query(
            f"""
            MERGE `{PROJECT}.{DATASET}.users` AS t
            USING (SELECT @uid AS user_id, @email AS email, @name AS display_name) AS s
            ON t.user_id = s.user_id
            WHEN MATCHED THEN UPDATE SET
              email = s.email, display_name = s.display_name,
              last_seen_at = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT
              (user_id, email, display_name, created_at, last_seen_at)
              VALUES (s.uid_placeholder, s.email, s.display_name,
                      CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
            """.replace("s.uid_placeholder", "s.user_id"),
            job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("uid", "STRING", user_id),
                bigquery.ScalarQueryParameter("email", "STRING", email),
                bigquery.ScalarQueryParameter("name", "STRING", name),
            ]),
        ).result()
        logger.info("user %s (%s)", name, user_id)

    # One MERGE for every pick. Batched because BigQuery meters DML per table per day,
    # and 284 separate statements would be a meaningful slice of that for no reason.
    struct_rows = []
    params = []
    for i, p in enumerate(picks):
        struct_rows.append(
            f"(@u{i}, @s{i}, @se{i}, @w{i}, @g{i}, @t{i}, @sel{i})"
        )
        params += [
            bigquery.ScalarQueryParameter(f"u{i}", "STRING", p["user_id"]),
            bigquery.ScalarQueryParameter(f"s{i}", "STRING", p["sport"]),
            bigquery.ScalarQueryParameter(f"se{i}", "INT64", p["season"]),
            bigquery.ScalarQueryParameter(f"w{i}", "INT64", p["week"]),
            bigquery.ScalarQueryParameter(f"g{i}", "STRING", p["game_id"]),
            bigquery.ScalarQueryParameter(f"t{i}", "STRING", p["pick_type"]),
            bigquery.ScalarQueryParameter(f"sel{i}", "STRING", p["selected"]),
        ]

    # Chunked so one statement never grows past what BigQuery will parse.
    CHUNK = 200
    for start in range(0, len(struct_rows), CHUNK):
        chunk_rows = struct_rows[start:start + CHUNK]
        chunk_params = params[start * 7:(start + len(chunk_rows)) * 7]
        client.query(
            f"""
            MERGE `{PROJECT}.{DATASET}.picks` AS t
            USING (
              SELECT
                CONCAT(user_id, '|', game_id, '|', pick_type) AS pick_id,
                p.*,
                g.spread_line AS spread_at_pick
              FROM UNNEST([
                STRUCT<user_id STRING, sport STRING, season INT64, week INT64,
                       game_id STRING, pick_type STRING, selected STRING>
                {', '.join(chunk_rows)}
              ]) AS p
              LEFT JOIN `{PROJECT}.{DATASET}.games` g USING (game_id)
            ) AS s
            ON t.pick_id = s.pick_id
            WHEN MATCHED THEN UPDATE SET
              selected = s.selected, updated_at = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT
              (pick_id, user_id, sport, season, week, game_id, pick_type, selected,
               spread_at_pick, created_at, updated_at)
              VALUES (s.pick_id, s.user_id, s.sport, s.season, s.week, s.game_id,
                      s.pick_type, s.selected, s.spread_at_pick,
                      CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
            """,
            job_config=bigquery.QueryJobConfig(query_parameters=chunk_params),
        ).result()
        logger.info("upserted %d picks", len(chunk_rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--write", action="store_true",
                    help="actually write; the default is a dry run")
    args = ap.parse_args()

    all_picks: list[dict] = []
    all_problems: list[str] = []
    for path in args.files:
        if not path.exists():
            all_problems.append(f"{path}: not found")
            continue
        picks, problems = parse_sheet(path)
        all_picks += picks
        all_problems += problems
        by_person: dict[str, int] = {}
        for p in picks:
            by_person[p["display_name"]] = by_person.get(p["display_name"], 0) + 1
        logger.info("%s: week %s, %d picks %s", path.name,
                    picks[0]["week"] if picks else "?", len(picks), by_person)

    for problem in all_problems:
        logger.warning(problem)

    if not all_picks:
        logger.error("nothing to import")
        return 1

    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT)

    known, unknown = verify_games(client, all_picks)
    if unknown:
        # Not fatal: a pick on a game the games table has never heard of simply cannot
        # be graded, so it is dropped loudly rather than stored to sit ungradeable.
        logger.warning("%d game ids are not in pickem.games and will be skipped: %s",
                       len(unknown), sorted(unknown)[:8])
        all_picks = [p for p in all_picks if p["game_id"] in known]

    print()
    print(f"  {len(all_picks)} picks ready across "
          f"{len({(p['season'], p['week']) for p in all_picks})} week(s)")
    for name in sorted({p["display_name"] for p in all_picks}):
        weeks = sorted({p["week"] for p in all_picks if p["display_name"] == name})
        count = sum(1 for p in all_picks if p["display_name"] == name)
        print(f"    {name:8} {count:4} picks, weeks {weeks}")

    if not args.write:
        print("\n  [dry run] pass --write to import")
        return 0

    write(client, all_picks)
    print("\n  imported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
