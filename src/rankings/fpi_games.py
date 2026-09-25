"""ESPN FPI game predictions — snapshot them before kickoff so they can be scored honestly.

FPI is ESPN's model, not ours. It is here for comparison on the football model page, next
to the production XGBoost, the margin ridge shadow and the betting line. It is never a
target: nothing in this repo is tuned toward it.

Where the numbers come from (probed 2026-09-25):

  * The site-API event summary carries a `predictor` block only while a game is
    upcoming. Once the game is played the block is gone.
  * The core API keeps it:
        sports.core.api.espn.com/v2/sports/football/leagues/{league}/events/{id}/
            competitions/{id}/predictor
    for completed games too, back through at least 2024. Each side has
    `gameProjection` (win %, 0-100), `teamChanceLoss`, `teamPredPtDiff` (predicted
    margin, points) and `matchupQuality`. NFL projections sum to slightly under 100
    because ESPN assigns a tie probability, so they are renormalised here.

Is the value on a completed game really pregame? Measured over 603 NFL games (2024 to
2026 week 3) and a 298-game FBS sample: it matches ESPN's own in-game win probability at
the first play within a median 0.15 points (NFL) and 0.0 (FBS), and differs by more than
3 points in 0.2% / 0.7% of games. A value recomputed after the game would drift from the
kickoff number; this one does not. So historical values are usable for a backtest.

They are still NOT pregame predictions in the sense the scoreboard needs, because we did
not hold them before kickoff: ESPN's `lastModified` on a completed game is often after the
game, and nothing proves the stored number was not revised. So this module keeps the two
apart by construction:

  * `snapshot()` only records games whose kickoff is still in the future, and stamps
    `predicted_at` with the capture time. Those rows count on the scoreboard.
  * `backfill()` records completed games with `source = 'backfill_after_kickoff'` and
    `predicted_at` = capture time, which is after kickoff — so the scoreboard's
    `predicted_at < kickoff` rule excludes them without any special case.

Table: {nfl_season|cfb_season}.fpi_game_predictions, append-only (one row per game per
snapshot; readers take the latest snapshot before kickoff). The load uses
CREATE_NEVER: this module will not create the table. The DDL is in
scripts/gcp/football/create_fpi_game_predictions.sql and needs a one-time approval.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

try:  # package layout locally, flat module tree in Cloud Functions
    from rankings.http import get_bytes as _get_bytes
except ImportError:  # pragma: no cover
    from http_transport import get_bytes as _get_bytes

logger = logging.getLogger(__name__)

CORE = ("https://sports.core.api.espn.com/v2/sports/football/leagues/{league}"
        "/events/{event}/competitions/{event}/predictor")
LEAGUE = {"nfl": "nfl", "cfb": "college-football"}
DATASET = {"nfl": "nfl_season", "cfb": "cfb_season"}
TABLE = "fpi_game_predictions"
MODEL_VERSION = "espn_fpi"

SOURCE_PREGAME = "pregame_snapshot"
SOURCE_BACKFILL = "backfill_after_kickoff"

# Column order is the table contract (see the DDL). The first four are the same names
# every model table uses, so the backend reads all of them with one query shape.
COLUMNS = [
    "game_id", "season", "week", "division",
    "home_team", "away_team", "kickoff",
    "home_win_probability", "predicted_home_margin",
    "home_game_projection", "away_game_projection", "matchup_quality",
    "espn_event_id", "espn_home_team_id", "espn_last_modified",
    "predicted_at", "source", "model_version",
]


def _stat_map(side: dict) -> dict[str, float]:
    return {s.get("name"): s.get("value") for s in side.get("statistics", []) or []}


def _team_id(side: dict) -> str | None:
    ref = (side.get("team") or {}).get("$ref", "")
    if "/teams/" not in ref:
        return None
    return ref.split("/teams/")[-1].split("?")[0] or None


def parse_predictor(payload: dict) -> dict | None:
    """One ESPN predictor payload -> the fields worth keeping, home-oriented.

    Returns None when either side is missing a projection — ESPN publishes the block
    before it has numbers for some games, and a half-filled row would score as 50/50.
    """
    home, away = payload.get("homeTeam"), payload.get("awayTeam")
    if not home or not away:
        return None
    hs, as_ = _stat_map(home), _stat_map(away)
    hp, ap = hs.get("gameProjection"), as_.get("gameProjection")
    if hp is None or ap is None or (hp + ap) <= 0:
        return None
    return {
        # Renormalised: NFL projections leave room for a tie (they sum to ~99.7).
        "home_win_probability": float(hp) / float(hp + ap),
        "predicted_home_margin": (float(hs["teamPredPtDiff"])
                                  if hs.get("teamPredPtDiff") is not None else None),
        "home_game_projection": float(hp),
        "away_game_projection": float(ap),
        "matchup_quality": (float(hs["matchupQuality"])
                            if hs.get("matchupQuality") is not None else None),
        "espn_home_team_id": _team_id(home),
        "espn_away_team_id": _team_id(away),
        "espn_last_modified": payload.get("lastModified"),
    }


def orient(parsed: dict, expected_home_id: str | None) -> dict:
    """Swap sides if ESPN's home team is our away team.

    Measured 0 mismatches across 603 NFL games including every neutral-site and
    international game, so this is a guard, not a fix. It only runs where the caller
    knows ESPN's id for our home team (the NFL, via the FPI team table); college game
    ids are ESPN's own events, so the orientation is ESPN's by definition.
    """
    if not expected_home_id or parsed.get("espn_home_team_id") in (None, expected_home_id):
        return parsed
    if parsed.get("espn_away_team_id") != expected_home_id:
        return parsed  # neither side matches; leave it and let the caller log
    out = dict(parsed)
    out["home_win_probability"] = 1.0 - parsed["home_win_probability"]
    out["home_game_projection"], out["away_game_projection"] = (
        parsed["away_game_projection"], parsed["home_game_projection"])
    if parsed.get("predicted_home_margin") is not None:
        out["predicted_home_margin"] = -parsed["predicted_home_margin"]
    out["espn_home_team_id"], out["espn_away_team_id"] = (
        parsed["espn_away_team_id"], parsed["espn_home_team_id"])
    logger.warning("FPI: ESPN listed the sides the other way round; swapped")
    return out


def fetch_predictor(sport: str, event_id: str, timeout: int = 30) -> dict | None:
    url = CORE.format(league=LEAGUE[sport], event=event_id)
    try:
        payload = json.loads(_get_bytes(url, timeout).decode())
    except Exception as exc:
        logger.warning("FPI predictor %s %s failed: %s", sport, event_id, str(exc)[:160])
        return None
    return parse_predictor(payload)


def _rows(sport: str, slate: pd.DataFrame, source: str, now: datetime,
          workers: int = 6) -> pd.DataFrame:
    """Fetch and assemble rows for a slate frame.

    `slate` needs: game_id, espn_event_id, season, week, home_team, away_team, kickoff
    (UTC); optional division and espn_home_team_id (for `orient`).
    """
    if slate.empty:
        return pd.DataFrame(columns=COLUMNS)
    records = slate.to_dict("records")

    def one(rec):
        parsed = fetch_predictor(sport, str(rec["espn_event_id"]))
        if parsed is None:
            return None
        parsed = orient(parsed, rec.get("espn_home_team_id"))
        return {
            "game_id": str(rec["game_id"]),
            "season": int(rec["season"]),
            "week": int(rec["week"]),
            "division": rec.get("division"),
            "home_team": rec["home_team"],
            "away_team": rec["away_team"],
            "kickoff": pd.Timestamp(rec["kickoff"]),
            **{k: parsed[k] for k in ("home_win_probability", "predicted_home_margin",
                                      "home_game_projection", "away_game_projection",
                                      "matchup_quality", "espn_home_team_id")},
            "espn_event_id": str(rec["espn_event_id"]),
            "espn_last_modified": (pd.Timestamp(parsed["espn_last_modified"])
                                   if parsed.get("espn_last_modified") else pd.NaT),
            "predicted_at": pd.Timestamp(now),
            "source": source,
            "model_version": MODEL_VERSION,
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = [r for r in ex.map(one, records) if r is not None]
    df = pd.DataFrame(out, columns=COLUMNS)
    missing = len(records) - len(df)
    if missing:
        logger.info("FPI %s: no predictor for %d of %d games", sport, missing, len(records))
    return df


def _utc(ts) -> pd.Series:
    s = pd.to_datetime(ts, errors="coerce", utc=True)
    return s


def snapshot(sport: str, slate: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Pregame snapshot: only games that have not kicked off yet, stamped with now.

    The kickoff filter is the whole guarantee. A game already under way, or finished, is
    dropped here — even if ESPN would still hand back a number — so every row this writes
    was captured strictly before its game started.
    """
    now = now or datetime.now(timezone.utc)
    if slate.empty:
        return pd.DataFrame(columns=COLUMNS)
    kick = _utc(slate["kickoff"])
    upcoming = slate[kick.notna() & (kick > pd.Timestamp(now))].copy()
    upcoming["kickoff"] = _utc(upcoming["kickoff"])
    dropped = len(slate) - len(upcoming)
    if dropped:
        logger.info("FPI snapshot: skipped %d games already started or without kickoff",
                    dropped)
    return _rows(sport, upcoming, SOURCE_PREGAME, now)


def backfill(sport: str, slate: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Completed games, for backtests. Marked so they can never pass as pregame."""
    now = now or datetime.now(timezone.utc)
    slate = slate.copy()
    slate["kickoff"] = _utc(slate["kickoff"])
    done = slate[slate["kickoff"].notna() & (slate["kickoff"] <= pd.Timestamp(now))]
    return _rows(sport, done, SOURCE_BACKFILL, now)


# --------------------------------------------------------------------------- slates
def nfl_slate(schedules: pd.DataFrame, now: datetime | None = None,
              horizon_days: int = 8, team_ids: dict[str, str] | None = None) -> pd.DataFrame:
    """Upcoming NFL games from an nflverse schedule frame.

    nflverse `gameday`/`gametime` are US Eastern wall-clock; converted to UTC here.
    `team_ids` maps nflverse abbreviation -> ESPN team id, used only by `orient`.
    """
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    df = schedules.copy()
    if "espn" not in df.columns:
        raise KeyError("schedule frame has no `espn` event id column")
    local = pd.to_datetime(df["gameday"].astype(str) + " "
                           + df["gametime"].fillna("13:00").astype(str),
                           errors="coerce")
    df["kickoff"] = local.dt.tz_localize("America/New_York", ambiguous="NaT",
                                         nonexistent="NaT").dt.tz_convert("UTC")
    df = df[df["espn"].notna() & df["kickoff"].notna()]
    df = df[(df["kickoff"] > now) & (df["kickoff"] <= now + timedelta(days=horizon_days))]
    out = pd.DataFrame({
        "game_id": df["game_id"].astype(str),
        "espn_event_id": pd.to_numeric(df["espn"], errors="coerce").astype("Int64").astype(str),
        "season": df["season"].astype(int),
        "week": df["week"].astype(int),
        "division": None,
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "kickoff": df["kickoff"],
    })
    if team_ids:
        out["espn_home_team_id"] = out["home_team"].map(team_ids)
    return out.reset_index(drop=True)


def cfb_slate(scheduled: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Upcoming college games from espn_data.fetch_scheduled (naive-UTC game_date).

    FPI covers FBS teams; an FCS-vs-FCS game simply comes back without a predictor.
    """
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if scheduled.empty:
        return pd.DataFrame()
    kick = pd.to_datetime(scheduled["game_date"], errors="coerce")
    if kick.dt.tz is None:
        kick = kick.dt.tz_localize("UTC")
    df = scheduled.assign(kickoff=kick)
    df = df[df["kickoff"] > now]
    return pd.DataFrame({
        "game_id": df["game_id"].astype(str),
        "espn_event_id": df["game_id"].astype(str),
        "season": df["season"].astype(int),
        "week": df["week"].astype(int),
        "division": df.get("division"),
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "kickoff": df["kickoff"],
    }).reset_index(drop=True)


def nfl_team_ids(season: int) -> dict[str, str]:
    """nflverse abbreviation -> ESPN team id, from the FPI team table (one cached call).
    Empty on failure: `orient` then trusts ESPN's ordering, which measured correct."""
    try:
        from rankings import fpi
    except ImportError:  # pragma: no cover - flat tree
        import fpi  # type: ignore
    try:
        teams = fpi.fetch("nfl", season)
    except Exception as exc:
        logger.warning("FPI team ids unavailable (%s); trusting ESPN side order", exc)
        return {}
    abbr = teams["team_abbr"].replace(fpi.ESPN_TO_NFLVERSE)
    return dict(zip(abbr, teams["espn_team_id"].astype(str)))


# --------------------------------------------------------------------------- write
def write_bq(rows: pd.DataFrame, sport: str, project: str = "hankstank") -> int:
    """Append a snapshot. Never creates the table (CREATE_NEVER) and never deletes.

    Append-only is deliberate: each snapshot is evidence of what FPI said at that time,
    and a reader takes the latest one before kickoff. A load job, not streaming, so no
    streaming-buffer interaction with anything else.
    """
    if rows.empty:
        return 0
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    table_id = f"{project}.{DATASET[sport]}.{TABLE}"
    cfg = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        create_disposition="CREATE_NEVER",
    )
    frame = rows[COLUMNS].copy()
    for col in ("kickoff", "espn_last_modified", "predicted_at"):
        frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    client.load_table_from_dataframe(frame, table_id, job_config=cfg).result()
    logger.info("FPI: appended %d rows -> %s", len(frame), table_id)
    return len(frame)


def enabled(req: dict) -> bool:
    """Off by default, like the ridge shadow: {"fpi_snapshot": true} or FPI_SNAPSHOT=1.
    Until the table exists, turning it on would only log a CREATE_NEVER failure."""
    import os

    return bool(req.get("fpi_snapshot")) or os.environ.get("FPI_SNAPSHOT") == "1"


def run_snapshot(sport: str, slate: pd.DataFrame, steps: dict, write: bool = True) -> None:
    """Snapshot + append, recorded in `steps`. Never fatal: FPI is a comparison, and a
    failure here must not cost the ingest or the production predictions anything."""
    try:
        rows = snapshot(sport, slate)
        steps["fpi_snapshot"] = write_bq(rows, sport) if write else len(rows)
    except Exception as exc:
        logger.error("FPI snapshot failed: %s", exc)
        steps["fpi_snapshot"] = {"error": str(exc)[:200]}


# --------------------------------------------------------------------------- CLI
def main() -> int:
    """Dry run by default: prints what would be written.

        python -m rankings.fpi_games --sport nfl                 # upcoming NFL, dry run
        python -m rankings.fpi_games --sport cfb --week 5        # a CFB week, dry run
        python -m rankings.fpi_games --sport nfl --backfill --season 2025 --csv out.csv
        ... --write-bq                                           # needs the table
    """
    import argparse
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", required=True, choices=sorted(LEAGUE))
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=None, help="CFB week (default: next unplayed)")
    ap.add_argument("--backfill", action="store_true",
                    help="completed games, source=backfill_after_kickoff")
    ap.add_argument("--write-bq", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--schedule-csv", default=None,
                    help="local nflverse games.csv instead of nflreadpy (for a laptop "
                         "whose TLS inspection breaks Python's GitHub fetch)")
    args = ap.parse_args()

    src = Path(__file__).resolve().parents[1]
    now = datetime.now(timezone.utc)

    if args.sport == "nfl":
        sys.path.insert(0, str(src / "nfl"))
        if args.schedule_csv:
            sched = pd.read_csv(args.schedule_csv, low_memory=False)
        else:
            from data import load_schedules  # type: ignore

            sched = load_schedules(refresh=True)
        if args.backfill:
            sched = sched[(sched["season"] == args.season) & sched["result"].notna()]
            slate = nfl_slate(sched, now=pd.Timestamp("1900-01-01", tz="UTC"),
                              horizon_days=10 ** 6)
            rows = backfill("nfl", slate, now)
        else:
            rows = snapshot("nfl", nfl_slate(sched, now, team_ids=nfl_team_ids(args.season)),
                            now)
    else:
        sys.path.insert(0, str(src / "cfb"))
        from espn_data import fetch_scheduled  # type: ignore

        if args.week is None:
            from pipeline import next_unplayed_week  # type: ignore

            args.week = next_unplayed_week(args.season)
        sched = fetch_scheduled(args.season, args.week)
        if args.backfill:
            slate = cfb_slate(sched, now=pd.Timestamp("1900-01-01", tz="UTC"))
            rows = backfill("cfb", slate, now)
        else:
            rows = snapshot("cfb", cfb_slate(sched, now), now)

    print(f"{len(rows)} FPI rows ({'backfill' if args.backfill else 'pregame snapshot'})")
    if not rows.empty:
        print(rows[["game_id", "week", "home_team", "away_team", "kickoff",
                    "home_win_probability", "predicted_home_margin"]].head(20)
              .to_string(index=False))
    if args.csv:
        rows.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    if args.write_bq:
        write_bq(rows, args.sport)
    else:
        print("dry run: nothing written to BigQuery (pass --write-bq)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
