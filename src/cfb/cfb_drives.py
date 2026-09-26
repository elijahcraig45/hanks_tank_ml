"""One row per college drive, from CollegeFootballData /drives, for the CFB drive simulator.

Stored in `cfb_historical.drives` (DDL: scripts/gcp/football/create_cfb_drive_sim_tables.sql).
The simulator refits a drive-outcome multinomial every week, so it needs drive rows, not
per-team aggregates: rates cannot be refit with field-position and game-script states.

CFBD quirks this handles (memory: cfbd-endpoint-parameter-quirks):
  * /drives has no game parameter, only year/week/seasonType, so it is fetched by week
    (~3,000 drives, FBS plus most FCS in one call; CFBD has no FCS-only drives for 2021);
  * gameId is ESPN's id, so games join to cfb_historical.games on game_id directly;
  * team names are CFBD school names ("Hawai'i"), not the ESPN abbreviations the stack
    keys on. Sides come from the BigQuery game via isHomeOffense, but CFBD's home/away
    disagrees with ESPN's in ~21 games per 7,000 (neutral sites), so each drive's
    offense name is first resolved through a committed name -> id crosswalk
    (cfbd_team_crosswalk.json, majority vote over 2021-2025 drives).

Paths:
  * weekly (`mode=drives` in main.py): fetch the season's completed weeks that have no
    drives yet (at most MAX_WEEKS_PER_RUN calls), replace those games' rows only;
  * once, locally, history 2021-2026:
        CFBD_API_KEY="$(cat ~/.cfbd_key)" python src/cfb/cfb_drives.py \\
            --seasons 2021-2026 --cfbd-cache <dir of cached CFBD json> --write
    (without --write it only prints counts / writes --out parquet).

The transform is the research one (research/football_2026_09/cfb_drive_sim/build_drives.py)
so the live model trains on the same rows the backtest did.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)

TABLE = "drives"
MAX_WEEKS_PER_RUN = 3
CROSSWALK_PATH = Path(__file__).resolve().parent / "cfbd_team_crosswalk.json"

DRIVE_COLUMNS = [
    "game_id", "season", "week", "season_type", "home_team", "away_team", "posteam",
    "defteam", "qtr", "game_half", "res", "hsr0", "yl", "sd0", "off_pts", "def_pts",
    "off_home", "yl_next", "dur", "drive_number", "cfbd_offense", "cfbd_defense", "division",
]

_TD = {"TD", "PASSING TD", "RUSHING TD"}
_FG = {"FG", "FG GOOD"}
_MFG = {"MISSED FG", "FG MISSED", "BLOCKED FG"}
_PUNT = {"PUNT", "BLOCKED PUNT"}
_TO = {"INT", "FUMBLE"}
_EOH = {"END OF HALF", "END OF GAME", "END OF 4TH QUARTER"}
_DEF_SCORE = {"INT TD", "FUMBLE RETURN TD", "FUMBLE TD", "PUNT RETURN TD", "PUNT TD",
              "MISSED FG TD", "INT RETURN TOUCH"}
_AMBIGUOUS = _DEF_SCORE | {"DOWNS TD", "END OF HALF TD", "END OF GAME TD", "FG TD",
                           "Uncategorized", "KICKOFF", "POSSESSION (FOR OT DRIVES)"}


def load_crosswalk() -> dict[str, str]:
    try:
        return json.loads(CROSSWALK_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("no CFBD team crosswalk; falling back to isHomeOffense")
        return {}


def classify(result: pd.Series, dop: pd.Series, dde: pd.Series) -> pd.Series:
    """CFBD driveResult + score deltas -> TD FG MFG PUNT TO OTD TOD SAF EOH (or None).

    Score deltas settle the ambiguous labels ("FUMBLE TD" is sometimes the offence's);
    unresolvable rows (mostly zero-point "Uncategorized") return None and are dropped.
    """
    r = result.fillna("Uncategorized")
    res = pd.Series(None, index=r.index, dtype=object)
    res[r.isin(_TD)] = "TD"
    res[r.isin(_FG)] = "FG"
    res[r.isin(_MFG)] = "MFG"
    res[r.isin(_PUNT)] = "PUNT"
    res[r.isin(_TO)] = "TO"
    res[r == "DOWNS"] = "TOD"
    res[r == "SF"] = "SAF"
    res[r.isin(_EOH)] = "EOH"
    amb = r.isin(_AMBIGUOUS)
    res[amb & (dop >= 6) & (dde < 6)] = "TD"
    res[amb & (dde >= 6) & (dop < 6)] = "OTD"
    res[r.isin(_DEF_SCORE) & res.isna()] = "OTD"
    res[amb & res.isna() & (dop == 3)] = "FG"
    res[amb & res.isna() & (dde == 2)] = "SAF"
    res[(r == "DOWNS TD") & res.isna()] = "TOD"
    res[r.isin({"END OF HALF TD", "END OF GAME TD"}) & res.isna()] = "EOH"
    return res


def transform(records: list[dict], games: pd.DataFrame,
              crosswalk: dict[str, str] | None = None) -> pd.DataFrame:
    """CFBD /drives records -> DRIVE_COLUMNS rows for games present in `games`.

    `games` is cfb_historical.games-shaped (game_id, season, week, is_postseason,
    home_team, away_team, division). Season/week come from the BigQuery game, not the
    CFBD call, so FCS playoff weeks line up with the schedule the model indexes on.
    """
    if not records:
        return pd.DataFrame(columns=DRIVE_COLUMNS)
    crosswalk = load_crosswalk() if crosswalk is None else crosswalk
    d = pd.json_normalize(records)
    d["game_id"] = d["gameId"].astype(str)
    g = games[["game_id", "season", "week", "is_postseason", "home_team", "away_team",
               "division"]].drop_duplicates("game_id")
    d = d.merge(g, on="game_id", how="inner")
    if d.empty:
        return pd.DataFrame(columns=DRIVE_COLUMNS)
    off_id = d["offense"].map(crosswalk)
    def_id = d["defense"].map(crosswalk)
    ok_home = (off_id == d.home_team) & (def_id == d.away_team)
    ok_away = (off_id == d.away_team) & (def_id == d.home_team)
    home_off = np.where(ok_home, True, np.where(ok_away, False, d["isHomeOffense"].astype(bool)))
    d["posteam"] = np.where(home_off, d.home_team, d.away_team)
    d["defteam"] = np.where(home_off, d.away_team, d.home_team)
    d["off_home"] = home_off.astype(bool)

    dop = d["endOffenseScore"] - d["startOffenseScore"]
    dde = d["endDefenseScore"] - d["startDefenseScore"]
    d["res"] = classify(d["driveResult"], dop, dde)
    keep = d["res"].notna()
    d, dop, dde = d[keep].copy(), dop[keep], dde[keep]

    per = d["startPeriod"].clip(lower=1)
    minutes = d.get("startTime.minutes", pd.Series(15, index=d.index)).fillna(15)
    seconds = d.get("startTime.seconds", pd.Series(0, index=d.index)).fillna(0)
    secq = (minutes * 60 + seconds).clip(0, 900)
    d["game_half"] = np.where(per <= 2, "Half1", np.where(per <= 4, "Half2", "Overtime"))
    d["hsr0"] = np.where(per.isin([1, 3]), 900 + secq, np.where(per.isin([2, 4]), secq, 900))
    d["qtr"] = per.astype(float)
    d["yl"] = d["startYardsToGoal"].astype(float).where(lambda s: s.between(1, 99))
    d["sd0"] = (d["startOffenseScore"] - d["startDefenseScore"]).astype(float)
    d["off_pts"] = np.where(d.res == "TD", dop.where(dop.isin([6, 7, 8]), 7),
                            np.where(d.res == "FG", 3, 0)).astype(float)
    d["def_pts"] = np.where(d.res == "OTD", dde.where(dde.isin([6, 7, 8]), 7),
                            np.where(d.res == "SAF", 2, 0)).astype(float)
    d["drive_number"] = pd.to_numeric(d["driveNumber"], errors="coerce")
    d = d.sort_values(["game_id", "drive_number", "qtr", "hsr0"],
                      ascending=[True, True, True, False])
    grp = d.groupby(["game_id", "game_half"])
    d["hsr_next"] = grp["hsr0"].shift(-1)
    d["yl_next"] = grp["yl"].shift(-1)
    d["dur"] = (d["hsr0"] - d["hsr_next"].fillna(0)).clip(0, 1800).astype(float)
    d["season_type"] = np.where(d["is_postseason"].fillna(0).astype(int) == 1, "POST", "REG")
    d["cfbd_offense"] = d["offense"]
    d["cfbd_defense"] = d["defense"]
    out = d[DRIVE_COLUMNS].reset_index(drop=True)
    out["season"] = out["season"].astype("int64")
    out["week"] = out["week"].astype("int64")
    out["drive_number"] = out["drive_number"].astype(float)
    return out


def fetch_week(season: int, week: int | None, season_type: str = "regular",
               ttl_hours: float | None = None) -> list[dict]:
    """One /drives call (disk-cached by stats.cfbd). week=None fetches a postseason."""
    from stats import cfbd

    params = {"year": int(season), "seasonType": season_type}
    if week is not None:
        params["week"] = int(week)
    return cfbd.get("/drives", params, ttl_hours=ttl_hours) or []


def missing_weeks(games: pd.DataFrame, have: pd.DataFrame, season: int) -> list[int]:
    """Regular-season weeks of `season` with completed games but no drive rows yet."""
    g = games[(games["season"] == season) & games["home_won"].notna()
              & (games["is_postseason"].fillna(0).astype(int) == 0)]
    have_ids = set(have["game_id"].astype(str)) if len(have) else set()
    need = g[~g["game_id"].astype(str).isin(have_ids)]
    return sorted(need["week"].astype(int).unique().tolist())


def ingest(season: int, dry_run: bool = False, weeks: list[int] | None = None) -> dict:
    """Weekly path: fetch completed weeks lacking drives, replace those games' rows.

    Scoped to game_id (backfill_cfb.replace_game_ids) into a table that must already
    exist (CREATE_NEVER). Under dry_run it fetches and transforms but writes nothing to
    BigQuery (the CFBD responses still land in the /tmp disk cache).
    """
    import cfb_config
    from pipeline import load_played_games

    games = load_played_games()
    have = load_drives(season, source="bq")
    todo = weeks if weeks is not None else missing_weeks(games, have, season)
    skipped = todo[MAX_WEEKS_PER_RUN:]
    todo = todo[:MAX_WEEKS_PER_RUN]
    frames = []
    for w in todo:
        # The current season's feed is still being corrected; completed weeks are cached
        # for 12 h rather than forever.
        frames.append(transform(fetch_week(season, w, ttl_hours=12), games))
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DRIVE_COLUMNS)
    info = {"season": season, "weeks": todo, "deferred_weeks": skipped, "drives": len(rows),
            "games": int(rows["game_id"].nunique()) if len(rows) else 0}
    if dry_run or rows.empty:
        return {**info, "written": 0, "dry_run": bool(dry_run)}
    from backfill_cfb import ensure_datasets, replace_game_ids

    ensure_datasets()
    info["written"] = replace_game_ids(rows, cfb_config.CTX.hist_dataset, TABLE,
                                       cluster_fields=["season", "posteam"],
                                       create_disposition="CREATE_NEVER")
    return info


def _plain_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """BigQuery hands back nullable Int64/Float64 extension columns. The simulator does
    numpy linear algebra on them, and under pandas 2.3 (the Cloud Function's pin) a
    matrix product with a masked array raises; pandas 3 happens to tolerate it. Convert
    to plain numpy dtypes: float where there are NULLs, int64 otherwise."""
    out = df.copy()
    for c in out.columns:
        dt = out[c].dtype
        if isinstance(dt, pd.api.extensions.ExtensionDtype) and pd.api.types.is_numeric_dtype(dt):
            s = out[c]
            if pd.api.types.is_integer_dtype(dt) and not s.isna().any():
                out[c] = s.astype("int64")
            elif pd.api.types.is_bool_dtype(dt) and not s.isna().any():
                out[c] = s.astype(bool)
            else:
                out[c] = s.astype("float64")
    return out


def load_drives(first_season: int, source: str = "auto") -> pd.DataFrame:
    """Drives for seasons >= first_season: BigQuery, or a local parquet
    (env CFB_DRIVES_PARQUET) when source is "auto" and BigQuery has nothing."""
    import os

    if source in ("auto", "bq"):
        try:
            import cfb_config
            from google.cloud import bigquery

            cfg = bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("s", "INT64", int(first_season))])
            table = f"{cfb_config.CTX.project}.{cfb_config.CTX.hist_dataset}.{TABLE}"
            df = bigquery.Client(project=cfb_config.CTX.project).query(
                f"SELECT * FROM `{table}` WHERE season >= @s", job_config=cfg).to_dataframe()
            if not df.empty or source == "bq":
                return _plain_numeric(df)
        except Exception as exc:
            if source == "bq":
                logger.warning("drives BigQuery read failed (%s)", exc)
                return pd.DataFrame(columns=DRIVE_COLUMNS)
            logger.warning("drives BigQuery read failed (%s); trying local parquet", exc)
    path = os.environ.get("CFB_DRIVES_PARQUET")
    if path and Path(path).exists():
        df = pd.read_parquet(path)
        return df[df["season"] >= first_season]
    return pd.DataFrame(columns=DRIVE_COLUMNS)


def main() -> int:
    ap = argparse.ArgumentParser(description="backfill cfb_historical.drives from CFBD /drives")
    ap.add_argument("--seasons", default="2021-2026")
    ap.add_argument("--cfbd-cache", help="directory of cached CFBD responses (reused; "
                    "only cache misses spend API calls)")
    ap.add_argument("--games-parquet", help="cfb_historical.games snapshot (else BigQuery)")
    ap.add_argument("--out", help="also write a local parquet here")
    ap.add_argument("--write", action="store_true", help="replace these games in BigQuery")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Local runs: stats/ and rankings/ are packages under src/ (flat in the deployed tree).
    sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

    from stats import cfbd

    if a.cfbd_cache:
        cfbd.CACHE = Path(a.cfbd_cache)
    cfbd.MAX_CALLS_PER_RUN = 120   # a deliberate one-off: ~100 weeks, nearly all cached

    if a.games_parquet:
        games = pd.read_parquet(a.games_parquet)
    else:
        from pipeline import load_played_games
        games = load_played_games()
    lo, hi = (int(x) for x in a.seasons.split("-"))
    frames = []
    for s in range(lo, hi + 1):
        gs = games[(games["season"] == s) & games["home_won"].notna()]
        reg = sorted(gs.loc[gs["is_postseason"].fillna(0).astype(int) == 0, "week"]
                     .astype(int).unique().tolist())
        recs = []
        for w in reg:
            recs += fetch_week(s, w)
        if (gs["is_postseason"].fillna(0).astype(int) == 1).any():
            recs += fetch_week(s, None, "postseason")
        d = transform(recs, games)
        frames.append(d)
        print(s, "weeks", reg, "drives", len(d), "games", d["game_id"].nunique(),
              "cfbd calls so far", cfbd.calls_used(), flush=True)
    d = pd.concat(frames, ignore_index=True)
    if a.out:
        d.to_parquet(a.out)
    if a.write:
        import cfb_config
        from backfill_cfb import replace_game_ids

        n = replace_game_ids(d, cfb_config.CTX.hist_dataset, TABLE,
                             cluster_fields=["season", "posteam"],
                             create_disposition="CREATE_NEVER")
        print("wrote", n)
    print("total drives", len(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
