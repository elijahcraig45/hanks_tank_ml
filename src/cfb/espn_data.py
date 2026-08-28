"""College football ingestion from ESPN's public API.

Chosen over CollegeFootballData because it needs no API key, and over per-date
scoreboard calls because the week-indexed endpoint returns a whole slate at once:
roughly 16 calls per season per division rather than ~150.

FBS (groups=80) and FCS (groups=81) are fetched separately and tagged with a
`division` column. They land in the same table with the same schema — the split that
matters is at model training, not storage.
"""

from __future__ import annotations

import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

import pandas as pd
import requests

from cfb_config import DIVISION_GROUPS, ESPN_BASE, FIRST_SEASON, RAW_CACHE

logger = logging.getLogger(__name__)

GAMES_CACHE = RAW_CACHE / "cfb_games.parquet"

REGULAR_SEASON, POSTSEASON = 2, 3
MAX_REGULAR_WEEK = 16
MAX_POST_WEEK = 5


# Transport selection.
#
# Python's requests fails on this machine: it sits behind a Netskope / Home Depot
# TLS-inspecting proxy whose root ("The Home Depot Low Assurance Root PR CAv2")
# predates the Authority Key Identifier requirement, so OpenSSL 3 rejects it even when
# explicitly trusted. curl validates the same chain fine against the macOS trust store.
#
# Both paths verify certificates — this is a trust-store difference, not a bypass. The
# probe runs once and the result is cached, because paying a 30s TLS timeout on every
# one of ~400 requests turns a 2-minute backfill into a 45-minute one. In Cloud
# Functions there is no interception, so the probe passes and requests is used.
_USE_CURL: bool | None = None


def _probe_transport() -> bool:
    global _USE_CURL
    if _USE_CURL is None:
        try:
            requests.get(f"{ESPN_BASE}/scoreboard", params={"limit": 1}, timeout=15)
            _USE_CURL = False
        except requests.exceptions.SSLError:
            logger.info("TLS interception detected — using curl transport")
            _USE_CURL = True
        except Exception:
            _USE_CURL = False
    return _USE_CURL


def _get_json(url: str, params: dict) -> dict:
    """Fetch JSON, verifying TLS via whichever transport trusts this network."""
    if not _probe_transport():
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    out = subprocess.run(
        ["curl", "-sS", "--fail", "--max-time", "30", f"{url}?{urlencode(params)}"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def _fetch_week(season: int, week: int, group: int, seasontype: int) -> list[dict]:
    params = {
        "dates": season, "seasontype": seasontype, "week": week,
        "groups": group, "limit": 400,
    }
    return _get_json(f"{ESPN_BASE}/scoreboard", params).get("events", [])


def _parse_event(ev: dict, season: int, week: int, division: str,
                 seasontype: int, require_completed: bool = True) -> dict | None:
    """Parse one ESPN event.

    With require_completed=False, scheduled-but-unplayed games are returned with null
    scores and home_won — that is what the weekly prediction path consumes.
    """
    try:
        comp = ev["competitions"][0]
        teams = comp["competitors"]
        home = next(t for t in teams if t["homeAway"] == "home")
        away = next(t for t in teams if t["homeAway"] == "away")

        # ESPN puts status on the event; some payloads repeat it on the competition.
        status = ev.get("status") or comp.get("status") or {}
        completed = bool(status.get("type", {}).get("completed"))

        if require_completed and not completed:
            return None

        if completed:
            hs, as_ = int(home.get("score")), int(away.get("score"))
            if hs == as_:
                return None  # overtime makes ties vanishingly rare, but guard anyway
            home_won = int(hs > as_)
            result = hs - as_
        else:
            hs = as_ = home_won = result = None

        return {
            "game_id": str(ev["id"]),
            "season": season,
            "week": week,
            "division": division,
            "is_postseason": int(seasontype == POSTSEASON),
            "game_date": pd.to_datetime(ev["date"]).tz_localize(None),
            "home_team": home["team"].get("abbreviation") or home["team"]["id"],
            "away_team": away["team"].get("abbreviation") or away["team"]["id"],
            "home_team_name": home["team"].get("displayName"),
            "away_team_name": away["team"].get("displayName"),
            "home_score": hs,
            "away_score": as_,
            "home_won": home_won,
            "result": result,
            "neutral_site": int(bool(comp.get("neutralSite"))),
            "conference_game": int(bool(comp.get("conferenceCompetition"))),
            "home_conference_id": (home["team"].get("conferenceId")),
            "away_conference_id": (away["team"].get("conferenceId")),
            "venue": comp.get("venue", {}).get("fullName"),
        }
    except Exception:
        return None


def fetch_history(first_season: int = FIRST_SEASON, last_season: int = 2025,
                  divisions: tuple[str, ...] = ("fbs", "fcs"),
                  refresh: bool = False, workers: int = 12) -> pd.DataFrame:
    """Fetch completed games for the given seasons and divisions, with a disk cache.

    Week fetches are independent, so they run concurrently — serially this is ~2s per
    call across ~400 calls, which is 15 minutes for no reason.
    """
    RAW_CACHE.mkdir(parents=True, exist_ok=True)

    cached = pd.read_parquet(GAMES_CACHE) if GAMES_CACHE.exists() and not refresh \
        else pd.DataFrame()

    # Build the work list, skipping season/division pairs already cached.
    jobs = []
    for division in divisions:
        group = DIVISION_GROUPS[division]
        for season in range(first_season, last_season + 1):
            if not cached.empty:
                have = cached[(cached["season"] == season)
                              & (cached["division"] == division)]
                if len(have) > 100:
                    continue
            for seasontype, max_week in ((REGULAR_SEASON, MAX_REGULAR_WEEK),
                                         (POSTSEASON, MAX_POST_WEEK)):
                for week in range(1, max_week + 1):
                    jobs.append((season, week, group, seasontype, division))

    if not jobs:
        logger.info("cfb cache already covers %d-%d", first_season, last_season)
        return cached

    logger.info("fetching %d week-slates with %d workers", len(jobs), workers)
    _probe_transport()  # resolve transport once, before the pool fans out

    rows: list[dict] = []

    def _one(job):
        season, week, group, seasontype, division = job
        try:
            events = _fetch_week(season, week, group, seasontype)
        except Exception as exc:
            logger.warning("%s %d wk%d (type %d) failed: %s",
                           division, season, week, seasontype, exc)
            return []
        wk = week if seasontype == REGULAR_SEASON else MAX_REGULAR_WEEK + week
        out = [_parse_event(ev, season, wk, division, seasontype) for ev in events]
        return [o for o in out if o]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for got in pool.map(_one, jobs):
            rows.extend(got)

    fresh = pd.DataFrame(rows)
    combined = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
    if combined.empty:
        return combined

    # FBS-vs-FCS games appear in BOTH group feeds. Divisions are iterated
    # fbs-first and we keep the first sighting, so a cross-division game is
    # tagged fbs rather than being relabelled by whichever feed ran last.
    combined = combined.drop_duplicates(subset=["game_id"], keep="first")
    combined = combined.sort_values(["season", "week", "game_date"]).reset_index(drop=True)
    combined.to_parquet(GAMES_CACHE, index=False)
    logger.info("cfb games cached: %d rows, %d seasons",
                len(combined), combined["season"].nunique())
    return combined


def load_games() -> pd.DataFrame:
    if GAMES_CACHE.exists():
        return pd.read_parquet(GAMES_CACHE)
    return fetch_history()


def fetch_scheduled(season: int, week: int,
                    divisions: tuple[str, ...] = ("fbs", "fcs")) -> pd.DataFrame:
    """Scheduled games for one week, played or not.

    The prediction path needs the slate before it happens; fetch_history only keeps
    completed games because it feeds the training set.
    """
    rows = []
    for division in divisions:
        group = DIVISION_GROUPS[division]
        for seasontype, offset in ((REGULAR_SEASON, 0), (POSTSEASON, MAX_REGULAR_WEEK)):
            wk = week - offset
            if wk < 1 or (seasontype == REGULAR_SEASON and week > MAX_REGULAR_WEEK):
                continue
            try:
                events = _fetch_week(season, wk, group, seasontype)
            except Exception as exc:
                logger.warning("%s %d wk%d failed: %s", division, season, week, exc)
                continue
            for ev in events:
                parsed = _parse_event(ev, season, week, division, seasontype,
                                      require_completed=False)
                if parsed:
                    rows.append(parsed)

    df = pd.DataFrame(rows)
    if not df.empty:
        # See fetch_history: cross-division games show up in both feeds; fbs is
        # iterated first and wins the tag.
        df = df.drop_duplicates(subset=["game_id"], keep="first")
    logger.info("scheduled %d %d wk%d: %d games", season, season, week, len(df))
    return df
