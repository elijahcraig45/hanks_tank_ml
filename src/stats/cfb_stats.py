"""College football team and player season stats from ESPN.

Fills the gap the college pipeline has always had: it carries games and ratings but no
box-score detail, so the site could only ever show "no stats feed for this sport". ESPN
publishes both, keyless, at season granularity.

Parsing note, shared with the FPI ingest: ESPN returns bare parallel arrays per team or
athlete, with field names declared once in a top-level `categories` block. So the
numbers are meaningless without zipping them against that block — there are no labelled
keys to read.

Team stats arrive as two splits per category, the team's own and its opponents'
(splitId 0 and 1). Both are kept and prefixed, because "opponent yards allowed" is the
only defensive information this feed carries.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from pathlib import Path

import pandas as pd

try:  # package layout locally, flat module tree in Cloud Functions
    from rankings.http import get_bytes as _get_bytes, cache_dir as _cache_dir
except ImportError:  # pragma: no cover
    from http_transport import get_bytes as _get_bytes, cache_dir as _cache_dir

logger = logging.getLogger(__name__)

# Writable wherever this runs; /workspace is read-only in Cloud Functions.
CACHE = _cache_dir("stats")

BASE = "https://site.web.api.espn.com/apis/common/v3/sports/football/college-football"
REGULAR_SEASON = 2

# ESPN's split ids on the team-stats endpoint: "0" is the team's own production and
# "900" is what its opponents did against it. (Not "1" — that is a different split
# family, and assuming it silently drops every defensive column.)
SPLIT_OWN = "0"
SPLIT_OPPONENT = "900"

# ESPN caps a page at 100 and paginates athletes into the hundreds of pages.
PAGE_SIZE = 100


def _curl(url: str, timeout: int = 90) -> bytes:
    """Delegates to the shared transport; see rankings.http for why."""
    return _get_bytes(url, timeout)


def _get(path: str, **params) -> dict:
    query = urllib.parse.urlencode(params)
    return json.loads(_curl(f"{BASE}/{path}?{query}").decode())


def _field_order(payload: dict) -> dict[str, list[str]]:
    """Category name -> ordered field names, declared once at the top level."""
    return {c["name"]: c.get("names", []) for c in payload.get("categories", [])}


def _flatten(entry: dict, order: dict[str, list[str]], prefix_by_split: bool) -> dict:
    """Zip an entity's parallel value arrays against the declared field names."""
    out: dict = {}
    for category in entry.get("categories", []):
        names = order.get(category.get("name"), [])
        split = str(category.get("splitId", SPLIT_OWN))
        prefix = "opp_" if (prefix_by_split and split == SPLIT_OPPONENT) else ""
        for name, value in zip(names, category.get("values", [])):
            out[f"{prefix}{name}"] = value
    return out


# ── Team season stats ───────────────────────────────────────────────────────
def fetch_team_stats(season: int, refresh: bool = False) -> pd.DataFrame:
    cached = CACHE / f"cfb_team_stats_{season}.json"
    if cached.exists() and not refresh:
        payload = json.loads(cached.read_text())
    else:
        payload = _get("statistics/byteam", season=season, seasontype=REGULAR_SEASON,
                       limit=400)
        cached.write_text(json.dumps(payload))

    order = _field_order(payload)
    rows = []
    for entry in payload.get("teams", []):
        team = entry.get("team", {})
        row = {
            "season": season,
            "team": team.get("displayName"),
            "team_abbr": team.get("abbreviation"),
            "espn_team_id": team.get("id"),
        }
        row.update(_flatten(entry, order, prefix_by_split=True))
        rows.append(row)

    df = pd.DataFrame(rows)
    # Identifiers, not quantities. Left to inference they land as INTEGER whenever a
    # round-trip through CSV makes them look numeric, which then conflicts with the
    # strings the live feed actually returns.
    for column in ("espn_team_id", "team_abbr", "team"):
        if column in df.columns:
            df[column] = df[column].astype("string")
    logger.info("CFB team stats %d: %d teams, %d columns", season, len(df), len(df.columns))
    return df


# ── League leaders ──────────────────────────────────────────────────────────
#
# There is no public ESPN endpoint that returns a full per-player college season table.
# `statistics/byathlete` sorts correctly by a stat but returns "-" for every value of
# it — including for the player it just ranked first — and passing `category=` to it
# 400s. So college gets leaders rather than a player database; the NFL, which has
# nflverse, gets the full table.
#
# The core API returns leaders as $ref links, so athlete and team names take a second
# fetch. Refs repeat heavily across categories, hence the dedupe and the shared cache.

CORE = ("https://sports.core.api.espn.com/v2/sports/football/leagues/"
        "college-football/seasons")

LEADER_LABELS = {
    "passingYards": "Passing yards",
    "passingTouchdowns": "Passing TDs",
    "quarterbackRating": "QB rating",
    "rushingYards": "Rushing yards",
    "rushingTouchdowns": "Rushing TDs",
    "receivingYards": "Receiving yards",
    "receptions": "Receptions",
    "receivingTouchdowns": "Receiving TDs",
    "totalTackles": "Tackles",
    "sacks": "Sacks",
    "interceptions": "Interceptions",
}


def _resolve_names(refs: set[str], workers: int = 12) -> dict[str, str]:
    """Fetch a batch of $ref URLs and pull a display name out of each."""
    from concurrent.futures import ThreadPoolExecutor

    def one(ref: str) -> tuple[str, str | None]:
        try:
            payload = json.loads(_curl(ref, timeout=30).decode())
            return ref, payload.get("displayName") or payload.get("name")
        except Exception:
            return ref, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {ref: name for ref, name in pool.map(one, refs) if name}


def fetch_leaders(season: int, refresh: bool = False) -> pd.DataFrame:
    """Top players per statistical category, long-form: one row per leader."""
    cached = CACHE / f"cfb_leaders_{season}.json"
    if cached.exists() and not refresh:
        payload = json.loads(cached.read_text())
    else:
        payload = json.loads(
            _curl(f"{CORE}/{season}/types/{REGULAR_SEASON}/leaders?lang=en").decode()
        )
        cached.write_text(json.dumps(payload))

    raw = []
    refs: set[str] = set()
    for category in payload.get("categories", []):
        name = category.get("name")
        if name not in LEADER_LABELS:
            continue
        for rank, leader in enumerate(category.get("leaders", []), start=1):
            athlete_ref = (leader.get("athlete") or {}).get("$ref")
            team_ref = (leader.get("team") or {}).get("$ref")
            if athlete_ref:
                refs.add(athlete_ref)
            if team_ref:
                refs.add(team_ref)
            raw.append({
                "season": season,
                "sport": "cfb",
                "category": name,
                "category_label": LEADER_LABELS[name],
                "rank": rank,
                "value": leader.get("value"),
                "display_value": leader.get("displayValue"),
                "_athlete_ref": athlete_ref,
                "_team_ref": team_ref,
            })

    names = _resolve_names(refs) if refs else {}
    for row in raw:
        row["player_name"] = names.get(row.pop("_athlete_ref"))
        row["team"] = names.get(row.pop("_team_ref"))

    df = pd.DataFrame(raw)
    for column in ("player_name", "team", "category", "category_label", "display_value"):
        if column in df.columns:
            df[column] = df[column].astype("string")
    logger.info("CFB leaders %d: %d rows across %d categories",
                season, len(df), df["category"].nunique() if not df.empty else 0)
    return df
