"""ESPN Football Power Index ingest — strength of record, strength of schedule, efficiency.

The Bradley-Terry board answers "who is best on the results". FPI answers adjacent
questions it deliberately does not model: how hard was the schedule, how good is the
resume, how efficient is each unit. Both are worth showing side by side, and keeping
them in separate tables keeps our rating independent of ESPN's — this is a supplement,
not an input.

Available for NFL and college football; there is no FPI for baseball, so MLB rankings
carry no SOR/SOS from here.

Endpoint shape: ESPN returns each team's numbers as bare parallel arrays, with the
field names living once in a top-level `categories` block. Parsing therefore means
zipping names to values per category rather than reading labelled keys.
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
CACHE = _cache_dir("rankings")

BASE = "https://site.web.api.espn.com/apis/fitt/v3/sports/football"
LEAGUE_PATH = {"nfl": "nfl", "cfb": "college-football"}

# Only the fields worth surfacing. ESPN exposes ~30; the rest are simulation
# probabilities that change hourly and would age badly in a nightly table.
FIELDS = {
    "fpi": "fpi",
    "fpirank": "fpi_rank",
    # NFL only; ESPN does not publish per-unit EPA for college.
    "epaoffense": "epa_offense",
    "epadefense": "epa_defense",
    "epaspecialteams": "epa_special_teams",
    "accomplishmentrank": "sor_rank",          # strength of record
    "avgsosrank": "sos_rank",                  # strength of schedule to date
    "sosremainingrank": "sos_remaining_rank",
    "gamecontrolrank": "game_control_rank",
    "totefficiency": "eff_total",
    "totefficiencyrank": "eff_total_rank",
    "offefficiency": "eff_offense",
    "offefficiencyrank": "eff_offense_rank",
    "defefficiency": "eff_defense",
    "defefficiencyrank": "eff_defense_rank",
    "stefficiency": "eff_special_teams",
    "stefficiencyrank": "eff_special_teams_rank",
    "projectedw": "projected_wins",
    "projectedl": "projected_losses",
    "probmakeplayoffs": "playoff_pct",
    "APRank/CFPRank": "ap_rank",
}


def _curl(url: str, timeout: int = 90) -> bytes:
    """Delegates to the shared transport; see rankings.http for why."""
    return _get_bytes(url, timeout)


def _fetch_page(sport: str, season: int, page: int, limit: int = 200) -> dict:
    query = urllib.parse.urlencode({"season": season, "limit": limit, "page": page})
    url = f"{BASE}/{LEAGUE_PATH[sport]}/powerindex?{query}"
    return json.loads(_curl(url).decode())


def fetch(sport: str, season: int, refresh: bool = False) -> pd.DataFrame:
    """One row per team, with the FPI fields named rather than positional."""
    if sport not in LEAGUE_PATH:
        raise ValueError(f"no FPI for sport: {sport}")

    cached = CACHE / f"fpi_{sport}_{season}.json"
    if cached.exists() and not refresh:
        pages = json.loads(cached.read_text())
    else:
        first = _fetch_page(sport, season, 1)
        total = int(first.get("pagination", {}).get("pages", 1))
        pages = [first] + [_fetch_page(sport, season, p) for p in range(2, total + 1)]
        cached.write_text(json.dumps(pages))

    # Field order is declared once per category at the top level, not per team.
    order: dict[str, list[str]] = {
        c["name"]: c.get("names", []) for c in pages[0].get("categories", [])
    }

    rows = []
    for page in pages:
        for entry in page.get("teams", []):
            team = entry.get("team", {})
            row = {
                "season": season,
                "sport": sport,
                "team": team.get("displayName"),
                "team_abbr": team.get("abbreviation"),
                "espn_team_id": team.get("id"),
            }
            for category in entry.get("categories", []):
                names = order.get(category.get("name"), [])
                for name, value in zip(names, category.get("values", [])):
                    column = FIELDS.get(name)
                    if column:
                        row[column] = value
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("no FPI rows for %s %d", sport, season)
        return df

    # Ranks come back as floats; they are ordinals and read badly as "1.0".
    #
    # Zero is not a rank. ESPN returns 0 for a metric it does not compute for a league
    # — strength of record and game control are college playoff concepts and come back
    # 0 for every NFL team — and for a season with no games played yet. Left as-is that
    # renders as a suspiciously good "0th"; as null the UI can say "not available",
    # which is the truth.
    for column in [c for c in df.columns if c.endswith("_rank")]:
        numeric = pd.to_numeric(df[column], errors="coerce")
        df[column] = numeric.where(numeric > 0).astype("Int64")

    logger.info("FPI %s %d: %d teams", sport, season, len(df))
    return df.sort_values("fpi_rank").reset_index(drop=True)


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", required=True, choices=sorted(LEAGUE_PATH))
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    df = fetch(args.sport, args.season, refresh=args.refresh)
    if df.empty:
        print("no rows")
        return 1

    show = ["fpi_rank", "team", "fpi", "sor_rank", "sos_rank", "eff_offense",
            "eff_defense", "projected_wins", "ap_rank"]
    show = [c for c in show if c in df.columns]
    print()
    print(f"ESPN FPI — {args.sport.upper()} {args.season} ({len(df)} teams)")
    print(df[show].head(args.top).to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ESPN and nflverse disagree on exactly two NFL abbreviations. Everything else in the
# 32 matches, so an explicit two-entry map beats a fuzzy name matcher.
ESPN_TO_NFLVERSE = {"LAR": "LA", "WSH": "WAS"}

ATTACH_COLUMNS = [
    "fpi", "fpi_rank", "sor_rank", "sos_rank", "sos_remaining_rank",
    "game_control_rank", "eff_total", "eff_offense", "eff_defense",
    "eff_special_teams", "epa_offense", "epa_defense", "epa_special_teams",
    "projected_wins", "projected_losses", "playoff_pct", "ap_rank",
]


def attach(board: pd.DataFrame, sport: str, season: int,
           refresh: bool = False) -> pd.DataFrame:
    """Left-join FPI onto a power-rankings board.

    Left join on purpose: our board is the source of truth for who is ranked. FPI covers
    FBS only, so every FCS row keeps null FPI columns rather than being dropped, and a
    fetch failure degrades to a board without the extra columns instead of no board.

    Matching differs by sport because the two feeds identify teams differently: college
    names come from ESPN on both sides and match exactly, while the NFL board uses
    nflverse abbreviations.
    """
    if sport not in LEAGUE_PATH:
        return board

    try:
        extra = fetch(sport, season, refresh=refresh)
    except Exception as exc:
        logger.warning("FPI unavailable for %s %d (%s); board keeps its own columns",
                       sport, season, exc)
        return board
    if extra.empty:
        return board

    extra = extra.copy()
    if sport == "nfl":
        extra["join_key"] = (
            extra["team_abbr"].replace(ESPN_TO_NFLVERSE)
        )
    else:
        extra["join_key"] = extra["team"]

    keep = ["join_key"] + [c for c in ATTACH_COLUMNS if c in extra.columns]
    merged = board.merge(
        extra[keep], how="left", left_on="team", right_on="join_key"
    ).drop(columns=["join_key"])

    matched = merged["fpi"].notna().sum() if "fpi" in merged.columns else 0
    logger.info("FPI attached to %d/%d %s rows", matched, len(merged), sport)
    return merged
