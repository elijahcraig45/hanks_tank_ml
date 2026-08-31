"""Conference membership and the human polls, attached to a rankings board for context.

Two things a rating cannot tell you and people immediately want: which conference a
team plays in, and how the humans see it. Neither feeds the fit — they are display
columns, so the board stays independent of the AP voters and of ESPN's own index.

Conference lookup is by ESPN group id. Games already carry each team's `conferenceId`,
and the group hierarchy resolves it to a name: 90 (Division I) -> 80/81 (FBS/FCS) ->
one child per conference.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

try:  # package layout locally, flat module tree in Cloud Functions
    from rankings.http import get_bytes as _get_bytes, cache_dir as _cache_dir
except ImportError:  # pragma: no cover
    from http_transport import get_bytes as _get_bytes, cache_dir as _cache_dir

logger = logging.getLogger(__name__)

CACHE = _cache_dir("rankings")

CORE = ("https://sports.core.api.espn.com/v2/sports/football/leagues/"
        "college-football/seasons")
SITE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"

DIVISION_GROUPS = {"fbs": 80, "fcs": 81}

# NFL conference and division are fixed and small, so a literal beats a feed: no
# request to fail, and relocations change the city, never the division.
NFL_DIVISIONS = {
    "BUF": ("AFC", "AFC East"),  "MIA": ("AFC", "AFC East"),
    "NE": ("AFC", "AFC East"),   "NYJ": ("AFC", "AFC East"),
    "BAL": ("AFC", "AFC North"), "CIN": ("AFC", "AFC North"),
    "CLE": ("AFC", "AFC North"), "PIT": ("AFC", "AFC North"),
    "HOU": ("AFC", "AFC South"), "IND": ("AFC", "AFC South"),
    "JAX": ("AFC", "AFC South"), "TEN": ("AFC", "AFC South"),
    "DEN": ("AFC", "AFC West"),  "KC": ("AFC", "AFC West"),
    "LAC": ("AFC", "AFC West"),  "LV": ("AFC", "AFC West"),
    "DAL": ("NFC", "NFC East"),  "NYG": ("NFC", "NFC East"),
    "PHI": ("NFC", "NFC East"),  "WAS": ("NFC", "NFC East"),
    "CHI": ("NFC", "NFC North"), "DET": ("NFC", "NFC North"),
    "GB": ("NFC", "NFC North"),  "MIN": ("NFC", "NFC North"),
    "ATL": ("NFC", "NFC South"), "CAR": ("NFC", "NFC South"),
    "NO": ("NFC", "NFC South"),  "TB": ("NFC", "NFC South"),
    "ARI": ("NFC", "NFC West"),  "LA": ("NFC", "NFC West"),
    "SF": ("NFC", "NFC West"),   "SEA": ("NFC", "NFC West"),
}


def _resolve(refs: list[str], workers: int = 10) -> list[dict]:
    def one(ref: str) -> dict | None:
        try:
            return json.loads(_get_bytes(ref, timeout=25).decode())
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for r in pool.map(one, refs) if r]


def cfb_conference_names(season: int, refresh: bool = False) -> dict[str, str]:
    """ESPN conference id -> conference name, across FBS and FCS."""
    cached = CACHE / f"cfb_conferences_{season}.json"
    if cached.exists() and not refresh:
        return json.loads(cached.read_text())

    names: dict[str, str] = {}
    for group in DIVISION_GROUPS.values():
        try:
            listing = json.loads(
                _get_bytes(f"{CORE}/{season}/types/2/groups/{group}/children?limit=100")
                .decode()
            )
        except Exception as exc:
            logger.warning("conference listing failed for group %s: %s", group, exc)
            continue
        for conf in _resolve([i["$ref"] for i in listing.get("items", [])]):
            if conf.get("id") and conf.get("name"):
                names[str(conf["id"])] = conf["name"]

    if names:
        cached.write_text(json.dumps(names))
    logger.info("CFB conferences %d: %d resolved", season, len(names))
    return names


def cfb_team_conferences(season: int, lookback: int = 2) -> dict[str, str]:
    """Team display name -> conference name, read off the cached games.

    Walks back through earlier seasons first and lets each newer one overwrite. A
    team's conference only appears once it has played, so deriving from the current
    season alone leaves an early-season board almost empty — 2026 FBS resolved 16 of
    138 teams in week one. Conferences are stable year to year, so last season is a
    good default, and this season overrides it wherever realignment actually moved a
    team.
    """
    import sys
    from pathlib import Path

    cfb_dir = Path(__file__).resolve().parents[1] / "cfb"
    if cfb_dir.is_dir():
        sys.path.insert(0, str(cfb_dir))
    from espn_data import load_games  # noqa: E402

    games = load_games()
    out: dict[str, str] = {}

    for year in range(season - lookback, season + 1):
        names = cfb_conference_names(year)
        if not names:
            continue
        window = games[games["season"] == year]
        if window.empty:
            continue
        for g in window.itertuples(index=False):
            for team, conf in (
                (g.home_team_name, getattr(g, "home_conference_id", None)),
                (g.away_team_name, getattr(g, "away_conference_id", None)),
            ):
                if team and conf is not None and str(conf) in names:
                    out[team] = names[str(conf)]

    logger.info("CFB conferences %d: %d teams mapped", season, len(out))
    return out


# ── Human polls ─────────────────────────────────────────────────────────────
#
# ESPN publishes several under one endpoint; these are the two people mean by "the
# rankings", plus the FCS coaches poll so that board is not left blank. Only the most
# recent published week is kept — a poll is a snapshot, and the latest is the one a
# reader means.
POLL_COLUMNS = {
    "ap": "ap_rank",
    "usa": "coaches_rank",
    "fcs": "fcs_coaches_rank",
}


def cfb_polls(season: int, refresh: bool = False) -> pd.DataFrame:
    """Latest AP / Coaches poll positions, one row per team."""
    cached = CACHE / f"cfb_polls_{season}.json"
    if cached.exists() and not refresh:
        payload = json.loads(cached.read_text())
    else:
        try:
            payload = json.loads(
                _get_bytes(f"{SITE}/rankings?season={season}&seasontype=2").decode()
            )
            cached.write_text(json.dumps(payload))
        except Exception as exc:
            logger.warning("poll fetch failed for %d: %s", season, exc)
            return pd.DataFrame()

    # Key on the ESPN team id, never the name. The poll payload carries `name` as the
    # NICKNAME ("Buckeyes"), and nicknames are not unique across divisions — keying on
    # it merged Georgia's AP #3 with an FCS school's coaches #25 into one row. The
    # display name the board uses is location + name.
    rows: dict[str, dict] = {}
    for poll in payload.get("rankings", []):
        column = POLL_COLUMNS.get(poll.get("type"))
        if not column:
            continue
        for entry in poll.get("ranks", []):
            team = entry.get("team") or {}
            team_id = str(team.get("id") or "")
            location, nickname = team.get("location"), team.get("name")
            full = f"{location} {nickname}".strip() if location and nickname else nickname
            if not team_id or not full:
                continue
            rows.setdefault(team_id, {"team": full, "espn_team_id": team_id})[column] = (
                entry.get("current")
            )

    df = pd.DataFrame(rows.values())
    if not df.empty:
        for column in POLL_COLUMNS.values():
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    logger.info("CFB polls %d: %d ranked teams", season, len(df))
    return df


# ── MLB ─────────────────────────────────────────────────────────────────────
MLB_TEAMS = ("https://statsapi.mlb.com/api/v1/teams"
             "?sportId=1&season={season}&fields=teams,id,name,division,league")


def mlb_team_divisions(season: int, refresh: bool = False) -> dict[str, tuple[str, str]]:
    """Team name -> (league, division), read live rather than hardcoded.

    A literal would be shorter, but team names change — Oakland Athletics became
    Athletics — and StatsAPI always reports the name the season actually used, which is
    the one the board is keyed on.
    """
    cached = CACHE / f"mlb_divisions_{season}.json"
    if cached.exists() and not refresh:
        payload = json.loads(cached.read_text())
    else:
        try:
            payload = json.loads(_get_bytes(MLB_TEAMS.format(season=season)).decode())
            cached.write_text(json.dumps(payload))
        except Exception as exc:
            logger.warning("MLB team lookup failed for %d: %s", season, exc)
            return {}

    out: dict[str, tuple[str, str]] = {}
    for team in payload.get("teams", []):
        name = team.get("name")
        league = (team.get("league") or {}).get("name")
        division = (team.get("division") or {}).get("name")
        if name and division:
            out[name] = (league or "", division)
    logger.info("MLB divisions %d: %d teams", season, len(out))
    return out


def attach(board: pd.DataFrame, sport: str, season: int) -> pd.DataFrame:
    """Add conference/division and poll columns to a board. Never fatal.

    Purely descriptive: none of it feeds the fit. A failure here costs the extra
    columns, never the board.
    """
    out = board.copy()

    try:
        if sport == "cfb":
            conferences = cfb_team_conferences(season)
            out["conference"] = out["team"].map(conferences)

            polls = cfb_polls(season)
            if not polls.empty:
                columns = ["team"] + [c for c in POLL_COLUMNS.values() if c in polls.columns]
                # ap_rank may already be present from ESPN's FPI feed; the poll payload
                # is the more direct source, so let it win rather than duplicating.
                out = out.drop(columns=[c for c in columns if c != "team" and c in out.columns])
                out = out.merge(polls[columns], on="team", how="left")

        elif sport == "nfl":
            out["conference"] = out["team"].map(
                {t: c for t, (c, _) in NFL_DIVISIONS.items()}
            )
            out["division_name"] = out["team"].map(
                {t: d for t, (_, d) in NFL_DIVISIONS.items()}
            )

        elif sport == "mlb":
            divisions = mlb_team_divisions(season)
            out["conference"] = out["team"].map(
                {t: league for t, (league, _) in divisions.items()}
            )
            out["division_name"] = out["team"].map(
                {t: div for t, (_, div) in divisions.items()}
            )
    except Exception as exc:
        logger.warning("context attach failed for %s %d: %s", sport, season, exc)

    return out
