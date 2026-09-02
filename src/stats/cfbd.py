"""CollegeFootballData client and transforms.

Lives in src/stats/ rather than in a package of its own, and that placement is
deliberate. deploy_cfb.sh and deploy_nfl.sh both stage `rankings/` and `stats/` as
packages into their images, so anything here is already present in both with no
deploy-script change. stats/build.py imports cfb_stats at module level in the NFL image
too, so a CFBD client living under src/cfb/ would make `import stats.build` crash there.

Everything below the client is pure: no network, no BigQuery. That is what lets the
flattening and the long-to-wide pivot be tested offline from committed fixtures, which
matters because those transforms are where the bugs live.

Budget: the account is metered per month and shared with the basketball product. One
league-wide call covers a whole season for most endpoints, so a weekly ingest is cheap —
but a loop over seasons is not, and that is exactly the shape of the bug that once wiped
cfb_historical.games. Hence a hard per-run ceiling that raises rather than warns.
"""

from __future__ import annotations

import json
import logging
import re
import os
import time
import urllib.parse
from pathlib import Path

import pandas as pd

try:  # package layout locally, flat module tree in Cloud Functions
    from rankings.http import get_bytes as _get_bytes, cache_dir as _cache_dir
except ImportError:  # pragma: no cover
    from http_transport import get_bytes as _get_bytes, cache_dir as _cache_dir

logger = logging.getLogger(__name__)

BASE = "https://api.collegefootballdata.com"
CACHE = _cache_dir("cfbd")

# Raises when exceeded. A weekly ingest uses about a dozen calls; anything approaching
# this means something is looping over seasons or weeks that should not be.
MAX_CALLS_PER_RUN = 40

# Classifications the site covers. /teams also returns DII and DIII, which it does not.
SITE_CLASSIFICATIONS = ("fbs", "fcs")

_calls = 0
_unauthorized = False


class CfbdKeyMissing(RuntimeError):
    """No API key configured. Callers degrade to a skipped step, never a failure."""


class CfbdUnauthorized(RuntimeError):
    """Key rejected, or the tier does not cover this endpoint."""


def api_key() -> str:
    """Resolve the key lazily.

    Lazy because importing this module must not require a key: stats/build.py is
    imported in the NFL image, where nothing CFBD-backed ever runs.
    """
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        path = Path.home() / ".config" / "cfbd" / "key"
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise CfbdKeyMissing(
            "CFBD_API_KEY is not set. Read it from Secret Manager with: "
            "gcloud secrets versions access latest --secret=cfbd-api-key "
            "--project=hankstank"
        )
    return key


def has_api_key() -> bool:
    try:
        api_key()
        return True
    except CfbdKeyMissing:
        return False


def calls_used() -> int:
    """Surfaced in the Cloud Function's result so monthly spend is visible in logs."""
    return _calls


def reset_call_counter() -> None:
    global _calls, _unauthorized
    _calls = 0
    _unauthorized = False


def _cache_path(path: str, params: dict) -> Path:
    slug = path.strip("/").replace("/", "_")
    query = urllib.parse.urlencode(sorted(params.items()))
    digest = str(abs(hash(query)) % (10 ** 12))
    return CACHE / f"{slug}_{digest}.json"


def get(path: str, params: dict | None = None, ttl_hours: float | None = None) -> object:
    """GET a CFBD endpoint, cached on disk.

    `ttl_hours=None` means cache forever, which is correct for a completed season: those
    numbers never change again. Only the current season should pass a TTL. That single
    rule is what keeps a full historical backfill affordable.
    """
    global _calls, _unauthorized

    params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
    cache_file = _cache_path(path, params)

    if cache_file.exists():
        fresh = ttl_hours is None or (
            time.time() - cache_file.stat().st_mtime < ttl_hours * 3600
        )
        if fresh:
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.info("%s: unreadable cache, refetching", cache_file.name)

    if _unauthorized:
        # One 401 means every later call this run would also fail. Skipping them stops
        # a misconfigured run from burning the whole monthly allowance on rejections.
        raise CfbdUnauthorized(f"skipping {path}: an earlier call was unauthorized")

    if _calls >= MAX_CALLS_PER_RUN:
        raise RuntimeError(
            f"CFBD call ceiling ({MAX_CALLS_PER_RUN}) hit at {path}. This is a guard "
            "against a loop over seasons or weeks — widen it deliberately, do not "
            "raise it to make a run pass."
        )

    url = f"{BASE}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    _calls += 1
    try:
        raw = _get_bytes(url, timeout=90,
                         headers={"Authorization": f"Bearer {api_key()}"})
    except Exception as exc:
        if "401" in str(exc) or "403" in str(exc):
            _unauthorized = True
            raise CfbdUnauthorized(
                f"{path} rejected — check the key and that the tier covers it"
            ) from exc
        raise

    payload = json.loads(raw.decode())
    try:
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        logger.debug("could not cache %s: %s", cache_file.name, exc)
    return payload


# --------------------------------------------------------------------------- #
# Transforms — pure, no network, no BigQuery
# --------------------------------------------------------------------------- #

def snake(name: str) -> str:
    """camelCase / dotted.path / ACRONYM -> snake_case.

    Acronyms are the whole difficulty. CFBD mixes camelCase field names
    (`successRate`) with all-caps stat types (`YDS`, `TD`, `INT`, `QB HUR`) and dotted
    nesting (`fieldPosition.averageStart`). Inserting a separator before every capital
    turns YDS into y_d_s and totalPPA into total_p_p_a, which is how a whole player
    table ends up with unusable column names — so a boundary is only taken where a
    capital actually starts a new word.
    """
    text = name.replace(".", "_").replace(" ", "_").replace("-", "_")
    # A capital beginning a lowercase word, after anything: totalPPA stays whole,
    # passingYards splits.
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    # A capital following a lowercase or digit: the totalPPA -> total_PPA boundary.
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def flatten(record: dict, prefix: str = "", sep: str = "_") -> dict:
    """Recursively flatten nested dicts into snake_case columns.

    CFBD returns advanced stats as nested offense/defense objects several levels deep
    (`offense.passingPlays.successRate`), and BigQuery wants columns. Lists are left
    alone: the only lists in these payloads are line scores, which belong as a repeated
    field rather than as fifteen columns.
    """
    out: dict = {}
    for key, value in (record or {}).items():
        name = f"{prefix}{snake(key)}"
        if isinstance(value, dict):
            out.update(flatten(value, prefix=f"{name}{sep}", sep=sep))
        else:
            out[name] = value
    return out


# CFBD spells this field without the second "i". Both spellings are accepted because a
# silent fix upstream would otherwise drop the column without anything failing.
_OPPORTUNITY_ALIASES = ("total_opportunies", "total_opportunities")


def _fix_known_typos(row: dict) -> dict:
    seen = [a for a in _OPPORTUNITY_ALIASES if a in row]
    if seen:
        row["total_opportunities"] = row.pop(seen[0])
        for extra in seen[1:]:
            row.pop(extra, None)
    return row


def split_off_def(record: dict, keys: tuple[str, ...] = ("offense", "defense")) -> dict:
    """Flatten an offense/defense record into own and `opp_`-prefixed columns.

    The `opp_` convention is not invented here: cfb_season.team_season_stats already
    uses it for ESPN's opponent split, and the site renders those columns. CFBD's
    `defense` block means the same thing — what opponents did — so it maps to the same
    prefix. Keeping one convention per table is why this is done at ingest and never in
    the feature builder, which uses off_/def_ instead.
    """
    own_key, opp_key = keys
    out: dict = {}
    for name, value in flatten(record.get(own_key) or {}).items():
        out[name] = value
    for name, value in flatten(record.get(opp_key) or {}).items():
        out[f"opp_{name}"] = value
    return _fix_known_typos(out)


def pivot_player_season(records: list[dict]) -> pd.DataFrame:
    """Pivot CFBD's long-form player stats into one row per player.

    The feed returns one row per (player, category, statType) — `passing/YDS`,
    `rushing/TD` — so a wide table has to be built rather than read. Columns are named
    `{category}_{statType}` so two categories publishing the same statType cannot
    collide.

    Transfers are the trap: a player who moved mid-season appears twice for the same
    (category, statType) with two team values. Summing would be wrong for every rate
    stat, so the most recent row wins and the duplicate count is logged rather than
    passing silently.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    required = {"season", "playerId", "player", "category", "statType", "stat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"player season records missing columns: {sorted(missing)}")

    df["column"] = (
        df["category"].astype(str).map(snake) + "_"
        + df["statType"].astype(str).map(snake)
    )
    df["value"] = pd.to_numeric(df["stat"], errors="coerce")

    dupes = df.duplicated(subset=["season", "playerId", "column"]).sum()
    if dupes:
        logger.info(
            "%d duplicate (player, stat) rows — most likely transfers; keeping the last",
            dupes,
        )

    wide = df.pivot_table(
        index=["season", "playerId"],
        columns="column",
        values="value",
        aggfunc="last",
        # dropna=False keeps a column the feed published even when every value failed
        # to parse. The default drops it, which means the table's shape would depend on
        # whether values happened to be numeric that week — silent schema drift, and
        # exactly what pinning dtypes elsewhere is meant to prevent.
        dropna=False,
    ).reset_index()
    wide.columns.name = None

    # Identity comes from the last row seen for a player, so a transfer resolves to the
    # team they finished the season with.
    identity = (
        df.sort_values(["season", "playerId"])
        .groupby(["season", "playerId"], as_index=False)
        .agg({"player": "last", "position": "last", "team": "last",
              "conference": "last"})
    )

    out = identity.merge(wide, on=["season", "playerId"], how="left")
    return out.rename(columns={"playerId": "player_id", "player": "player_name"})


def as_frame(rows: list[dict], text_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Build a frame with dtypes pinned, so BigQuery autodetect is deterministic.

    Autodetect over a frame whose dtypes are fixed is equivalent to an explicit schema,
    and unlike 85 hand-written SchemaFields it can be checked offline. The failure this
    prevents is real: an all-null column infers as float64, lands as FLOAT, and then
    rejects the load or silently changes type the week it gains a value.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in df.columns:
        if col in text_columns:
            df[col] = df[col].astype("string")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
