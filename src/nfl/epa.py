"""Team-week EPA aggregates from nflverse play-by-play.

EPA is the single largest signal in NFL modelling — point differential and win streaks
are crude proxies for what EPA measures directly. The v1 feature set (Elo, Pythagorean,
point diff, streaks) failed to meaningfully beat Elo alone on walk-forward CV, which is
what motivated building this.

Play-by-play is ~50k rows/season and ~20s to download per season, so the raw pull is
aggregated down to one row per (season, week, team) and cached. The Cloud Function
reads the aggregate, never the raw pbp.
"""

from __future__ import annotations

import logging

import polars as pl

from config import RAW_CACHE

logger = logging.getLogger(__name__)

EPA_CACHE = RAW_CACHE / "team_week_epa.parquet"


def _aggregate_season(pbp: pl.DataFrame) -> pl.DataFrame:
    """Collapse one season of plays into per-team, per-week offensive and defensive rates."""
    # Scrimmage plays only: EPA is undefined/noisy on specials and no-plays.
    plays = pbp.filter(
        pl.col("epa").is_not_null()
        & pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("play_type").is_in(["pass", "run"])
    )

    explosive = (
        ((pl.col("play_type") == "pass") & (pl.col("yards_gained") >= 20))
        | ((pl.col("play_type") == "run") & (pl.col("yards_gained") >= 10))
    ).cast(pl.Int8)

    turnover = (
        pl.col("interception").fill_null(0) + pl.col("fumble_lost").fill_null(0)
    ).cast(pl.Int8)

    plays = plays.with_columns(
        explosive.alias("_explosive"),
        turnover.alias("_turnover"),
        pl.col("success").fill_null(0).cast(pl.Float64).alias("_success"),
    )

    off = plays.group_by(["season", "week", "posteam"]).agg(
        pl.col("epa").mean().alias("off_epa_play"),
        pl.col("epa").filter(pl.col("play_type") == "pass").mean().alias("off_pass_epa"),
        pl.col("epa").filter(pl.col("play_type") == "run").mean().alias("off_rush_epa"),
        pl.col("_success").mean().alias("off_success_rate"),
        pl.col("_explosive").mean().alias("off_explosive_rate"),
        pl.col("_turnover").sum().alias("off_turnovers"),
        pl.len().alias("off_plays"),
    ).rename({"posteam": "team"})

    dfn = plays.group_by(["season", "week", "defteam"]).agg(
        pl.col("epa").mean().alias("def_epa_play"),
        pl.col("epa").filter(pl.col("play_type") == "pass").mean().alias("def_pass_epa"),
        pl.col("epa").filter(pl.col("play_type") == "run").mean().alias("def_rush_epa"),
        pl.col("_success").mean().alias("def_success_rate"),
        pl.col("_explosive").mean().alias("def_explosive_rate"),
        pl.col("_turnover").sum().alias("def_takeaways"),
        pl.len().alias("def_plays"),
    ).rename({"defteam": "team"})

    return off.join(dfn, on=["season", "week", "team"], how="full", coalesce=True)


def build_team_week_epa(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Aggregate EPA for the given seasons, caching the (small) result."""
    RAW_CACHE.mkdir(parents=True, exist_ok=True)

    if EPA_CACHE.exists() and not refresh:
        cached = pl.read_parquet(EPA_CACHE)
        have = set(cached["season"].unique().to_list())
        missing = [s for s in seasons if s not in have]
        if not missing:
            return cached
        logger.info("EPA cache missing seasons: %s", missing)
        frames = [cached]
    else:
        missing = list(seasons)
        frames = []

    import nflreadpy as nfl

    for season in missing:
        logger.info("aggregating EPA for %d", season)
        try:
            pbp = nfl.load_pbp(seasons=[season])
        except Exception as exc:
            logger.warning("pbp fetch failed for %d (%s), skipping", season, exc)
            continue
        frames.append(_aggregate_season(pbp))

    out = pl.concat(frames, how="diagonal_relaxed").unique(
        subset=["season", "week", "team"], keep="last"
    ).sort(["season", "week", "team"])

    out.write_parquet(EPA_CACHE)
    logger.info("team-week EPA: %d rows covering %d seasons",
                out.height, out["season"].n_unique())
    return out
