"""Extract one row per drive from nflverse pbp (2008-2025). Pregame-safe: only used as
training data for games strictly after each drive."""
import polars as pl, sys
from pathlib import Path
S = Path("/private/tmp/claude-501/-Users-VTNX82W-Documents-personalDev-mlb/c6d2d5d7-e0c5-4f76-86af-92e5653efdcd/scratchpad")
OUT = S / "drive_sim" / "nfl_drives.parquet"
RES = {"Touchdown": "TD", "Field goal": "FG", "Missed field goal": "MFG", "Punt": "PUNT",
       "Turnover": "TO", "Opp touchdown": "OTD", "Turnover on downs": "TOD", "Safety": "SAF",
       "End of half": "EOH"}
frames = []
for s in range(2008, 2026):
    p = pl.read_parquet(S / f"pbp/pbp_{s}.parquet", columns=[
        "game_id", "season", "week", "season_type", "home_team", "away_team", "posteam", "defteam",
        "qtr", "game_half", "half_seconds_remaining", "play_type", "yardline_100", "fixed_drive",
        "fixed_drive_result", "total_home_score", "total_away_score", "play_id"])
    p = p.filter(pl.col("fixed_drive").is_not_null() & pl.col("posteam").is_not_null()).sort(["game_id", "play_id"])
    # score before each play = running score on previous row within game
    p = p.with_columns([
        pl.col("total_home_score").shift(1).over("game_id").fill_null(0).alias("hs0"),
        pl.col("total_away_score").shift(1).over("game_id").fill_null(0).alias("as0")])
    scrim = pl.col("play_type").is_in(["run", "pass", "punt", "field_goal", "no_play", "qb_kneel", "qb_spike"])
    d = p.group_by(["game_id", "fixed_drive"], maintain_order=True).agg([
        pl.col("season").first(), pl.col("week").first(), pl.col("season_type").first(),
        pl.col("home_team").first(), pl.col("away_team").first(),
        pl.col("posteam").first(), pl.col("defteam").first(), pl.col("qtr").first(),
        pl.col("game_half").first(), pl.col("fixed_drive_result").first().alias("res"),
        pl.col("half_seconds_remaining").first().alias("hsr0"),
        pl.col("yardline_100").filter(scrim).first().alias("yl"),
        pl.col("hs0").first(), pl.col("as0").first(),
        pl.col("total_home_score").last().alias("hs1"), pl.col("total_away_score").last().alias("as1"),
        pl.col("yardline_100").filter(scrim).last().alias("yl_end"),
    ])
    frames.append(d)
    print(s, d.height, flush=True)
d = pl.concat(frames).with_columns(pl.col("res").replace_strict(RES, default=None).alias("res"))
d = d.with_columns([
    (pl.col("posteam") == pl.col("home_team")).alias("off_home"),
    (pl.col("hs1") - pl.col("hs0")).alias("dh"), (pl.col("as1") - pl.col("as0")).alias("da")])
d = d.with_columns([
    pl.when(pl.col("off_home")).then(pl.col("dh")).otherwise(pl.col("da")).alias("off_pts"),
    pl.when(pl.col("off_home")).then(pl.col("da")).otherwise(pl.col("dh")).alias("def_pts"),
    pl.when(pl.col("off_home")).then(pl.col("hs0") - pl.col("as0")).otherwise(pl.col("as0") - pl.col("hs0")).alias("sd0"),
])
# duration = clock until next drive's start within same half (last drive: rest of half)
d = d.with_columns([
    pl.col("hsr0").shift(-1).over(["game_id", "game_half"]).alias("hsr_next"),
    pl.col("yl").shift(-1).over(["game_id", "game_half"]).alias("yl_next"),
])
d = d.with_columns((pl.col("hsr0") - pl.col("hsr_next").fill_null(0)).clip(0, 1800).alias("dur"))
d.write_parquet(OUT)
print(d.height, d["res"].value_counts())
