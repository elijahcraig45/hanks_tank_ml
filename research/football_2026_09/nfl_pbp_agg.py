"""Pull nflverse pbp season by season, aggregate to team-game unit stats + player-game usage."""
import sys, polars as pl, nflreadpy as nfl
from pathlib import Path
OUT = Path(sys.argv[1]); OUT.mkdir(exist_ok=True)
for s in range(2008, 2026):
    f1, f2 = OUT / f"tg_{s}.parquet", OUT / f"pg_{s}.parquet"
    if f1.exists() and f2.exists():
        continue
    p = pl.read_parquet(f"/private/tmp/claude-501/-Users-VTNX82W-Documents-personalDev-mlb/c6d2d5d7-e0c5-4f76-86af-92e5653efdcd/scratchpad/pbp/pbp_{s}.parquet")
    p = p.filter(pl.col("epa").is_not_null() & pl.col("posteam").is_not_null()
                 & pl.col("play_type").is_in(["pass", "run"]))
    neutral = (pl.col("wp").is_between(0.2, 0.8) & (pl.col("qtr") <= 3) & (pl.col("down") <= 2))
    isr = (pl.col("play_type") == "run")
    tg = p.group_by(["season", "week", "game_id", "posteam", "defteam"]).agg(
        pl.len().alias("plays"),
        isr.sum().alias("rush_n"),
        (~isr).sum().alias("pass_n"),
        pl.col("epa").filter(isr).sum().alias("rush_epa_sum"),
        pl.col("epa").filter(~isr).sum().alias("pass_epa_sum"),
        neutral.sum().alias("neu_n"),
        (neutral & isr).sum().alias("neu_rush_n"),
    )
    tg.write_parquet(f1)
    r = p.filter(isr & pl.col("rusher_player_id").is_not_null()).group_by(
        ["season", "week", "game_id", "posteam", "rusher_player_id"]).agg(
        pl.len().alias("carries"), pl.col("epa").sum().alias("epa"), pl.col("rusher_player_name").first().alias("name"))
    r = r.rename({"rusher_player_id": "player_id"}).with_columns(pl.lit("rush").alias("role"))
    q = p.filter((~isr) & pl.col("passer_player_id").is_not_null()).group_by(
        ["season", "week", "game_id", "posteam", "passer_player_id"]).agg(
        pl.len().alias("carries"), pl.col("epa").sum().alias("epa"), pl.col("passer_player_name").first().alias("name"))
    q = q.rename({"passer_player_id": "player_id"}).with_columns(pl.lit("pass").alias("role"))
    pl.concat([r, q]).write_parquet(f2)
    print(s, tg.height, flush=True)
