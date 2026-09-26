"""One row per CFBD drive, joined to cfb_historical.games by game_id (ESPN id).
Team ids come from BQ via (game_id, isHomeOffense); CFBD names are only checked for
consistency. Output columns mirror the NFL nfl_drives.parquet so sim_core can share code."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/VTNX82W/Documents/personalDev/mlb/hanks_tank_ml/src/cfb")
from espn_data import resolve_team_divisions

d = pd.read_parquet("cfb_drives_raw.parquet")
g = pd.read_parquet("cfb_games_bq.parquet")
g = g[g.season <= 2025]
d["game_id"] = d.gameId.astype(str)
n0 = len(d)
d = d.merge(g[["game_id", "home_team", "away_team", "neutral_site", "division", "home_score", "away_score"]],
            on="game_id", how="left")
matched = d.home_team.notna()
print(f"drives {n0}, matched to BQ games {matched.sum()} ({matched.mean():.4f}); "
      f"games {d.game_id.nunique()}, matched {d[matched].game_id.nunique()}")
d = d[matched].copy()
# CFBD's home/away disagrees with ESPN's in some games (neutral sites, feed errors), so
# isHomeOffense cannot be trusted blindly. Majority-vote a name->id crosswalk over
# drives, then orient each game by which BQ side its offense names map to.
h = np.where(d.isHomeOffense, d.home_team, d.away_team)
votes = pd.concat([pd.DataFrame({"name": d.offense, "id": h}),
                   pd.DataFrame({"name": d.defense, "id": np.where(d.isHomeOffense, d.away_team, d.home_team)})])
votes = votes.groupby(["name", "id"]).size().rename("n").reset_index().sort_values("n", ascending=False)
xw = votes.drop_duplicates("name").set_index("name").id
d["off_id"] = d.offense.map(xw); d["def_id"] = d.defense.map(xw)
ok_home = (d.off_id == d.home_team) & (d.def_id == d.away_team)
ok_away = (d.off_id == d.away_team) & (d.def_id == d.home_team)
d["_s"] = np.where(ok_home, 1, np.where(ok_away, -1, 0))
gs = d.groupby("game_id")._s.agg(lambda x: (x == 1).sum() - (x == -1).sum())
orient = np.sign(gs)  # +1: offense names resolve directly; ambiguous -> fall back to isHomeOffense
drv = d.offense.map(xw).eq(d.home_team) | d.offense.map(xw).eq(d.away_team)
d["posteam"] = np.where(d._s == 1, d.home_team, np.where(d._s == -1, d.away_team,
                        np.where(d.isHomeOffense, d.home_team, d.away_team)))
d["defteam"] = np.where(d.posteam == d.home_team, d.away_team, d.home_team)
flipped = (d.posteam == d.home_team) != d.isHomeOffense
unres = d._s == 0
print(f"crosswalk {len(xw)} CFBD names -> {xw.nunique()} BQ ids; ids with >1 name: {(xw.reset_index().groupby('id').name.nunique()>1).sum()}")
print(f"drives whose side resolved by name {1-unres.mean():.4f}; CFBD home/away flipped vs ESPN in "
      f"{d[flipped].game_id.nunique()} games ({flipped.mean():.4f} of drives)")
d["isHomeOffense"] = d.posteam == d.home_team
xw.rename("bq_id").to_csv("team_crosswalk.csv")
# ----- outcome
d = d.copy()
dop = d.endOffenseScore - d.startOffenseScore; dde = d.endDefenseScore - d.startDefenseScore
r = d.driveResult.fillna("Uncategorized")
res = pd.Series(None, index=d.index, dtype=object)
res[r.isin(["TD", "PASSING TD", "RUSHING TD"])] = "TD"
res[r.isin(["FG", "FG GOOD"])] = "FG"
res[r.isin(["MISSED FG", "FG MISSED", "BLOCKED FG"])] = "MFG"
res[r.isin(["PUNT", "BLOCKED PUNT"])] = "PUNT"
res[r.isin(["INT", "FUMBLE"])] = "TO"
res[r.isin(["DOWNS"])] = "TOD"
res[r.isin(["SF"])] = "SAF"
res[r.isin(["END OF HALF", "END OF GAME", "END OF 4TH QUARTER"])] = "EOH"
amb = r.isin(["INT TD", "FUMBLE RETURN TD", "FUMBLE TD", "PUNT RETURN TD", "PUNT TD", "MISSED FG TD",
              "INT RETURN TOUCH", "DOWNS TD", "END OF HALF TD", "END OF GAME TD", "FG TD",
              "Uncategorized", "KICKOFF", "POSSESSION (FOR OT DRIVES)"])
defsc = r.isin(["INT TD", "FUMBLE RETURN TD", "FUMBLE TD", "PUNT RETURN TD", "PUNT TD", "MISSED FG TD", "INT RETURN TOUCH"])
res[amb & (dop >= 6) & (dde < 6)] = "TD"
res[amb & (dde >= 6) & (dop < 6)] = "OTD"
res[defsc & res.isna()] = "OTD"          # return TD whose score delta the feed missed
res[amb & res.isna() & (dop == 3)] = "FG"
res[amb & res.isna() & (dde == 2)] = "SAF"
res[r.isin(["DOWNS TD"]) & res.isna()] = "TOD"
res[r.isin(["END OF HALF TD", "END OF GAME TD"]) & res.isna()] = "EOH"
print("unresolved (dropped):", res.isna().sum(), r[res.isna()].value_counts().head(5).to_dict())
d["res"] = res
keepm = d.res.notna()
d = d[keepm].copy(); dop = dop[keepm]; dde = dde[keepm]

# ----- clock / state
per = d.startPeriod.clip(lower=1)
secq = (d["startTime.minutes"].fillna(15) * 60 + d["startTime.seconds"].fillna(0)).clip(0, 900)
d["game_half"] = np.where(per <= 2, "Half1", np.where(per <= 4, "Half2", "Overtime"))
d["hsr0"] = np.where(per.isin([1, 3]), 900 + secq, np.where(per.isin([2, 4]), secq, 900))
d["qtr"] = per
d["yl"] = d.startYardsToGoal.astype(float).where(lambda s: s.between(1, 99))
d["sd0"] = d.startOffenseScore - d.startDefenseScore
d["off_pts"] = np.where(d.res == "TD", dop.where(dop.isin([6, 7, 8]), 7), np.where(d.res == "FG", 3, 0))
d["def_pts"] = np.where(d.res == "OTD", dde.where(dde.isin([6, 7, 8]), 7), np.where(d.res == "SAF", 2, 0))
d["off_home"] = d.isHomeOffense.astype(bool)
d = d.sort_values(["game_id", "driveNumber", "qtr", "hsr0"], ascending=[True, True, True, False])
d["hsr_next"] = d.groupby(["game_id", "game_half"]).hsr0.shift(-1)
d["yl_next"] = d.groupby(["game_id", "game_half"]).yl.shift(-1)
d["dur"] = (d.hsr0 - d.hsr_next.fillna(0)).clip(0, 1800)
d["season_type"] = np.where(d.season_type == "postseason", "POST", "REG")
div = resolve_team_divisions(g)
d["off_fbs"] = (d.posteam.map(div) == "fbs").astype(int); d["def_fbs"] = (d.defteam.map(div) == "fbs").astype(int)
keep = ["game_id", "season", "week", "season_type", "home_team", "away_team", "posteam", "defteam", "qtr",
        "game_half", "res", "hsr0", "yl", "sd0", "off_pts", "def_pts", "off_home", "yl_next", "dur",
        "off_fbs", "def_fbs", "division", "driveNumber"]
d[keep].reset_index(drop=True).to_parquet("cfb_drives.parquet")
print(d.res.value_counts(normalize=True).round(4).to_dict())
# drives per game (regulation) by season & division, and game-level points reconstruction check
reg = d[d.game_half != "Overtime"]
pg = reg.groupby(["season", "division", "game_id"]).size().groupby(["season", "division"]).mean().unstack().round(2)
print("regulation drives per game (both teams):\n", pg)
pts = d.assign(h=np.where(d.off_home, d.off_pts, d.def_pts), a=np.where(d.off_home, d.def_pts, d.off_pts)).groupby("game_id")[["h", "a"]].sum()
chk = pts.join(g.set_index("game_id")[["home_score", "away_score"]])
print("reconstructed total within 3 pts of actual:", ((chk.h + chk.a - chk.home_score - chk.away_score).abs() <= 3).mean().round(3),
      "; mean recon-actual total", (chk.h + chk.a - chk.home_score - chk.away_score).mean().round(2))
