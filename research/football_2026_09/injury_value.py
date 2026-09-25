"""How much is a starting RB worth vs a starting QB? NFL 2008-2025, nflverse pbp aggregates.

Starter = the team-season's leader in carries (RB; excluding the team's leading passer)
or dropbacks (QB). Present = >=8 carries / >=15 dropbacks; absent = 0 (partial games dropped).
Outcomes are opponent-adjusted (subtract opponent's season mean allowed/earned, excluding
this game) and compared within team-season, so team quality cancels. CIs bootstrap team-seasons.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd

SP = Path(__file__).resolve().parent
REPO = Path("/Users/VTNX82W/Documents/personalDev/machineLEARNING/hanks_tank_ml")
tg = pd.concat(pd.read_parquet(f) for f in glob.glob(str(SP / "nflagg/tg_*.parquet")))
pg = pd.concat(pd.read_parquet(f) for f in glob.glob(str(SP / "nflagg/pg_*.parquet")))
sch = pd.read_parquet(REPO / "data/nfl/raw/schedules.parquet")
sch = sch[sch.result.notna() & (sch.game_type == "REG")]
tg = tg[tg.game_id.isin(sch.game_id)]
pg = pg[pg.game_id.isin(sch.game_id)]

# per team-game outcomes (pbp team codes are consistent within season)
tg["rush_epa"] = tg.rush_epa_sum / tg.rush_n
tg["pass_epa"] = tg.pass_epa_sum / tg.pass_n
tg["tot_epa"] = (tg.rush_epa_sum + tg.pass_epa_sum) / tg.plays
tg["tot_epa_sum"] = tg.rush_epa_sum + tg.pass_epa_sum
# net EPA per game: my offense minus opponent's offense
opp = tg[["game_id", "posteam", "tot_epa_sum"]].rename(columns={"posteam": "defteam", "tot_epa_sum": "opp_off"})
tg = tg.merge(opp, on=["game_id", "defteam"])
tg["net_epa"] = tg.tot_epa_sum - tg.opp_off
# points margin from pbp-free schedule: attach via home/away scores
s2 = pd.concat([sch[["game_id", "home_team", "result"]].rename(columns={"home_team": "t"}).assign(m=lambda d: d.result),
                sch[["game_id", "away_team", "result"]].rename(columns={"away_team": "t"}).assign(m=lambda d: -d.result)])
# pbp codes may differ from schedule codes for relocated teams; match by game and sign via net_epa sides
tg = tg.merge(s2[["game_id", "t", "m"]], left_on=["game_id", "posteam"], right_on=["game_id", "t"], how="left")
miss = tg.m.isna()
if miss.any():  # relocated codes: margin = -(opponent's margin)
    other = tg[["game_id", "posteam", "m"]].rename(columns={"posteam": "defteam", "m": "m_opp"})
    tg = tg.merge(other, on=["game_id", "defteam"], how="left")
    tg["m"] = tg.m.fillna(-tg.m_opp)


def opp_adjust(col, by_def=True):
    """value - (opponent's season mean of what it allows/earns, leave-one-out) + league mean."""
    key = "defteam"
    grp = tg.groupby(["season", key])[col]
    loo = (grp.transform("sum") - tg[col]) / (grp.transform("count") - 1)
    return tg[col] - loo + tg.groupby("season")[col].transform("mean")


for c in ["rush_epa", "pass_epa", "tot_epa"]:
    tg[c + "_adj"] = opp_adjust(c)
# margin/net: opponent strength = opponent's own season mean (leave-one-out) of its margin
for c in ["m", "net_epa"]:
    own = tg.groupby(["season", "posteam"])[c]
    oppmean = ((own.transform("sum") - tg[c]) / (own.transform("count") - 1)).rename("x")
    lk = pd.DataFrame({"season": tg.season, "defteam": tg.posteam, "game_id": tg.game_id, "om": oppmean})
    tg = tg.merge(lk, on=["season", "game_id", "defteam"], how="left")
    tg[c + "_adj"] = tg[c] + tg.om
    tg = tg.drop(columns="om")

inj = pd.concat(pd.read_parquet(f) for f in glob.glob(str(SP / "inj/injuries_*.parquet")))
inj = inj[inj.report_status.isin(["Out", "Doubtful"]) & (inj.game_type == "REG")]
first_out = inj.groupby(["season", "gsis_id"]).week.min().rename("first_out").reset_index()
last_wk = tg.groupby("season").week.transform("max")
tg = tg[tg.week < last_wk]  # drop final regular-season week (starters rested)

OUTC = ["rush_epa_adj", "pass_epa_adj", "tot_epa_adj", "net_epa_adj", "m_adj", "rush_n"]
gnum = tg[["season", "posteam", "game_id", "week"]].copy()
gnum["gn"] = gnum.groupby(["season", "posteam"]).week.rank(method="first")
pg = pg.merge(gnum[["game_id", "posteam", "gn"]], on=["game_id", "posteam"], how="left")
for defn in ("any_absence", "injury_report", "opening_starter+injury_report"):
  for role in ("rush", "pass"):
    col = "carries"
    p = pg[pg.role == role]
    src = p[p.gn <= 4] if defn.startswith("opening") else p
    tot = src.groupby(["season", "posteam", "player_id"]).agg(n=(col, "sum"), name=("name", "first")).reset_index()
    if role == "rush":  # drop QBs: exclude each team's leading passer
        qsrc = pg[pg.role == "pass"]; qsrc = qsrc[qsrc.gn <= 4] if defn.startswith("opening") else qsrc
        qb = qsrc.groupby(["season", "posteam", "player_id"]).carries.sum().reset_index()
        qb = qb.sort_values("carries").groupby(["season", "posteam"]).tail(1)
        tot = tot.merge(qb[["season", "posteam", "player_id"]].assign(isqb=1), how="left").fillna({"isqb": 0})
        tot = tot[tot.isqb == 0]
    star = tot.sort_values("n").groupby(["season", "posteam"]).tail(1)[["season", "posteam", "player_id", "name", "n"]]
    use = p.groupby(["game_id", "posteam", "player_id"])[col].sum().rename("u").reset_index()
    x = tg.merge(star, on=["season", "posteam"]).merge(use, on=["game_id", "posteam", "player_id"], how="left")
    x["u"] = x.u.fillna(0)
    x["absent"] = (x.u == 0).astype(int)
    if defn.endswith("injury_report"):
        x = x.merge(first_out.rename(columns={"gsis_id": "player_id"}), on=["season", "player_id"], how="left")
        x = x[(x.absent == 0) | (x.week >= x.first_out)]   # absence must follow an Out/Doubtful listing
    ok = x.groupby(["season", "posteam"]).absent.transform(lambda a: 0 < a.sum() < len(a))
    x = x[ok]
    ts = x.groupby(["season", "posteam", "absent"])[OUTC].mean().unstack("absent")
    nabs = x.groupby(["season", "posteam"]).absent.sum()
    diff = (ts.xs(1, axis=1, level=1) - ts.xs(0, axis=1, level=1))
    w = nabs.reindex(diff.index).values
    rng = np.random.default_rng(0)
    print(f"\n[{defn}] {'RB' if role=='rush' else 'QB'} starter absent - present, within team-season, opp-adjusted")
    print(f"  team-seasons: {len(diff)}, absent games: {int(nabs.sum())}")
    for c in OUTC:
        v = diff[c].values; okv = ~np.isnan(v); v, w_ = v[okv], w[okv]
        est = np.average(v, weights=w_)
        bs = [np.average(v[i], weights=w_[i]) for i in (rng.integers(0, len(v), len(v)) for _ in range(2000))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {c:14s} {est:+.4f}  (95% CI {lo:+.4f}, {hi:+.4f})")
