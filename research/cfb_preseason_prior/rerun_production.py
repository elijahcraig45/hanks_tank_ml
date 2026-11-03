"""    python3 research/cfb_preseason_prior/rerun_production.py   (needs data/games_raw.csv from data/games.sql)

Re-run the frozen prior backtest with the PRODUCTION code path (program_prior + fit(prior=), majority_divisions per season).
Same games, baseline, sigma and week-block CIs as research/cfb_preseason_prior. 2025 is never scored."""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src" / "nfl")); sys.path.insert(0, str(HERE.parents[1] / "src" / "cfb"))
import margin_ridge as mr, pipeline as cp
spec = importlib.util.spec_from_file_location("eval_prior", str(HERE / "eval_prior.py")); ep = importlib.util.module_from_spec(spec); spec.loader.exec_module(ep)
g, rg = ep.load()
CFG = mr.CFB_RIDGE
out = []
for s in (2022, 2023, 2024, 2026):
    prior = mr.program_prior(rg, s, CFG, cp.majority_divisions(g[g.season <= s]))
    for wk in sorted(rg.loc[rg.season == s, "week"].unique()):
        if s == 2026 and wk > 9:
            continue
        t = mr.time_index(s, wk)
        sel = (rg.season == s) & (rg.week == wk) & rg.margin.notna()
        base, pri = mr.fit(rg, t, CFG), mr.fit(rg, t, CFG, prior=prior)
        if base is None or not sel.any():
            continue
        f = rg.loc[sel, ["game_id", "season", "week", "division", "home_won"]].copy()
        f["base"] = ep.ll_vec(f.home_won, mr.win_prob(base.predict_margin(rg[sel]), CFG.sigma))
        f["prior"] = ep.ll_vec(f.home_won, mr.win_prob(pri.predict_margin(rg[sel]), CFG.sigma))
        out.append(f)
ev = pd.concat(out, ignore_index=True)
ev["bucket"] = ep.bucket(ev.week)
ep.table(ev[ev.season <= 2024], "EVAL 2022-2024, production code, majority divisions (prior - baseline)")
ep.table(ev[ev.season == 2026], "2026 weeks 1-9, production code, majority divisions")
