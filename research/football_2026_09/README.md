# Football experiments, 2026-09-25

Research scripts behind the football findings. They are kept for reproducibility and not wired into production.

| Script | Question | Result |
|---|---|---|
| `football_eval.py` | Does a margin ridge beat production XGBoost? | Yes. CFB FBS 2025: 0.4801 vs 0.5313 log loss. NFL 2017-24: 0.6342 vs 0.6455. |
| `matchup_eval.py`, `nfl_pbp_agg.py` | Do run/pass unit matchups add anything over the ridge? | No. The split and the interactions are worse; the best add-on is within noise. |
| `injury_value.py` | What is a starting RB or QB worth? | RB out: −0.4 pts (CI spans 0). QB out: −2.0 pts. |
| `drive_sim/` | Drive-based Monte Carlo simulator (NFL). | Ties the ridge on winners. Adds nothing to the market total. Better margin-distribution shape only. |

The teaching write-ups are in `mlb/ml_writeups/*.html`, outside this repo.

The scripts were written in a session scratch directory. Input paths (nflverse pbp parquet,
cached CFBD JSON) may point there. Re-point them at `data/` before re-running. The pbp
files come from nflverse releases and are not committed.
