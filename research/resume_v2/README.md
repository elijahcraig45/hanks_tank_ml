# Resume v2 experiments

Scripts behind `docs/RESUME_V2_REPORT.md`. They read ESPN game results through `rankings.evaluate.load_games("cfb")` and the
released `rankings` package on the homelab (`/opt/pipeline/src/hanks_tank_ml/src`), so run them there as the `pipeline` user:

    W=/var/lib/pipeline/work/resumeexp          # copy these scripts and src/rankings/resume_v2.py (into $W/mod) here first
    python -B -I $W/exp_v2.py  $W $W/mod 3 100  # round 1 (pre-registered): 29 arms x 72 boards, ~30 s
    python -B -I $W/exp_v2b.py $W $W/mod 3 100  # round 2 (exploratory)
    python -B -I $W/analyze_v2.py $W v2  "A add 0.5"              # tables for round 1
    python -B -I $W/analyze_v2.py $W v2b "R0 add .5 sum v1order"  # tables for round 2
    python -B -I $W/exp_v1x.py $W $W/mod 3 100   # round 3: the shipping candidate on the PRODUCTION module (needs $W/prod/resume.py)
    python -B -I $W/ci_v1x.py $W                 # round 3 intervals and the decision rule inputs
    python -B -I $W/ci_v2.py $W                  # paired block-bootstrap intervals against the shipped board
    python -B -I $W/extras_v2.py $W              # by season stage, benchmark size
    python -B -I $W/live_v2.py $W                # the current-season board under each credit

Outputs (`v2*_rows.parquet`, `v2*_pairs.parquet`, `*_summary_*.csv`) are written next to the scripts. Splits: tune 2017-19 and
2021, eval 2022-25. Seeds are fixed per (season, week).
