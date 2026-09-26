cd "$(dirname "$0")"
PY=/Users/VTNX82W/Documents/personalDev/mlb/hanks_tank_ml/.venv/bin/python
until grep -q DONE runs/tune.log; do sleep 10; done
for C in 0.3 1.0; do $PY run_sim.py c $C 16 1 2022 2022 >> runs/tune.log 2>&1; done
echo EXT_DONE >> runs/tune.log
