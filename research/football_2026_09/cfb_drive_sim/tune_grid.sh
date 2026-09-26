cd "$(dirname "$0")"
PY=/Users/VTNX82W/Documents/personalDev/mlb/hanks_tank_ml/.venv/bin/python
for V in c b; do for C in 0.01 0.03 0.1; do for T in 8 16 32; do
  $PY run_sim.py $V $C $T 1 2022 2022 >> runs/tune.log 2>&1
done; done; done
echo DONE >> runs/tune.log
