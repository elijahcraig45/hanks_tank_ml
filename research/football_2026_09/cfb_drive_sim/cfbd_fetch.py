"""Thin wrapper over src/stats/cfbd.py: key read from ~/.cfbd_key inside the process only,
never printed. Counts real (non-cache) calls in calls_log.txt."""
import os, sys, json
from pathlib import Path
REPO = Path("/Users/VTNX82W/Documents/personalDev/mlb/hanks_tank_ml/src")
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent
ca = HERE.parent / "ca.pem"
os.environ.setdefault("REQUESTS_CA_BUNDLE", str(ca)); os.environ.setdefault("SSL_CERT_FILE", str(ca))
os.environ["CFBD_API_KEY"] = Path.home().joinpath(".cfbd_key").read_text().strip()
from stats import cfbd
cfbd.MAX_CALLS_PER_RUN = 150

def get(path, params):
    before = cfbd.calls_used()
    out = cfbd.get(path, params)
    if cfbd.calls_used() > before:
        with open(HERE / "calls_log.txt", "a") as f:
            f.write(json.dumps({"path": path, "params": params}) + "\n")
    return out
