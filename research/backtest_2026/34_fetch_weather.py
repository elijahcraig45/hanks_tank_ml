"""Fetch per-game weather for the arena (2015-2021), one call per date.

Uses curl via subprocess because the corporate TLS proxy breaks Python's cert chain
against statsapi.mlb.com (curl succeeds; urllib does not) -- the same quirk
season_2026_pipeline.py works around.
"""
import json, subprocess, sys, time
import pandas as pd

arena = pd.read_parquet("data/odds/arena.parquet")
dates = sorted(pd.to_datetime(arena.game_date).dt.strftime("%Y-%m-%d").unique())
print(f"{len(dates)} distinct game dates to fetch")

def fetch(d):
    u = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}"
         f"&hydrate=weather,venue(fieldInfo)")
    for attempt in range(3):
        r = subprocess.run(["curl", "-s", "-m", "25", u], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            try:
                return json.loads(r.stdout)
            except json.JSONDecodeError:
                pass
        time.sleep(1.5 * (attempt + 1))
    return None

rows, miss = [], 0
t0 = time.time()
for i, d in enumerate(dates):
    j = fetch(d)
    if not j:
        miss += 1; continue
    for day in j.get("dates", []):
        for g in day.get("games", []):
            w = g.get("weather") or {}
            fi = (g.get("venue") or {}).get("fieldInfo") or {}
            rows.append(dict(game_pk=g.get("gamePk"), game_date=d,
                             temp=w.get("temp"), condition=w.get("condition"),
                             wind=w.get("wind"), roof=fi.get("roofType")))
    if (i + 1) % 150 == 0:
        print(f"  {i+1}/{len(dates)} dates, {len(rows)} games, {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows).drop_duplicates("game_pk")
df.to_parquet("data/odds/weather_2015_2021.parquet", index=False)
print(f"\nfetched {len(df)} games ({miss} dates failed) in {time.time()-t0:.0f}s")
print(f"temp present {df.temp.notna().mean()*100:.1f}%  wind {df.wind.notna().mean()*100:.1f}%  "
      f"roof {df.roof.notna().mean()*100:.1f}%")
print("\nroof types:"); print(df.roof.value_counts().to_string())
print("\nsample wind strings:"); print(df.wind.dropna().value_counts().head(12).to_string())
