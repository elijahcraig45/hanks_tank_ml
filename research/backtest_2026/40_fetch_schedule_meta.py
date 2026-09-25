"""Per-game schedule metadata 2015-2026 from MLB statsapi (read-only HTTP, curl).

venue_id (true park identity, since statcast abbreviations hide relocations such as
TB 2025 / ATH 2025), scheduledInnings (7-inning doubleheaders in 2020-21), weather
(temp/wind/condition as posted pregame), roof type, series position. One call per month.
"""
import json, subprocess, time
import pandas as pd

rows = []
for yr in range(2015, 2027):
    for m in range(3, 11):
        s, e = f"{yr}-{m:02d}-01", f"{yr}-{m:02d}-{31 if m in (3,5,7,8,10) else 30}"
        u = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R&startDate={s}"
             f"&endDate={e}&hydrate=weather,venue(fieldInfo)")
        j = None
        for a in range(3):
            r = subprocess.run(["curl", "-s", "-m", "60", u], capture_output=True, text=True)
            try:
                j = json.loads(r.stdout); break
            except Exception:
                time.sleep(2)
        if not j:
            print("FAILED", s); continue
        for d in j.get("dates", []):
            for g in d.get("games", []):
                w = g.get("weather") or {}
                v = g.get("venue") or {}
                rows.append(dict(game_pk=g["gamePk"], official_date=g.get("officialDate"),
                                 game_date_utc=g.get("gameDate"),
                                 status=(g.get("status") or {}).get("detailedState"),
                                 venue_id=v.get("id"), venue_name=v.get("name"),
                                 roof=(v.get("fieldInfo") or {}).get("roofType"),
                                 temp=w.get("temp"), wind=w.get("wind"), condition=w.get("condition"),
                                 sched_inn=g.get("scheduledInnings"), dh=g.get("doubleHeader"),
                                 day_night=g.get("dayNight"), games_in_series=g.get("gamesInSeries"),
                                 series_game=g.get("seriesGameNumber"),
                                 home_id=g["teams"]["home"]["team"]["id"],
                                 away_id=g["teams"]["away"]["team"]["id"]))
    print(yr, len(rows), flush=True)
df = pd.DataFrame(rows).drop_duplicates("game_pk", keep="last")
df.to_parquet("data/backtest_2026/rich/sched_meta.parquet", index=False)
print(df.shape, df.temp.notna().mean(), df.sched_inn.value_counts().to_dict())
