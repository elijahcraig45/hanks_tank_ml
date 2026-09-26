import cfbd_fetch as F, pandas as pd, json
WEEKS = {2021: 15, 2022: 15, 2023: 15, 2024: 16, 2025: 16}
rows = []
for s, nw in WEEKS.items():
    for w in range(1, nw + 1):
        d = F.get('/drives', {'year': s, 'seasonType': 'regular', 'week': w})
        for x in d: x['season'] = s; x['week'] = w; x['season_type'] = 'regular'
        rows += d
    d = F.get('/drives', {'year': s, 'seasonType': 'postseason'})
    for x in d: x['season'] = s; x['week'] = 17; x['season_type'] = 'postseason'
    rows += d
    print(s, len(rows), flush=True)
df = pd.json_normalize(rows)
df.to_parquet('cfb_drives_raw.parquet')
print(df.shape)
