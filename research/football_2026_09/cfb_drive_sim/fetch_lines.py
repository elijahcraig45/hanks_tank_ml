import cfbd_fetch as F, pandas as pd, numpy as np
rows = []
for s in range(2021, 2026):
    for st in ("regular", "postseason"):
        for gm in F.get('/lines', {'year': s, 'seasonType': st}):
            ls = gm.get('lines') or []
            def med(k):
                v = [x.get(k) for x in ls if x.get(k) is not None]
                return float(np.median(v)) if v else np.nan
            rows.append(dict(game_id=str(gm['id']), season=s, season_type=st,
                             spread_cfbd=med('spread'), total_line=med('overUnder'),
                             spread_open=med('spreadOpen'), total_open=med('overUnderOpen'),
                             home_ml=med('homeMoneyline'), away_ml=med('awayMoneyline'),
                             providers=len(ls), home=gm['homeTeam'], away=gm['awayTeam'],
                             hs=gm.get('homeScore'), as_=gm.get('awayScore')))
L = pd.DataFrame(rows)
L["spread_line"] = -L.spread_cfbd       # flip: positive = home favoured (stack convention)
m = L.hs.notna() & L.spread_line.notna()
print("corr(spread_line, home margin) =", np.corrcoef(L.spread_line[m], (L.hs - L.as_)[m])[0, 1].round(3))
print(L.groupby(['season','season_type']).agg(n=('game_id','size'), sp=('spread_line','count'), tot=('total_line','count')))
L.to_parquet('cfb_lines.parquet')
