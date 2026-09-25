WITH p AS (
  SELECT game_pk, game_date,
    IF(inning_topbot='Top', away_team, home_team) AS bat_team,
    IF(inning_topbot='Top', home_team, away_team) AS fld_team,
    description, events, pitch_type, release_speed, launch_speed, launch_angle,
    woba_value, woba_denom, estimated_woba_using_speedangle AS xw
  FROM `hankstank.mlb_2026_season.statcast_pitches`
  WHERE game_type='R'
),
pa AS (SELECT *, launch_speed IS NOT NULL AND launch_angle IS NOT NULL
               AND events IS NOT NULL AND events NOT IN ('strikeout','walk','hit_by_pitch','intent_walk','strikeout_double_play','catcher_interf') AS bbe
      FROM p),
bat AS (
  SELECT bat_team AS team, game_date,
    SUM(IF(woba_denom>0, woba_value, 0)) AS woba_num, SUM(IF(woba_denom>0, woba_denom, 0)) AS woba_den,
    SUM(IF(bbe, launch_speed, 0)) AS ev_sum, COUNTIF(bbe) AS bbe_n,
    COUNTIF(bbe AND launch_speed>=95) AS hh_n,
    COUNTIF(bbe AND launch_speed>=98
            AND launch_angle BETWEEN GREATEST(8, 26-(launch_speed-98)) AND LEAST(50, 30+2*(launch_speed-98))) AS brl_n
  FROM pa GROUP BY 1,2),
fld AS (
  SELECT fld_team AS team, game_date,
    COUNTIF(description IN ('swinging_strike','swinging_strike_blocked','foul_tip','missed_bunt')) AS whiffs,
    COUNTIF(description IN ('swinging_strike','swinging_strike_blocked','foul_tip','foul','foul_bunt','missed_bunt','hit_into_play','bunt_foul_tip')) AS swings,
    SUM(IF(pitch_type IN ('FF','SI'), release_speed, 0)) AS fbv_sum, COUNTIF(pitch_type IN ('FF','SI') AND release_speed IS NOT NULL) AS fbv_n,
    COUNTIF(events IN ('strikeout','strikeout_double_play')) AS k,
    COUNTIF(events IN ('walk','intent_walk')) AS bb, COUNTIF(events='hit_by_pitch') AS hbp,
    COUNTIF(events='home_run') AS hr,
    COUNTIF(bbe AND launch_angle BETWEEN 25 AND 50) AS fb,
    SUM(CASE WHEN events IN ('field_out','strikeout','force_out','fielders_choice_out','sac_fly','sac_bunt','other_out','fielders_choice') THEN 1
             WHEN events IN ('grounded_into_double_play','double_play','strikeout_double_play','sac_fly_double_play','sac_bunt_double_play') THEN 2
             WHEN events='triple_play' THEN 3 ELSE 0 END) AS outs
  FROM pa GROUP BY 1,2),
td AS (SELECT * FROM bat FULL OUTER JOIN fld USING (team, game_date)),
cum AS (
  SELECT team, game_date,
    SUM(woba_num) OVER w AS woba_num, SUM(woba_den) OVER w AS woba_den,
    SUM(ev_sum) OVER w AS ev_sum, SUM(bbe_n) OVER w AS bbe_n, SUM(hh_n) OVER w AS hh_n, SUM(brl_n) OVER w AS brl_n,
    SUM(whiffs) OVER w AS whiffs, SUM(swings) OVER w AS swings, SUM(fbv_sum) OVER w AS fbv_sum, SUM(fbv_n) OVER w AS fbv_n,
    SUM(k) OVER w AS k, SUM(bb) OVER w AS bb, SUM(hbp) OVER w AS hbp, SUM(hr) OVER w AS hr, SUM(fb) OVER w AS fb, SUM(outs) OVER w AS outs
  FROM td
  WINDOW w AS (PARTITION BY team ORDER BY UNIX_DATE(game_date) RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)),
gm AS (SELECT game_pk, ANY_VALUE(game_date) game_date, ANY_VALUE(home_team) h, ANY_VALUE(away_team) a
       FROM `hankstank.mlb_2026_season.statcast_pitches` WHERE game_type='R' GROUP BY 1)
SELECT gm.game_pk, gm.game_date, 'home' side, c.* EXCEPT(team, game_date) FROM gm JOIN cum c ON c.team=gm.h AND c.game_date=gm.game_date
UNION ALL
SELECT gm.game_pk, gm.game_date, 'away' side, c.* EXCEPT(team, game_date) FROM gm JOIN cum c ON c.team=gm.a AND c.game_date=gm.game_date
