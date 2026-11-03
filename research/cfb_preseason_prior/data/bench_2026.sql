-- 2026 benchmarks for the one-shot check: market line and ESPN FPI per game (latest FPI row).
WITH b AS (SELECT game_id, ANY_VALUE(spread_line) spread FROM `hankstank.cfb_season.betting_lines`
           WHERE season = 2026 GROUP BY 1),
f AS (SELECT game_id, home_win_probability fpi FROM `hankstank.cfb_season.fpi_game_predictions`
      WHERE season = 2026 QUALIFY ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY predicted_at DESC) = 1)
SELECT game_id, spread, fpi FROM b FULL OUTER JOIN f USING (game_id)
