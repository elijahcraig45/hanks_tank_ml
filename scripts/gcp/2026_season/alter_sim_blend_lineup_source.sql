-- Adds lineup_source to game_predictions_sim_blend (2026-09-28). Optional and safe in
-- either order with the code deploy: blend.append drops columns a table lacks, so until
-- this runs the column is simply not written.
--   posted              both lineups as posted before first pitch
--   previous_game       both sides fell back to the team's previous game's lineup (T-35 retry)
--   previous_game_home  / previous_game_away: only that side fell back
-- NULL for rows written before the column existed (all of them posted lineups).
ALTER TABLE `hankstank.mlb_2026_season.game_predictions_sim_blend`
  ADD COLUMN IF NOT EXISTS lineup_source STRING;
