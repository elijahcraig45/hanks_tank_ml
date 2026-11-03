SELECT game_id, season, week, division, is_postseason, game_date, home_team, away_team,
  home_team_name, away_team_name, home_score, away_score, home_won, result, neutral_site,
  conference_game, cross_division
FROM `hankstank.cfb_historical.games`
