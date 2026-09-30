-- Record WHICH artifact produced each MLB prediction. The label 'v10' has covered different artifacts over time (the model file was overwritten on
-- 2026-04-29), so the label alone cannot reproduce history; the sha256 can. Nullable: old rows stay NULL. The pipeline only writes this column if
-- the table already has it, so running this before or after the code deploy is safe.
ALTER TABLE `hankstank.mlb_2026_season.game_predictions`
  ADD COLUMN IF NOT EXISTS model_sha256 STRING OPTIONS (description = 'sha256 of the model artifact bytes that produced this row');
