import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_validation import DataValidator  # noqa: E402


class DataValidatorTests(unittest.TestCase):
    @patch("data_validation.bigquery.Client")
    def test_run_returns_zero_when_no_findings(self, _mock_client):
        validator = DataValidator()
        with patch.object(validator, "check_row_counts"), \
             patch.object(validator, "check_game_duplicates"), \
             patch.object(validator, "check_statcast_duplicates"), \
             patch.object(validator, "check_freshness"), \
             patch.object(validator, "check_score_sanity"), \
             patch.object(validator, "check_team_consistency"):
            self.assertEqual(validator.run(), 0)

    @patch("data_validation.bigquery.Client")
    def test_run_returns_warning_code_when_only_warnings_exist(self, _mock_client):
        validator = DataValidator()

        def add_warning():
          validator.warnings.append("games not fresh")

        with patch.object(validator, "check_row_counts", side_effect=add_warning), \
             patch.object(validator, "check_game_duplicates"), \
             patch.object(validator, "check_statcast_duplicates"), \
             patch.object(validator, "check_freshness"), \
             patch.object(validator, "check_score_sanity"), \
             patch.object(validator, "check_team_consistency"):
            self.assertEqual(validator.run(), 2)

    @patch("data_validation.bigquery.Client")
    def test_run_returns_error_code_when_errors_exist(self, _mock_client):
        validator = DataValidator()

        def add_error():
          validator.errors.append("duplicate games")

        with patch.object(validator, "check_row_counts"), \
             patch.object(validator, "check_game_duplicates", side_effect=add_error), \
             patch.object(validator, "check_statcast_duplicates"), \
             patch.object(validator, "check_freshness"), \
             patch.object(validator, "check_score_sanity"), \
             patch.object(validator, "check_team_consistency"):
            self.assertEqual(validator.run(), 1)


if __name__ == "__main__":
    unittest.main()
