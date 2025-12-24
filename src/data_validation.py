#!/usr/bin/env python3
"""
MLB Data Quality Validation Pipeline

Validates data integrity across all BigQuery tables for MLB historical data.
Designed to run on historical data (2015-2025) and new 2026 season data as it arrives.

Usage:
    python data_validation.py --year 2025              # Validate single year
    python data_validation.py --year all               # Validate all years
    python data_validation.py --year 2026 --tables games_historical rosters_historical  # Specific tables for 2026
    python data_validation.py --report-only            # Generate report from last run
"""

from google.cloud import bigquery
from datetime import datetime, timedelta
import argparse
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import sys


class Severity(Enum):
    """Validation result severity levels"""
    PASS = "PASS"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    check_name: str
    table: str
    year: Optional[int]
    severity: Severity
    passed: bool
    message: str
    actual_value: Optional[Any] = None
    expected_value: Optional[Any] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class MLBDataValidator:
    """Comprehensive data validation for MLB BigQuery tables"""
    
    def __init__(self, project_id: str = "hankstank", dataset: str = "mlb_historical_data"):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset = dataset
        self.full_dataset = f"{project_id}.{dataset}"
        self.results: List[ValidationResult] = []
        
        # Expected data ranges for validation
        self.expected_ranges = {
            'years': (2015, 2026),
            'teams_per_year': 30,
            'games_per_year': (2400, 2900),  # Include postseason/spring
            'score_max': 35,
            'roster_size': (26, 70),  # Off-season can include 40-man + invites
        }
    
    def run_validation(self, years: List[int] = None, tables: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive validation suite
        
        Args:
            years: List of years to validate, or None for all years
            tables: List of tables to validate, or None for all tables
            
        Returns:
            Summary of validation results
        """
        print("=" * 80)
        print("MLB DATA VALIDATION PIPELINE")
        print("=" * 80)
        print(f"Project: {self.project_id}")
        print(f"Dataset: {self.dataset}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Determine which tables to validate
        all_tables = [
            'teams_historical',
            'team_stats_historical', 
            'player_stats_historical',
            'standings_historical',
            'games_historical',
            'rosters_historical',
            'transactions_historical',
            'statcast_pitches'
        ]
        tables_to_check = tables if tables else all_tables
        
        # Determine which years to validate
        if years is None or 'all' in [str(y).lower() for y in years]:
            years_to_check = list(range(2015, datetime.now().year + 1))
        else:
            years_to_check = [int(y) for y in years]
        
        print(f"Validating tables: {', '.join(tables_to_check)}")
        print(f"Validating years: {years_to_check}")
        print("=" * 80)
        print()
        
        # Run validations for each table
        for table in tables_to_check:
            print(f"\n{'─' * 80}")
            print(f"TABLE: {table}")
            print(f"{'─' * 80}")
            
            try:
                # Core validations for all tables
                self._check_table_exists(table)
                self._check_schema_integrity(table)
                
                # Year-specific validations
                for year in years_to_check:
                    if table in ['teams_historical', 'team_stats_historical', 'standings_historical']:
                        self._validate_team_based_table(table, year)
                    elif table == 'player_stats_historical':
                        self._validate_player_stats(year)
                    elif table == 'games_historical':
                        self._validate_games(year)
                    elif table == 'rosters_historical':
                        self._validate_rosters(year)
                    elif table == 'transactions_historical':
                        self._validate_transactions(year)
                    elif table == 'statcast_pitches':
                        self._validate_statcast(year)
                
                # Cross-year validations
                self._check_data_continuity(table, years_to_check)
                
            except Exception as e:
                self.results.append(ValidationResult(
                    check_name=f"table_validation_{table}",
                    table=table,
                    year=None,
                    severity=Severity.CRITICAL,
                    passed=False,
                    message=f"Validation failed with error: {str(e)}"
                ))
        
        # Generate summary report
        return self._generate_report()
    
    def _check_table_exists(self, table: str):
        """Verify table exists in BigQuery"""
        try:
            table_ref = self.client.get_table(f"{self.full_dataset}.{table}")
            self.results.append(ValidationResult(
                check_name="table_exists",
                table=table,
                year=None,
                severity=Severity.PASS,
                passed=True,
                message=f"Table exists with {table_ref.num_rows:,} rows"
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                check_name="table_exists",
                table=table,
                year=None,
                severity=Severity.CRITICAL,
                passed=False,
                message=f"Table not found: {str(e)}"
            ))
    
    def _check_schema_integrity(self, table: str):
        """Verify table has expected schema"""
        required_columns = {
            'teams_historical': ['team_id', 'year', 'team_name'],
            'team_stats_historical': ['team_id', 'year', 'wins', 'losses'],
            'player_stats_historical': ['player_id', 'year', 'team_id'],
            'standings_historical': ['team_id', 'year', 'wins', 'losses', 'pct'],
            'games_historical': ['game_pk', 'game_date', 'year', 'home_team_id', 'away_team_id'],
            'rosters_historical': ['player_id', 'team_id', 'year'],
            'transactions_historical': ['transaction_id', 'date', 'person_id'],
            'statcast_pitches': ['game_pk', 'pitch_type', 'release_speed']
        }
        
        if table not in required_columns:
            return
        
        try:
            table_ref = self.client.get_table(f"{self.full_dataset}.{table}")
            actual_columns = [field.name for field in table_ref.schema]
            missing_columns = [col for col in required_columns[table] if col not in actual_columns]
            
            if not missing_columns:
                self.results.append(ValidationResult(
                    check_name="schema_integrity",
                    table=table,
                    year=None,
                    severity=Severity.PASS,
                    passed=True,
                    message=f"All required columns present ({len(actual_columns)} total columns)"
                ))
            else:
                self.results.append(ValidationResult(
                    check_name="schema_integrity",
                    table=table,
                    year=None,
                    severity=Severity.CRITICAL,
                    passed=False,
                    message=f"Missing required columns: {', '.join(missing_columns)}"
                ))
        except Exception as e:
            self.results.append(ValidationResult(
                check_name="schema_integrity",
                table=table,
                year=None,
                severity=Severity.CRITICAL,
                passed=False,
                message=f"Schema check failed: {str(e)}"
            ))
    
    def _validate_team_based_table(self, table: str, year: int):
        """Validate tables with team-year grain (teams, team_stats, standings_historical)"""
        
        # Check record count
        query = f"""
        SELECT COUNT(*) as record_count
        FROM `{self.full_dataset}.{table}`
        WHERE year = {year}
        """
        result = list(self.client.query(query).result())[0]
        record_count = result['record_count']
        
        # team_stats has multiple rows per team (batting/pitching, sometimes per game_type)
        if table == 'team_stats_historical':
            min_expected, max_expected = 60, 120
        else:
            min_expected = max_expected = self.expected_ranges['teams_per_year']
        
        if min_expected <= record_count <= max_expected:
            self.results.append(ValidationResult(
                check_name="record_count",
                table=table,
                year=year,
                severity=Severity.PASS,
                passed=True,
                message=f"Record count in expected range: {record_count}",
                actual_value=record_count
            ))
        elif record_count == 0:
            # No data for future years is expected
            if year > datetime.now().year:
                self.results.append(ValidationResult(
                    check_name="record_count",
                    table=table,
                    year=year,
                    severity=Severity.PASS,
                    passed=True,
                    message=f"No data for future year (expected)"
                ))
            else:
                self.results.append(ValidationResult(
                    check_name="record_count",
                    table=table,
                    year=year,
                    severity=Severity.CRITICAL,
                    passed=False,
                    message=f"No records found for {year}",
                    actual_value=record_count
                ))
        else:
            severity = Severity.CRITICAL
            self.results.append(ValidationResult(
                check_name="record_count",
                table=table,
                year=year,
                severity=severity,
                passed=False,
                message=f"Record count {record_count} outside expected range",
                actual_value=record_count
            ))
        
        # Check for NULLs in key fields
        if table in ['team_stats_historical', 'standings_historical']:
            # team_stats has batting rows with NULL wins/losses (only pitching has them)
            if table == 'team_stats_historical':
                query = f"""
                SELECT 
                    COUNTIF(team_id IS NULL) as null_team_id
                FROM `{self.full_dataset}.{table}`
                WHERE year = {year}
                """
            else:
                query = f"""
                SELECT 
                    COUNTIF(team_id IS NULL) as null_team_id,
                    COUNTIF(wins IS NULL) as null_wins,
                    COUNTIF(losses IS NULL) as null_losses
                FROM `{self.full_dataset}.{table}`
                WHERE year = {year}
                """
            result = list(self.client.query(query).result())[0]
            
            for field, null_count in result.items():
                if null_count == 0:
                    self.results.append(ValidationResult(
                        check_name=f"null_check_{field.replace('null_', '')}",
                        table=table,
                        year=year,
                        severity=Severity.PASS,
                        passed=True,
                        message=f"No NULL values in {field.replace('null_', '')}"
                    ))
                else:
                    self.results.append(ValidationResult(
                        check_name=f"null_check_{field.replace('null_', '')}",
                        table=table,
                        year=year,
                        severity=Severity.CRITICAL,
                        passed=False,
                        message=f"Found {null_count} NULL values in {field.replace('null_', '')}",
                        actual_value=null_count
                    ))
    
    def _validate_player_stats(self, year: int):
        """Validate player_stats_historical table"""
        table = 'player_stats_historical'
        
        query = f"""
        SELECT COUNT(*) as record_count
        FROM `{self.full_dataset}.{table}`
        WHERE year = {year}
        """
        result = list(self.client.query(query).result())[0]
        record_count = result['record_count']
        
        # Players vary: ~120-250 per year
        min_expected, max_expected = 120, 250
        
        if record_count == 0 and year > datetime.now().year:
            self.results.append(ValidationResult(
                check_name="record_count",
                table=table,
                year=year,
                severity=Severity.PASS,
                passed=True,
                message=f"No data for future year (expected)"
            ))
        elif min_expected <= record_count <= max_expected:
            self.results.append(ValidationResult(
                check_name="record_count",
                table=table,
                year=year,
                severity=Severity.PASS,
                passed=True,
                message=f"Player count in range: {record_count}"
            ))
        elif record_count == 0:
            self.results.append(ValidationResult(
                check_name="record_count",
                table=table,
                year=year,
                severity=Severity.CRITICAL,
                passed=False,
                message=f"No player stats found"
            ))
        else:
            self.results.append(ValidationResult(
                check_name="record_count",
                table=table,
                year=year,
                severity=Severity.WARNING,
                passed=False,
                message=f"Player count {record_count} outside typical 120-250 range"
            ))
    
    def _validate_games(self, year: int):
        """Validate games_historical table"""
        table = 'games_historical'
        
        # Check game count
        query = f"""
        SELECT COUNT(*) as game_count
        FROM `{self.full_dataset}.{table}`
        WHERE year = {year}
        """
        result = list(self.client.query(query).result())[0]
        game_count = result['game_count']
        
        min_expected, max_expected = self.expected_ranges['games_per_year']
        
        # Special case: 2020 COVID season
        if year == 2020:
            min_expected, max_expected = 900, 1100
        
        if year > datetime.now().year:
            # Future year - expect no data
            if game_count == 0:
                self.results.append(ValidationResult(
                    check_name="game_count",
                    table=table,
                    year=year,
                    severity=Severity.PASS,
                    passed=True,
                    message=f"No games for future year (expected)"
                ))
            else:
                self.results.append(ValidationResult(
                    check_name="game_count",
                    table=table,
                    year=year,
                    severity=Severity.WARNING,
                    passed=True,
                    message=f"Found {game_count} games for future year (early season data)"
                ))
        elif year == datetime.now().year:
            # Current year - check if season is in progress
            month = datetime.now().month
            if month < 3:
                # Before season starts
                expected_min = 0
                expected_max = 100
            elif month >= 3 and month < 10:
                # During season - proportional check
                expected_min = 0
                expected_max = max_expected
            else:
                # After season
                expected_min = min_expected
                expected_max = max_expected
            
            if expected_min <= game_count <= expected_max:
                self.results.append(ValidationResult(
                    check_name="game_count",
                    table=table,
                    year=year,
                    severity=Severity.PASS,
                    passed=True,
                    message=f"Game count within expected range: {game_count}",
                    actual_value=game_count
                ))
            else:
                self.results.append(ValidationResult(
                    check_name="game_count",
                    table=table,
                    year=year,
                    severity=Severity.WARNING,
                    passed=False,
                    message=f"Game count outside expected range: {game_count} (expected {expected_min}-{expected_max})",
                    actual_value=game_count
                ))
        else:
            # Historical year - should have full season
            if min_expected <= game_count <= max_expected:
                self.results.append(ValidationResult(
                    check_name="game_count",
                    table=table,
                    year=year,
                    severity=Severity.PASS,
                    passed=True,
                    message=f"Game count within expected range: {game_count}",
                    actual_value=game_count
                ))
            else:
                severity = Severity.WARNING if game_count > min_expected * 0.9 else Severity.CRITICAL
                self.results.append(ValidationResult(
                    check_name="game_count",
                    table=table,
                    year=year,
                    severity=severity,
                    passed=False,
                    message=f"Game count outside expected range: {game_count} (expected {min_expected}-{max_expected})",
                    actual_value=game_count
                ))
        
        if game_count == 0:
            return  # Skip further checks if no data
        
        # Check for duplicate game_pks
        query = f"""
        SELECT COUNT(*) - COUNT(DISTINCT game_pk) as duplicates
        FROM `{self.full_dataset}.{table}`
        WHERE year = {year}
        """
        result = list(self.client.query(query).result())[0]
        duplicates = result['duplicates']
        
        if duplicates == 0:
            self.results.append(ValidationResult(
                check_name="duplicate_check",
                table=table,
                year=year,
                severity=Severity.PASS,
                passed=True,
                message="No duplicate game_pks found"
            ))
        else:
            self.results.append(ValidationResult(
                check_name="duplicate_check",
                table=table,
                year=year,
                severity=Severity.CRITICAL,
                passed=False,
                message=f"Found {duplicates} duplicate game_pks",
                actual_value=duplicates
            ))
        
        # Check score validity
        query = f"""
        SELECT 
            COUNTIF(home_score < 0 OR home_score > {self.expected_ranges['score_max']}) as invalid_home_scores,
            COUNTIF(away_score < 0 OR away_score > {self.expected_ranges['score_max']}) as invalid_away_scores,
            MAX(home_score) as max_home_score,
            MAX(away_score) as max_away_score
        FROM `{self.full_dataset}.{table}`
        WHERE year = {year}
        """
        result = list(self.client.query(query).result())[0]
        
        if result['invalid_home_scores'] == 0 and result['invalid_away_scores'] == 0:
            self.results.append(ValidationResult(
                check_name="score_validity",
                table=table,
                year=year,
                severity=Severity.PASS,
                passed=True,
                message=f"All scores valid (max: home={result['max_home_score']}, away={result['max_away_score']})"
            ))
        else:
            self.results.append(ValidationResult(
                check_name="score_validity",
                table=table,
                year=year,
                severity=Severity.CRITICAL,
                passed=False,
                message=f"Invalid scores: {result['invalid_home_scores']} home, {result['invalid_away_scores']} away"
            ))
        
        # Check for NULL critical fields
        query = f"""
        SELECT 
            COUNTIF(game_pk IS NULL) as null_game_pk,
            COUNTIF(game_date IS NULL) as null_game_date,
            COUNTIF(home_team_id IS NULL) as null_home_team,
            COUNTIF(away_team_id IS NULL) as null_away_team
        FROM `{self.full_dataset}.{table}`
        WHERE year = {year}
        """
        result = list(self.client.query(query).result())[0]
        
        null_issues = {k: v for k, v in result.items() if v > 0}
        if not null_issues:
            self.results.append(ValidationResult(
                check_name="null_check",
                table=table,
                year=year,
                severity=Severity.PASS,
                passed=True,
                message="No NULL values in critical fields"
            ))
        else:
            self.results.append(ValidationResult(
                check_name="null_check",
                table=table,
                year=year,
                severity=Severity.CRITICAL,
                passed=False,
                message=f"NULL values found: {null_issues}"
            ))
    
    def _validate_rosters(self, year: int):
        """Validate rosters_historical table"""
        table = 'rosters_historical'
        
        # Check roster size per team
        query = f"""
        SELECT 
            team_id,
            COUNT(DISTINCT player_id) as roster_size
        FROM `{self.full_dataset}.{table}`
        WHERE year = {year}
        GROUP BY team_id
        HAVING roster_size < {self.expected_ranges['roster_size'][0]} 
            OR roster_size > {self.expected_ranges['roster_size'][1]}
        """
        result = list(self.client.query(query).result())
        
        if len(result) == 0:
            self.results.append(ValidationResult(
                check_name="roster_size",
                table=table,
                year=year,
                severity=Severity.PASS,
                passed=True,
                message=f"All team rosters within expected range ({self.expected_ranges['roster_size'][0]}-{self.expected_ranges['roster_size'][1]})"
            ))
        else:
            teams_with_issues = [f"Team {r['team_id']}: {r['roster_size']} players" for r in result]
            self.results.append(ValidationResult(
                check_name="roster_size",
                table=table,
                year=year,
                severity=Severity.WARNING,
                passed=False,
                message=f"Teams with unusual roster sizes: {'; '.join(teams_with_issues)}"
            ))
    
    def _validate_transactions(self, year: int):
        """Validate transactions_historical table"""
        table = 'transactions_historical'
        
        # Just check if data exists - transactions vary widely
        # Note: date field is STRING in format 'YYYY-MM-DD'
        query = f"""
        SELECT COUNT(*) as transaction_count
        FROM `{self.full_dataset}.{table}`
        WHERE year = {year}
        """
        result = list(self.client.query(query).result())[0]
        transaction_count = result['transaction_count']
        
        if transaction_count > 0:
            self.results.append(ValidationResult(
                check_name="transaction_count",
                table=table,
                year=year,
                severity=Severity.PASS,
                passed=True,
                message=f"Found {transaction_count:,} transactions"
            ))
        elif year > datetime.now().year:
            self.results.append(ValidationResult(
                check_name="transaction_count",
                table=table,
                year=year,
                severity=Severity.PASS,
                passed=True,
                message="No transactions for future year (expected)"
            ))
        else:
            self.results.append(ValidationResult(
                check_name="transaction_count",
                table=table,
                year=year,
                severity=Severity.WARNING,
                passed=False,
                message=f"No transactions found for {year}"
            ))
    
    def _validate_statcast(self, year: int):
        """Validate statcast_pitches table"""
        table = 'statcast_pitches'
        
        # Check if data exists
        query = f"""
        SELECT COUNT(*) as pitch_count
        FROM `{self.full_dataset}.{table}`
        WHERE EXTRACT(YEAR FROM game_date) = {year}
        LIMIT 1
        """
        try:
            result = list(self.client.query(query).result())[0]
            pitch_count = result['pitch_count']
            
            if pitch_count > 0:
                self.results.append(ValidationResult(
                    check_name="pitch_count",
                    table=table,
                    year=year,
                    severity=Severity.PASS,
                    passed=True,
                    message=f"Statcast data exists for {year}"
                ))
            else:
                severity = Severity.PASS if year > datetime.now().year else Severity.WARNING
                self.results.append(ValidationResult(
                    check_name="pitch_count",
                    table=table,
                    year=year,
                    severity=severity,
                    passed=year > datetime.now().year,
                    message=f"No Statcast data for {year}"
                ))
        except Exception as e:
            self.results.append(ValidationResult(
                check_name="pitch_count",
                table=table,
                year=year,
                severity=Severity.WARNING,
                passed=False,
                message=f"Could not validate Statcast data: {str(e)}"
            ))
    
    def _check_data_continuity(self, table: str, years: List[int]):
        """Check for gaps in year coverage"""
        if len(years) < 2:
            return
        
        query = f"""
        SELECT DISTINCT year
        FROM `{self.full_dataset}.{table}`
        WHERE year BETWEEN {min(years)} AND {max(years)}
        ORDER BY year
        """
        
        try:
            result = list(self.client.query(query).result())
            actual_years = sorted([r['year'] for r in result])
            expected_years = list(range(min(years), max(years) + 1))
            
            # Filter out future years
            current_year = datetime.now().year
            expected_years = [y for y in expected_years if y <= current_year]
            
            missing_years = [y for y in expected_years if y not in actual_years]
            
            if not missing_years:
                self.results.append(ValidationResult(
                    check_name="data_continuity",
                    table=table,
                    year=None,
                    severity=Severity.PASS,
                    passed=True,
                    message=f"Complete year coverage: {min(actual_years)}-{max(actual_years)}"
                ))
            else:
                self.results.append(ValidationResult(
                    check_name="data_continuity",
                    table=table,
                    year=None,
                    severity=Severity.WARNING,
                    passed=False,
                    message=f"Missing years: {missing_years}"
                ))
        except Exception as e:
            pass  # Table might not have year column
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        # Count by severity
        passed = [r for r in self.results if r.severity == Severity.PASS]
        warnings = [r for r in self.results if r.severity == Severity.WARNING]
        critical = [r for r in self.results if r.severity == Severity.CRITICAL]
        
        print(f"\n✅ PASSED:   {len(passed):>4} checks")
        print(f"⚠️  WARNING:  {len(warnings):>4} checks")
        print(f"❌ CRITICAL: {len(critical):>4} checks")
        
        # Show warnings
        if warnings:
            print(f"\n{'─' * 80}")
            print("WARNINGS")
            print(f"{'─' * 80}")
            for result in warnings:
                year_str = f"[{result.year}]" if result.year else ""
                print(f"⚠️  {result.table:25} {year_str:8} {result.check_name:25} {result.message}")
        
        # Show critical failures
        if critical:
            print(f"\n{'─' * 80}")
            print("CRITICAL FAILURES")
            print(f"{'─' * 80}")
            for result in critical:
                year_str = f"[{result.year}]" if result.year else ""
                print(f"❌ {result.table:25} {year_str:8} {result.check_name:25} {result.message}")
        
        # Overall status
        print(f"\n{'=' * 80}")
        if critical:
            print("❌ VALIDATION FAILED - Critical issues found")
            overall_status = "FAILED"
        elif warnings:
            print("⚠️  VALIDATION PASSED WITH WARNINGS")
            overall_status = "WARNING"
        else:
            print("✅ VALIDATION PASSED - All checks successful")
            overall_status = "PASSED"
        print("=" * 80)
        
        # Save detailed report to JSON
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'summary': {
                'total_checks': len(self.results),
                'passed': len(passed),
                'warnings': len(warnings),
                'critical': len(critical)
            },
            'results': [asdict(r) for r in self.results]
        }
        
        report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description='MLB Data Quality Validation Pipeline')
    parser.add_argument('--year', nargs='+', default=['all'],
                      help='Year(s) to validate (e.g., 2025, or "all" for all years)')
    parser.add_argument('--tables', nargs='+', default=None,
                      help='Specific tables to validate (default: all tables)')
    parser.add_argument('--report-only', action='store_true',
                      help='Only generate report from previous run')
    
    args = parser.parse_args()
    
    validator = MLBDataValidator()
    
    if args.report_only:
        print("Report-only mode not yet implemented")
        return
    
    # Convert year arguments
    years = None if 'all' in [str(y).lower() for y in args.year] else [int(y) for y in args.year]
    
    # Run validation
    report = validator.run_validation(years=years, tables=args.tables)
    
    # Exit with error code if critical failures
    if report['summary']['critical'] > 0:
        sys.exit(1)
    elif report['summary']['warnings'] > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
