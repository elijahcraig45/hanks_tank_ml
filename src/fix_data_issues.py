#!/usr/bin/env python3
"""
Fix Data Quality Issues

Repairs issues found by data validation pipeline.
Run this after validation identifies problems.

Usage:
    python fix_data_issues.py --issue duplicates --table games_historical --year 2025
"""

from google.cloud import bigquery
from datetime import datetime
import argparse
from typing import List


class DataFixer:
    """Fix data quality issues in BigQuery"""
    
    def __init__(self, project_id: str = "hankstank", dataset: str = "mlb_historical_data"):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset = dataset
        self.full_dataset = f"{project_id}.{dataset}"
    
    def fix_duplicate_games(self, year: int, dry_run: bool = True):
        """
        Remove duplicate game_pk entries intelligently:
        - Keep final/completed games, remove postponed versions
        - Preserve different game types (spring training, postseason, etc.)
        - Only remove true duplicates (same game_pk + game_type)
        
        Args:
            year: Year to deduplicate
            dry_run: If True, only show what would be deleted
        """
        print(f"\n{'=' * 80}")
        print(f"FIXING DUPLICATE GAMES FOR {year}")
        print(f"{'=' * 80}\n")
        
        # Analyze duplicates by type
        analyze_query = f"""
        WITH duplicates AS (
            SELECT 
                game_pk,
                game_type,
                COUNT(*) as occurrence_count,
                COUNTIF(status_code IN ('F', 'FR', 'FO')) as final_count,
                COUNTIF(status_code IN ('DR', 'DI', 'DS')) as postponed_count
            FROM `{self.full_dataset}.games_historical`
            WHERE year = {year}
            GROUP BY game_pk, game_type
            HAVING COUNT(*) > 1
        )
        SELECT 
            COUNT(*) as total_duplicate_games,
            SUM(occurrence_count) as total_duplicate_rows,
            SUM(final_count) as final_versions,
            SUM(postponed_count) as postponed_versions
        FROM duplicates
        """
        
        analysis = list(self.client.query(analyze_query).result())[0]
        
        if analysis['total_duplicate_games'] == 0:
            print("✅ No duplicates found!")
            return
        
        print(f"Found {analysis['total_duplicate_games']} games with duplicates:")
        print(f"  • Total duplicate rows: {analysis['total_duplicate_rows']}")
        print(f"  • Final/Completed versions: {analysis['final_versions']}")
        print(f"  • Postponed versions: {analysis['postponed_versions']}")
        print()
        
        # Find what would be deleted
        preview_query = f"""
        WITH ranked_games AS (
            SELECT 
                game_pk,
                game_type,
                game_date,
                status_code,
                status_description,
                home_team_name,
                away_team_name,
                home_score,
                away_score,
                -- Prioritize: Final (F, FR, FO) > In Progress > Postponed (DR, DI, DS)
                ROW_NUMBER() OVER (
                    PARTITION BY game_pk, game_type 
                    ORDER BY 
                        CASE 
                            WHEN status_code IN ('F', 'FR', 'FO') THEN 1
                            WHEN status_code IN ('DR', 'DI', 'DS') THEN 3
                            ELSE 2
                        END,
                        game_date DESC,
                        (COALESCE(home_score, 0) + COALESCE(away_score, 0)) DESC
                ) as rn
            FROM `{self.full_dataset}.games_historical`
            WHERE year = {year}
        ),
        duplicates_only AS (
            SELECT 
                game_pk,
                CASE WHEN rn = 1 THEN 'KEEP' ELSE 'DELETE' END as action,
                status_code,
                status_description,
                game_date,
                home_team_name,
                away_team_name,
                CAST(home_score AS STRING) as home_score,
                CAST(away_score AS STRING) as away_score
            FROM ranked_games
            WHERE game_pk IN (
                SELECT game_pk 
                FROM ranked_games 
                GROUP BY game_pk, game_type
                HAVING COUNT(*) > 1
            )
        )
        SELECT 
            COUNT(*) as would_delete,
            COUNTIF(status_code IN ('DR', 'DI', 'DS')) as postponed_deletes,
            COUNTIF(status_code = 'F') as final_deletes
        FROM duplicates_only
        WHERE action = 'DELETE'
        """
        
        preview = list(self.client.query(preview_query).result())[0]
        
        print(f"Deduplication strategy:")
        print(f"  ✅ Keep: Final/Completed games (status F, FR, FO)")
        print(f"  ❌ Remove: Postponed versions when final exists (status DR, DI, DS)")
        print(f"  ✅ Keep: All unique game_type combinations (spring training, postseason, etc.)")
        print()
        print(f"Impact:")
        print(f"  • Would delete {preview['would_delete']} rows total")
        print(f"  • Postponed games to remove: {preview['postponed_deletes']}")
        print(f"  • Final games to remove: {preview['final_deletes']} (exact duplicates only)")
        print()
        
        if dry_run:
            print(f"⚠️  DRY RUN MODE - No changes will be made")
            print(f"Run with --execute to apply fixes\n")
        else:
            print(f"🔧 EXECUTING FIX...\n")
            
            # Create temporary table with deduplicated data
            dedup_query = f"""
            CREATE OR REPLACE TABLE `{self.full_dataset}.games_historical_deduped` AS
            WITH ranked_games AS (
                SELECT 
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY game_pk, game_type 
                        ORDER BY 
                            CASE 
                                WHEN status_code IN ('F', 'FR', 'FO') THEN 1
                                WHEN status_code IN ('DR', 'DI', 'DS') THEN 3
                                ELSE 2
                            END,
                            game_date DESC,
                            (COALESCE(home_score, 0) + COALESCE(away_score, 0)) DESC
                    ) as rn
                FROM `{self.full_dataset}.games_historical`
            )
            SELECT * EXCEPT(rn)
            FROM ranked_games
            WHERE rn = 1
            """
            
            job = self.client.query(dedup_query)
            job.result()
            
            print("✅ Created deduplicated table: games_historical_deduped")
            
            # Get row counts
            old_count_query = f"SELECT COUNT(*) as cnt FROM `{self.full_dataset}.games_historical`"
            new_count_query = f"SELECT COUNT(*) as cnt FROM `{self.full_dataset}.games_historical_deduped`"
            
            old_count = list(self.client.query(old_count_query).result())[0]['cnt']
            new_count = list(self.client.query(new_count_query).result())[0]['cnt']
            
            print(f"Old table: {old_count:,} rows")
            print(f"New table: {new_count:,} rows")
            print(f"Removed: {old_count - new_count:,} duplicates")
            
            # Verify we kept the right games
            verify_query = f"""
            SELECT 
                COUNTIF(status_code IN ('F', 'FR', 'FO')) as final_games,
                COUNTIF(status_code IN ('DR', 'DI', 'DS')) as postponed_games,
                COUNT(DISTINCT game_type) as game_types
            FROM `{self.full_dataset}.games_historical_deduped`
            WHERE year = {year}
            """
            verification = list(self.client.query(verify_query).result())[0]
            
            print(f"\nVerification:")
            print(f"  • Final/Completed games: {verification['final_games']:,}")
            print(f"  • Postponed games (unique): {verification['postponed_games']:,}")
            print(f"  • Game types preserved: {verification['game_types']}")
            
            # Backup old table
            backup_table = f"{self.full_dataset}.games_historical_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"\n📦 Creating backup: {backup_table}")
            
            backup_job = self.client.copy_table(
                f"{self.full_dataset}.games_historical",
                backup_table
            )
            backup_job.result()
            
            # Replace original table
            print(f"🔄 Replacing original table...")
            
            replace_query = f"""
            CREATE OR REPLACE TABLE `{self.full_dataset}.games_historical` AS
            SELECT * FROM `{self.full_dataset}.games_historical_deduped`
            """
            
            self.client.query(replace_query).result()
            
            # Drop temp table
            self.client.delete_table(f"{self.full_dataset}.games_historical_deduped")
            
            print(f"\n✅ COMPLETE! Duplicates removed intelligently.")
            print(f"   • Kept final/completed games")
            print(f"   • Removed postponed versions")
            print(f"   • Preserved all game types (spring training, postseason, etc.)")
            print(f"\n📁 Backup saved to: {backup_table}")
            print(f"   You can restore with:")
            print(f"   bq cp {backup_table} {self.full_dataset}.games_historical")
    
    def show_duplicate_games_details(self, year: int):
        """Show details of duplicate games for inspection"""
        query = f"""
        WITH ranked_games AS (
            SELECT 
                game_pk,
                game_type,
                game_date,
                status_code,
                status_description,
                home_team_name,
                away_team_name,
                home_score,
                away_score,
                ROW_NUMBER() OVER (
                    PARTITION BY game_pk, game_type 
                    ORDER BY 
                        CASE 
                            WHEN status_code IN ('F', 'FR', 'FO') THEN 1
                            WHEN status_code IN ('DR', 'DI', 'DS') THEN 3
                            ELSE 2
                        END,
                        game_date DESC,
                        (COALESCE(home_score, 0) + COALESCE(away_score, 0)) DESC
                ) as rn
            FROM `{self.full_dataset}.games_historical`
            WHERE year = {year}
        )
        SELECT 
            game_pk,
            game_type,
            game_date,
            status_code,
            status_description,
            home_team_name,
            away_team_name,
            CAST(home_score AS STRING) as home_score,
            CAST(away_score AS STRING) as away_score,
            rn
        FROM ranked_games
        WHERE game_pk IN (
            SELECT game_pk 
            FROM ranked_games 
            GROUP BY game_pk, game_type
            HAVING COUNT(*) > 1
        )
        ORDER BY game_pk, rn
        LIMIT 30
        """
        
        results = list(self.client.query(query).result())
        
        if not results:
            print(f"\n✅ No duplicate games found for {year}!")
            return
        
        print(f"\n{'=' * 80}")
        print(f"DUPLICATE GAMES DETAILS FOR {year}")
        print(f"{'=' * 80}\n")
        print(f"Game Types: R=Regular Season, D=Division Series, L=League Championship,")
        print(f"            F=World Series, W=Wild Card, S=Spring Training")
        print()
        
        current_game = None
        for row in results:
            if current_game != row['game_pk']:
                current_game = row['game_pk']
                print(f"\n🎮 Game {row['game_pk']} (Type: {row['game_type']}):")
            
            action = "✅ KEEP" if row['rn'] == 1 else "❌ DELETE"
            score_str = f"{row['home_score']:>4} - {row['away_score']:<4}"
            print(f"  {action} | {row['game_date']} | {row['status_description']:15} | "
                  f"{row['home_team_name']:20} {score_str} {row['away_team_name']:20}")


def main():
    parser = argparse.ArgumentParser(description='Fix data quality issues in MLB historical data')
    parser.add_argument('--issue', required=True, choices=['duplicates'], help='Issue type to fix')
    parser.add_argument('--year', type=int, help='Year to fix')
    parser.add_argument('--execute', action='store_true', help='Actually apply fixes (default is dry-run)')
    parser.add_argument('--details', action='store_true', help='Show details of issues')
    
    args = parser.parse_args()
    
    fixer = DataFixer()
    
    if args.issue == 'duplicates':
        if args.details:
            if not args.year:
                parser.error('--year is required for showing details')
            fixer.show_duplicate_games_details(args.year)
        else:
            if not args.year:
                parser.error('--year is required for fixing duplicates')
            fixer.fix_duplicate_games(args.year, dry_run=not args.execute)


if __name__ == "__main__":
    main()
