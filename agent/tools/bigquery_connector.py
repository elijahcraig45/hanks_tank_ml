"""
BigQuery connector for ML Agent
Handles all database operations: queries, feature loading, metadata tracking
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


class BigQueryConnector:
    """Unified interface to BigQuery for agent operations"""
    
    def __init__(self, project_id: Optional[str] = None, credentials_path: Optional[str] = None):
        """Initialize BigQuery client"""
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self.client = bigquery.Client(
                project=self.project_id,
                credentials=credentials
            )
        else:
            self.client = bigquery.Client(project=self.project_id)
        
        self.dataset_id = "mlb_historical_data"
        logger.info(f"Initialized BigQuery connector for project: {self.project_id}")
    
    def get_training_data(self, version: str = "v3", year_range: tuple = (2015, 2024)) -> pd.DataFrame:
        """Fetch training data for model version"""
        start_year, end_year = year_range
        
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.training_features_{version}`
        WHERE game_date BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
        ORDER BY game_date ASC
        """
        
        logger.info(f"Querying training data for {version} ({start_year}-{end_year})")
        df = self.client.query(query).to_pandas()
        logger.info(f"Retrieved {len(df)} training samples for {version}")
        
        return df
    
    def get_validation_data(self, version: str = "v3", year: int = 2025) -> pd.DataFrame:
        """Fetch validation data for model version"""
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.training_features_{version}`
        WHERE EXTRACT(YEAR FROM game_date) = {year}
        ORDER BY game_date ASC
        """
        
        logger.info(f"Querying validation data for {version} ({year})")
        df = self.client.query(query).to_pandas()
        logger.info(f"Retrieved {len(df)} validation samples for {version}")
        
        return df
    
    def get_latest_games(self, days: int = 7) -> pd.DataFrame:
        """Get most recent games for live predictions"""
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        
        query = f"""
        SELECT 
            game_id,
            game_date,
            home_team,
            away_team,
            home_runs,
            away_runs,
            outcome
        FROM `{self.project_id}.{self.dataset_id}.games`
        WHERE game_date BETWEEN '{from_date}' AND '{to_date}'
        ORDER BY game_date DESC
        """
        
        logger.info(f"Querying recent games ({from_date} to {to_date})")
        df = self.client.query(query).to_pandas()
        logger.info(f"Retrieved {len(df)} recent games")
        
        return df
    
    def get_model_performance_history(self, model_version: str, days: int = 30) -> pd.DataFrame:
        """Track model performance over time"""
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        query = f"""
        SELECT 
            prediction_date,
            model_version,
            accuracy,
            precision,
            recall,
            f1_score,
            predictions_count
        FROM `{self.project_id}.{self.dataset_id}.model_performance`
        WHERE model_version = '{model_version}'
            AND prediction_date >= '{from_date}'
        ORDER BY prediction_date DESC
        """
        
        logger.info(f"Querying performance history for {model_version}")
        df = self.client.query(query).to_pandas()
        logger.info(f"Retrieved {len(df)} performance records")
        
        return df
    
    def get_feature_importance(self, model_version: str) -> Dict[str, float]:
        """Retrieve feature importance for model version"""
        query = f"""
        SELECT 
            feature_name,
            importance_score
        FROM `{self.project_id}.{self.dataset_id}.feature_importance`
        WHERE model_version = '{model_version}'
        ORDER BY importance_score DESC
        """
        
        logger.info(f"Querying feature importance for {model_version}")
        df = self.client.query(query).to_pandas()
        
        return dict(zip(df["feature_name"], df["importance_score"]))
    
    def check_data_freshness(self) -> Dict[str, Any]:
        """Check how recent the training data is"""
        query = f"""
        SELECT 
            'games' as table_name,
            MAX(game_date) as latest_date,
            COUNT(*) as total_records
        FROM `{self.project_id}.{self.dataset_id}.games`
        UNION ALL
        SELECT 
            'team_stats',
            MAX(date),
            COUNT(*)
        FROM `{self.project_id}.{self.dataset_id}.team_stats`
        UNION ALL
        SELECT 
            'player_stats',
            MAX(date),
            COUNT(*)
        FROM `{self.project_id}.{self.dataset_id}.player_stats`
        """
        
        df = self.client.query(query).to_pandas()
        
        result = {}
        for _, row in df.iterrows():
            table_name = row["table_name"]
            latest_date = row["latest_date"]
            total_records = row["total_records"]
            days_old = (datetime.now().date() - latest_date.date()).days
            
            result[table_name] = {
                "latest_date": str(latest_date),
                "days_old": days_old,
                "total_records": int(total_records),
                "is_fresh": days_old <= 1,
                "is_stale": days_old > 7
            }
        
        logger.info(f"Data freshness check: {result}")
        return result
    
    def get_team_stats(self, team: str, lookback_days: int = 30) -> Dict[str, float]:
        """Get recent team performance metrics"""
        from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        query = f"""
        SELECT 
            team,
            AVG(wins) as avg_wins_per_game,
            AVG(runs_scored) as avg_runs_scored,
            AVG(runs_allowed) as avg_runs_allowed,
            AVG(win_pct) as win_percentage
        FROM `{self.project_id}.{self.dataset_id}.team_stats`
        WHERE team = '{team}'
            AND date >= '{from_date}'
        GROUP BY team
        """
        
        df = self.client.query(query).to_pandas()
        
        if len(df) > 0:
            return df.iloc[0].to_dict()
        return {}
    
    def get_player_stats(self, player_id: str, stat_type: str = "batting") -> Dict[str, float]:
        """Get recent player performance stats"""
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.player_stats`
        WHERE player_id = '{player_id}'
            AND stat_type = '{stat_type}'
        ORDER BY date DESC
        LIMIT 1
        """
        
        df = self.client.query(query).to_pandas()
        
        if len(df) > 0:
            return df.iloc[0].to_dict()
        return {}
    
    def log_prediction(self, prediction_data: Dict) -> bool:
        """Log prediction for performance tracking"""
        table_id = f"{self.project_id}.{self.dataset_id}.predictions_log"
        
        rows_to_insert = [
            {
                "prediction_id": prediction_data.get("prediction_id"),
                "model_version": prediction_data.get("model_version"),
                "game_id": prediction_data.get("game_id"),
                "predicted_outcome": prediction_data.get("predicted_outcome"),
                "confidence": prediction_data.get("confidence"),
                "prediction_date": datetime.now().isoformat(),
                "actual_outcome": None,  # Fill in later when game result is known
            }
        ]
        
        try:
            errors = self.client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error(f"Insert errors: {errors}")
                return False
            logger.info(f"Logged prediction: {prediction_data.get('game_id')}")
            return True
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            return False
    
    def log_agent_decision(self, decision_data: Dict) -> bool:
        """Log agent decisions for audit trail"""
        table_id = f"{self.project_id}.{self.dataset_id}.agent_decisions"
        
        rows_to_insert = [
            {
                "decision_id": decision_data.get("decision_id"),
                "agent_role": decision_data.get("agent_role"),
                "decision_type": decision_data.get("decision_type"),
                "decision_text": decision_data.get("decision_text"),
                "confidence": decision_data.get("confidence"),
                "requires_confirmation": decision_data.get("requires_confirmation"),
                "user_approved": decision_data.get("user_approved"),
                "decision_timestamp": datetime.now().isoformat(),
                "result": decision_data.get("result"),
            }
        ]
        
        try:
            errors = self.client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error(f"Insert errors: {errors}")
                return False
            logger.info(f"Logged agent decision")
            return True
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
            return False


# Example usage
if __name__ == "__main__":
    connector = BigQueryConnector()
    
    # Check data freshness
    freshness = connector.check_data_freshness()
    print("Data Freshness:")
    print(json.dumps(freshness, indent=2, default=str))
    
    # Get model performance
    perf = connector.get_model_performance_history("v3", days=30)
    print(f"\nRecent performance: {len(perf)} records")
    print(perf.head())
