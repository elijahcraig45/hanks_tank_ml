"""
Agent Configuration
Local settings, environment variables, model parameters
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AgentConfig:
    """Central configuration for ML Agent system"""
    
    # API Configuration
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"
    
    # BigQuery Configuration
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
    GCP_CREDENTIALS_PATH: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    BIGQUERY_DATASET: str = "mlb_historical_data"
    
    # Model Configuration
    MODEL_VERSIONS: list = ["v1", "v2", "v3"]
    PRIMARY_MODEL_VERSION: str = "v3"  # V3 XGBoost (54.6% accuracy)
    RETRAIN_THRESHOLD_ACCURACY: float = 0.546  # Alert if drops below this
    
    # Data Configuration
    TRAINING_YEARS: tuple = (2015, 2024)
    VALIDATION_YEAR: int = 2025
    MIN_TRAINING_SAMPLES: int = 1000
    DATA_FRESHNESS_THRESHOLD_DAYS: int = 1  # Alert if data > 1 day old
    
    # Agent Behavior Configuration
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_CONFIRMATION_TIMEOUT_SEC: int = 300  # 5 minutes
    AGENT_LOG_DIRECTORY: str = "agent/logs"
    AGENT_DECISION_LOG: str = "agent/logs/decisions.jsonl"
    
    # Feature Configuration
    V1_FEATURE_COUNT: int = 5
    V2_FEATURE_COUNT: int = 44
    V3_FEATURE_COUNT: int = 57
    
    # Confidence Thresholds
    CONFIDENCE_HIGH_PERCENTILE: float = 0.90  # Top 90% confident predictions
    CONFIDENCE_HIGH_ACCURACY: float = 0.565   # Expected accuracy at 90th percentile
    CONFIDENCE_MEDIUM_PERCENTILE: float = 0.50
    CONFIDENCE_MEDIUM_ACCURACY: float = 0.563  # Expected accuracy at 50th percentile
    
    # Baseball Domain Configuration
    TEAM_COUNT: int = 30
    GAMES_PER_SEASON: int = 162
    MIN_PITCHER_REST_DAYS: int = 3
    PITCHER_FATIGUE_THRESHOLD_PITCH_COUNT: int = 100
    
    # Local LLM Configuration (Optional: Ollama)
    USE_LOCAL_LLM: bool = False
    LOCAL_LLM_MODEL: str = "mistral:7b"  # or "llama2:13b"
    LOCAL_LLM_BASE_URL: str = "http://localhost:11434"
    LOCAL_LLM_TIMEOUT_SEC: int = 120
    
    # Monitoring Configuration
    ENABLE_PERFORMANCE_MONITORING: bool = True
    MONITOR_CHECK_INTERVAL_MIN: int = 60  # Check every 60 minutes
    ENABLE_DATA_QUALITY_ALERTS: bool = True
    ENABLE_MODEL_DRIFT_ALERTS: bool = True
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith("_") and key.isupper()
        }
    
    def validate(self) -> tuple[bool, str]:
        """Validate critical configuration"""
        issues = []
        
        if not self.ANTHROPIC_API_KEY:
            issues.append("ANTHROPIC_API_KEY not set")
        
        if not self.GCP_PROJECT_ID:
            issues.append("GCP_PROJECT_ID not set")
        
        if self.CONFIDENCE_HIGH_PERCENTILE <= self.CONFIDENCE_MEDIUM_PERCENTILE:
            issues.append("Confidence percentiles not in ascending order")
        
        if not os.path.exists(self.AGENT_LOG_DIRECTORY):
            os.makedirs(self.AGENT_LOG_DIRECTORY, exist_ok=True)
        
        return (len(issues) == 0, "; ".join(issues) if issues else "All checks passed")


# Load configuration
config = AgentConfig()

# Validation check
is_valid, message = config.validate()
if not is_valid:
    print(f"Configuration Warning: {message}")
