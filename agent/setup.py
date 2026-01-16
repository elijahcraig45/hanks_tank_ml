"""
ML Agent System - Setup and Installation Guide
Run this to bootstrap the agent system on your machine
"""

import os
import sys
import json
import subprocess
from pathlib import Path


class AgentSetup:
    """Bootstrap and configuration for AI Agent system"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.agent_dir = self.project_root / "agent"
        self.logs_dir = self.agent_dir / "logs"
        self.knowledge_dir = self.agent_dir / "knowledge"
    
    def print_section(self, title: str):
        """Pretty print section headers"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    
    def check_python_version(self) -> bool:
        """Verify Python 3.9+"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 9:
            print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
            return True
        print(f"✗ Python {version.major}.{version.minor} - requires 3.9+")
        return False
    
    def check_dependencies(self) -> bool:
        """Check if key packages are installed"""
        required = [
            ("anthropic", "Anthropic API client"),
            ("google-cloud-bigquery", "BigQuery connector"),
            ("pandas", "Data manipulation"),
            ("numpy", "Numerical computing"),
            ("scikit-learn", "ML utilities"),
            ("xgboost", "XGBoost models"),
        ]
        
        all_ok = True
        for package, description in required:
            try:
                __import__(package)
                print(f"✓ {package:30} ({description})")
            except ImportError:
                print(f"✗ {package:30} - Not installed")
                all_ok = False
        
        return all_ok
    
    def install_dependencies(self) -> bool:
        """Install required packages"""
        self.print_section("Installing Dependencies")
        
        packages = [
            "anthropic>=0.7.0",
            "google-cloud-bigquery>=3.13.0",
            "google-auth-oauthlib>=1.1.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "xgboost>=2.0.0",
        ]
        
        print("Installing packages...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q"
            ] + packages)
            print("✓ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Installation failed: {e}")
            return False
    
    def setup_directories(self) -> bool:
        """Create agent directory structure"""
        self.print_section("Setting Up Directories")
        
        dirs = [
            self.agent_dir,
            self.agent_dir / "core",
            self.agent_dir / "tools",
            self.agent_dir / "knowledge",
            self.agent_dir / "config",
            self.logs_dir,
            self.logs_dir / "decisions",
            self.knowledge_dir,
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created {d.relative_to(self.project_root)}")
        
        return True
    
    def setup_environment(self) -> bool:
        """Configure environment variables"""
        self.print_section("Environment Configuration")
        
        env_file = self.project_root / ".env.agent"
        
        if env_file.exists():
            print("✓ .env.agent already exists (keeping existing)")
            return True
        
        env_template = """# ML Agent Configuration
# Get these from your GCP and Anthropic accounts

# Anthropic API
ANTHROPIC_API_KEY=your_api_key_here

# Google Cloud Platform
GCP_PROJECT_ID=your_project_id_here
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Optional: Local LLM (Ollama)
USE_LOCAL_LLM=false
LOCAL_LLM_MODEL=mistral:7b
LOCAL_LLM_BASE_URL=http://localhost:11434

# Agent Behavior
AGENT_LOG_DIRECTORY=agent/logs
PRIMARY_MODEL_VERSION=v3
RETRAIN_THRESHOLD_ACCURACY=0.546
"""
        
        with open(env_file, "w") as f:
            f.write(env_template)
        
        print(f"✓ Created {env_file.name} (edit with your credentials)")
        print("\nRequired configuration:")
        print("  1. ANTHROPIC_API_KEY - from https://console.anthropic.com")
        print("  2. GCP_PROJECT_ID - your Google Cloud project")
        print("  3. GOOGLE_APPLICATION_CREDENTIALS - path to service account JSON")
        print("\nOptional:")
        print("  4. Ollama for local LLM (download from https://ollama.ai)")
        
        return True
    
    def create_example_queries(self) -> bool:
        """Create example agent queries"""
        self.print_section("Creating Example Usage")
        
        examples = {
            "retrain_check.json": {
                "query": "Should I retrain the V3 model? Check for performance degradation and data quality issues.",
                "description": "Checks if model needs retraining"
            },
            "feature_analysis.json": {
                "query": "What new features should we engineer? Look at current feature importance and suggest improvements.",
                "description": "Suggests new features based on analysis"
            },
            "prediction_confidence.json": {
                "query": "Analyze the confidence distribution for yesterday's predictions. Which games had highest/lowest confidence?",
                "description": "Analyzes prediction confidence"
            },
            "baseball_insights.json": {
                "query": "What baseball domain insights should we incorporate? Check for relevant pitcher fatigue, park factors, or seasonal trends.",
                "description": "Gets domain-specific insights"
            },
            "data_health.json": {
                "query": "Check data health: Is BigQuery data fresh? Are there missing values or anomalies in recent training data?",
                "description": "Validates data quality"
            }
        }
        
        examples_dir = self.agent_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        for filename, data in examples.items():
            filepath = examples_dir / filename
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            print(f"✓ Created {examples_dir.name}/{filename}")
        
        return True
    
    def create_readme(self) -> bool:
        """Create agent README"""
        self.print_section("Creating Documentation")
        
        readme = """# ML Model Management Agent

Autonomous AI agent for managing MLB prediction models with BigQuery integration.

## Features

- **Model Management**: Train, validate, and compare V1/V2/V3 models
- **Performance Monitoring**: Track accuracy, precision, recall, F1 scores
- **Data Quality**: Validate training data and detect anomalies
- **Confidence Analysis**: Compute prediction confidence intervals
- **Domain Knowledge**: Leverage baseball analytics and sabermetrics
- **Self-Improvement**: Iterate on models and features based on performance
- **Decision Tracking**: Full audit trail of all agent actions

## Quick Start

1. **Setup**:
   ```bash
   python agent/setup.py
   ```

2. **Configure**:
   - Edit `.env.agent` with your credentials
   - Set ANTHROPIC_API_KEY and GCP_PROJECT_ID

3. **Run**:
   ```bash
   python agent/main.py
   ```

4. **Example Queries**:
   - "Should I retrain the V3 model?"
   - "What new features should we engineer?"
   - "Check data health in BigQuery"

## Architecture

```
Agent Orchestrator (Claude API)
    ├── Data Agent (BigQuery queries)
    ├── ML Agent (Model training/validation)
    ├── Analysis Agent (Metrics, confidence)
    ├── Confirmation Agent (User approval)
    └── Tool System (Execution layer)

Knowledge Base
    ├── Pitcher fatigue analysis
    ├── Park factors
    ├── Injury impact
    └── Weather effects

Data Layer
    ├── BigQuery (26,900 games, 2015-2025)
    ├── Current models (V1/V2/V3)
    └── Predictions log
```

## Tools Available

### Query & Analysis
- `query_bigquery` - Execute SQL queries
- `analyze_model_confidence` - Percentile analysis
- `get_baseball_stats` - Live stats and news
- `validate_data_quality` - Check data health

### Model Operations
- `train_model` - Train/retrain models
- `search_knowledge_base` - Find domain knowledge
- `confirm_action` - Request user approval

## Configuration

Key settings in `agent/config/agent_config.py`:

- **Models**: V1 (54.0%), V2 (54.4%), V3 (54.6%)
- **Data**: 2015-2024 training, 2025 validation
- **Retraining threshold**: 0.546 (54.6% accuracy)
- **Confidence high percentile**: 90th (56.5% accuracy)

## Cost Optimization

- **Local LLM**: Use Ollama for reasoning, reduces API costs by 80%
- **Batch API**: Use for non-urgent analysis tasks
- **BigQuery**: Cached queries, query optimization built-in

Estimated monthly cost:
- Full API: ~$200/month
- With local LLM: ~$40/month
- Batch only: ~$10/month

## Monitoring & Logging

- Decision logs: `agent/logs/decisions.jsonl`
- Agent reasoning: `agent/logs/agent_decisions.log`
- Performance tracking: BigQuery `agent_decisions` table

## Examples

See `agent/examples/` for query templates:
- `retrain_check.json` - Model retraining
- `feature_analysis.json` - Feature engineering
- `prediction_confidence.json` - Confidence analysis
- `baseball_insights.json` - Domain insights
- `data_health.json` - Data quality

## Next Steps

1. Implement tool functions in `agent/tools/`
2. Add BigQuery table schemas to config
3. Integrate with existing model training scripts
4. Set up scheduled agent runs (e.g., daily)
5. Create monitoring dashboard

## Support

For issues or questions, check:
- `agent/logs/` for detailed logs
- BigQuery `mlb_historical_data` for data validation
- Model performance history for trends
"""
        
        readme_path = self.agent_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme)
        
        print(f"✓ Created {readme_path.relative_to(self.project_root)}")
        return True
    
    def run_setup(self) -> bool:
        """Execute full setup"""
        self.print_section("ML Model Management Agent - Setup")
        
        print("Checking system requirements...")
        if not self.check_python_version():
            return False
        
        self.print_section("Checking Installed Packages")
        if not self.check_dependencies():
            response = input("\n✓ Install missing dependencies? (y/n): ")
            if response.lower() == "y":
                if not self.install_dependencies():
                    return False
            else:
                print("✗ Cannot proceed without dependencies")
                return False
        else:
            print("✓ All dependencies installed")
        
        if not self.setup_directories():
            return False
        
        if not self.setup_environment():
            return False
        
        if not self.create_example_queries():
            return False
        
        if not self.create_readme():
            return False
        
        self.print_section("Setup Complete! ✓")
        print("""
Next steps:
1. Edit .env.agent with your API credentials
2. Run: python agent/main.py
3. Try a query: "Should I retrain the V3 model?"

Documentation: agent/README.md
Example queries: agent/examples/
        """)
        
        return True


if __name__ == "__main__":
    setup = AgentSetup()
    success = setup.run_setup()
    sys.exit(0 if success else 1)
