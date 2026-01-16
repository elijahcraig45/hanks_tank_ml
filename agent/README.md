"""
ML MODEL MANAGEMENT AGENT - README
Autonomous AI system for baseball prediction models
"""

# ML MODEL MANAGEMENT AGENT

Autonomous AI agent for managing, training, and optimizing MLB game outcome prediction models. This system uses Claude API with tool use (function calling) to make intelligent decisions about model retraining, feature engineering, and prediction analysis.

## 🎯 What It Does

- **Autonomous Retraining**: Monitors model performance and automatically suggests retraining when accuracy degrades
- **Feature Engineering**: Analyzes feature importance and suggests new engineered features
- **Confidence Analysis**: Computes prediction confidence intervals and percentile-based filtering
- **Data Quality**: Validates training data and detects anomalies or data drift
- **Baseball Domain Knowledge**: Integrates sabermetrics and domain expertise into decisions
- **Decision Tracking**: Full audit trail of all agent decisions with reasoning
- **Cost Optimization**: Uses local LLM for routine tasks, API for complex reasoning

## 🏗️ Architecture

```
┌─────────────────┐
│   User CLI      │  (interactive commands or scripts)
└────────┬────────┘
         │
┌────────▼────────────────────────────────────────┐
│  Agent Orchestrator (Claude API with Tool Use)  │
│  - Agentic loop (up to 10 iterations)           │
│  - Tool invocation & execution                  │
│  - Decision logging                             │
└────────┬────────────────────────────────────────┘
         │
    ┌────┴────┬────────────┬─────────────┬──────────────┐
    │          │            │             │              │
┌───▼──┐  ┌───▼───┐  ┌────▼────┐  ┌─────▼──────┐  ┌────▼────┐
│BigQ  │  │Model  │  │Analysis  │  │Confirmation│  │Knowledge│
│Tools │  │Tools  │  │Tools     │  │Tools       │  │Search   │
└───┬──┘  └───┬───┘  └────┬────┘  └─────┬──────┘  └────┬────┘
    │         │           │             │              │
    └─────────┴───────────┴─────────────┴──────────────┘
              │
    ┌─────────▼──────────────┐
    │  Existing Ecosystem    │
    │  ✓ Models (V1/V2/V3)   │
    │  ✓ Training scripts    │
    │  ✓ Features (57 total) │
    │  ✓ BigQuery data       │
    │  ✓ Predictions (2026)  │
    └────────────────────────┘
```

## 📦 Installation

### Quick Setup (5 minutes)

```bash
# 1. Install bootstrap
cd agent
python setup.py

# 2. Configure credentials
nano .env.agent
# Set: ANTHROPIC_API_KEY, GCP_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS

# 3. Start agent
python main.py

# 4. Try a command
agent> query "Should I retrain the V3 model?"
```

### Full Setup with Optional Local LLM

```bash
# Install Ollama from https://ollama.ai
ollama pull mistral:7b

# Or for more capability:
ollama pull llama2:13b

# Update .env.agent
USE_LOCAL_LLM=true
LOCAL_LLM_MODEL=mistral:7b
LOCAL_LLM_BASE_URL=http://localhost:11434
```

## 🚀 Usage

### Interactive CLI

```bash
$ python agent/main.py

agent> query "Should I retrain the V3 model?"
agent> analyze
agent> features        # Suggest new features
agent> health         # Check data quality
agent> knowledge pitcher fatigue
agent> config
agent> logs
agent> help
agent> exit
```

### Command Line

```bash
# Check system status
python agent/main.py --command status

# Run specific analysis
python agent/main.py --command retrain
python agent/main.py --command analyze
python agent/main.py --command features
python agent/main.py --command health

# Single query
python agent/main.py --query "What is V3 model accuracy?"

# Search knowledge base
python agent/main.py --knowledge-search "pitcher fatigue"
```

## 📊 Current Model Status

| Model | Algorithm | Features | Accuracy | Status |
|-------|-----------|----------|----------|--------|
| V1 | LogisticRegression | 5 | 54.0% | Baseline |
| V2 | LogisticRegression | 44 | 54.4% | +0.4% |
| V3 | XGBoost | 57 | 54.6% | ⭐ Production |

### Confidence Analysis Results

| Percentile | Coverage | Accuracy | F1 Score | Use Case |
|-----------|----------|----------|----------|----------|
| 50th | 50% of games | 56.3% | 0.677 | Standard predictions |
| 90th | 10.2% of games | 56.5% | 0.689 | **Optimal** |
| 95th | 5% of games | 57.6% | 0.694 | High confidence only |
| 99th | 1% of games | 57.7% | 0.645 | Extreme confidence |

## 🛠️ Tools & Functions

### Query Tools
- `query_bigquery(query: str)` - Execute SQL queries
- `validate_data_quality(data_type: str)` - Check data health

### Model Operations
- `train_model(version: str)` - Train/retrain models
- `analyze_model_confidence(model_version: str)` - Percentile analysis

### Baseball Data
- `get_baseball_stats(query_type: str)` - Live stats, injuries, news
- `search_knowledge_base(query: str)` - Find domain knowledge

### User Interaction
- `confirm_action(action: str, impact: str)` - Request approval

## 📚 Baseball Domain Knowledge

The agent has built-in understanding of:

- **Pitcher Fatigue**: ERA increases ~0.5 with high pitch count
- **Park Factors**: Fenway +15%, Coors +20% run scoring
- **Weather Effects**: +5-7 ft ball flight per 15°F temperature change
- **Batter Form**: Hot streaks give +12% win probability advantage
- **Injury Impact**: Star player loss = -4% to -6% team win rate
- **Strength of Schedule**: Affects win probability by ±4-6%
- **Streak Dynamics**: Binomial distribution, high streaks are statistical noise

## 💰 Cost Analysis

### Scenario Comparison

| Setup | Monthly | Setup Time | Pros | Cons |
|-------|---------|-----------|------|------|
| **API Only** | $300+ | 5 min | Simple | Expensive |
| **Local + Fallback** ⭐ | $30-50 | 2 hrs | Cost-effective | Setup required |
| **Batch Only** | $5-10 | 1 hr | Cheapest | 24hr latency |

**Recommended**: Local LLM (Mistral 7B) + API Fallback = **80-90% savings**

## 📋 Configuration

Key settings in `config/agent_config.py`:

```python
# Models
PRIMARY_MODEL_VERSION = "v3"
RETRAIN_THRESHOLD_ACCURACY = 0.546

# Data
TRAINING_YEARS = (2015, 2024)
VALIDATION_YEAR = 2025
DATA_FRESHNESS_THRESHOLD_DAYS = 1

# Confidence
CONFIDENCE_HIGH_PERCENTILE = 0.90
CONFIDENCE_HIGH_ACCURACY = 0.565
```

## 🔧 Connecting to Your Existing System

The agent integrates with your existing models and scripts:

```
Your System                  Agent System
─────────────               ──────────────
models/*.pkl           →     call train_model()
src/build_v3_*         →     call build features
src/train_v3_*         →     call train
src/analyze_*          →     call analyze
src/predict_*          →     call predict
data/training/*.pq     →     load features
BigQuery               →     query_bigquery()
```

### Required BigQuery Tables

The agent expects these tables in `mlb_historical_data`:

- `games` - Game data (game_id, date, teams, scores)
- `training_features_v3` - ML features (57 features + outcome)
- `model_performance` - Accuracy tracking over time
- `player_stats` - Player performance data
- `team_stats` - Team statistics
- `agent_decisions` - Agent decision audit trail

## 📈 Implementation Roadmap

### Phase 1: Current ✅
- Agent framework
- Tool definitions
- CLI interface
- Configuration system

### Phase 2: Connect Tools (1-2 weeks)
- [ ] Implement BigQuery connector
- [ ] Integrate training scripts
- [ ] Add confirmation workflows

### Phase 3: Knowledge Base (2-3 weeks)
- [ ] Vector embeddings
- [ ] Live data APIs
- [ ] Research integration

### Phase 4: Self-Improvement (3-4 weeks)
- [ ] Feature suggestions
- [ ] Hyperparameter optimization
- [ ] Learning loops

### Phase 5: Production (2-3 weeks)
- [ ] Docker containerization
- [ ] Scheduled runs
- [ ] Monitoring dashboard
- [ ] Alerting system

## 📝 Example Interactions

### Example 1: Retrain Decision
```
agent> query "Should I retrain?"

Agent analyzes:
  1. Recent accuracy: 54.3% (threshold: 54.6%)
  2. Data freshness: OK (1 day old)
  3. Domain factors: Temp spike affects park factors
  
Recommendation: RETRAIN
  - Confidence: 85%
  - Reason: Address park factor shifts
  - Expected gain: +0.2-0.4%

Requires confirmation: [yes/no]
```

### Example 2: Feature Analysis
```
agent> query "What new features should we add?"

Agent analyzes:
  1. Current feature importance
  2. Recent prediction errors
  3. Domain knowledge
  4. Baseball research papers
  
Suggestions:
  1. Add weather interaction terms
  2. Improve pitcher fatigue encoding
  3. Track streaks more granularly
  4. Add park-weather interactions
```

### Example 3: Data Quality
```
agent> health

Agent checks:
  1. BigQuery data freshness: ✓ Same day
  2. Training data quality: ✓ 99.8% complete
  3. Outlier detection: ⚠ 12 games flagged
  4. Schema validation: ✓ All fields present
  
Status: HEALTHY (minor anomaly detected)
```

## 🔐 Security & Privacy

- Credentials stored in `.env.agent` (add to `.gitignore`)
- BigQuery uses service account authentication
- Optional: All processing stays local with Ollama
- Decisions logged in BigQuery for compliance

## 📊 Monitoring & Logs

- **Decision Log**: `agent/logs/decisions.jsonl`
- **Agent Logs**: `agent/logs/agent_decisions.log`
- **BigQuery**: `mlb_historical_data.agent_decisions` table
- **CLI Status**: `python agent/main.py --command status`

## 🐛 Troubleshooting

### Agent not responding
```bash
# Check logs
tail -f agent/logs/agent_decisions.log

# Verify API key
echo $ANTHROPIC_API_KEY

# Test BigQuery connection
python -c "from tools.bigquery_connector import BigQueryConnector; BigQueryConnector()"
```

### Tool execution failed
```bash
# Check specific tool implementation
grep -n "def _tool_" core/agent_manager.py

# Verify tool function in agent_manager.py has implementation (not placeholder)
```

### High costs
```bash
# Enable local LLM
USE_LOCAL_LLM=true

# Start Ollama
ollama serve

# Agent will automatically use local model
```

## 📞 Support & Resources

- **Architecture**: `agent/ARCHITECTURE_GUIDE.py` (comprehensive design)
- **Quick Start**: `agent/QUICK_START.md` (30-minute setup)
- **Examples**: `agent/examples/` (query templates)
- **Config**: `agent/config/agent_config.py` (all settings)
- **CLI Help**: `python agent/main.py --help`

## 🤝 Contributing

To add new functionality:

1. Define tool in `core/agent_manager.py` (add to tools list)
2. Implement tool in `tools/` subdirectory
3. Update `config/agent_config.py` if needed
4. Test with: `python agent/main.py --command test`
5. Log results and iterate

## 📄 License

Same as main hanks_tank_ml project

## 🎉 Next Steps

1. **Read**: `QUICK_START.md` (5 min)
2. **Setup**: `python setup.py` (5 min)
3. **Configure**: Edit `.env.agent` (2 min)
4. **Test**: `python main.py --command status` (1 min)
5. **Implement**: Tools in `tools/` (ongoing)
6. **Deploy**: As scheduled service (Week 2+)

---

**You're building the future of autonomous ML ops! 🚀**

Questions? Check the logs or read ARCHITECTURE_GUIDE.py for details.
