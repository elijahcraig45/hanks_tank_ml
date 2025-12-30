# Hank's Tank MLB Data Platform

Automated MLB data pipeline for historical collection, real-time tracking, and ML model training. Designed to be future-proof and cost-effective on Google Cloud Platform.

## 📚 Key Documentation

- **[Production Architecture](docs/PRODUCTION_ARCHITECTURE.md)**: High-level system design, cloud infrastructure, and cost estimates.
- **[Data Collection Design](docs/DATA_COLLECTION_SYSTEM_DESIGN.md)**: Detailed API specifications for the data collector.
- **[ML Curriculum](ml_curriculum/CURRICULUM.md)**: Learning path for building the ML models.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Prototypes

We use Jupyter notebooks for prototyping and testing data collection:

```bash
# Launch the test notebook
jupyter notebook notebooks/data_collection_test.ipynb
```

### 3. Automation Scripts

Scripts for daily operations are located in `scripts/`:
- `setup_2026_automation.sh`: Initial setup
- `run_daily_2026.sh`: Daily data collection job

## 📊 System Overview

```
MLB Stats API / Statcast
          ↓
    [Data Collector]
          ↓
    [Validation Layer]
          ↓
      [BigQuery]
          ↓
   [Feature Engine]
          ↓
      [ML Models]
```

## 📁 Project Structure

- `src/`: Core Python source code for pipelines and collectors.
- `docs/`: Detailed system documentation and design specs.
- `notebooks/`: Prototyping and testing environments.
- `research/`: Baseball analytics research and feature engineering plans.
- `scripts/`: Shell scripts for automation and deployment.
