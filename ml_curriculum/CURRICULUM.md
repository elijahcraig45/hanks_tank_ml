# MLB Prediction ML Curriculum

**Goal:** Master data engineering and data science to independently build and deploy MLB prediction models

**Prerequisites:** Python, SQL  
**Timeline:** 12-16 weeks (self-paced)  
**Project:** Build production-ready game outcome and player performance prediction system

---

## Course Structure

### Module 1: Data Engineering Foundations (Weeks 1-4)

#### **Lesson 1: Introduction to Data Pipelines & ETL**
- What is a data pipeline?
- ETL vs ELT processes
- Batch vs streaming data
- Data quality and validation
- **Hands-on:** Analyze your MLB data pipeline, identify improvements
- **Exercise:** Build a simple data quality checker for games_historical

#### **Lesson 2: BigQuery Deep Dive**
- BigQuery architecture and pricing
- Query optimization techniques
- Partitioning and clustering
- Window functions and CTEs
- **Hands-on:** Optimize slow queries in your dataset
- **Exercise:** Create materialized views for feature engineering

#### **Lesson 3: Data Modeling & Schema Design**
- Star schema vs snowflake schema
- Normalization vs denormalization
- Fact and dimension tables
- Slowly changing dimensions (SCD)
- **Hands-on:** Redesign your schema for optimal ML queries
- **Exercise:** Create a denormalized feature table

#### **Lesson 4: Workflow Orchestration**
- Airflow/Dagster fundamentals
- DAG design patterns
- Error handling and retries
- Monitoring and alerting
- **Hands-on:** Convert your backfill scripts to Airflow DAGs
- **Exercise:** Build a daily feature refresh pipeline

---

### Module 2: Statistics & Probability (Weeks 5-6)

#### **Lesson 5: Probability Fundamentals**
- Probability distributions (normal, binomial, Poisson)
- Expected value and variance
- Bayes' theorem
- Sampling and confidence intervals
- **Hands-on:** Analyze win probability distributions
- **Exercise:** Calculate confidence intervals for team win rates

#### **Lesson 6: Statistical Inference**
- Hypothesis testing (t-test, chi-square)
- P-values and statistical significance
- Type I and Type II errors
- A/B testing framework
- **Hands-on:** Test if home field advantage is statistically significant
- **Exercise:** Compare team performance across eras

#### **Lesson 7: Correlation & Causation**
- Correlation coefficients (Pearson, Spearman)
- Confounding variables
- Simpson's paradox
- Causal inference basics
- **Hands-on:** Find correlations between team stats and wins
- **Exercise:** Identify spurious correlations in baseball data

---

### Module 3: Machine Learning Fundamentals (Weeks 7-10)

#### **Lesson 8: Supervised Learning Basics**
- Classification vs regression
- Training, validation, test splits
- Overfitting and underfitting
- Bias-variance tradeoff
- **Hands-on:** Build first game prediction model (logistic regression)
- **Exercise:** Tune model to avoid overfitting

#### **Lesson 9: Feature Engineering**
- Feature scaling and normalization
- Encoding categorical variables
- Handling missing data
- Creating interaction features
- Feature selection techniques
- **Hands-on:** Engineer top 20 features for game prediction
- **Exercise:** Compare feature importance across models

#### **Lesson 10: Model Evaluation Metrics**
- Accuracy, precision, recall, F1
- ROC curves and AUC
- Confusion matrices
- Log loss and Brier score
- Calibration curves
- **Hands-on:** Evaluate your model comprehensively
- **Exercise:** Build a model comparison dashboard

#### **Lesson 11: Tree-Based Models**
- Decision trees fundamentals
- Random forests
- Gradient boosting (XGBoost, LightGBM)
- Feature importance interpretation
- **Hands-on:** Build XGBoost game predictor
- **Exercise:** Hyperparameter tuning with cross-validation

#### **Lesson 12: Ensemble Methods**
- Bagging vs boosting
- Stacking and blending
- Weighted averaging
- Diversity in ensembles
- **Hands-on:** Create stacked ensemble
- **Exercise:** Optimize ensemble weights

---

### Module 4: Deep Learning (Weeks 11-13)

#### **Lesson 13: Neural Networks Fundamentals**
- Perceptrons and activation functions
- Backpropagation
- Loss functions and optimizers
- Regularization (dropout, L1/L2)
- **Hands-on:** Build feed-forward network in TensorFlow
- **Exercise:** Predict game outcomes with neural network

#### **Lesson 14: Recurrent Neural Networks (RNN/LSTM)**
- Sequence modeling concepts
- Vanishing gradient problem
- LSTM and GRU architecture
- Many-to-one predictions
- **Hands-on:** Build LSTM for game sequence prediction
- **Exercise:** Capture team momentum with LSTM

#### **Lesson 15: Attention & Transformers**
- Attention mechanism intuition
- Self-attention and multi-head attention
- Transformer architecture
- Position encoding
- **Hands-on:** Build transformer for game prediction
- **Exercise:** Visualize attention weights (which games matter?)

#### **Lesson 16: Graph Neural Networks**
- Graph representation learning
- Message passing
- Graph Attention Networks (GAT)
- Node embeddings
- **Hands-on:** Model teams as graph structure
- **Exercise:** Learn division rivalry patterns

---

### Module 5: MLOps & Production (Weeks 14-16)

#### **Lesson 17: Model Training at Scale**
- Distributed training
- Hyperparameter optimization (Optuna, Ray Tune)
- Experiment tracking (MLflow, W&B)
- Model versioning
- **Hands-on:** Set up experiment tracking for your models
- **Exercise:** Automated hyperparameter search

#### **Lesson 18: Model Deployment**
- Model serialization (pickle, ONNX)
- REST API design (FastAPI)
- Containerization (Docker)
- Cloud deployment (GCP Cloud Run)
- **Hands-on:** Deploy prediction API
- **Exercise:** Load test your API

#### **Lesson 19: Monitoring & Maintenance**
- Data drift detection
- Model performance monitoring
- A/B testing
- Online learning
- **Hands-on:** Build monitoring dashboard
- **Exercise:** Implement auto-retraining pipeline

#### **Lesson 20: Production Best Practices**
- CI/CD for ML models
- Feature stores
- Model governance
- Cost optimization
- **Hands-on:** Full production pipeline
- **Exercise:** Production readiness checklist

---

## Supplementary Topics

### **Special Topic A: Time Series Analysis**
- Autocorrelation and seasonality
- ARIMA models
- Prophet for forecasting
- Application to player performance trends

### **Special Topic B: Bayesian Methods**
- Prior and posterior distributions
- Bayesian updating
- Markov Chain Monte Carlo (MCMC)
- Application to win probability

### **Special Topic C: Reinforcement Learning**
- MDP (Markov Decision Process)
- Q-learning
- Policy gradients
- Application to game strategy optimization

---

## Learning Resources

### Books
- **Data Engineering:**
  - "Designing Data-Intensive Applications" by Martin Kleppmann
  - "The Data Warehouse Toolkit" by Ralph Kimball
  
- **Statistics:**
  - "Practical Statistics for Data Scientists" by Bruce & Bruce
  - "The Signal and the Noise" by Nate Silver (baseball focus!)
  
- **Machine Learning:**
  - "Hands-On Machine Learning" by Aurélien Géron
  - "Deep Learning" by Goodfellow, Bengio, Courville
  - "The Hundred-Page Machine Learning Book" by Andriy Burkov

### Online Courses (Supplementary)
- Fast.ai Practical Deep Learning
- Andrew Ng's Machine Learning Specialization
- Google Cloud BigQuery Training

### Baseball Analytics Specific
- **Websites:**
  - FanGraphs (sabermetrics education)
  - Baseball Prospectus (PECOTA methodology)
  - MLB Statcast Search
  
- **Books:**
  - "The Book: Playing the Percentages in Baseball"
  - "Moneyball" by Michael Lewis
  - "The MVP Machine" by Ben Lindbergh

---

## Assessment & Milestones

### Milestone 1 (Week 4): Data Pipeline
✅ Build automated daily data refresh  
✅ Create data quality monitoring  
✅ Design optimized feature table schema

### Milestone 2 (Week 8): Baseline Model
✅ Train logistic regression baseline (>54% accuracy)  
✅ Implement proper cross-validation  
✅ Create evaluation framework

### Milestone 3 (Week 12): Advanced Model
✅ Build ensemble model (>60% accuracy)  
✅ Implement LSTM for sequences  
✅ Feature importance analysis

### Milestone 4 (Week 16): Production System
✅ Deploy prediction API  
✅ Automated retraining pipeline  
✅ Monitoring dashboard  
✅ Make 2026 season predictions!

---

## Daily Study Plan

### Weekday Schedule (2-3 hours)
- **30 min:** Read lesson material
- **60 min:** Watch supplementary videos/tutorials
- **60 min:** Hands-on exercises with your data
- **30 min:** Document learnings and questions

### Weekend Schedule (4-6 hours)
- **2 hours:** Deep dive on challenging concepts
- **2 hours:** Build project components
- **1 hour:** Review week's progress
- **1 hour:** Plan next week

---

## Success Metrics

**By End of Curriculum:**
- [ ] Game prediction model: >60% accuracy, >0.65 AUC
- [ ] Player prediction model: R² >0.40 for batting, >0.45 for pitching
- [ ] Deployed production API serving predictions
- [ ] Automated daily pipeline refreshing features
- [ ] Monitoring system tracking model health
- [ ] Portfolio project demonstrating skills

**Stretch Goals:**
- [ ] Beat FiveThirtyEight's prediction accuracy
- [ ] Publish blog post on methodology
- [ ] Open-source prediction model
- [ ] Contribute to baseball analytics community

---

## Getting Help

- **Documentation:** Each lesson has references
- **Community:** Reddit r/MLQuestions, r/Sabermetrics
- **Stack Overflow:** For specific code issues
- **GitHub Issues:** Track your questions and learnings

---

**Let's get started! Proceed to Lesson 1 →**
