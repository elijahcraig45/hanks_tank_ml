"""Sport-agnostic model constructors and evaluation.

Split out of train_nfl_models.py so CFB can import these without dragging in NFL's
data loader — `data` and `config` are ambiguous module names once two football
packages are on sys.path, and this module imports neither.

Ported from the MLB harness in train_v8_models.py, retuned for football's much smaller
sample: NFL has ~7k rows against MLB's ~27k, so the trees are shallower and the
regularization heavier.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_lr(C: float = 0.1):
    """Scaled L2 logistic regression. At N~7k with ~80 features this is genuinely
    competitive with the tree models, not a formality."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, solver="lbfgs"),
    )


def build_xgb():
    """XGBoost tuned for ~8x less data than the MLB model: depth 3 not 4,
    min_child_weight 20 not 5, and real gamma/lambda so it cannot memorize."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        max_depth=3,
        min_child_weight=20,
        learning_rate=0.03,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.6,
        gamma=0.5,
        reg_lambda=5.0,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )


def evaluate(y_true, proba) -> dict:
    pred = (proba > 0.5).astype(int)
    return {
        "acc": float(np.mean(pred == y_true)),
        "auc": float(roc_auc_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_true, proba)),
    }
