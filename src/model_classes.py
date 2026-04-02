"""
Shared model class definitions.
Must be importable by both train_v5_models.py and predict_today_games.py
so that pickle serialization/deserialization works correctly.
"""
import numpy as np


class StackedV5Model:
    """
    Thin wrapper around the V5 stacked ensemble so predict_proba works
    like a single sklearn model. Must live at module level to be picklable.
    """
    def __init__(self, base_models: dict, meta):
        self.base_models = base_models
        self.meta = meta

    def predict_proba(self, X):
        base_probas = np.column_stack([
            m.predict_proba(X)[:, 1]
            for m in self.base_models.values()
        ])
        return self.meta.predict_proba(base_probas)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
