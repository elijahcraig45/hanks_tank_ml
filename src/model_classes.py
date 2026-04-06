"""
Shared model class definitions.
Must be importable by both train_v*_models.py and predict_today_games.py
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


class StackedV6Model:
    """
    V6 stacked ensemble — same architecture as V5 but with additional
    pitcher arsenal and venue-split features. Kept as a separate class
    so pickled models deserialize correctly when V5 and V6 coexist.
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


class StackedV7Model:
    """
    V7 stacked ensemble — extends V6 with bullpen health, moon phase,
    circadian offset, and pitcher venue splits. Separate class required
    for correct pickle round-trip alongside V5/V6 model artifacts.
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
