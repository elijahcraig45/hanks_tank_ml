import pytest


@pytest.fixture(autouse=True)
def _no_live_model_control(monkeypatch):
    """No test may read the real control dataset. get_state() returns the empty (fail-open)
    state without touching Google auth when this is set; tests that exercise the read inject
    a client, and tests that need a state patch model_control.get_state."""
    monkeypatch.setenv("MODEL_CONTROL_DISABLED", "1")
    monkeypatch.delenv("CONTROL_DATASET", raising=False)
