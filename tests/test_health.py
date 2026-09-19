from fastapi.testclient import TestClient

from app import dataset as dataset_module
from app.main import app, scheduler

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startup_loads_dataset():
    """Using TestClient as a context manager runs the lifespan handler."""
    with TestClient(app) as started_client:
        assert started_client.get("/health").status_code == 200
        # The dataset is in memory once startup has completed.
        assert len(dataset_module.get_dataset()) == 750
        assert scheduler.is_running
    assert not scheduler.is_running
