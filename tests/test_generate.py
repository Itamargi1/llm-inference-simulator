"""Tests for POST /generate.

TestClient is used as a context manager throughout so the lifespan handler
runs and the dataset is loaded, exactly as it is under a real server.
"""

import contextlib
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.dataset import get_dataset
from app.main import app
from app.simulation.completion import target_completion_tokens
from app.tokenizer import estimate_tokens


@pytest.fixture(scope="module", autouse=True)
def instant_scheduler():
    """Swap the app's scheduler for one whose GPU never actually waits.

    The scheduler, its FIFO queue and its worker thread are all real - only
    the simulated GPU delay is removed, so endpoint tests do not spend
    minutes asleep. Simulated delay itself is covered in test_gpu.py with a
    recording sleeper, in test_scheduler.py with gated sleepers, and end to
    end by the real HTTP checks.
    """
    import app.main as main_module
    from app.simulation.gpu import DEFAULT_GPU_PROFILE, SimulatedGPU
    from app.simulation.scheduler import SingleGPUScheduler

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            main_module,
            "scheduler",
            SingleGPUScheduler(
                gpu=SimulatedGPU(DEFAULT_GPU_PROFILE), sleeper=lambda seconds: None
            ),
        )
        yield


@contextlib.contextmanager
def running_server(asgi_app):
    """Serve `asgi_app` on a free port in a background thread.

    Needed for genuine client concurrency: TestClient funnels every call
    through one blocking portal, so it cannot show requests contending.
    """
    import socket

    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    config = uvicorn.Config(
        asgi_app, host="127.0.0.1", port=port, log_level="error", lifespan="on"
    )
    server = uvicorn.Server(config)
    # Signal handlers can only be installed on the main thread.
    server.install_signal_handlers = lambda: None

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(15)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as started_client:
        yield started_client


def test_valid_prompt_id_returns_200(client: TestClient):
    response = client.post("/generate", json={"prompt_id": 1})
    assert response.status_code == 200


def test_response_schema(client: TestClient):
    body = client.post("/generate", json={"prompt_id": 1}).json()
    assert set(body) == {
        "request_id",
        "prompt_id",
        "content",
        "prompt_tokens",
        "completion_tokens",
    }
    assert isinstance(body["content"], str) and body["content"]
    assert isinstance(body["prompt_tokens"], int)
    assert isinstance(body["completion_tokens"], int)
    assert isinstance(body["request_id"], str) and body["request_id"]
    assert body["prompt_id"] == 1


def test_internal_state_is_not_exposed(client: TestClient):
    """Lifecycle internals stay inside the simulation domain."""
    body = client.post("/generate", json={"prompt_id": 1}).json()
    for hidden in ("state", "queue_time", "latency", "gpu_id", "generated_tokens"):
        assert hidden not in body


def test_request_id_is_unique_per_call(client: TestClient):
    """Same prompt, two invocations, two different request IDs."""
    first = client.post("/generate", json={"prompt_id": 200}).json()
    second = client.post("/generate", json={"prompt_id": 200}).json()

    assert first["prompt_id"] == second["prompt_id"] == 200
    assert first["request_id"] != second["request_id"]
    # Everything else about the two responses is identical and deterministic.
    assert first["content"] == second["content"]
    assert first["prompt_tokens"] == second["prompt_tokens"]
    assert first["completion_tokens"] == second["completion_tokens"]


def test_request_ids_are_unique_across_many_calls(client: TestClient):
    ids = {
        client.post("/generate", json={"prompt_id": pid}).json()["request_id"]
        for pid in range(1, 31)
    }
    assert len(ids) == 30


def test_unknown_prompt_id_returns_404(client: TestClient):
    response = client.post("/generate", json={"prompt_id": 999999})
    assert response.status_code == 404
    assert "999999" in response.json()["detail"]


def test_malformed_request_returns_422(client: TestClient):
    assert client.post("/generate", json={}).status_code == 422
    assert client.post("/generate", json={"prompt_id": "abc"}).status_code == 422


def test_prompt_tokens_match_dataset_prompt(client: TestClient):
    """Checked against the real record, for a spread of prompt sizes."""
    dataset = get_dataset()
    records = sorted(dataset.records, key=lambda r: len(r.prompt))
    sample = [records[0], records[len(records) // 2], records[-1]]
    for record in sample:
        body = client.post("/generate", json={"prompt_id": record.id}).json()
        assert body["prompt_tokens"] == estimate_tokens(record.prompt)


def test_completion_tokens_match_returned_content(client: TestClient):
    """The reported count must describe the payload actually returned."""
    dataset = get_dataset()
    for record in dataset.records[:25]:
        body = client.post("/generate", json={"prompt_id": record.id}).json()
        assert body["completion_tokens"] == estimate_tokens(body["content"])


def test_completion_tokens_are_close_to_target(client: TestClient):
    dataset = get_dataset()
    for record in dataset.records[:25]:
        body = client.post("/generate", json={"prompt_id": record.id}).json()
        target = target_completion_tokens(body["prompt_tokens"], record.category)
        assert abs(body["completion_tokens"] - target) <= 2


def test_repeated_requests_are_identical_apart_from_request_id(client: TestClient):
    first = client.post("/generate", json={"prompt_id": 200}).json()
    second = client.post("/generate", json={"prompt_id": 200}).json()
    assert first.pop("request_id") != second.pop("request_id")
    assert first == second


def test_successful_request_reaches_completed(monkeypatch, client: TestClient):
    """Every served request must end its lifecycle in COMPLETED.

    The state is not exposed in the API, so it is observed by capturing the
    SimulatedRequest objects the endpoint creates. The shared client is used
    deliberately: a second TestClient would run the lifespan again and stop
    the scheduler this module's other tests depend on.
    """
    import app.main as main_module
    from app.simulation.request import RequestState, SimulatedRequest

    created = []

    def capturing(*args, **kwargs):
        request = SimulatedRequest(*args, **kwargs)
        created.append(request)
        return request

    monkeypatch.setattr(main_module, "SimulatedRequest", capturing)

    for prompt_id in (1, 200, 750):
        assert client.post("/generate", json={"prompt_id": prompt_id}).status_code == 200

    assert len(created) == 3
    for request in created:
        assert request.state == RequestState.COMPLETED
        assert request.is_terminal
        assert request.is_decode_complete
        assert request.generated_tokens == request.target_completion_tokens


def test_failed_lookup_creates_no_simulated_request(monkeypatch, client: TestClient):
    """A 404 is a client error, not a simulated REJECTED request."""
    import app.main as main_module
    from app.simulation.request import SimulatedRequest

    created = []

    def capturing(*args, **kwargs):
        request = SimulatedRequest(*args, **kwargs)
        created.append(request)
        return request

    monkeypatch.setattr(main_module, "SimulatedRequest", capturing)

    assert client.post("/generate", json={"prompt_id": 999999}).status_code == 404
    assert created == []


def test_completion_tokens_respect_category_bounds(client: TestClient):
    """No response may fall below base_tokens or above the cap."""
    from app.config import COMPLETION_RULES

    dataset = get_dataset()
    records = sorted(dataset.records, key=lambda r: len(r.prompt))
    for record in [records[0], records[len(records) // 2], records[-1]]:
        body = client.post("/generate", json={"prompt_id": record.id}).json()
        rule = COMPLETION_RULES[record.category]
        assert (
            rule["base_tokens"] - 2
            <= body["completion_tokens"]
            <= rule["max_tokens"] + 2
        )


def test_different_categories_produce_different_wording(client: TestClient):
    dataset = get_dataset()
    seen = {}
    for record in dataset.records:
        if record.category not in seen:
            body = client.post("/generate", json={"prompt_id": record.id}).json()
            seen[record.category] = body["content"].split(".")[0]
        if len(seen) == 3:
            break
    assert len(set(seen.values())) == 3


def test_health_still_works(client: TestClient):
    assert client.get("/health").json() == {"status": "ok"}


def test_generate_waits_for_the_simulated_service_time():
    """The endpoint really drives the batching loop - verified without waiting.

    A recording sleeper captures every wait the request caused: one prefill
    wait, then one wait per decode step. Their sum must equal the isolated
    service time the GPU predicts for that prompt.
    """
    import app.main as main_module
    from app.dataset import load_dataset
    from app.simulation.completion import target_completion_tokens
    from app.simulation.gpu import DEFAULT_GPU_PROFILE, SimulatedGPU
    from app.simulation.request import SimulatedRequest
    from app.simulation.scheduler import SingleGPUScheduler
    from app.tokenizer import estimate_tokens

    recorded: list[float] = []
    gpu = SimulatedGPU(DEFAULT_GPU_PROFILE)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            main_module,
            "scheduler",
            SingleGPUScheduler(gpu=gpu, sleeper=recorded.append),
        )
        with TestClient(main_module.app) as local_client:
            response = local_client.post("/generate", json={"prompt_id": 8})
    assert response.status_code == 200

    record = load_dataset().get(8)
    prompt_tokens = estimate_tokens(record.prompt)
    target = target_completion_tokens(prompt_tokens, record.category)
    expected = gpu.estimate_timing(
        SimulatedRequest(
            prompt_id=8,
            category=record.category,
            prompt_tokens=prompt_tokens,
            target_completion_tokens=target,
        )
    )

    # One prefill wait, then one wait per decode step.
    assert len(recorded) == 1 + target
    assert recorded[0] == pytest.approx(gpu.prefill_setup_seconds(
        SimulatedRequest(8, record.category, prompt_tokens, target)
    ))
    assert sum(recorded) == pytest.approx(expected.total_seconds)


def test_concurrent_api_requests_share_a_bounded_batch():
    """Concurrent callers must batch together, up to the configured cap.

    Run against a real uvicorn server in a background thread, because
    TestClient drives the app through a single blocking portal and therefore
    cannot express genuine client concurrency.

    Sampling the scheduler's active count from inside its sleeper shows both
    halves of the claim: requests really do run together (peak > 1), and the
    batch never exceeds MAX_ACTIVE_REQUESTS.
    """
    import json
    import threading
    import urllib.request

    import app.main as main_module
    from app.config import MAX_ACTIVE_REQUESTS
    from app.simulation.gpu import DEFAULT_GPU_PROFILE, SimulatedGPU
    from app.simulation.scheduler import SingleGPUScheduler

    peak = 0
    samples: list[int] = []

    def sampling_sleeper(seconds: float) -> None:
        nonlocal peak
        active = scheduler.active_count
        samples.append(active)
        peak = max(peak, active)
        time.sleep(0.002)  # brief, so requests genuinely overlap

    scheduler = SingleGPUScheduler(
        gpu=SimulatedGPU(DEFAULT_GPU_PROFILE), sleeper=sampling_sleeper
    )

    results: list[dict] = []
    results_lock = threading.Lock()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(main_module, "scheduler", scheduler)
        with running_server(main_module.app) as base_url:
            barrier = threading.Barrier(6)

            def call(prompt_id: int) -> None:
                barrier.wait()  # fire together
                request = urllib.request.Request(
                    f"{base_url}/generate",
                    data=json.dumps({"prompt_id": prompt_id}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = json.load(response)
                with results_lock:
                    results.append(body)

            threads = [
                threading.Thread(target=call, args=(pid,)) for pid in range(1, 7)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(60)

    assert len(results) == 6, "not every concurrent request returned"
    assert peak > 1, "requests did not batch together"
    assert peak <= MAX_ACTIVE_REQUESTS, f"batch grew to {peak}"
    assert max(samples) <= MAX_ACTIVE_REQUESTS

    request_ids = {body["request_id"] for body in results}
    assert len(request_ids) == 6

    # Deterministic per-prompt output is unchanged by concurrency.
    for body in results:
        assert body["completion_tokens"] == estimate_tokens(body["content"])


def test_larger_prompts_request_longer_waits():
    """Service time must track workload size, end to end through the API."""
    import app.main as main_module
    from app.dataset import load_dataset
    from app.simulation.gpu import DEFAULT_GPU_PROFILE, SimulatedGPU
    from app.simulation.scheduler import SingleGPUScheduler

    dataset = load_dataset()
    records = sorted(dataset.records, key=lambda r: len(r.prompt))
    smallest, largest = records[0].id, records[-1].id

    waits = {}
    for prompt_id in (smallest, largest):
        recorded: list[float] = []
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                main_module,
                "scheduler",
                SingleGPUScheduler(
                    gpu=SimulatedGPU(DEFAULT_GPU_PROFILE), sleeper=recorded.append
                ),
            )
            with TestClient(main_module.app) as local_client:
                local_client.post("/generate", json={"prompt_id": prompt_id})
        waits[prompt_id] = sum(recorded)

    assert waits[largest] > waits[smallest]
