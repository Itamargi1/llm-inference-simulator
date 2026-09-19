"""FastAPI application entry point for the LLM inference simulator.

The dataset is loaded at startup; POST /generate submits work to a scheduler
that serves one simulated GPU as a continuous batch; GET /metrics reports
what the simulation observed. No real model or GPU is used.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.dataset import get_prompt, init_dataset
from app.schemas import GenerateRequest, GenerateResponse
from app.simulation.completion import build_content, target_completion_tokens
from app.simulation.request import RequestState, SimulatedRequest
from app.simulation.scheduler import SingleGPUScheduler
from app.tokenizer import estimate_tokens

# One scheduler for the whole application, representing one simulated GPU
# service. Created here and started in the lifespan - never per request.
# Module-level so tests can substitute one driven by a no-op sleeper without
# disabling simulated delay globally.
scheduler = SingleGPUScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the dataset and start the scheduler before serving requests.

    The dataset is read from disk exactly once; a failure raises and prevents
    startup, since serving traffic against a broken dataset would produce
    misleading simulation results.

    The scheduler's worker thread is started here and stopped on shutdown, so
    it does not outlive the application or leak between tests.
    """
    init_dataset()
    scheduler.start()
    try:
        yield
    finally:
        scheduler.stop()


API_DESCRIPTION = """
This page is an interactive interface for the local **LLM inference simulator**.

No real language model and no GPU are used. The service simulates how an
inference server would behave: requests queue, are prefilled, then decode one
token per scheduler step. Requests therefore take a measurable amount of time
on purpose.

This page is **optional**. It calls exactly the same API as the `curl` and
PowerShell examples in the project README.

### How to use this page

1. Expand an endpoint below.
2. Click **Try it out** to enable its inputs.
3. Enter input if the endpoint requires any.
4. Click **Execute** to send the request.
5. Read the result under **Server response**.

### Main endpoints

- `GET /health` - check that the simulator is running.
- `POST /generate` - run one simulated inference request.
- `GET /metrics` - inspect simulator metrics.

### Helpful terms

- **No parameters** means the endpoint needs no input. `/health` and
  `/metrics` both show this, which is expected.
- **Responses** lists the possible HTTP outcomes of an endpoint.
- **Schemas** (at the bottom of the page) are reference definitions of the
  request and response JSON. They are documentation, not live data.
- **HTTPValidationError** describes the shape of a possible `422` invalid
  input response. Seeing it listed does **not** mean an error occurred.
"""

app = FastAPI(
    title="LLM Inference Simulator",
    description=API_DESCRIPTION,
    version="0.8.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    summary="Check that the simulator is running",
    description=(
        "No input is required, so **No parameters** is expected here.\n\n"
        "Click **Try it out**, then **Execute**. A successful result is "
        "`{\"status\": \"ok\"}`.\n\n"
        "This is only a liveness check: it confirms the process is up and "
        "serving. It does not check the dataset, the scheduler, or any "
        "other dependency."
    ),
)
def health() -> dict[str, str]:
    """Liveness probe.

    Kept deliberately trivial: it reports that the process is up and serving.
    It is not a readiness or dependency-health check.
    """
    return {"status": "ok"}


@app.get(
    "/metrics",
    summary="Inspect simulator metrics",
    description=(
        "No input is required, so **No parameters** is expected here.\n\n"
        "Click **Try it out**, then **Execute**.\n\n"
        "The response reports request counts, queue depth, active requests, "
        "batch occupancy, simulated GPU utilization, queue latency, time to "
        "first token, total latency, recent throughput, and token work.\n\n"
        "**These values describe the simulation, not real hardware.** "
        "`simulated_gpu_utilization` comes from the scheduler's own busy and "
        "idle bookkeeping; no GPU is queried. The README explains what each "
        "metric means and why it matters."
    ),
)
def metrics() -> dict:
    """Simulator metrics as JSON.

    **These describe the simulation, not hardware.** In particular
    `simulated_gpu_utilization` comes from the scheduler's own busy/idle
    bookkeeping - no GPU is queried and no NVML/nvidia-smi call is made.

    Current gauges are read from the scheduler at request time, so they
    cannot drift out of step with the real queue.
    """
    return scheduler.metrics.snapshot(scheduler)


@app.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Run one simulated inference request",
    description=(
        "The main simulator endpoint.\n\n"
        "Click **Try it out**, enter a dataset `prompt_id` (for example "
        "`30`), then click **Execute**.\n\n"
        "The request intentionally takes a measurable amount of time. That "
        "delay represents simulated queueing, prefill, and decode work. No "
        "real language model or GPU is called.\n\n"
        "The returned `content` is deterministic placeholder text, and the "
        "token counts use the simulator's own approximation rather than a "
        "real tokenizer.\n\n"
        "Outcomes:\n\n"
        "- `200` - the simulated request completed.\n"
        "- `404` - no prompt with that `prompt_id` exists in the dataset.\n"
        "- `422` - `prompt_id` was missing or not an integer."
    ),
)
def generate(request: GenerateRequest) -> GenerateResponse:
    """Simulate one inference request.

    The route owns only the front of the lifecycle - RECEIVED, TOKENIZED,
    WAITING - and then hands the request to the scheduler, which owns
    PREFILL, DECODING and COMPLETED.

    Concurrent callers contend for one simulated GPU: this thread blocks
    until its queued request has been served. Up to MAX_ACTIVE_REQUESTS
    requests decode together, so callers share the batch rather than
    strictly queueing behind one another.

    The simulation intentionally omits chunked prefill, KV-cache accounting
    and bounded-queue rejection.
    """
    # RECEIVED: look the prompt up in the dataset held in memory since
    # startup, so serving never touches disk.
    record = get_prompt(request.prompt_id)
    if record is None:
        # Not the REJECTED state: no request exists yet, and an unknown
        # prompt_id is a client error rather than an admission decision.
        raise HTTPException(
            status_code=404, detail=f"Unknown prompt_id: {request.prompt_id}"
        )

    prompt_tokens = estimate_tokens(record.prompt)
    target_tokens = target_completion_tokens(prompt_tokens, record.category)

    simulated = SimulatedRequest(
        prompt_id=record.id,
        category=record.category,
        prompt_tokens=prompt_tokens,
        target_completion_tokens=target_tokens,
    )

    # TOKENIZED: prompt and completion sizes are now known.
    simulated.transition_to(RequestState.TOKENIZED)

    # WAITING: join the FIFO queue. A real wait, until a batch slot frees.
    simulated.transition_to(RequestState.WAITING)

    # The scheduler drives PREFILL -> DECODING -> COMPLETED. This call blocks
    # until the job finishes, which is what makes concurrent callers contend
    # for the one simulated GPU.
    job = scheduler.submit(simulated)
    job.wait()

    if not job.succeeded:
        # The worker already marked the request FAILED and woke us, so a
        # caller never hangs on an abandoned job.
        raise HTTPException(
            status_code=500,
            detail=f"Simulated inference failed: {job.error}",
        )

    content = build_content(record.category, record.id, target_tokens)

    # Measured from the text actually returned, so the reported count
    # describes the real payload rather than restating the decode target.
    # The two differ by up to ~2 tokens: the placeholder is trimmed on a
    # word boundary.
    completion_tokens = estimate_tokens(content)

    return GenerateResponse(
        request_id=simulated.request_id,
        prompt_id=record.id,
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
