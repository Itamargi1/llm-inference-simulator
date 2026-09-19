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


app = FastAPI(
    title="LLM Inference Simulator",
    description="A local simulation of a self-hosted vLLM-style inference service.",
    version="0.8.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe.

    Kept deliberately trivial: it reports that the process is up and serving.
    It is not a readiness or dependency-health check.
    """
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    """Simulator metrics as JSON.

    **These describe the simulation, not hardware.** In particular
    `simulated_gpu_utilization` comes from the scheduler's own busy/idle
    bookkeeping - no GPU is queried and no NVML/nvidia-smi call is made.

    Current gauges are read from the scheduler at request time, so they
    cannot drift out of step with the real queue.
    """
    return scheduler.metrics.snapshot(scheduler)


@app.post("/generate", response_model=GenerateResponse)
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
