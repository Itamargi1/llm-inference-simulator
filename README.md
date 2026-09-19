# LLM Inference Simulator

A deterministic local simulator for the request flow of an LLM inference
service. It models dataset lookup, approximate token counts, FIFO queueing,
prefill, continuous batched decode, latency, throughput, and utilization.

No model is loaded and no GPU is accessed. All GPU work, token generation, and
timing are simulated. The queue, worker thread, synchronization, HTTP server,
and wall-clock waits are real application behavior.

## Architecture

```text
POST /generate
      |
      v
dataset lookup -> approximate prompt tokens -> completion target
      |
      v
SimulatedRequest -> FIFO waiting queue -> one simulated GPU worker
                                             |
                              sequential prefill on admission
                                             |
                              continuous batched decode (up to 4)
                                             |
                                             v
                                      HTTP response

scheduler events -> MetricsCollector -> GET /metrics
```

The application loads the dataset once during FastAPI startup and starts one
background scheduler worker. A generation request transitions through:

```text
RECEIVED -> TOKENIZED -> WAITING -> PREFILL -> DECODING -> COMPLETED
```

`FAILED` is used when simulated service raises unexpectedly. `REJECTED` exists
in the state model and metrics schema but is not used because the FIFO queue is
unbounded.

## Dataset

[`data/prompts.json`](data/prompts.json) contains 750 deterministic synthetic
records with this shape:

```json
{"id": 1, "category": "summarization", "prompt": "..."}
```

The IDs and prompts are unique. The three categories contain 250 records each:

- `summarization`
- `document_understanding`
- `customer_support`

Prompt lengths range from 72 to 9,955 characters, including 63 records above
4,000 characters. Document-understanding prompts are generated so every asked
question has its supporting clause in the same prompt.

[`scripts/generate_dataset.py`](scripts/generate_dataset.py) uses a fixed seed,
stable ordering, sorted JSON keys, UTF-8, and LF line endings. Running it
reproduces the checked-in dataset byte for byte:

```bash
python scripts/generate_dataset.py
```

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the pinned runtime and test dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

## Run the server

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at
<http://127.0.0.1:8000/docs>.

## Public API

### `GET /health`

A process liveness endpoint. It does not claim dependency readiness.

```json
{"status": "ok"}
```

### `POST /generate`

Accepts one dataset ID:

```json
{"prompt_id": 1}
```

The request blocks while its simulated prefill and decode work is served. A
successful response has this unchanged public shape:

```json
{
  "request_id": "c07da4b9-6c54-4d82-9603-c0f611784b06",
  "prompt_id": 1,
  "content": "Simulated summary for prompt 1. (1) The source material is condensed into its main points. (2) Key decisions and their owners are listed",
  "prompt_tokens": 117,
  "completion_tokens": 34
}
```

`prompt_id` identifies dataset content. `request_id` identifies one invocation,
so repeated calls for the same prompt receive different request IDs but the
same deterministic content. `completion_tokens` is measured from the returned
text; the internal completion target represents simulated decode work and can
differ slightly because content is trimmed at a word boundary.

An unknown `prompt_id` returns HTTP 404. A missing or invalid request body
returns FastAPI's HTTP 422 validation response.

Example:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt_id":1}'
```

### `GET /metrics`

Returns in-memory JSON metrics for the current process:

```json
{
  "uptime_seconds": 12.4,
  "requests": {
    "submitted": 8,
    "completed": 8,
    "failed": 0,
    "rejected": 0
  },
  "current": {
    "queue_depth": 0,
    "active_requests": 0,
    "max_active_requests": 4,
    "batch_occupancy": 0.0,
    "gpu_busy": false,
    "simulated_gpu_utilization": 0.42
  },
  "latency_seconds": {
    "queue": {"avg": 0.15, "p50": 0.02, "p95": 0.48, "count": 8},
    "ttft": {"avg": 0.31, "p50": 0.21, "p95": 0.71, "count": 8},
    "total": {"avg": 0.82, "p50": 0.74, "p95": 1.39, "count": 8}
  },
  "throughput": {
    "requests_per_second_recent": 0.65,
    "window_seconds": 12.4
  },
  "tokens": {
    "prompt_tokens_processed": 8058,
    "completion_tokens_generated": 749
  }
}
```

The numbers above illustrate the schema; runtime values depend on traffic.

Metrics are defined as follows:

| Metric | Definition |
|---|---|
| Queue latency | Admission time minus enqueue time |
| TTFT | First decoded token time minus enqueue time; includes queueing, prefill, and the first decode step's compute delay |
| Total latency | Completion time minus enqueue time |
| Batch occupancy | Active requests divided by the configured maximum, capped at 1 |
| `gpu_busy` | Whether any request is actively in prefill or decode; queued-only work does not count |
| Simulated GPU utilization | Cumulative scheduler busy time divided by process uptime |
| Recent throughput | Completions in the trailing 60-second window; during startup, elapsed uptime is the denominator |
| Prompt tokens processed | Actual simulated prefill work whose prefill completed |
| Completion tokens generated | Actual simulated decode work produced by decode steps |

Request and token counters are cumulative for the process lifetime. Latency
statistics use nearest-rank percentiles over the most recent 1,000 successful
requests. Failed requests do not enter successful latency samples, but any
prefill or decode work completed before failure remains in the token-work
counters. Metrics failures are isolated from request execution.

## Simulation model

All coefficients are centralized in [`app/config.py`](app/config.py). They are
assumptions chosen to create useful relative behavior, not measurements of any
specific accelerator, model, tokenizer, or serving stack.

| Assumption | Value |
|---|---:|
| Characters per approximate token | 4.0 |
| Prefill throughput | 5,000 tokens/s |
| Decode throughput | 100 tokens/s |
| Fixed per-request overhead | 0.010 s |
| Maximum active requests | 4 |

### Token approximation

```text
prompt_tokens = ceil(characters / 4.0)
```

This is character based; it does not implement vocabulary lookup, byte-pair
encoding, chat templates, or model-specific tokenization.

### Completion target

```text
target = min(max_tokens, round(base_tokens + prompt_tokens * ratio))
```

| Category | Base tokens | Ratio | Maximum tokens |
|---|---:|---:|---:|
| `summarization` | 20 | 0.12 | 300 |
| `document_understanding` | 12 | 0.05 | 180 |
| `customer_support` | 20 | 0.10 | 220 |

The target controls decode work. Returned text is deterministic placeholder
content sized near that target and then measured independently.

### Prefill and decode timing

For one isolated request:

```text
prefill setup = fixed_overhead + prompt_tokens / prefill_throughput
decode        = target_completion_tokens / decode_throughput
service       = prefill setup + decode
```

`POST /generate` waits for these simulated stages; the service duration is not
merely calculated and returned.

### FIFO queueing and continuous batching

Admission is FIFO. The worker prefills admitted requests one at a time, without
overlapping prefill and decode. Each decode step then advances every active
request by one token and incurs one shared delay:

```text
decode_step_seconds = 1 / decode_tokens_per_second
```

A step costs the same with one to four active requests, so aggregate simulated
decode throughput scales linearly with occupancy up to four. This is a
deliberate simplification; real batching efficiency is nonlinear. When a
request completes, its slot is filled on the next scheduler iteration without
waiting for the other active requests to finish.

The decode-step delay occurs before generated tokens become visible. Therefore
TTFT includes the compute delay of the first token-producing step.

## Monitoring and alerting recommendations

The project exposes data but does not run an alerting system. In a production
wrapper, monitor combinations and trends rather than isolated values:

- Growing queue depth together with rising queue latency indicates demand is
  exceeding service capacity.
- Rising TTFT identifies queue or prefill pressure earlier than total latency
  alone.
- Sustained latency above an explicit service objective should alert; choose
  thresholds only after establishing a workload baseline.
- High utilization with a stable, low queue is healthy. High utilization with
  a growing queue indicates saturation.
- Low utilization with a growing queue suggests a scheduler or dependency
  problem rather than insufficient simulated compute capacity.
- Alert on increasing failure rate and correlate it with changes in offered
  load and deployments.
- Falling throughput at comparable offered load, or persistently low batch
  occupancy while requests wait, indicates degraded scheduling efficiency.

## Assumptions and limitations

- One simulated GPU and one scheduler worker.
- No real model, generated semantics, accelerator, CUDA runtime, or hardware
  telemetry.
- No KV-cache, VRAM, batch-token capacity, preemption, prefix caching, or
  chunked prefill model.
- No multiple GPUs, load balancing, bounded queue, timeout, or rejection
  policy.
- No streaming response; callers receive deterministic placeholder text after
  all simulated work completes.
- Linear decode batching and fixed throughput ignore nonlinear hardware and
  workload effects.
- Metrics are process-local and reset on restart; no Prometheus, Grafana,
  OpenTelemetry, or persistent time-series storage is included.
- `/health` is liveness only.

## Tests

The suite covers dataset validation and reproducibility, token approximation,
completion sizing, request-state invariants, GPU timing, FIFO admission,
continuous batching, failure handling, metrics semantics, HTTP validation, and
scheduler shutdown.

With the virtual environment active:

```bash
python -m pip check
python -m pytest -q
```

## Production system design

The local application in this repository is a simulation. A proposed production architecture for scaling a real vLLM deployment to 10,000 requests per minute is described in:

[`docs/technical_memo.md`](docs/technical_memo.md)
