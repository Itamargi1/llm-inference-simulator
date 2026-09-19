# Technical Memo: Scaling the LLM Inference Service to 10,000 Requests per Minute

## Overview

The production target is 10,000 requests per minute:

$$
10{,}000 / 60 \approx 167 \text{ requests/second}
$$

I propose serving **Qwen3-8B** with **vLLM**. Qwen3-8B has 8.2 billion parameters, is Apache 2.0 licensed, supports thinking and non-thinking modes, and is supported by vLLM. For summarization, document understanding, and customer support I would start in non-thinking mode, because these workloads benefit from predictable latency and output length rather than long reasoning traces.

The number of GPUs cannot be derived from request rate alone. Capacity also depends on prompt length, output length, concurrency, latency objectives, and cache reuse. The configuration below is therefore an initial design to be validated with representative traffic using `vllm bench serve`, which measures throughput, TTFT, TPOT, end-to-end latency, and goodput against latency objectives.

For initial planning I assume a typical request carries roughly 500 to 1,000 input tokens and up to 128 output tokens. Longer document requests are supported but should be tracked separately, since they consume much more prefill compute and KV-cache memory.

## 1. Proposed Architecture

```text
Clients
   |
   v
Application / API Gateway
   |
   v
vLLM Router
   |
   +------------+------------+------------+
   |            |            |            |
   v            v            v            v
vLLM         vLLM         vLLM         vLLM
Replica      Replica      Replica      Replica
Qwen3-8B     Qwen3-8B     Qwen3-8B     Qwen3-8B
GPU          GPU          GPU          GPU
   |            |            |            |
   +------------+------------+------------+
                     |
              Kubernetes Cluster
```

### Request flow

A client calls the application layer, which handles authentication, request validation, request IDs, rate limits, and maximum input and output limits. It forwards valid requests to the router rather than selecting GPUs itself. The router picks a replica, the replica's vLLM scheduler batches the request with other active sequences, the GPU runs prefill and then decode, and the response returns along the same path.

### Required hardware

| Resource          | Proposed configuration                                      |
| ----------------- | ----------------------------------------------------------- |
| Model             | Qwen3-8B                                                    |
| Mode              | Non-thinking                                                |
| Precision         | BF16                                                        |
| GPU               | NVIDIA H100 SXM 80GB                                        |
| Initial GPU count | 12                                                          |
| vLLM replicas     | 12                                                          |
| GPUs per replica  | 1                                                           |
| Physical layout   | At least 3 GPU nodes                                        |
| CPU               | Approximately 10 vCPUs per replica as a starting allocation |
| Host RAM          | Approximately 64 GB per replica as a starting allocation    |
| Local storage     | NVMe for model and container caches                         |

At BF16, Qwen3-8B weights occupy roughly 16.4 GB before runtime overhead. An H100 SXM provides 80 GB of memory and 3.35 TB/s of bandwidth, leaving substantial room for KV cache and serving overhead, so one GPU per replica is sufficient and tensor parallelism is not needed. The CPU and host-memory figures are starting allocations, consistent with vLLM Production Stack examples for a single-GPU replica, not strict requirements.

### Capacity planning

**The 12-replica figure is an initial capacity hypothesis, not a proven requirement.** It must be validated before launch.

Sizing follows from measured goodput:

$$
N_{\text{replicas}}
=
\left\lceil
\frac{\text{target request rate} \times \text{headroom factor}}
{\text{measured goodput per replica}}
\right\rceil
$$

With an illustrative 30% headroom for spikes and failures:

$$
167 \times 1.3 \approx 217 \text{ good requests/second}
$$

Spread across 12 replicas, the hypothesis requires each replica to sustain:

$$
217 / 12 \approx 18.1 \text{ good requests/second}
$$

"Good" means requests that meet the chosen TTFT and end-to-end latency objectives, not raw completions. I would measure this with `vllm bench serve` under the production prompt and output length distribution, then act on the result:

* if measured goodput per replica is below roughly 18.1 good requests/second, increase the replica count;
* if it is above, fewer than 12 replicas may be sufficient.

I am not quoting a benchmark result here, and I would not derive production capacity from the timing coefficients used in Parts 1 and 2, which are deliberate simulation assumptions rather than hardware measurements.

### Load balancing

Requests pass through a redundant vLLM Production Stack router.

Round-robin distributes evenly but ignores live replica load. I would use **load-aware routing based on live replica load**, which reacts to queue depth and running sequences. If production traffic turns out to contain substantial repeated prompt prefixes, for example shared system prompts or repeated documents, I would additionally enable **KV-cache-aware routing through the Production Stack's LMCache integration**. That is extra infrastructure and configuration, so it is worth enabling only once prefix reuse is measured, and it needs a load component so that cache-holding replicas do not become hotspots.

Within each replica, vLLM performs continuous batching so multiple sequences share GPU execution.

The application should also enforce maximum input and output lengths, preventing a few very large requests from consuming a disproportionate share of GPU memory.

### Observability pipeline

```text
Application / Router
        |
        +------ logs and traces ------> OpenTelemetry / log backend
        |
vLLM Replicas
        |
        +------ Prometheus metrics ---> Prometheus ---> Grafana / Alerts
        |
GPU Nodes
        |
        +------ GPU telemetry --------> Prometheus ---> Grafana / Alerts
```

vLLM exposes Prometheus metrics including running and waiting requests, queue time, TTFT, end-to-end latency, prefill and decode time, prompt and generated token counts, KV-cache utilization, and preemptions. These correspond closely to the queueing, latency, token, and throughput metrics the simulator exposes in Part 2. The GPU layer adds real hardware telemetry: compute utilization, memory usage, temperature, hardware errors, and node health.

Dashboards and alerts should combine signals rather than read one metric in isolation:

```text
high GPU utilization + low stable queue  = efficient use
high GPU utilization + growing queue     = insufficient capacity
low GPU utilization + growing queue      = routing, scheduling, or dependency problem
```

## 2. Scaling Challenges and Proposed Solutions

**Requests are not a uniform unit of work.** A short support prompt generating 50 tokens is far cheaper than a document request with thousands of input tokens. Capacity must therefore be tracked in prompt tokens per second and generated tokens per second alongside requests per second, with the prompt and output length distributions monitored, and load tests must reproduce the production distribution rather than one fixed prompt size.

**KV-cache pressure limits concurrency.** Each active sequence holds KV-cache memory, so long contexts and high concurrency can cap the batch well before the weights stop fitting. I would monitor vLLM's KV-cache utilization and preemption metrics directly, cap context and output lengths, and if long document jobs degrade short interactive ones, split them into separate serving pools so large requests cannot crowd out latency-sensitive traffic.

**Throughput and latency trade off against each other.** Larger batches raise aggregate throughput but can raise per-request latency, so the objective is not maximum utilization but the maximum number of requests meeting their objectives. I would fix target p95 TTFT and end-to-end latency, then benchmark at increasing arrival rates using the goodput constraints in vLLM's benchmark tooling.

**Autoscaling reacts slowly.** A GPU replica must load model weights before serving, so scale-up is far slower than for a stateless web service. I would keep warm spare capacity rather than relying on reactive scaling alone, use KEDA with Prometheus signals such as waiting-request count for sustained load increases, and scale down conservatively to avoid churn.

## 3. Reliability

### Traffic spikes

External APIs previously absorbed spikes. A self-hosted cluster has fixed immediate capacity, so an unbounded queue converts a spike into unbounded latency. I would layer:

1. approximately 30% capacity headroom in normal operation;
2. autoscaling when queue pressure stays elevated;
3. bounded queues and request deadlines;
4. per-user or per-service rate limits;
5. maximum prompt and output lengths;
6. a retryable error when the interactive path is overloaded, instead of letting queue latency grow without limit.

Not every request needs the same treatment. Interactive support requests need low latency, while many summarization and document jobs can run asynchronously from a durable queue as capacity frees up. If privacy and cost policy allow, an external provider could remain available purely as an emergency overflow path, not as part of normal serving.

### Partial GPU failure

Replicas are independent and spread across at least three nodes, and because Qwen3-8B fits on one H100 each replica owns one GPU. A single GPU failure therefore removes one replica, not the service.

The router stops sending new requests to an unhealthy replica based on readiness and health checks. Kubernetes can then restart the worker on the recovered GPU, or reschedule it onto another healthy GPU **if spare GPU capacity is available**. Until replacement capacity is ready, the remaining replicas continue serving at reduced cluster capacity, which is the reason for holding headroom. Kubernetes reschedules onto existing hardware; it does not create GPU capacity.

If evaluation later shows a larger model is required, the deployment may need tensor parallelism across several GPUs. In that design one GPU failure disables the whole tensor-parallel replica, so I would run several independent replica groups and remove a failed group from routing while the others continue.

### Degraded model performance

This covers two different problems, with different responses.

**Serving degradation** shows up as rising queue time, rising TTFT, slower token generation, falling throughput, increased failures, or heavy KV-cache pressure and preemptions. Causes include overload, an unhealthy GPU, a routing fault, or a regression from a new vLLM or model configuration. The response depends on the cause: remove unhealthy replicas, add capacity when demand genuinely rose, or roll back a recent serving change when degradation began after a deployment.

**Model-quality degradation** is different: the service can stay fast and healthy while answers get worse, for example weaker summaries, lower document question-answering accuracy, or more hallucinated or incorrect content, often following a model, prompt-template, or quantization change. I would keep a representative offline evaluation set for the three workloads and evaluate any such change against it before rollout. New versions then go out as a canary, with quality, latency, and error metrics compared against the stable version, and traffic returned to the previous version if quality drops beyond the accepted threshold.

If Qwen3-8B itself is not accurate enough for the business requirement, adding GPUs does not solve it. That is a model-selection problem: evaluate a stronger model, then repeat the benchmark and hardware-sizing process for it.

## Conclusion

The design is horizontal: independent single-GPU vLLM replicas behind a load-aware router, with continuous batching inside each replica, observability spanning application, inference, and hardware layers, and headroom to absorb load variation and GPU failures.

The 12-replica starting point is a hypothesis to be confirmed or corrected by benchmarking against the required latency objectives before launch.

Parts 1 and 2 simulate the same core concepts at small scale: queueing, prefill, batched decode, latency, throughput, token work, and utilization. In production the assumed timing model is replaced by real vLLM scheduling, GPU execution, KV-cache management, routing, autoscaling, and hardware telemetry.
