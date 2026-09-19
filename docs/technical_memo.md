# Technical Memo: Scaling the LLM Inference Service to 10,000 Requests per Minute

## Overview

The production target is 10,000 requests per minute, or approximately:

$$
10{,}000 / 60 \approx 167 \text{ requests/second}
$$

I propose serving **Qwen3-8B** using **vLLM**. Qwen3-8B has 8.2 billion parameters, is available under the Apache 2.0 license, supports both thinking and non-thinking modes, and can be served through vLLM. For summarization, document understanding, and customer support, I would initially use non-thinking mode because these workloads generally benefit from predictable latency and output length rather than long reasoning traces.

The exact number of GPUs cannot be determined from request rate alone. Capacity also depends on prompt length, generated output length, concurrency, latency requirements, cache reuse, and the model configuration. I would therefore use the hardware configuration below as an initial production design and validate it with representative traffic using `vllm bench serve` before launch. vLLM's benchmark tooling can measure throughput as well as TTFT, TPOT, end-to-end latency, and goodput against latency objectives.

For initial capacity planning, I assume a typical request contains approximately 500 to 1,000 input tokens and up to 128 output tokens. Longer document requests are supported, but should be monitored separately because they consume substantially more prefill compute and KV-cache memory.

## 1. Proposed Architecture

The proposed request flow is:

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

The application layer is responsible for authentication, request validation, request IDs, rate limits, and maximum input/output limits. It forwards valid inference requests to the inference cluster rather than selecting GPUs itself.

The inference cluster consists of multiple independent vLLM replicas. vLLM supports data-parallel deployment where model weights are replicated across GPUs and each replica processes independent request batches.

### Model and hardware

I would start with:

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

Qwen3-8B has approximately 8.2 billion parameters. At BF16 precision, the raw model weights require roughly 16.4 GB before runtime overhead. An H100 SXM provides 80 GB of GPU memory and 3.35 TB/s of memory bandwidth, leaving substantial memory for KV cache and serving overhead.

The CPU and host-memory values are starting allocations rather than strict requirements. Current vLLM Production Stack deployment examples use approximately 10 CPUs and 64 GiB of memory for a one-GPU serving replica.

### Capacity planning

The target is 167 requests per second, but I would not assume a fixed number of requests per second per H100 without benchmarking the actual workload.

Instead, I would benchmark one replica with representative prompt and output lengths and calculate:

$$
N_{\text{replicas}}
=
\left\lceil
\frac{\text{target request rate} \times \text{headroom factor}}
{\text{measured goodput per replica}}
\right\rceil
$$

For example, with 30% capacity headroom:

$$
\text{capacity target}
=
167 \times 1.3
\approx
217 \text{ requests/second}
$$

The 12-GPU configuration is therefore an initial capacity target, not a claim that exactly 12 GPUs are always required. Before production launch I would run representative traffic with `vllm bench serve` and increase or decrease the number of replicas according to measured goodput and the required latency objectives.

This is preferable to deriving production capacity from the timing coefficients used in Parts 1 and 2, because those values were deliberately simulation assumptions rather than measurements of real hardware.

### Load balancing

Requests should pass through a redundant vLLM Production Stack router.

A basic round-robin strategy distributes requests evenly, but it ignores both current replica load and KV-cache reuse. I would therefore use **load-aware routing with KV-cache awareness**.

vLLM's load-aware routing considers both cached prompt tokens and current load. This prevents a common problem where many requests sharing a popular prefix are all sent to one cache-holder while other replicas remain underused. When the cached replica becomes too busy, traffic can be sent to a less-loaded replica instead.

Within each replica, vLLM handles continuous batching so multiple requests can share GPU execution.

The application should also enforce maximum input length and maximum output length. This prevents a small number of exceptionally large requests from consuming a disproportionate amount of GPU memory or KV-cache capacity.

## Observability Pipeline

The production observability pipeline should extend the ideas implemented in the simulator:

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

vLLM exposes Prometheus-compatible production metrics including:

* requests running and waiting
* queue time
* TTFT
* end-to-end request latency
* prefill and decode time
* prompt and generated token counts
* KV-cache utilization
* request preemptions
* successful request counts

These metrics closely correspond to the simplified queueing, token, throughput, and latency metrics implemented in Part 2 of the simulator.

The GPU layer should additionally expose real hardware telemetry such as GPU utilization, GPU memory usage, temperature, hardware errors, and node health.

The most important dashboards and alerts should combine signals rather than examine one metric in isolation. For example:

```text
high GPU utilization + low stable queue
= efficient use of the hardware

high GPU utilization + growing queue
= insufficient serving capacity

low GPU utilization + growing queue
= likely routing, scheduling, or dependency problem
```

## 2. Scaling Challenges and Proposed Solutions

### Different requests have very different costs

Requests per second alone is not a sufficient capacity metric.

A short customer-support prompt generating 50 tokens is much cheaper than a long document-understanding request containing thousands of input tokens and producing hundreds of output tokens.

I would therefore monitor:

* requests per second
* prompt tokens per second
* generated tokens per second
* prompt-length distribution
* output-length distribution
* TTFT
* total latency

Capacity testing should reproduce the actual production distribution rather than benchmark only one fixed prompt size.

### KV-cache pressure

Each active sequence requires KV-cache memory. High concurrency and long contexts can therefore limit the number of requests that can be served simultaneously even when the model weights fit comfortably in GPU memory.

vLLM exposes KV-cache utilization and preemption metrics, which should be monitored directly.

I would set maximum context and output lengths and monitor KV-cache pressure. If long document-processing requests significantly interfere with short interactive requests, I would consider separate serving pools. For example:

```text
interactive customer-support pool
        +
long-document processing pool
```

This prevents very large requests from degrading latency for short interactive requests.

### Throughput versus latency

Continuous batching increases aggregate GPU throughput, but waiting for larger batches can increase latency.

The goal should therefore not be maximum GPU utilization at any cost. The goal is to maximize the number of requests that meet the service-level objectives.

In practice, I would define acceptable p95 TTFT and total latency and benchmark the cluster at increasing request rates. vLLM's benchmark tooling supports goodput constraints based on TTFT, TPOT, and end-to-end latency.

### Autoscaling delay

GPU replicas take longer to start than ordinary stateless web services because the model must be loaded into GPU memory.

For that reason, I would maintain a minimum amount of warm spare capacity instead of relying entirely on reactive autoscaling.

For sustained increases in load, Kubernetes can scale the number of vLLM replicas. The vLLM Production Stack supports KEDA autoscaling using Prometheus metrics such as the number of waiting requests.

Scale-down should be conservative so that replicas are not repeatedly destroyed and recreated during fluctuating traffic.

## 3. Reliability

### Traffic spikes

External APIs previously absorbed traffic spikes for us. A self-hosted GPU cluster has a fixed amount of immediately available compute, so allowing requests to accumulate in an unlimited queue would eventually produce unacceptable latency.

I would use several layers of protection:

1. Maintain approximately 30% capacity headroom during normal operation.
2. Autoscale when queue pressure remains elevated.
3. Use bounded queues and request deadlines.
4. Apply per-user or per-service rate limits.
5. Limit maximum prompt and output length.
6. Return a retryable error when the interactive service is overloaded rather than allowing queue latency to grow without limit.

Not all requests need the same treatment. Interactive customer-support requests require low latency, while some summarization or document-processing requests can be asynchronous. Those asynchronous jobs can be placed in a durable queue and processed as GPU capacity becomes available.

If company privacy and cost policies allow it, an external LLM provider could also remain available as an emergency overflow path. It would not be part of normal serving, but could protect availability during an exceptional capacity incident.

### Partial GPU failure

The cluster contains multiple independent inference replicas distributed across several GPUs and physical nodes.

Because Qwen3-8B fits on one H100, each replica uses one GPU. If one GPU fails, only that replica becomes unavailable.

The router should stop sending new requests to an unhealthy replica based on readiness and health information. Kubernetes can restart the worker or reschedule it on another healthy GPU. The remaining replicas continue serving traffic with temporarily reduced total capacity.

Replicas should be spread across multiple physical nodes so that a node failure does not remove the whole inference service.

This is one advantage of using independent single-GPU replicas when the model fits on one GPU. It provides simple horizontal scaling and clear fault isolation.

If model-quality evaluation later shows that Qwen3-8B is insufficient and a larger model is required, the deployment may need tensor parallelism across multiple GPUs. In that design, failure of one GPU can make the entire tensor-parallel replica unavailable. I would therefore deploy several independent tensor-parallel replica groups. A failed group can be removed from routing while the remaining groups continue serving traffic.

### Degraded model performance

I would treat **serving performance** and **model quality** as two different problems.

#### Serving degradation

Examples include:

* increasing queue time
* increasing TTFT
* slower token generation
* falling throughput
* rising failure rate
* excessive KV-cache pressure or preemptions

These may indicate overload, an unhealthy GPU, a routing problem, or a regression introduced by a new vLLM or model configuration.

The response should depend on the cause. The system can remove unhealthy replicas, add capacity when demand is genuinely higher, or roll back a recent serving configuration when the degradation started after deployment.

#### Model-quality degradation

A model can also remain fast and technically healthy while producing worse answers.

For summarization, document understanding, and customer support, I would maintain a representative offline evaluation set containing expected examples and quality criteria. Changes to the model, prompt template, quantization, or serving configuration should be evaluated against this set before full deployment.

New versions should then use a canary deployment. A small percentage of traffic is sent to the candidate version while quality, latency, and error metrics are compared with the stable version. If quality decreases beyond an accepted threshold, traffic is returned to the previous version.

If Qwen3-8B itself does not provide sufficient accuracy for the business requirements, adding more GPUs does not solve the problem. That is a **model-selection problem**. I would evaluate a stronger model, such as a larger Qwen model, and then repeat the capacity benchmark and hardware-sizing process for that model.

This separates two questions:

```text
Is the model being served reliably?
        versus
Is the selected model good enough for the task?
```

Both need to be monitored, but they require different solutions.

## Conclusion

I would begin with Qwen3-8B in non-thinking mode, served through vLLM using independent H100 GPU replicas behind a load-aware router.

An initial deployment of 12 H100 80GB GPUs provides a reasonable starting point for capacity testing and horizontal fault isolation. The exact production replica count should then be determined by benchmarking representative traffic against the required TTFT and end-to-end latency objectives.

The architecture scales horizontally, uses continuous batching inside each vLLM replica, monitors both inference and hardware behavior, and keeps enough spare capacity to tolerate load variation and individual GPU failures.

Parts 1 and 2 of this project simulate the same core concepts at a smaller scale: queueing, prefill, batched decode, latency, throughput, token work, and utilization. In production, the assumed timing model is replaced by real vLLM scheduling, GPU execution, KV-cache management, routing, autoscaling, and hardware observability.
