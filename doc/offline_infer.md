# Key takeaways

## Teampreture and Top-k strategy
Tempreture: The randomness of selecting next token. Higher tempreture indicating higer probability of selecting tokens other than the best match (highest probability).
Top-p (nucleus sampling，p=0.95): Cut out tokens with lower probability to match the inout sequence.

## Metrics
- Queue and engine state

num_requests_running / num_requests_waiting (Gauge): both 0 -> all requests finished.
engine_sleep_state (Gauge): 1 means idle; multiple entries are from different labels/times.
- Caching

kv_cache_usage_perc (Gauge): 0 at the end.
prefix_cache_queries/hits (Counter): 640/0 → no prefix reuse.
external_prefix_cache_*, mm_cache_*: 0 → not used.
request_prefill_kv_computed_tokens (Hist): sum=640, count=128 → 5 prompt tokens/request.
- Work volume and success

prompt_tokens / generation_tokens (Counter): 640 / 2048 → ~5 in, ~16 out per request (128 requests total).
request_success (Counter): shows 128 successful (multiple time series entries).
num_preemptions: 0.
- Request parameters

request_params_n (Hist): n=1 (single hypothesis).
request_max_num_generation_tokens / request_params_max_tokens (Hist): capped around 16; cumulative buckets ≥20 show all requests.
- Latency and time breakdown

time_to_first_token_seconds (Hist): sum≈5319, count=128 → mean ≈41.6s; all TTFB in (40s, 80s].
e2e_request_latency_seconds (Hist): sum≈5339, count=128 → mean ≈41.7s; all in (40s, 50s].
request_queue_time_seconds: ~0 → negligible queuing.
request_inference_time_seconds: sum≈65.47 → ≈0.51s/request total compute.
prefill ≈0.35s/request; decode ≈0.16s/request.
inter_token_latency_seconds: sum≈21.00, count=1920 → ≈10.9 ms/token.
request_time_per_output_token_seconds: ≈10.9 ms/token (per-request aggregation).
- Scheduler/iteration

iteration_tokens_total (Hist): sum≈2688 (=640+2048), count=17 → 1 prefill iteration + 16 decode iterations.
- Config/info

cache_config_info (Gauge): “1” with labels to expose cache config.

- Takeaways

- Steady-state compute is fast (~10.9 ms/token; ~0.51 s/request for 5-in/16-out).
- End-to-end and TTFB are dominated by cold start/model load; warm up once and re-measure for steady-state.
- No prefix cache hits; enable/seed prefix cache to benefit from reuse if your workload repeats prefixes.

## GPU memory estimating
Total memory usage: Model weights + KV cache.
Model weights: $Parameter * Size of Data Type$. For Qwen3-1.7B, 1.7B * 2 (FP16) ~= 3.4 GB
KV chache (https://lmcache.ai/kv_cache_calculator.html)
With: $B$: batch size; $L$: sequence length; $N$: number of transformer layers; $H$: number of heads; D: dims per head($hidden size / H$); $S$: size of data type

For each layer: $\text{Key cache} = \text{Value cache} = B \times H \times L \times D \times S$

Hence, each layer: $Cahce_{KV} = 2 \times B \times H \times L \times D \times S$

Total KV cache: $N \times Cache_{KV}$

Actual usage reported by vLLM: Model ~ 3.215 GB, CUDA graph ~ 0.21 GB, ravailable KV cache ~ 24.20 GB (RTX 5090, 0.9 utilization), actual KV cache usage ~ 226,528 tokens slots
