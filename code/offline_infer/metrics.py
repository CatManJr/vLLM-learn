# ~/offline/metrics.py
# key metrics used in offline inference
import os
os.environ["VLLM_USE_V1"] = "1"  # ensure using vLLM version 1. The operation should be before importing vllm
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # set HF mirror for faster model downloading in China mainland
# os.environ["HF_ENDPOINT"] = "https://huggingface.co" # original HF endpoint as I live in the US
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector

import time

# Sample prompts.
prompts = ["The future of AI is"] * 128
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95) # tempreture is the randomness of the sampling
# top-p sampling means to sample from the smallest set of tokens whose cumulative probability exceeds the threshold p.

def main():
    # Create an LLM.
    llm = LLM(
        model="Qwen/Qwen3-1.7B",
        max_model_len=4096,
        max_num_seqs=128,
        gpu_memory_utilization=0.9,
        disable_log_stats=False,
    )
    
    # Estimated memory usage: 1.7B * 2 Bytes + KV cache
    # Total SizeKV = N × SizeKV per layer = 2 x Batch_size × Heads x Length(sequence) x Dims(hidden size / heads) × N x Size of(float)

    # Generate texts from the prompts.
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t1 = time.perf_counter()
    new_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    throughput = new_tokens / (t1 - t0)
    print(f"Throughput = {throughput:.2f} tok/s")

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Dump all metrics
    for metric in llm.get_metrics():
        if isinstance(metric, Gauge):
            print(f"{metric.name} (gauge) = {metric.value}")
        elif isinstance(metric, Counter):
            print(f"{metric.name} (counter) = {metric.value}")
        elif isinstance(metric, Vector):
            print(f"{metric.name} (vector) = {metric.values}")
        elif isinstance(metric, Histogram):
            print(f"{metric.name} (histogram)")
            print(f"    sum = {metric.sum}")
            print(f"    count = {metric.count}")
            for bucket_le, value in metric.buckets.items():
                print(f"    {bucket_le} = {value}")


if __name__ == "__main__":
    main()