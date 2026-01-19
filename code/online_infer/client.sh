#!/usr/bin/env bash
# available model list
curl -s --noproxy '*' http://127.0.0.1:13311/v1/models | jq .

# chat/completions
curl -s --noproxy '*' http://127.0.0.1:13311/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "introduce vLLM in 20 words"}],
    "max_tokens": 30,
    "temperature": 0.6
  }' | jq -r '.choices[0].message.content'
