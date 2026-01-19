# Archieving my learning progress of vllm

## Dependency
`NVIDIA GPU`
`Linux (better for triton)`
`vLLM python library (and its dependencies recursively)`

## Install vllm by uv
```bash
uv init
uv add vllm
```

## vLLM version
As vLLM-project is working on upgrading to version 1, this archive mainly focuses on version 1.
Set: 
```python
os.environ["VLLM_USE_V1"] = "1" 
```
to declare the inference engine as version 1.