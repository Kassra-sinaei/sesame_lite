#!/usr/bin/env bash
# Local OpenAI-compatible server for Qwen3-14B (official AWQ 4-bit weights) in its own venv (.venv-llm).
# GPU share is capped so pi0.5 (~12 GB) and the simulator fit beside it on the 48 GB A6000.
#   bash scripts/llm/serve_qwen.sh            # then: --llm qwen (endpoint http://127.0.0.1:8001/v1)
cd "$(dirname "$0")/../.."
export PATH="$PWD/.venv-llm/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer JIT-compiles kernels; no nvcc in this container
exec .venv-llm/bin/vllm serve models/Qwen3-14B-AWQ \
  --served-model-name qwen3-14b --host 127.0.0.1 --port "${QWEN_PORT:-8001}" \
  --gpu-memory-utilization "${QWEN_GPU_FRACTION:-0.30}" --max-model-len 16384 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes \
  --seed 0 --generation-config vllm --attention-backend FLASH_ATTN
