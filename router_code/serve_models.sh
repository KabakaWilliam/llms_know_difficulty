#!/usr/bin/env bash
set -euo pipefail

API_KEY="token-abc123"
HOST="0.0.0.0"
LOGDIR="./vllm_logs"; mkdir -p "$LOGDIR"

# GPU0: 1.5B + 7B (shared)
# GPU1: 14B alone
# GPU2: unused

pids=()
trap 'for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done' EXIT INT TERM

echo "Starting 1.5B on GPU0:8001"
CUDA_VISIBLE_DEVICES=0 vllm serve "Qwen/Qwen2.5-Math-1.5B-Instruct" \
  --host "$HOST" --port 8001 \
  --dtype auto --api-key "$API_KEY" \
  --gpu-memory-utilization 0.35 \
  > "${LOGDIR}/Qwen__Qwen2.5-Math-1.5B-Instruct_8001.log" 2>&1 &
pids+=("$!")
sleep 2

echo "Starting 7B on GPU0:8002"
CUDA_VISIBLE_DEVICES=0 vllm serve "Qwen/Qwen2.5-Math-7B-Instruct" \
  --host "$HOST" --port 8002 \
  --dtype auto --api-key "$API_KEY" \
  --gpu-memory-utilization 0.55 \
  > "${LOGDIR}/Qwen__Qwen2.5-Math-7B-Instruct_8002.log" 2>&1 &
pids+=("$!")
sleep 2

echo "Starting 14B on GPU1:8003"
CUDA_VISIBLE_DEVICES=1 vllm serve "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --host "$HOST" --port 8003 \
  --dtype auto --api-key "$API_KEY" \
  --gpu-memory-utilization 0.70 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  > "${LOGDIR}/deepseek-ai__DeepSeek-R1-Distill-Qwen-14B_8003.log" 2>&1 &
pids+=("$!")
sleep 2

echo "Up:"
echo "  1.5B -> http://$HOST:8001/v1"
echo "  7B   -> http://$HOST:8002/v1"
echo "  14B  -> http://$HOST:8003/v1"
wait
echo "All servers have exited."