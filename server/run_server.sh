#!/bin/sh
# POSIX-compliant vLLM OpenAI server launcher
# Fail fast on errors
set -e

# Default configuration
MODEL="${MODEL:-gpt-oss-20}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# Print resolved configuration
echo "=================================================="
echo "vLLM OpenAI Server Configuration"
echo "=================================================="
echo "Model:              ${MODEL}"
echo "Host:               ${HOST}"
echo "Port:               ${PORT}"
echo "Max Model Length:   ${MAX_MODEL_LEN:-<auto>}"
echo "GPU Memory Util:    ${GPU_MEM_UTIL:-<default>}"
echo "KV Cache Dtype:     ${KV_CACHE_DTYPE:-<auto>}"
echo "HF Token:           $([ -n "${HF_TOKEN}" ] && echo "✓ Set" || echo "✗ Not set")"
echo "=================================================="
echo ""

# Check if HF_TOKEN is required (warn but don't fail)
if [ -z "${HF_TOKEN}" ]; then
    echo "Warning: HF_TOKEN not set. Private/gated models will fail to load."
    echo "Set HF_TOKEN environment variable if accessing private repositories."
    echo ""
fi

# Build vLLM command
CMD="python -m vllm.entrypoints.openai.api_server"
CMD="${CMD} --model ${MODEL}"
CMD="${CMD} --host ${HOST}"
CMD="${CMD} --port ${PORT}"
CMD="${CMD} --trust-remote-code"

# Add optional flags if set
if [ -n "${MAX_MODEL_LEN}" ]; then
    CMD="${CMD} --max-model-len ${MAX_MODEL_LEN}"
fi

if [ -n "${GPU_MEM_UTIL}" ]; then
    CMD="${CMD} --gpu-memory-utilization ${GPU_MEM_UTIL}"
fi

if [ -n "${KV_CACHE_DTYPE}" ]; then
    CMD="${CMD} --kv-cache-dtype ${KV_CACHE_DTYPE}"
fi

# Export HF_TOKEN if set (for Hugging Face authentication)
if [ -n "${HF_TOKEN}" ]; then
    export HF_TOKEN
fi

echo "Starting vLLM server..."
echo "Command: ${CMD}"
echo ""

# Execute the command
exec ${CMD}
