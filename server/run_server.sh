#!/bin/sh
# POSIX-compliant vLLM OpenAI server launcher
# Fail fast on errors
set -e

# Load .env file if it exists
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "${ENV_FILE}" ]; then
    echo "Loading configuration from .env..."
    # Export variables from .env (skip comments and empty lines)
    set -a
    . "${ENV_FILE}"
    set +a
    echo ""
fi

# Default configuration
MODEL="${MODEL:-facebook/opt-1.3b}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

# Simple chat template for models without native chat templates (like OPT)
# This just concatenates messages with newlines
CHAT_TEMPLATE='{% for message in messages %}{{ message.content }}
{% endfor %}'

# Print resolved configuration
echo ""
echo "=================================================="
echo "vLLM OpenAI Server Configuration"
echo "=================================================="
echo "Model:              ${MODEL}"
echo "Host:               ${HOST}"
echo "Port:               ${PORT}"
echo "Max Model Length:   ${MAX_MODEL_LEN}"
echo "GPU Memory Util:    ${GPU_MEM_UTIL}"
echo "KV Cache Dtype:     ${KV_CACHE_DTYPE}"
if [ -n "${HF_TOKEN}" ]; then
    echo "HF Token:           ✓ Set"
else
    echo "HF Token:           ✗ Not set"
fi
echo "=================================================="
echo ""

# Start vLLM server with all parameters
exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --chat-template "${CHAT_TEMPLATE}"
