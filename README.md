# vllm-bench

vllm-bench is an async load generator and benchmarking tool for vLLM inference servers. By default, it targets **`gpt-oss-20`** from Hugging Face, but supports any model compatible with vLLM's OpenAI-compatible API.

## Project Structure

```
vllm-bench/
├── server/          # vLLM server launcher scripts and Docker config
│   ├── run_server.sh      # POSIX shell script to launch vLLM
│   └── docker-compose.yml # Docker Compose configuration
├── bench/           # Benchmarking logic and load generators
│   ├── bench.py           # Main benchmark orchestrator
│   ├── load_real_prompts.py  # Load prompts from HF datasets
│   ├── scenarios.yaml     # Benchmark scenario definitions
│   └── summarize.py       # Generate statistics
├── results/         # Benchmark results (gitignored)
│   └── plots/       # Generated visualization plots
├── inspect_prompts.py  # View/inspect prompts before benchmarking
├── REAL_PROMPTS.md  # Documentation on real prompt datasets
├── Makefile         # Build and task automation
├── requirements.txt # Python dependencies (httpx, pydantic, etc.)
├── .gitignore       # Excludes venv, results, __pycache__
└── README.md        # This file
```

## Real Human Prompts

**vllm-bench uses real human-generated prompts** from Hugging Face datasets instead of synthetic text. This provides more realistic benchmarking that reflects actual production workloads.

**Default dataset:** [Databricks Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- 3,863 prompts available (50-500 tokens)
- High-quality, diverse instruction-following tasks
- Covers Q&A, summarization, extraction, reasoning

**View prompts before benchmarking:**
```bash
# Inspect 10 prompts from Dolly dataset
python inspect_prompts.py --show 10

# Save all 500 prompts to file
python inspect_prompts.py --save-text results/prompts.txt

# Try different dataset
python inspect_prompts.py --dataset openassistant --show 10
```

**Available datasets:**
- `dolly` (default) - 3,863 prompts, best availability
- `sharegpt` - 330 prompts, Alpaca instruction-following
- `openassistant` - 341 prompts, conversational

See [REAL_PROMPTS.md](REAL_PROMPTS.md) for full documentation.

## Setup Instructions

### 1. Install Dependencies

```bash
# Create mamba environment and install packages
make install

# Or manually:
mamba create -n vllm-bench python=3.10 -y
mamba activate vllm-bench
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Create .env file from template
make init-env

# Edit .env and add your Hugging Face token
nano .env  # or vim, code, etc.
```

Your `.env` file should look like:
```bash
# Required
HF_TOKEN=hf_your_actual_token_here

# Model configuration
MODEL=gpt-oss-20
PORT=8000
HOST=0.0.0.0

# Performance tuning
GPU_MEM_UTIL=0.90
MAX_MODEL_LEN=4096
KV_CACHE_DTYPE=auto
```

**Get your Hugging Face token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" permissions
3. Copy it to your `.env` file

### 3. Accept Model License (if needed)

### 3. Accept Model License (if needed)

If you're using **private or gated models** (like `gpt-oss-20`):

1. Visit the model page: `https://huggingface.co/gpt-oss-20`
2. Click "Agree and access repository" if prompted
3. Your HF_TOKEN (set in `.env`) will grant access

### 4. Override Default Model (Optional)

You can override `.env` settings via command line or by editing `.env`:

```bash
# Option 1: Edit .env file
nano .env
# Change: MODEL=meta-llama/Llama-2-7b-chat-hf

# Option 2: Override on command line
MODEL=mistralai/Mistral-7B-Instruct-v0.2 make server
```

## Quick Start

### Start the vLLM Server

**Option A: Using Make (reads from .env)**
```bash
# Server reads configuration from .env file
make server

# Or override specific variables
MODEL=meta-llama/Llama-2-7b-chat-hf make server
```

**Option B: Using the shell script directly**
```bash
# Still reads from .env if you source it
source .env
./server/run_server.sh

# Or set variables inline
HF_TOKEN=hf_xxxxx MODEL=gpt-oss-20 PORT=8000 ./server/run_server.sh
```

**Option C: Using Docker Compose**
```bash
cd server
# Docker Compose will read .env from parent directory
docker-compose up

# Or set inline
HF_TOKEN=hf_xxxxx MODEL=gpt-oss-20 docker-compose up
```

The server will start on `http://0.0.0.0:8000` by default (or your configured PORT).

### Run Benchmarks

```bash
# Using Make
make bench        # Run all scenarios
make summarize    # Generate statistics
make plots        # Create visualizations
make all          # Run bench + summarize + plots

# Or run Python scripts directly (with mamba environment activated)
mamba activate vllm-bench
python bench/bench.py                    # All scenarios
python bench/bench.py --name short_chat  # Single scenario
python bench/bench.py --telemetry        # With GPU monitoring

# View results
ls results/
ls results/plots/
ls results/telemetry/  # GPU metrics if telemetry enabled
```

### GPU Telemetry (Optional)

The benchmark suite includes optional GPU monitoring that tracks:
- GPU utilization (%)
- VRAM usage (MB)
- Temperature (°C)
- Power consumption (W)

**To enable GPU telemetry:**

1. Install the optional dependency:
   ```bash
   pip install nvidia-ml-py3
   ```

2. Run benchmarks with `--telemetry` flag:
   ```bash
   python bench/bench.py --telemetry
   ```

3. View telemetry results in `results/telemetry/<run_id>.csv`

**Note:** Telemetry requires NVIDIA GPUs and NVML support. If unavailable, the benchmark will display a warning and continue without telemetry.

## Smoke Test

Quick validation to ensure the benchmark suite is working correctly.

### Prerequisites

- vLLM server running locally (or adjust `base_url` in scenarios.yaml)
- Dependencies installed: `make install` or `pip install -r requirements.txt`

### Test Procedure

**Step 1: Start the vLLM Server**

```bash
# In terminal 1
HF_TOKEN=hf_xxxxx MODEL=gpt-oss-20 ./server/run_server.sh

# Wait for server to be ready (look for "Application startup complete")
```

**Step 2: Create a Minimal Test Scenario**

Create `bench/test_scenario.yaml`:

```yaml
defaults:
  base_url: "http://localhost:8000/v1"
  model: "gpt-oss-20"
  temperature: 0.0
  stream: true

scenarios:
  smoke_test:
    description: "Minimal smoke test"
    prefill_tokens: 256
    max_new_tokens: 128
    concurrencies: [8]
    num_requests: 50
```

**Step 3: Run the Smoke Test**

```bash
# In terminal 2
python bench/bench.py --scenarios bench/test_scenario.yaml --name smoke_test
```

**Step 4: Verify Outputs**

Check that the following files were created:

```bash
# 1. Run CSV exists and contains data
ls -lh results/smoke_test-cc8-pref256-max128-*.csv
wc -l results/smoke_test-cc8-pref256-max128-*.csv
# Expected: 51 lines (1 header + 50 requests)

# 2. Summary files generated
python bench/summarize.py results/smoke_test-cc8-pref256-max128-*.csv
ls -lh results/summary.csv results/summary.md

# 3. Plots generated
python bench/plot.py results/smoke_test-cc8-pref256-max128-*.csv
ls -lh results/plots/smoke_test-cc8-pref256-max128-*_ttft_cdf.png
ls -lh results/plots/smoke_test-cc8-pref256-max128-*_latency_cdf.png
# Expected: Two PNG files (TTFT and latency CDFs)
```

**Step 5: Inspect Results**

```bash
# View summary
cat results/summary.md

# Quick stats from CSV
python -c "
import pandas as pd
df = pd.read_csv('results/smoke_test-cc8-pref256-max128-*.csv', glob='results/smoke_test-*.csv')
successful = df[df['error'].isna() | (df['error'] == '')]
print(f'Success rate: {len(successful)}/{len(df)} ({len(successful)/len(df)*100:.1f}%)')
print(f'TTFT p50: {successful[\"ttft_ms\"].quantile(0.5):.0f}ms')
print(f'TTFT p95: {successful[\"ttft_ms\"].quantile(0.95):.0f}ms')
print(f'Latency p50: {successful[\"latency_ms\"].quantile(0.5):.0f}ms')
print(f'Latency p95: {successful[\"latency_ms\"].quantile(0.95):.0f}ms')
"
```

### Expected Results ("What Good Looks Like")

**Success Criteria:**

✅ **All requests complete** - Success rate should be 100% (or very close)  
✅ **CSV file created** - Contains 51 lines (header + 50 rows)  
✅ **No errors column values** - Most/all error fields should be empty  
✅ **TTFT values present** - All successful requests have ttft_ms > 0  
✅ **Summary generates** - summary.csv and summary.md created without errors  
✅ **Plots render** - Two PNG files created (TTFT and latency CDFs)  

**Expected Metrics (gpt-oss-20, GPU-dependent):**

For a well-tuned server on modern GPU (A100/H100):

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| **Success Rate** | 95-100% | Should be nearly perfect for low concurrency |
| **TTFT p50** | 50-200ms | First token should arrive quickly |
| **TTFT p95** | 100-400ms | Even tail latency should be reasonable |
| **Latency p50** | 1000-3000ms | Total time for 128 tokens |
| **Latency p95** | 1500-5000ms | Tail latency still manageable |
| **Throughput** | 5-15 req/s | At concurrency 8 |

**TTFT Stability Check:**

Run multiple concurrency levels to verify TTFT remains stable:

```yaml
concurrencies: [8, 16, 32]
num_requests: 50
```

**What good looks like:**
- TTFT p50 should remain relatively stable as concurrency increases (±50ms variance)
- TTFT p95 may increase slightly but shouldn't explode (< 2x at 4x concurrency)
- If TTFT doubles or triples with modest concurrency increase, indicates batching/scheduling issues

**Example good output:**
```
Concurrency 8:  TTFT p50=120ms, p95=180ms
Concurrency 16: TTFT p50=135ms, p95=210ms  ✓ Stable
Concurrency 32: TTFT p50=160ms, p95=280ms  ✓ Acceptable
```

**Example bad output (indicates problem):**
```
Concurrency 8:  TTFT p50=120ms, p95=180ms
Concurrency 16: TTFT p50=450ms, p95=890ms  ✗ Unstable - investigate batching
Concurrency 32: TTFT p50=1200ms, p95=2500ms ✗ Severe degradation - server overloaded
```

### Troubleshooting

**If requests fail:**
- Check server is running: `curl http://localhost:8000/v1/models`
- Check server logs for errors
- Verify HF_TOKEN is valid for gpt-oss-20
- Ensure GPU has sufficient memory

**If TTFT is very high (>1s):**
- Check GPU utilization with telemetry: `--telemetry`
- May indicate batching delays or resource contention
- Try reducing `max-num-batched-tokens` on server

**If plots don't render:**
- Check matplotlib backend: `python -c "import matplotlib; print(matplotlib.get_backend())"`
- Try non-interactive backend: `export MPLBACKEND=Agg`

**If CSV is empty:**
- Check for Python exceptions in benchmark output
- Verify scenarios.yaml is valid: `python -c "import yaml; print(yaml.safe_load(open('bench/test_scenario.yaml')))"`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `gpt-oss-20` | Hugging Face model to load |
| `HF_TOKEN` | _(none)_ | Hugging Face access token for private/gated repos |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server host |
| `MAX_MODEL_LEN` | _(auto)_ | Maximum model context length |
| `GPU_MEM_UTIL` | _(default)_ | GPU memory utilization fraction (e.g., 0.9) |
| `KV_CACHE_DTYPE` | _(auto)_ | KV cache data type (e.g., fp8, fp16) |

## Experiments: Tuning vLLM for gpt-oss-20

This section covers key vLLM server parameters to experiment with for optimal performance. Each parameter affects different aspects of throughput, latency, and concurrency handling.

### Key Parameters to Sweep

#### 1. `--gpu-memory-utilization` (GPU_MEM_UTIL)

Controls the fraction of GPU memory allocated for the KV cache.

**Values to try:** `0.85`, `0.90`, `0.95`

**Effects:**
- **Higher values (0.95):**
  - ✅ Larger KV cache → supports more concurrent requests
  - ✅ Higher batch sizes possible
  - ❌ Risk of OOM errors if model weights + KV cache exceed GPU memory
  - 📊 **Impact:** Increases concurrency ceiling, may reduce p95/p99 latency under high load

- **Lower values (0.85):**
  - ✅ More memory headroom, safer operation
  - ❌ Smaller KV cache → fewer concurrent requests supported
  - 📊 **Impact:** Lower concurrency ceiling, may increase p95/p99 as requests queue

**Recommendation:** Start with `0.90`, increase to `0.95` if stable.

#### 2. `--max-model-len` (MAX_MODEL_LEN)

Maximum context length (prompt + generation) the model supports.

**Values to try:** `2048`, `4096`, `8192` (if model supports)

**Effects:**
- **Shorter lengths (2048):**
  - ✅ Smaller KV cache per request → more concurrent requests fit
  - ✅ Better TTFT for most requests
  - ❌ Rejects requests exceeding limit
  - 📊 **Impact:** Improves throughput for short/medium contexts, increases concurrency ceiling

- **Longer lengths (8192):**
  - ✅ Supports long-context workloads
  - ❌ Larger KV cache per request → fewer concurrent requests
  - ❌ Worse p95/p99 latency under load
  - 📊 **Impact:** Reduces concurrency ceiling, increases memory per request

**Recommendation:** Set based on workload (e.g., `4096` for RAG, `2048` for chat).

#### 3. `--kv-cache-dtype`

Data type for KV cache storage.

**Values to try:** `auto`, `fp16`, `fp8` (if GPU supports FP8)

**Effects:**
- **`fp8` (if supported on H100, A100):**
  - ✅ 2x memory reduction vs fp16 → doubles effective KV cache capacity
  - ✅ Significantly higher concurrency ceiling
  - ⚠️ Minor quality degradation (usually negligible)
  - 📊 **Impact:** Major improvement in concurrency, reduces p95/p99 under load

- **`fp16`:**
  - ✅ Standard precision, no quality loss
  - ❌ Higher memory usage
  - 📊 **Impact:** Baseline performance

- **`auto`:**
  - Uses model's native dtype (usually fp16)

**Recommendation:** Use `fp8` if available (H100/A100) for production workloads.

#### 4. `--max-num-batched-tokens`

Maximum number of tokens processed in a single batch (prefill + decode).

**Values to try:** `8192`, `16384`, `32768`

**Effects:**
- **Higher values (32768):**
  - ✅ Larger batches → better GPU utilization
  - ✅ Higher throughput under load
  - ❌ Increased TTFT for requests waiting in batch
  - ❌ Higher p95/p99 latency variance
  - 📊 **Impact:** Improves throughput at high concurrency, but increases latency tail

- **Lower values (8192):**
  - ✅ Lower TTFT, more predictable latency
  - ❌ Reduced throughput, GPU underutilization
  - 📊 **Impact:** Better p50/p95 latency, lower overall throughput

**Recommendation:** Start with `16384`, increase if throughput-bound.

#### 5. `--max-num-seqs`

Maximum number of sequences (requests) processed concurrently.

**Values to try:** `128`, `256`, `512`

**Effects:**
- **Higher values (512):**
  - ✅ More concurrent requests → higher throughput
  - ❌ Requires more GPU memory (KV cache for each sequence)
  - ❌ May increase p95/p99 as scheduler complexity grows
  - 📊 **Impact:** Increases concurrency ceiling if memory allows

- **Lower values (128):**
  - ✅ Lower memory usage, simpler scheduling
  - ❌ May reject requests when full
  - 📊 **Impact:** Sets hard concurrency limit

**Recommendation:** Set based on `gpu-memory-utilization` and `max-model-len` constraints.

### Experiment Design

**Example configurations to benchmark:**

```bash
# Baseline: Conservative settings
GPU_MEM_UTIL=0.85 MAX_MODEL_LEN=4096 KV_CACHE_DTYPE=auto make server

# Optimized for throughput: High concurrency
GPU_MEM_UTIL=0.95 MAX_MODEL_LEN=2048 KV_CACHE_DTYPE=fp8 make server

# Optimized for latency: Predictable response times
GPU_MEM_UTIL=0.90 MAX_MODEL_LEN=4096 KV_CACHE_DTYPE=fp16 make server
```

**What to measure:**
- TTFT p50/p95/p99 across concurrency levels
- Latency p50/p95/p99 across concurrency levels
- Maximum sustainable concurrency (where p99 < SLA threshold)
- Tokens/s throughput at different loads

### Expected Trade-offs

| Configuration | Concurrency Ceiling | TTFT p50 | p95/p99 Latency | Throughput |
|--------------|--------------------:|---------:|----------------:|-----------:|
| Conservative (0.85, fp16, 4k) | Low | Best | Best | Moderate |
| Balanced (0.90, fp16, 4k) | Medium | Good | Good | Good |
| High-throughput (0.95, fp8, 2k) | High | Moderate | Higher variance | Best |
| Long-context (0.90, fp16, 8k) | Low | Worst | Worst | Low |

**General guidelines:**
- **For chat/short queries:** Optimize for low TTFT and high concurrency (smaller max-model-len, fp8 cache)
- **For RAG/medium context:** Balance memory and latency (4k context, fp8 if available)
- **For long documents:** Accept lower concurrency (8k context, fp16, lower gpu-mem-util)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you'd like to add.

## License

This project is licensed under the MIT License. See the LICENSE file for details.