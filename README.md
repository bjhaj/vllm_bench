# vllm-bench

Async load generator and benchmarking tool for vLLM inference servers with realistic RAG workloads.

## Overview

**vllm-bench** is designed to benchmark vLLM inference servers under realistic production conditions. Unlike synthetic benchmarks that use random text or repeated prompts, vllm-bench simulates **real-world RAG (Retrieval-Augmented Generation) workloads** with long context windows, multi-turn conversations, and unique prompts that eliminate KV cache contamination.

### Key Design Principles

1. **Realistic Workloads**: Uses RAG-style prompts combining Wikipedia context passages with human-authored questions from the Dolly dataset. Average prompt length ~1371 tokens simulates production RAG scenarios.

2. **Cache-Free Testing**: Maintains unique prompt tracking across all concurrency levels within a benchmark run. Each prompt is used exactly once, ensuring performance metrics reflect true inference costs rather than cache hits.

3. **Comprehensive Metrics**: Tracks Time to First Token (TTFT), end-to-end latency, throughput, token distributions, and temporal patterns across different concurrency levels.

4. **Production-Ready Analysis**: Generates 7 visualization types including CDFs, histograms, and time series to identify performance degradation, tail latency issues, and capacity limits.

### What You Can Measure

- **TTFT (Time to First Token)**: Responsiveness for interactive applications
- **End-to-end latency**: Full request completion time across percentiles (p50, p90, p95, p99)
- **Throughput**: Requests per second and tokens per second under varying load
- **Concurrency scaling**: How performance degrades as concurrent requests increase
- **Token distribution**: Input/output length patterns and their impact on performance
- **Temporal stability**: Performance consistency over time, warmup effects, degradation

### Use Cases

- **Capacity Planning**: Determine maximum sustainable concurrency for your hardware
- **SLA Validation**: Verify p95/p99 latency meets requirements under production load
- **Configuration Tuning**: Compare vLLM parameters (GPU memory utilization, KV cache settings, batch sizes)
- **Model Comparison**: Benchmark different models on the same hardware
- **Regression Testing**: Detect performance regressions across vLLM versions
- **Cost Optimization**: Understand tokens/sec per GPU to optimize infrastructure costs

## Project Structure

```
vllm-bench/
├── server/          # vLLM server launcher scripts and Docker config
│   ├── run_server.sh      # POSIX shell script to launch vLLM
│   └── docker-compose.yml # Docker Compose configuration
├── bench/           # Benchmarking logic and load generators (modular structure)
│   ├── core/        # Core benchmarking logic
│   │   ├── benchmark.py   # Main benchmark orchestrator (was bench.py)
│   │   └── telemetry.py   # GPU monitoring utilities
│   ├── prompts/     # Prompt generation and loading
│   │   ├── generate_rag_prompts.py  # Generate RAG prompts from Dolly dataset
│   │   ├── load_jsonl_prompts.py    # Load pre-generated JSONL prompts
│   │   └── load_real_prompts.py     # Load prompts from HF datasets
│   ├── analysis/    # Post-benchmark analysis and visualization
│   │   ├── summarize.py        # Generate statistics from results
│   │   ├── plot.py             # Generate 7 visualization plots (enhanced)
│   │   ├── plot_summary.py     # Plot comparison across multiple runs
│   │   └── inspect_responses.py # View model outputs for debugging
│   ├── deprecated/  # Legacy utilities (archived)
│   │   └── util_text.py
│   ├── scenarios.yaml     # Benchmark scenario definitions
│   └── README.md          # Detailed bench/ documentation
├── data/            # Datasets and generated prompts
│   ├── rag_prompts_1500.jsonl  # Pre-generated unique RAG prompts
│   └── [results]    # Benchmark results CSVs (gitignored)
├── results/         # Legacy results directory (gitignored)
│   └── plots/       # Generated visualization plots
├── docs/            # Documentation
│   └── REAL_PROMPTS.md  # Documentation on real prompt datasets
├── Makefile         # Build and task automation
├── requirements.txt # Python dependencies (httpx, pydantic, etc.)
├── .env             # Environment configuration (gitignored)
├── .gitignore       # Excludes venv, results, __pycache__
└── README.md        # This file
```

## Features

- **RAG-style prompts**: 1500 unique pre-generated prompts from Dolly dataset + Wikipedia context (avg ~1371 tokens)
- **Cache-free benchmarking**: Unique prompt tracking ensures zero cache reuse across runs
- **Comprehensive visualization**: 7 plot types (TTFT/Latency/Throughput CDFs, histograms, time series)
- **Auto-detection**: `plot.py` automatically finds most recent benchmark results
- **Current model**: `mistralai/Mistral-7B-Instruct-v0.3` (8192 token context)
- **Optional GPU telemetry**: Track utilization, VRAM, temperature, power

## Quick Start

### 1. Install Dependencies

```bash
make install
# Or manually: mamba create -n vllm-bench python=3.10 -y && pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:
```bash
HF_TOKEN=hf_your_token_here
MODEL=mistralai/Mistral-7B-Instruct-v0.3
GPU_MEM_UTIL=0.85
MAX_MODEL_LEN=8192
```

### 3. Start vLLM Server

```bash
make server
# Or: source .env && ./server/run_server.sh
```

### 4. Run Benchmarks

```bash
# Run benchmarks with all scenarios
python3 bench/core/benchmark.py

# Generate visualizations (auto-detects most recent results)
python3 bench/analysis/plot.py

# Generate statistics
python3 bench/analysis/summarize.py data/*.csv
```

## Visualization Outputs

The `plot.py` tool generates 7 plot types:

1. **TTFT CDF** - Time to First Token distribution
2. **Latency CDF** - End-to-end latency distribution
3. **Throughput CDF** - Tokens/second distribution
4. **Prompt Tokens Histogram** - Input length distribution
5. **Completion Tokens Histogram** - Output length distribution
6. **Throughput Over Time** - Generation speed time series
7. **TTFT Over Time** - Responsiveness time series

All saved to `results/plots/`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `mistralai/Mistral-7B-Instruct-v0.3` | Model to load |
| `HF_TOKEN` | _(required)_ | HuggingFace token |
| `GPU_MEM_UTIL` | `0.85` | GPU memory utilization |
| `MAX_MODEL_LEN` | `8192` | Max context length |
| `PORT` | `8000` | Server port |

See `bench/README.md` for detailed documentation.

## License

MIT License. See LICENSE file for details.