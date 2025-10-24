# bench/ Directory

This directory contains the core benchmarking logic, prompt handling, and analysis tools for vllm-bench.

## Folder Structure

```
bench/
├── core/                    # Core benchmarking engine
│   ├── benchmark.py         # Main benchmark orchestrator (async load generation)
│   └── telemetry.py         # GPU monitoring utilities (NVML integration)
├── prompts/                 # Prompt generation and loading
│   ├── generate_rag_prompts.py  # Generate RAG prompts from Dolly dataset
│   ├── load_jsonl_prompts.py    # Load pre-generated JSONL prompts
│   └── load_real_prompts.py     # Load prompts from HF datasets (legacy)
├── analysis/                # Post-benchmark analysis and visualization
│   ├── summarize.py         # Generate statistics from benchmark results
│   ├── plot.py              # Generate 7 visualization plots (enhanced)
│   ├── plot_summary.py      # Compare multiple benchmark runs
│   └── inspect_responses.py # View model outputs for debugging
├── deprecated/              # Legacy utilities (archived)
│   └── util_text.py         # Old text generation utilities
└── scenarios.yaml           # Benchmark scenario definitions
```

## Core Components

### benchmark.py (Main Orchestrator)

The heart of the benchmarking system. Features:

- **Async load generation**: Configurable concurrency levels
- **Unique prompt tracking**: Ensures zero cache reuse within a run
- **Streaming support**: Handles SSE streaming responses
- **Telemetry integration**: Optional GPU monitoring
- **Flexible prompt loading**: Supports JSONL or HF datasets

**Key Features:**
- Maintains `used_prompt_indices` set to prevent prompt reuse across concurrency levels
- Displays progress: "Total unique prompts used: X/1500 (Y%)"
- Exports per-request metrics: TTFT, latency, throughput, token counts, errors
- Configurable via `scenarios.yaml`

**Key Features:**
- Maintains `used_prompt_indices` set to prevent prompt reuse across concurrency levels
- Displays progress: "Total unique prompts used: X/1500 (Y%)"
- Exports per-request metrics: TTFT, latency, throughput, token counts, errors
- Configurable via `scenarios.yaml`

**Usage:**
```bash
# Run all scenarios
python3 bench/core/benchmark.py

# Run specific scenario
python3 bench/core/benchmark.py --name rag_heavy

# With GPU monitoring
python3 bench/core/benchmark.py --telemetry

# Custom scenarios file
python3 bench/core/benchmark.py --scenarios custom_scenarios.yaml
```

**Import in code:**
```python
from bench.core.benchmark import run_benchmark
```

### telemetry.py (GPU Monitoring)

Tracks GPU metrics during benchmark runs:
- GPU utilization (%)
- VRAM usage (MB)
- Temperature (°C)
- Power consumption (W)

Requires `nvidia-ml-py3`. Gracefully degrades if unavailable.

## Prompt Handling

### generate_rag_prompts.py

Generates realistic RAG-style prompts from the Dolly dataset combined with Wikipedia context.

**Features:**
- Combines multiple text chunks until target token length
- Adds Dolly question and optional conversation history (0-4 turns)
- Outputs JSONL format: `{"messages": [...], "stats": {...}}`
- Default output: `../../data/rag_prompts.jsonl`

**Usage:**
```bash
# Generate 1500 prompts (default)
python3 bench/prompts/generate_rag_prompts.py

# Generate 500 prompts
python3 bench/prompts/generate_rag_prompts.py --num-prompts 500

# Custom output path
python3 bench/prompts/generate_rag_prompts.py --output custom_prompts.jsonl

# Customize target tokens and max history
python3 bench/prompts/generate_rag_prompts.py --target-tokens 2000 --max-history 2
```

**Output format:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Context:\n[Wikipedia passage]\n\nQuestion: [Dolly question]"}
  ],
  "stats": {
    "estimated_tokens": 1425,
    "num_history_turns": 2
  }
}
```

### load_jsonl_prompts.py

Loads pre-generated JSONL prompts (from `generate_rag_prompts.py`).

**Usage in code:**
```python
from bench.prompts.load_jsonl_prompts import load_prompts_jsonl

prompts, stats = load_prompts_jsonl("data/rag_prompts_1500.jsonl")
# Returns: (list of message arrays, list of token stats)
```

### load_real_prompts.py (Legacy)

Loads prompts directly from HF datasets (Dolly, ShareGPT, OpenAssistant). Kept for backward compatibility with simple prompt formats.

## Analysis Tools

### plot.py (Enhanced Visualization)

**NEW: 7 comprehensive plot types** (upgraded from 2):

1. **TTFT CDF** (`*_ttft_cdf.png`) - Time to First Token cumulative distribution
   - Shows p50, p90, p95, p99 percentiles
   - Useful for understanding responsiveness

2. **Latency CDF** (`*_latency_cdf.png`) - End-to-end latency cumulative distribution
   - Shows full request completion times
   - Identifies tail latency issues

3. **Throughput CDF** (`*_throughput_cdf.png`) - Tokens per second distribution
   - Shows generation speed variation
   - Useful for capacity planning

4. **Prompt Tokens Histogram** (`*_prompt_tokens_hist.png`)
   - Distribution of input token lengths
   - Validates prompt dataset characteristics

5. **Completion Tokens Histogram** (`*_completion_tokens_hist.png`)
   - Distribution of output token lengths
   - Shows model generation behavior

6. **Throughput Over Time** (`*_throughput_timeseries.png`)
   - Time series of tokens/second
   - Detects performance degradation or warmup effects

7. **TTFT Over Time** (`*_ttft_timeseries.png`)
   - Time series of time to first token
   - Shows scheduling/batching behavior over time

**Usage:**
```bash
# Auto-detect most recent CSV
python3 bench/analysis/plot.py

# Specific CSV file
python3 bench/analysis/plot.py data/benchmark_20250115_143022.csv

# Multiple files (separate plots for each)
python3 bench/analysis/plot.py data/*.csv
```

**Auto-detection logic:**
- Searches `data/` directory for CSV files matching benchmark naming pattern
- Sorts by timestamp in filename
- Selects most recent
- Useful for quick iteration: run benchmark → `python3 bench/analysis/plot.py` → view results

**Output location:**
All plots are saved to `results/plots/` with the same base filename as the input CSV.

### summarize.py

Generates statistical summaries from benchmark CSVs.

**Outputs:**
- `results/summary.csv` - Tabular statistics
- `results/summary.md` - Human-readable markdown

**Metrics computed:**
- Request counts (total, successful, failed)
- Success rate (%)
- Latency percentiles (p50, p90, p95, p99)
- TTFT percentiles (p50, p90, p95, p99)
- Token statistics (mean prompt tokens, completion tokens)
- Throughput (requests/sec, tokens/sec)

**Usage:**
```bash
# Single CSV
python3 bench/analysis/summarize.py data/benchmark_*.csv

# Multiple CSVs (aggregated summary)
python3 bench/analysis/summarize.py data/*.csv
```

### plot_summary.py

Compares multiple benchmark runs (cross-scenario or cross-configuration analysis).

**Usage:**
```bash
python3 bench/analysis/plot_summary.py data/*.csv
```

### inspect_responses.py

Views actual model outputs for debugging and qualitative analysis.

**Usage:**
```bash
# Show first 10 responses
python3 bench/analysis/inspect_responses.py data/benchmark_*.csv --num-samples 10

# Filter by error status
python3 bench/analysis/inspect_responses.py data/benchmark_*.csv --errors-only
```

## Unique Prompt Tracking

**Problem:** Repeated prompts lead to KV cache hits, artificially inflating performance metrics.

**Solution:** The benchmarking system maintains a `used_prompt_indices` set that persists across all concurrency levels within a single benchmark run.

**How it works:**
1. Load all 1500 prompts at start: `all_prompts = load_prompts_jsonl(...)`
2. Initialize tracking set: `used_prompt_indices = set()`
3. For each concurrency level:
   - Calculate needed prompts: `num_needed = concurrency × requests_per_user`
   - Find available prompts: `available = set(range(len(all_prompts))) - used_prompt_indices`
   - Sample unique subset: `random.sample(available, num_needed)`
   - Mark as used: `used_prompt_indices.update(selected_indices)`
   - Run benchmark with selected prompts
4. Progress tracking: Display "Total unique prompts used: X/1500 (Y%)"

**Example run:**
```
Starting benchmark: rag_heavy
Concurrency: 4, Requests: 100
Total unique prompts used: 40/1500 (2.7%)
...
Concurrency: 8, Requests: 200
Total unique prompts used: 120/1500 (8.0%)
...
Concurrency: 16, Requests: 400
Total unique prompts used: 280/1500 (18.7%)
...
Concurrency: 32, Requests: 800
Total unique prompts used: 600/1500 (40.0%)
```

**Benefits:**
- Zero KV cache reuse across concurrency levels
- Performance metrics reflect true inference costs
- Validates model behavior without cache contamination
- Pool of 1500 prompts supports extensive testing (40% utilization typical)

**Configuration:**
Customize in `scenarios.yaml`:
```yaml
defaults:
  base_url: "http://localhost:8000/v1"
  model: "mistralai/Mistral-7B-Instruct-v0.3"
  prompt_path: "../data/rag_prompts_1500.jsonl"  # Prompt pool
  
scenarios:
  rag_heavy:
    concurrencies: [4, 8, 16, 32]
    requests_per_user: 10  # Total requests = concurrency × requests_per_user
```

## Import Paths

Due to the nested structure, imports use `sys.path` manipulation in `benchmark.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.prompts.load_jsonl_prompts import load_prompts_jsonl
from bench.core.telemetry import TelemetryCollector
```

For scripts outside `bench/`, use:
```python
from bench.core.benchmark import run_benchmark
from bench.prompts.generate_rag_prompts import generate_rag_prompts
from bench.analysis.plot import create_plots
```

## Deprecated

The `deprecated/` folder contains archived utilities no longer actively maintained:
- `util_text.py` - Old synthetic text generation (replaced by RAG prompts)

These are kept for reference but not used in current workflows.
