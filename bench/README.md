# Bench Directory Structure

Organized structure for vLLM benchmarking system.

## Directory Layout

```
bench/
├── core/                   # Core benchmarking engine
│   ├── benchmark.py        # Main benchmark runner
│   └── telemetry.py        # GPU monitoring
│
├── prompts/                # Prompt generation and loading
│   ├── generate_rag_prompts.py   # Generate RAG prompts
│   ├── load_jsonl_prompts.py     # Load prompts from JSONL
│   └── load_real_prompts.py      # Load from datasets (Dolly, etc.)
│
├── analysis/               # Post-benchmark analysis tools
│   ├── summarize.py        # Generate summary statistics
│   ├── plot_summary.py     # Visualization tools
│   ├── plot.py             # Additional plotting utilities
│   └── inspect_responses.py # Inspect model outputs
│
├── deprecated/             # Archived/deprecated code
│   └── util_text.py        # Old synthetic text generation
│
└── scenarios.yaml          # Benchmark scenario configurations
```

## Usage

### Running Benchmarks

From project root:
```bash
python3 run_benchmark.py --name rag_realistic
```

Or using the module directly:
```bash
python3 -m bench.core.benchmark --name rag_realistic
```

### Generating Prompts

```bash
cd bench/prompts
python3 generate_rag_prompts.py -n 1500 -t 1500 --history-turns 2 -o ../../data/rag_prompts_1500.jsonl
```

### Analysis

```bash
cd bench/analysis
python3 summarize.py results/*.csv
python3 inspect_responses.py --num 5
```

## Import Paths

Files have been updated to use proper module imports:

```python
# Core
from bench.core.benchmark import main
from bench.core.telemetry import GPUTelemetryMonitor

# Prompts
from bench.prompts.load_jsonl_prompts import load_prompts_from_jsonl
from bench.prompts.load_real_prompts import load_real_prompts

# Analysis
from bench.analysis.summarize import summarize_results
```

## Changes Made

1. **Created subfolders**: `core/`, `prompts/`, `analysis/`, `deprecated/`
2. **Moved files** to appropriate directories
3. **Updated imports** in `core/benchmark.py` to use proper module paths
4. **Updated relative paths** for data files (e.g., `../data/` → `../../data/`)
5. **Created `run_benchmark.py`** wrapper for easier execution

## Backward Compatibility

Old commands still work from the project root:
```bash
python3 bench/core/benchmark.py --name rag_realistic
```
