# Using Real Prompts for Benchmarking

## Overview

Instead of using synthetic prompts, you can now load real human-generated prompts from open Hugging Face datasets for more realistic benchmarking.

## Quick Start

The benchmark automatically loads real prompts from the Dolly dataset:

```bash
# Run benchmark with real prompts (default)
python bench/bench.py short_chat
```

## Available Datasets

### 1. **Alpaca** (Default: `sharegpt`)
- **Dataset**: `tatsu-lab/alpaca`
- **Size**: 52K instruction-following examples
- **Quality**: High-quality, diverse instructions
- **Best for**: General instruction-following benchmarks

### 2. **Dolly**
- **Dataset**: `databricks/databricks-dolly-15k`  
- **Size**: 15K examples
- **Quality**: Human-generated, diverse tasks
- **Best for**: Question answering, summarization, extraction

### 3. **OpenAssistant**
- **Dataset**: `OpenAssistant/oasst1`
- **Size**: 88K messages (filtered for initial prompts)
- **Quality**: Conversational, human-feedback trained
- **Best for**: Chat/conversational benchmarks

## Inspecting Prompts

```bash
# View 10 real prompts from Alpaca
python inspect_prompts.py --use-real --show 10

# Try Dolly dataset
python inspect_prompts.py --use-real --dataset dolly --show 10

# Try OpenAssistant
python inspect_prompts.py --use-real --dataset openassistant --show 10

# Save real prompts to file
python inspect_prompts.py --use-real --save-text results/real_prompts.txt
```

## Implementation Details

### How It Works

The benchmark (`bench.py`) automatically:
1. Loads real prompts from Dolly dataset (default)
2. Filters by token count (min: 50, max: max(500, prefill*2))
3. Uses deterministic seeding for reproducibility
4. Fails gracefully if insufficient prompts are available

### Token Filtering

Prompts are filtered to match your scenario's `prefill_tokens`:
- **Min tokens**: `prefill_tokens / 2` (e.g., 128 for 256 target)
- **Max tokens**: `prefill_tokens * 2` (e.g., 512 for 256 target)
- Estimates: ~0.75 tokens per word

### Format

All prompts are converted to OpenAI chat format:
```python
[
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "<actual prompt text>"}
]
```

## Changing Dataset

To use a different dataset, edit `bench/bench.py`:

```python
# Change line ~548
messages_batch = load_real_prompts(
    num_prompts=num_requests,
    dataset_name="openassistant",  # Change to: dolly, sharegpt, openassistant
    min_tokens=50,
    max_tokens=max(500, prefill_tokens * 2),
    seed=hash(scenario_name) % 10000
)
```

## Dataset Statistics

After loading, the benchmark shows:
- ✅ Number of prompts loaded
- 📊 Token distribution (avg, min, max)
- 🎯 How well prompts match target length

## Requirements

```bash
pip install datasets  # Already installed
```

## Troubleshooting

**Issue**: Not enough prompts found  
**Solution**: Widen the token range or use a different dataset

**Issue**: Dataset download fails  
**Solution**: Check internet connection, Hugging Face may be down

**Issue**: "No module named 'datasets'"  
**Solution**: Run `pip install datasets`

## Examples

### Short prompts (128 tokens)
```bash
python bench/bench.py short_chat  # Uses Alpaca, 50-256 token range
```

### Medium prompts (512 tokens)
```bash
# Edit scenarios.yaml to set prefill_tokens: 512
# Will automatically filter for 256-1024 token prompts
python bench/bench.py medium_chat
```

### Inspect what you're using
```bash
python inspect_prompts.py --use-real --save-text results/my_prompts.txt
```

---

## Why Use Real Prompts?

✅ **More realistic**: Actual user queries, not repetitive synthetic text  
✅ **Better diversity**: Wide variety of tasks and lengths  
✅ **Easier**: No need to craft synthetic prompt generation logic  
✅ **Reproducible**: Deterministic sampling with seeds  
✅ **Validated**: Datasets are human-curated and quality-checked
