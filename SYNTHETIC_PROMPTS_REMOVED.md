# Removed Synthetic Prompts

## Summary

All synthetic prompt generation has been **removed** from vllm-bench. The benchmark now exclusively uses **real human-generated prompts** from Hugging Face datasets.

## What Was Removed

### 1. **util_text.py** (No longer used)
- ❌ `SEED_TEXT` constant with "quick brown fox" repetitive text
- ❌ `build_text_approx_tokens()` - synthetic text generation
- ❌ `build_chat_messages()` - synthetic message creation
- ❌ `build_batch_chat_messages()` - batch synthetic generation

### 2. **bench.py Changes**
```diff
- from util_text import build_chat_messages
- 
- # Try loading real prompts first, fall back to synthetic if unavailable
- try:
-     messages_batch = load_real_prompts(...)
- except Exception as e:
-     console.print(f"[yellow]Warning: Could not load real prompts ({e}), using synthetic prompts[/yellow]")
-     from util_text import build_batch_chat_messages
-     messages_batch = build_batch_chat_messages(...)

+ # Load real prompts from dataset
+ from load_real_prompts import load_real_prompts
+ messages_batch = load_real_prompts(...)
+ 
+ if len(messages_batch) < num_requests:
+     console.print(f"[red]Error: Only found {len(messages_batch)} prompts, need {num_requests}[/red]")
+     return
```

### 3. **inspect_prompts.py Changes**
```diff
- from util_text import build_batch_chat_messages, estimate_tokens
- 
- parser.add_argument('--use-real', action='store_true', help='Use real prompts from dataset instead of synthetic')
- 
- if args.use_real:
-     messages_batch = load_real_prompts(...)
- else:
-     messages_batch = build_batch_chat_messages(...)

+ def estimate_tokens(text: str) -> int:
+     """Estimate token count (rough: ~0.75 tokens per word)."""
+     return int(len(text.split()) * 0.75)
+ 
+ # Always use real prompts
+ messages_batch = load_real_prompts(...)
```

## Why This Change?

### ❌ Problems with Synthetic Prompts
1. **Unrealistic**: Repetitive "quick brown fox" text doesn't represent real user queries
2. **Poor diversity**: All prompts had similar structure and content
3. **Misleading benchmarks**: Performance on synthetic text ≠ performance on real workloads
4. **Wasted effort**: Maintaining synthetic generation code with no benefit

### ✅ Benefits of Real Prompts
1. **Realistic**: Actual human queries from production-like scenarios
2. **Diverse**: Wide variety of tasks, lengths, and complexity
3. **Validated**: Curated, high-quality datasets from Hugging Face
4. **Simpler code**: No need to maintain synthetic generation logic
5. **Better insights**: Benchmark results reflect real-world performance

## What Now Uses Real Prompts

### Default Dataset: Dolly
- **Source**: `databricks/databricks-dolly-15k`
- **Available**: 3,863 prompts (50-500 tokens)
- **Quality**: Human-generated instruction-following tasks
- **Coverage**: Q&A, summarization, classification, extraction

### Alternative Datasets
- **Alpaca**: 330 prompts (via `sharegpt` dataset name)
- **OpenAssistant**: 341 prompts (conversational)

## Files Status

| File | Status | Notes |
|------|--------|-------|
| `util_text.py` | ⚠️ **Deprecated** | Still exists but unused; can be deleted |
| `bench.py` | ✅ **Updated** | Only uses `load_real_prompts.py` |
| `inspect_prompts.py` | ✅ **Updated** | Only uses `load_real_prompts.py` |
| `load_real_prompts.py` | ✅ **Active** | Primary prompt source |

## Migration Guide

If you have custom code using synthetic prompts:

### Before
```python
from util_text import build_batch_chat_messages

messages = build_batch_chat_messages(
    prefill_tokens=256,
    num_messages=500,
    base_seed=42
)
```

### After
```python
from load_real_prompts import load_real_prompts

messages = load_real_prompts(
    num_prompts=500,
    dataset_name="dolly",
    min_tokens=50,
    max_tokens=500,
    seed=42
)
```

## Cleanup (Optional)

You can safely delete these unused files:

```bash
# Remove deprecated synthetic prompt generator
rm bench/util_text.py

# Remove old synthetic prompt outputs (if any)
rm results/prompts_short_chat.txt  # If it contains synthetic prompts
```

## Verify

Check that everything works:

```bash
# Should load 10 real prompts from Dolly
python inspect_prompts.py --show 3 --num-requests 10

# Should work without errors
python bench/bench.py --name short_chat
```

---

**Result**: vllm-bench now exclusively uses real, human-generated prompts for more realistic and meaningful benchmarking! 🎉
