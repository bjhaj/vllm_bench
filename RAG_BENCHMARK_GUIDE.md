# RAG Benchmarking with Realistic Prompts

## Overview

The vllm-bench now supports **realistic RAG benchmarking** with:
- ✅ Pre-generated prompts from Dolly dataset (context + questions)
- ✅ Conversation history (simulating multi-turn dialogues)
- ✅ Precise token counting (using facebook/opt-1.3b tokenizer)
- ✅ Scaled workloads (requests scale with concurrency)
- ✅ Actual token count logging in CSV results

**No more synthetic prompts!** Everything uses real data with measured token counts.

---

## Quick Start

### 1. Generate RAG Prompts

```bash
# Generate 500 prompts with ~1700 tokens and 2 history turns
cd bench
python3 generate_rag_prompts.py \
  --num-prompts 500 \
  --target-tokens 1700 \
  --history-turns 2 \
  --output ../data/rag_prompts.jsonl

# Or use shorthand
python3 generate_rag_prompts.py -n 500 -t 1700 --history-turns 2 -o ../data/rag_prompts.jsonl
```

**Output:**
- JSONL file with prompts containing:
  - Dolly text chunks as "RAG context"
  - Dolly questions appended to context
  - Conversation history (user/assistant turns)
  - Precise token counts per prompt

### 2. Run RAG Benchmark

```bash
# Run the realistic RAG scenario
python3 bench/bench.py rag_realistic
```

**What happens:**
- Loads prompts from `data/rag_prompts.jsonl`
- Tests concurrency levels: [4, 8, 16, 32]
- **Scaled workload**: 10 requests per user
  - 4 users = 40 requests
  - 8 users = 80 requests
  - 16 users = 160 requests
  - 32 users = 320 requests

### 3. View Results

```bash
# CSV contains actual token counts
head -5 results/rag_realistic-cc4-pref1700-max256-*.csv

# Check token count distribution
python3 -c "
import pandas as pd
df = pd.read_csv('results/rag_realistic-cc4-*.csv')
print(df[['actual_prefill_tokens', 'prompt_tokens', 'completion_tokens']].describe())
"
```

---

## Architecture

### 1. Prompt Generation (`generate_rag_prompts.py`)

**Input:**
- Dolly dataset (text chunks + questions)
- Target token count (e.g., 1700)
- History turns (e.g., 2)

**Process:**
1. Load Dolly dataset text chunks
2. Concatenate chunks until reaching target tokens
3. Append a Dolly question to the context
4. Add conversation history (random user/assistant turns)
5. Tokenize with facebook/opt-1.3b tokenizer
6. Save to JSONL with precise token counts

**Output JSONL format:**
```json
{
  "prompt_id": 0,
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant..."},
    {"role": "user", "content": "Can you explain that more?"},
    {"role": "assistant", "content": "Let me break that down:"},
    {"role": "user", "content": "Context:\n<text>\n\nQuestion: <question>"}
  ],
  "token_stats": {
    "system_tokens": 18,
    "history_tokens": 28,
    "wiki_context_tokens": 1450,
    "question_tokens": 15,
    "total_prefill_tokens": 1511
  },
  "target_prefill_tokens": 1700,
  "history_turns": 2
}
```

### 2. Prompt Loading (`load_jsonl_prompts.py`)

**Functions:**
- `load_prompts_from_jsonl()` - Load N prompts from file
- `load_prompts_from_jsonl_repeated()` - Load with cycling if N > available
- `get_jsonl_stats()` - Get statistics about JSONL file

**Returns:**
- `messages_batch`: List of message lists (OpenAI format)
- `token_stats_batch`: List of token statistics per prompt

### 3. Benchmark Execution (`bench.py`)

**RAG Mode:**
```yaml
scenarios:
  rag_realistic:
    rag: true  # Enable RAG mode
    prompt_file: "../data/rag_prompts.jsonl"
    target_prefill_tokens: 1700
    history_turns: 2
    dolly_questions: true
    max_new_tokens: 256
    concurrencies: [4, 8, 16, 32]
    requests_per_user: 10  # Scaled workload
```

**Scaled Workload:**
- `requests_per_user`: Each concurrent user sends N requests
- Total requests = concurrency × requests_per_user
- Simulates realistic load where more users = more total traffic

**CSV Output:**
```csv
run_id,qid,prefill_tokens,max_new_tokens,ttft_ms,latency_ms,prompt_tokens,completion_tokens,actual_prefill_tokens,error,timestamp
rag_realistic-cc4-pref1700-max256-20251023_195823,0,1700,256,45.2,1230.5,1511,128,1511,,2025-10-23T19:58:23.123456
```

**New field:**
- `actual_prefill_tokens`: Precise tokenized count from JSONL token_stats

---

## Configuration Reference

### Scenario YAML Fields

```yaml
scenarios:
  my_rag_scenario:
    # RAG configuration
    rag: true                                  # Enable RAG mode
    prompt_file: "../data/my_prompts.jsonl"   # Path to JSONL file
    target_prefill_tokens: 1700               # Target context size (informational)
    history_turns: 2                          # Number of history turns (informational)
    dolly_questions: true                     # Using Dolly questions (informational)
    
    # Workload configuration
    concurrencies: [4, 8, 16, 32]             # Concurrency levels to test
    requests_per_user: 10                     # Requests per user (scaled workload)
    # OR
    num_requests: 500                         # Fixed total (legacy, not scaled)
    
    # Generation parameters
    max_new_tokens: 256                       # Max tokens to generate
```

### Generate Prompts Options

```bash
python3 generate_rag_prompts.py --help

Options:
  -n, --num-prompts INT        Number of prompts to generate (default: 500)
  -t, --target-tokens INT      Target prefill tokens (default: 1700)
  --history-turns INT          Number of conversation turns (default: 2)
  -o, --output PATH            Output JSONL file (default: ../data/rag_prompts.jsonl)
  --tokenizer STR              Tokenizer model (default: facebook/opt-1.3b)
  --seed INT                   Random seed (default: 42)
```

---

## Examples

### Example 1: Generate Different Prompt Sizes

```bash
# Short RAG (1.5k tokens, no history)
python3 generate_rag_prompts.py -n 500 -t 1500 --history-turns 0 -o ../data/rag_short.jsonl

# Medium RAG (1.7k tokens, 2 turns)
python3 generate_rag_prompts.py -n 500 -t 1700 --history-turns 2 -o ../data/rag_medium.jsonl

# Heavy RAG (2.5k tokens, 4 turns)
python3 generate_rag_prompts.py -n 300 -t 2500 --history-turns 4 -o ../data/rag_heavy.jsonl
```

### Example 2: Create Custom Scenario

**Edit `bench/scenarios.yaml`:**
```yaml
scenarios:
  my_custom_rag:
    description: "Custom RAG with heavy context"
    rag: true
    prompt_file: "../data/rag_heavy.jsonl"
    target_prefill_tokens: 2500
    history_turns: 4
    dolly_questions: true
    max_new_tokens: 512
    concurrencies: [2, 4, 8]
    requests_per_user: 5  # 2×5=10, 4×5=20, 8×5=40 total requests
```

**Run it:**
```bash
python3 bench/bench.py my_custom_rag
```

### Example 3: Analyze Token Counts

```bash
# Generate prompts
python3 generate_rag_prompts.py -n 100 -t 1700 --history-turns 2

# Inspect what was generated
python3 load_jsonl_prompts.py ../data/rag_prompts.jsonl

# Run benchmark
python3 bench/bench.py rag_realistic

# Analyze actual vs target tokens
python3 << 'EOF'
import pandas as pd
import glob

files = glob.glob('results/rag_realistic-*.csv')
df = pd.concat([pd.read_csv(f) for f in files])

print("Token Count Analysis:")
print("="*60)
print(f"Target prefill: 1700 tokens")
print(f"Actual prefill: {df['actual_prefill_tokens'].mean():.1f} ± {df['actual_prefill_tokens'].std():.1f}")
print(f"Range: {df['actual_prefill_tokens'].min()}-{df['actual_prefill_tokens'].max()}")
print(f"\nAPI reported prompt_tokens: {df['prompt_tokens'].mean():.1f}")
print(f"Completion tokens: {df['completion_tokens'].mean():.1f}")
EOF
```

---

## Comparison: Old vs New

### Old System (Removed)
❌ Synthetic prompts ("quick brown fox" repeated)  
❌ Word-based token estimation (~0.75 tokens/word)  
❌ Fixed workload (500 requests regardless of concurrency)  
❌ No conversation history  
❌ No precise token counts  

### New System (Current)
✅ Real Dolly dataset (context + questions)  
✅ Precise tokenization (facebook/opt-1.3b tokenizer)  
✅ Scaled workload (requests scale with users)  
✅ Conversation history (simulates multi-turn)  
✅ Actual token counts logged in CSV  

---

## Token Count Fields Explained

| Field | Source | Description |
|-------|--------|-------------|
| `prefill_tokens` | Scenario config | Target/estimated prefill tokens |
| `actual_prefill_tokens` | JSONL token_stats | **Precise tokenized count** from prompt generation |
| `prompt_tokens` | API response | Tokens counted by vLLM server |
| `completion_tokens` | API response | Tokens generated by model |

**Why three different counts?**
- `prefill_tokens`: What you asked for (config)
- `actual_prefill_tokens`: What you got (measured during generation)
- `prompt_tokens`: What the server saw (may differ due to tokenizer differences)

---

## Troubleshooting

### Prompt file not found

```
Error: Prompt file not found: ../data/rag_prompts.jsonl
Generate it with: python bench/generate_rag_prompts.py -n 500 -t 1700
```

**Solution:**
```bash
cd bench
python3 generate_rag_prompts.py -n 500 -t 1700 --history-turns 2 -o ../data/rag_prompts.jsonl
```

### Not enough prompts

If your JSONL has 100 prompts but you need 320 (32 users × 10 req/user), the loader will **cycle/repeat** prompts automatically.

**Better solution:** Generate more prompts:
```bash
python3 generate_rag_prompts.py -n 500 -t 1700 --history-turns 2
```

### Token count mismatch

`actual_prefill_tokens` ≠ `prompt_tokens` is normal! They use different tokenizers:
- `actual_prefill_tokens`: facebook/opt-1.3b tokenizer (during generation)
- `prompt_tokens`: vLLM server tokenizer (may be slightly different)

Difference is usually < 5%.

---

## Performance Tips

### 1. Balance Context Size vs Throughput

- **Short context (1.5k tokens)**: Higher throughput, more concurrent requests
- **Long context (2.5k+ tokens)**: Lower throughput, higher memory usage

### 2. Adjust Concurrency for Scaled Workload

```yaml
# Light load: 2 users × 10 req = 20 requests
concurrencies: [2, 4, 8]
requests_per_user: 10

# Heavy load: 32 users × 20 req = 640 requests
concurrencies: [8, 16, 32]
requests_per_user: 20
```

### 3. Monitor Actual Token Counts

```bash
# After benchmark, check distribution
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('results/rag_realistic-*.csv')
print(df['actual_prefill_tokens'].describe())
EOF
```

---

## Next Steps

1. **Generate prompts**: `python3 generate_rag_prompts.py -n 500 -t 1700 --history-turns 2`
2. **Run benchmark**: `python3 bench/bench.py rag_realistic`
3. **Analyze results**: Check CSV `actual_prefill_tokens` column
4. **Tune workload**: Adjust `requests_per_user` and `concurrencies` in scenarios.yaml

🎉 **You now have realistic RAG benchmarking with precise token measurements!**
