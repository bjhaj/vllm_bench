"""
Load pre-generated RAG prompts from JSONL files.

Replaces the old load_real_prompts logic with a simple JSONL reader
that loads prompts with Wikipedia context, Dolly questions, and conversation history.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def load_prompts_from_jsonl(
    jsonl_file: Path,
    num_prompts: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42
) -> Tuple[List[List[Dict[str, str]]], List[Dict]]:
    """
    Load prompts from JSONL file.
    
    Args:
        jsonl_file: Path to JSONL file with pre-generated prompts
        num_prompts: Number of prompts to load (None = all)
        shuffle: Whether to shuffle prompts
        seed: Random seed for shuffling
        
    Returns:
        (messages_batch, token_stats_batch)
        - messages_batch: List of message lists (OpenAI chat format)
        - token_stats_batch: List of token statistics per prompt
    """
    if not jsonl_file.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")
    
    # Load all records
    records = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    # Shuffle if requested
    if shuffle:
        import random
        random.seed(seed)
        random.shuffle(records)
    
    # Limit number of prompts
    if num_prompts is not None:
        records = records[:num_prompts]
    
    # Extract messages and token stats
    messages_batch = [record["messages"] for record in records]
    token_stats_batch = [record["token_stats"] for record in records]
    
    return messages_batch, token_stats_batch


def load_prompts_from_jsonl_repeated(
    jsonl_file: Path,
    num_prompts: int,
    seed: int = 42
) -> Tuple[List[List[Dict[str, str]]], List[Dict]]:
    """
    Load prompts from JSONL, repeating/cycling if needed.
    
    Useful when num_prompts > available prompts in file.
    """
    messages_batch, token_stats_batch = load_prompts_from_jsonl(
        jsonl_file,
        num_prompts=None,  # Load all
        shuffle=True,
        seed=seed
    )
    
    if len(messages_batch) >= num_prompts:
        return messages_batch[:num_prompts], token_stats_batch[:num_prompts]
    
    # Need to repeat prompts
    import random
    random.seed(seed)
    
    repeated_messages = []
    repeated_stats = []
    
    while len(repeated_messages) < num_prompts:
        # Shuffle each cycle for variety
        indices = list(range(len(messages_batch)))
        random.shuffle(indices)
        
        for idx in indices:
            if len(repeated_messages) >= num_prompts:
                break
            repeated_messages.append(messages_batch[idx])
            repeated_stats.append(token_stats_batch[idx])
    
    return repeated_messages, repeated_stats


def get_jsonl_stats(jsonl_file: Path) -> Dict:
    """Get statistics about a JSONL prompt file."""
    if not jsonl_file.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")
    
    records = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    total_tokens = [r["token_stats"]["total_prefill_tokens"] for r in records]
    wiki_tokens = [r["token_stats"]["wiki_context_tokens"] for r in records]
    history_tokens = [r["token_stats"]["history_tokens"] for r in records]
    
    return {
        "num_prompts": len(records),
        "total_tokens_mean": sum(total_tokens) / len(total_tokens) if total_tokens else 0,
        "total_tokens_min": min(total_tokens) if total_tokens else 0,
        "total_tokens_max": max(total_tokens) if total_tokens else 0,
        "wiki_tokens_mean": sum(wiki_tokens) / len(wiki_tokens) if wiki_tokens else 0,
        "history_tokens_mean": sum(history_tokens) / len(history_tokens) if history_tokens else 0,
    }


if __name__ == "__main__":
    # Test loading
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_jsonl_prompts.py <jsonl_file>")
        sys.exit(1)
    
    jsonl_path = Path(sys.argv[1])
    
    print(f"Loading prompts from: {jsonl_path}")
    print("=" * 80)
    
    # Get stats
    stats = get_jsonl_stats(jsonl_path)
    print(f"\nFile statistics:")
    print(f"  Prompts: {stats['num_prompts']}")
    print(f"  Total tokens: {stats['total_tokens_mean']:.1f} (avg), {stats['total_tokens_min']}-{stats['total_tokens_max']} (range)")
    print(f"  Wiki tokens: {stats['wiki_tokens_mean']:.1f} (avg)")
    print(f"  History tokens: {stats['history_tokens_mean']:.1f} (avg)")
    
    # Load first 3 prompts
    messages, token_stats = load_prompts_from_jsonl(jsonl_path, num_prompts=3)
    
    print(f"\nFirst 3 prompts:")
    print("=" * 80)
    
    for i, (msgs, stats) in enumerate(zip(messages, token_stats), 1):
        print(f"\nPrompt #{i}:")
        print(f"  Total tokens: {stats['total_prefill_tokens']}")
        print(f"  Messages: {len(msgs)}")
        for msg in msgs:
            role = msg['role']
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"    - {role}: {content}")
