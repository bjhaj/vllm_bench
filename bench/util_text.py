"""
Text and prompt building utilities for benchmarking.

Provides fast, deterministic generation of chat messages approximating
target token counts without requiring external tokenizers.
"""

import hashlib
from typing import List, Dict, Any


# Deterministic seed text for generating prompts
SEED_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In the realm of artificial intelligence and machine learning, "
    "large language models have revolutionized natural language processing. "
    "These sophisticated systems can understand context, generate coherent text, "
    "and perform various tasks with remarkable accuracy. "
)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using a simple heuristic.
    
    Approximation: ~0.75 tokens per word on average for English text.
    This is a rough estimate but good enough for benchmarking purposes.
    
    Args:
        text: Input text string
        
    Returns:
        Estimated token count
    """
    words = text.split()
    return int(len(words) * 0.75)


def build_text_approx_tokens(target_tokens: int, seed: int = 0) -> str:
    """
    Build deterministic text approximating target token count.
    
    Uses a seed to generate reproducible content by repeating and hashing
    seed text until approximate token count is reached.
    
    Args:
        target_tokens: Approximate number of tokens desired
        seed: Integer seed for deterministic generation
        
    Returns:
        Generated text string
    """
    if target_tokens <= 0:
        return ""
    
    # Hash the seed for variation
    seed_hash = hashlib.md5(f"seed_{seed}".encode()).hexdigest()[:8]
    prefix = f"Context_{seed_hash}: "
    
    # Start with prefix
    text_parts = [prefix]
    current_tokens = estimate_tokens(prefix)
    
    # Repeat seed text until we reach target
    repetitions = 0
    while current_tokens < target_tokens:
        chunk = SEED_TEXT
        
        # Add variation every few repetitions
        if repetitions > 0 and repetitions % 5 == 0:
            variation_seed = hashlib.md5(f"{seed}_{repetitions}".encode()).hexdigest()[:6]
            chunk = f"Section_{variation_seed}: {SEED_TEXT}"
        
        text_parts.append(chunk)
        current_tokens = estimate_tokens("".join(text_parts))
        repetitions += 1
    
    full_text = "".join(text_parts)
    
    # Trim to approximate target if we overshot significantly
    words = full_text.split()
    target_words = int(target_tokens / 0.75)
    
    if len(words) > target_words * 1.2:  # More than 20% over
        words = words[:target_words]
        full_text = " ".join(words)
    
    return full_text


def build_chat_messages(
    prefill_tokens: int,
    seed: int = 0,
    system_prompt: str | None = None
) -> List[Dict[str, str]]:
    """
    Build OpenAI chat format messages approximating target prefill tokens.
    
    Returns a list of messages in OpenAI chat completion format:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    
    Args:
        prefill_tokens: Target number of prefill tokens
        seed: Integer seed for deterministic generation
        system_prompt: Optional custom system prompt. If None, uses default.
        
    Returns:
        List of chat messages in OpenAI format
    """
    # Default system prompt (typically ~50-60 tokens)
    if system_prompt is None:
        system_prompt = (
            "You are a helpful AI assistant. Provide accurate, concise, and relevant "
            "responses based on the given context. Use the information provided to "
            "answer questions thoroughly."
        )
    
    system_tokens = estimate_tokens(system_prompt)
    
    # Reserve tokens for user message
    user_tokens = prefill_tokens - system_tokens
    
    if user_tokens < 0:
        # System prompt is longer than target, just use system prompt
        user_tokens = 10
    
    # Build user message content
    user_content = build_text_approx_tokens(user_tokens, seed=seed)
    
    # Add a question at the end for more realistic chat
    questions = [
        "Please summarize the key points from the above context.",
        "What are the main themes discussed in this text?",
        "Based on the context provided, what conclusions can be drawn?",
        "Analyze the information presented and provide insights.",
        "What is the significance of the details mentioned above?",
    ]
    question_idx = seed % len(questions)
    user_content = f"{user_content}\n\nQuestion: {questions[question_idx]}"
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_batch_chat_messages(
    prefill_tokens: int,
    num_messages: int,
    base_seed: int = 0
) -> List[List[Dict[str, str]]]:
    """
    Build a batch of deterministic chat messages for load testing.
    
    Each message will have approximately the same token count but
    different content (via different seeds).
    
    Args:
        prefill_tokens: Target number of prefill tokens per message
        num_messages: Number of unique messages to generate
        base_seed: Base seed for generation (each message gets base_seed + i)
        
    Returns:
        List of chat message lists
    """
    return [
        build_chat_messages(prefill_tokens, seed=base_seed + i)
        for i in range(num_messages)
    ]


if __name__ == "__main__":
    # Example usage and verification
    print("=== Text Builder Examples ===\n")
    
    # Example 1: Short text
    text_256 = build_text_approx_tokens(256)
    print(f"Target: 256 tokens")
    print(f"Estimated: {estimate_tokens(text_256)} tokens")
    print(f"Text preview: {text_256[:100]}...\n")
    
    # Example 2: Chat messages
    messages = build_chat_messages(prefill_tokens=512, seed=42)
    total_tokens = sum(estimate_tokens(msg["content"]) for msg in messages)
    print(f"Target: 512 tokens")
    print(f"Estimated: {total_tokens} tokens")
    print(f"Messages: {len(messages)}")
    for msg in messages:
        print(f"  - {msg['role']}: {len(msg['content'])} chars, ~{estimate_tokens(msg['content'])} tokens")
    print()
    
    # Example 3: Batch generation
    batch = build_batch_chat_messages(prefill_tokens=1024, num_messages=3, base_seed=100)
    print(f"Generated {len(batch)} messages with ~1024 tokens each")
    for i, msgs in enumerate(batch):
        total = sum(estimate_tokens(msg["content"]) for msg in msgs)
        print(f"  Message {i}: ~{total} tokens")
