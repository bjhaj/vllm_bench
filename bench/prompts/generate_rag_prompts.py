#!/usr/bin/env python3
"""
Generate realistic RAG prompts with conversation history.

Combines:
- Wikipedia paragraphs (concatenated to target token length)
- Dolly questions (appended to Wikipedia context)
- Conversation history (simulates multi-turn dialogue)
- Precise token counting (using facebook/opt-1.3b tokenizer)

Output: JSONL file with pre-generated prompts ready for benchmarking
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer


# Conversation history templates (realistic assistant responses)
ASSISTANT_RESPONSES = [
    "I understand. Let me help you with that.",
    "That's a great question. Here's what I found:",
    "Based on the information available, ",
    "I can explain that for you.",
    "Let me break that down:",
    "Here's a summary:",
    "That's an interesting topic. ",
    "I'd be happy to clarify that.",
    "According to my knowledge,",
    "Let me provide more details:",
]

USER_FOLLOWUPS = [
    "Can you explain that more?",
    "Tell me more about that.",
    "What else should I know?",
    "Can you clarify?",
    "Go on.",
    "Interesting. What about",
    "How does that relate to",
    "Can you give an example?",
    "What's the context here?",
    "Why is that important?",
]


def load_tokenizer(model_name: str = "facebook/opt-1.3b"):
    """Load tokenizer for precise token counting."""
    print(f"📥 Loading tokenizer: {model_name}")
    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(tokenizer, text: str) -> int:
    """Accurately count tokens using the model's tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=True))


def load_wikipedia_paragraphs(num_articles: int = 10000, seed: int = 42) -> List[str]:
    """Load paragraphs from Wikipedia dataset (using Dolly context as fallback)."""
    print(f"📥 Loading text corpus for RAG context...")
    
    # Use Dolly dataset's context fields as Wikipedia-like text
    # This avoids Wikipedia dataset compatibility issues
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    random.seed(seed)
    
    paragraphs = []
    for item in dataset:
        # Get context field (often contains informative text)
        context = item.get("context", "").strip()
        if context and len(context) > 100:
            paragraphs.append(context)
        
        # Also use instruction+output as pseudo-articles
        instruction = item.get("instruction", "").strip()
        output = item.get("output", "").strip()
        
        if output and len(output) > 100:
            paragraphs.append(output)
        
        if instruction and output:
            combined = f"{instruction}\n\n{output}"
            if len(combined) > 200:
                paragraphs.append(combined)
    
    # Shuffle and deduplicate
    paragraphs = list(set(paragraphs))
    random.shuffle(paragraphs)
    
    # Limit to reasonable size
    paragraphs = paragraphs[:min(len(paragraphs), num_articles * 5)]
    
    print(f"✅ Loaded {len(paragraphs)} text chunks from Dolly dataset")
    return paragraphs


def load_dolly_questions(num_questions: int = 5000, seed: int = 42) -> List[str]:
    """Load questions from Dolly dataset."""
    print(f"📥 Loading Dolly questions...")
    
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    random.seed(seed)
    
    questions = []
    for item in dataset:
        instruction = item.get("instruction", "").strip()
        if instruction and len(instruction) > 10:
            # Add question mark if not present
            if not instruction.endswith("?"):
                instruction = instruction + "?"
            questions.append(instruction)
    
    random.shuffle(questions)
    questions = questions[:num_questions]
    
    print(f"✅ Loaded {len(questions)} questions from Dolly")
    return questions


def build_wikipedia_context(
    tokenizer,
    paragraphs: List[str],
    target_tokens: int,
    tolerance: float = 0.1
) -> Tuple[str, int]:
    """
    Concatenate random Wikipedia paragraphs to reach target token count.
    
    Returns:
        (context_text, actual_token_count)
    """
    min_tokens = int(target_tokens * (1 - tolerance))
    max_tokens = int(target_tokens * (1 + tolerance))
    
    # Randomly sample paragraphs
    selected = random.sample(paragraphs, min(len(paragraphs), 50))
    
    context_parts = []
    current_tokens = 0
    
    for para in selected:
        para_tokens = count_tokens(tokenizer, para)
        
        if current_tokens + para_tokens > max_tokens:
            # Check if we're in acceptable range
            if current_tokens >= min_tokens:
                break
            # Try to fit partial paragraph if needed
            continue
        
        context_parts.append(para)
        current_tokens += para_tokens
        
        if current_tokens >= min_tokens:
            break
    
    context_text = "\n\n".join(context_parts)
    actual_tokens = count_tokens(tokenizer, context_text)
    
    return context_text, actual_tokens


def build_conversation_history(
    tokenizer,
    num_turns: int,
    max_history_tokens: int = 200
) -> List[Dict[str, str]]:
    """
    Generate realistic conversation history.
    
    Returns:
        List of message dicts with role and content
    """
    if num_turns == 0:
        return []
    
    history = []
    current_tokens = 0
    
    for i in range(num_turns):
        # User message
        user_msg = random.choice(USER_FOLLOWUPS)
        user_tokens = count_tokens(tokenizer, user_msg)
        
        # Assistant response
        assistant_msg = random.choice(ASSISTANT_RESPONSES)
        assistant_tokens = count_tokens(tokenizer, assistant_msg)
        
        turn_tokens = user_tokens + assistant_tokens
        
        if current_tokens + turn_tokens > max_history_tokens:
            break
        
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
        
        current_tokens += turn_tokens
    
    return history


def generate_rag_prompt(
    tokenizer,
    paragraphs: List[str],
    questions: List[str],
    target_prefill_tokens: int,
    history_turns: int = 0,
    question_idx: int = 0
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Generate a single RAG prompt with Wikipedia context + Dolly question + history.
    
    Returns:
        (messages, token_stats)
    """
    # Reserve tokens for question and history
    question = questions[question_idx % len(questions)]
    question_tokens = count_tokens(tokenizer, question)
    
    history_tokens = 0
    if history_turns > 0:
        # Estimate history tokens (will refine below)
        history_tokens = history_turns * 30  # Rough estimate
    
    # Target tokens for Wikipedia context
    context_target_tokens = target_prefill_tokens - question_tokens - history_tokens - 50  # Buffer
    
    # Build Wikipedia context
    wiki_context, wiki_tokens = build_wikipedia_context(
        tokenizer,
        paragraphs,
        max(100, context_target_tokens),
        tolerance=0.15
    )
    
    # Build conversation history
    history = build_conversation_history(
        tokenizer,
        history_turns,
        max_history_tokens=min(300, target_prefill_tokens // 4)
    )
    
    # Construct final message
    system_msg = "You are a helpful AI assistant. Answer questions based on the provided context."
    system_tokens = count_tokens(tokenizer, system_msg)
    
    # Format user message with context and question
    user_content = f"Context:\n{wiki_context}\n\nQuestion: {question}"
    user_tokens = count_tokens(tokenizer, user_content)
    
    # Build full message list
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    
    # Calculate total tokens
    total_tokens = system_tokens + sum(count_tokens(tokenizer, msg["content"]) for msg in history) + user_tokens
    
    token_stats = {
        "system_tokens": system_tokens,
        "history_tokens": sum(count_tokens(tokenizer, msg["content"]) for msg in history),
        "wiki_context_tokens": wiki_tokens,
        "question_tokens": question_tokens,
        "total_prefill_tokens": total_tokens,
    }
    
    return messages, token_stats


def generate_rag_dataset(
    num_prompts: int,
    target_prefill_tokens: int,
    history_turns: int,
    output_file: Path,
    tokenizer_name: str = "facebook/opt-1.3b",
    seed: int = 42
):
    """Generate full RAG dataset and save to JSONL."""
    print(f"\n{'='*80}")
    print(f"Generating {num_prompts} RAG prompts")
    print(f"  Target prefill: {target_prefill_tokens} tokens")
    print(f"  History turns: {history_turns}")
    print(f"  Output: {output_file}")
    print(f"{'='*80}\n")
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_name)
    
    # Load Wikipedia paragraphs
    paragraphs = load_wikipedia_paragraphs(num_articles=10000, seed=seed)
    
    # Load Dolly questions
    questions = load_dolly_questions(num_questions=5000, seed=seed)
    
    # Generate prompts
    print(f"\n🔧 Generating {num_prompts} RAG prompts...")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for i in range(num_prompts):
            messages, token_stats = generate_rag_prompt(
                tokenizer,
                paragraphs,
                questions,
                target_prefill_tokens,
                history_turns,
                question_idx=i
            )
            
            # Write to JSONL
            record = {
                "prompt_id": i,
                "messages": messages,
                "token_stats": token_stats,
                "target_prefill_tokens": target_prefill_tokens,
                "history_turns": history_turns,
            }
            
            f.write(json.dumps(record) + "\n")
            
            if (i + 1) % 100 == 0:
                avg_tokens = sum(count_tokens(tokenizer, json.dumps(messages)) for _ in [record]) / 1
                print(f"  Generated {i+1}/{num_prompts} prompts (avg tokens: {token_stats['total_prefill_tokens']})")
    
    print(f"\n✅ Saved {num_prompts} prompts to {output_file}")
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"Dataset Statistics:")
    print(f"{'='*80}")
    
    with open(output_file, 'r') as f:
        records = [json.loads(line) for line in f]
    
    total_tokens = [r["token_stats"]["total_prefill_tokens"] for r in records]
    wiki_tokens = [r["token_stats"]["wiki_context_tokens"] for r in records]
    history_tokens = [r["token_stats"]["history_tokens"] for r in records]
    
    print(f"  Total prompts: {len(records)}")
    print(f"  Target prefill: {target_prefill_tokens} tokens")
    print(f"\n  Actual prefill tokens:")
    print(f"    Mean: {sum(total_tokens) / len(total_tokens):.1f}")
    print(f"    Min:  {min(total_tokens)}")
    print(f"    Max:  {max(total_tokens)}")
    print(f"\n  Wikipedia context tokens:")
    print(f"    Mean: {sum(wiki_tokens) / len(wiki_tokens):.1f}")
    print(f"    Min:  {min(wiki_tokens)}")
    print(f"    Max:  {max(wiki_tokens)}")
    print(f"\n  History tokens:")
    print(f"    Mean: {sum(history_tokens) / len(history_tokens):.1f}")
    print(f"    Min:  {min(history_tokens)}")
    print(f"    Max:  {max(history_tokens)}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate RAG prompts with Wikipedia context + Dolly questions + conversation history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate 500 prompts with 1.7k tokens, 2 history turns
  python generate_rag_prompts.py --num-prompts 500 --target-tokens 1700 --history-turns 2
  
  # Generate 1000 prompts with 1.5k tokens, no history
  python generate_rag_prompts.py --num-prompts 1000 --target-tokens 1500 --history-turns 0
  
  # Custom output file
  python generate_rag_prompts.py -n 200 -t 1900 --history-turns 3 -o data/rag_heavy.jsonl
        '''
    )
    
    parser.add_argument('-n', '--num-prompts', type=int, default=500, help='Number of prompts to generate')
    parser.add_argument('-t', '--target-tokens', type=int, default=1700, help='Target prefill tokens (1500-1900 recommended)')
    parser.add_argument('--history-turns', type=int, default=2, help='Number of conversation history turns')
    parser.add_argument('-o', '--output', type=str, default='../../data/rag_prompts.jsonl', help='Output JSONL file')
    parser.add_argument('--tokenizer', type=str, default='facebook/opt-1.3b', help='Tokenizer model name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    generate_rag_dataset(
        num_prompts=args.num_prompts,
        target_prefill_tokens=args.target_tokens,
        history_turns=args.history_turns,
        output_file=output_path,
        tokenizer_name=args.tokenizer,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
