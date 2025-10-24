#!/usr/bin/env python3
"""
Inspect and display the prompts generated for benchmark scenarios.

Usage:
    python inspect_prompts.py
    python inspect_prompts.py --scenario rag_medium
    python inspect_prompts.py --show 10
    python inspect_prompts.py --save-json prompts.json
"""

import sys
import argparse
import json
from pathlib import Path

# Add bench directory to path
sys.path.insert(0, 'bench')


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough: ~0.75 tokens per word)."""
    return int(len(text.split()) * 0.75)


def display_prompt_preview(messages_batch, num_show=5):
    """Display preview of prompts."""
    print(f"\n📋 Showing {min(num_show, len(messages_batch))} of {len(messages_batch)} prompts:\n")
    print("=" * 80)
    
    for i in range(min(num_show, len(messages_batch))):
        messages = messages_batch[i]
        
        print(f"\n🔹 Prompt #{i+1}:")
        print("-" * 80)
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            tokens = estimate_tokens(content)
            
            print(f"\n  Role: {role}")
            print(f"  Length: {len(content)} chars")
            print(f"  Estimated tokens: {tokens}")
            print(f"  Content preview:")
            print(f"    {content[:200]}...")
            if len(content) > 200:
                print(f"    ... [+{len(content) - 200} more characters]")
        
        print("-" * 80)


def display_statistics(messages_batch):
    """Display statistics about the prompts."""
    print("\n📊 Prompt Statistics:")
    print("=" * 80)
    
    # Calculate per-message token counts
    token_counts = []
    for messages in messages_batch:
        total_tokens = sum(estimate_tokens(msg["content"]) for msg in messages)
        token_counts.append(total_tokens)
    
    # Calculate statistics
    avg_tokens = sum(token_counts) / len(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    
    # Character statistics
    all_lengths = []
    for messages in messages_batch:
        for msg in messages:
            all_lengths.append(len(msg["content"]))
    
    avg_chars = sum(all_lengths) / len(all_lengths)
    min_chars = min(all_lengths)
    max_chars = max(all_lengths)
    
    print(f"  Total prompts: {len(messages_batch)}")
    print(f"\n  Token counts:")
    print(f"    Average: {avg_tokens:.1f} tokens")
    print(f"    Min: {min_tokens} tokens")
    print(f"    Max: {max_tokens} tokens")
    print(f"    Range: {max_tokens - min_tokens} tokens")
    
    print(f"\n  Character counts:")
    print(f"    Average: {avg_chars:.1f} chars")
    print(f"    Min: {min_chars} chars")
    print(f"    Max: {max_chars} chars")
    
    # Distribution
    print(f"\n  Distribution:")
    buckets = [0, 0, 0, 0]  # <90%, 90-95%, 95-105%, >105% of target
    target = token_counts[0]  # Use first as reference
    
    for count in token_counts:
        pct = (count / target) * 100
        if pct < 90:
            buckets[0] += 1
        elif pct < 95:
            buckets[1] += 1
        elif pct <= 105:
            buckets[2] += 1
        else:
            buckets[3] += 1
    
    print(f"    < 90% of target: {buckets[0]} ({buckets[0]/len(token_counts)*100:.1f}%)")
    print(f"    90-95% of target: {buckets[1]} ({buckets[1]/len(token_counts)*100:.1f}%)")
    print(f"    95-105% of target: {buckets[2]} ({buckets[2]/len(token_counts)*100:.1f}%)")
    print(f"    > 105% of target: {buckets[3]} ({buckets[3]/len(token_counts)*100:.1f}%)")
    
    print("=" * 80)


def save_prompts_json(messages_batch, output_file):
    """Save prompts to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(messages_batch, f, indent=2)
    
    print(f"\n✅ All {len(messages_batch)} prompts saved to: {output_path}")


def save_prompts_text(messages_batch, output_file):
    """Save prompts to human-readable text file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Generated Prompts ({len(messages_batch)} total)\n")
        f.write("=" * 80 + "\n\n")
        
        for i, messages in enumerate(messages_batch):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"PROMPT #{i+1}\n")
            f.write(f"{'=' * 80}\n\n")
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                tokens = estimate_tokens(content)
                
                f.write(f"Role: {role}\n")
                f.write(f"Estimated tokens: {tokens}\n")
                f.write(f"Content:\n")
                f.write(f"{'-' * 80}\n")
                f.write(f"{content}\n")
                f.write(f"{'-' * 80}\n\n")
    
    print(f"\n✅ All {len(messages_batch)} prompts saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect benchmark prompts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Inspect 500 prompts from Dolly dataset (default)
  python inspect_prompts.py
  
  # Show first 10 prompts
  python inspect_prompts.py --show 10
  
  # Try different dataset
  python inspect_prompts.py --dataset openassistant
  
  # Longer prompts for RAG scenario
  python inspect_prompts.py --prefill 4000 --num-requests 200
  
  # Save to JSON
  python inspect_prompts.py --save-json results/prompts_dolly.json
  
  # Save to readable text file
  python inspect_prompts.py --save-text results/prompts_dolly.txt
        '''
    )
    
    parser.add_argument('--scenario', default='short_chat', help='Scenario name (for seed)')
    parser.add_argument('--prefill', type=int, default=256, help='Target prefill tokens')
    parser.add_argument('--num-requests', type=int, default=500, help='Number of prompts to generate')
    parser.add_argument('--show', type=int, default=5, help='Number of prompts to display')
    parser.add_argument('--save-json', type=str, help='Save prompts to JSON file')
    parser.add_argument('--save-text', type=str, help='Save prompts to text file')
    parser.add_argument('--stats-only', action='store_true', help='Only show statistics')
    parser.add_argument('--dataset', default='dolly', help='Dataset to use (dolly, sharegpt/alpaca, openassistant)')
    
    args = parser.parse_args()
    
    # Load real prompts from dataset
    print(f"\n🔧 Loading {args.num_requests} real prompts from {args.dataset} dataset...")
    print(f"   Target: ~{args.prefill} tokens per prompt")
    
    from load_real_prompts import load_real_prompts
    messages_batch = load_real_prompts(
        num_prompts=args.num_requests,
        dataset_name=args.dataset,
        min_tokens=50,  # Fixed minimum for consistency
        max_tokens=max(500, args.prefill * 2),  # Flexible maximum
        seed=hash(args.scenario) % 10000
    )
    
    if len(messages_batch) < args.num_requests:
        print(f"\n⚠️  Warning: Only found {len(messages_batch)} prompts (requested {args.num_requests})")
        print(f"   Try: --dataset dolly (has more prompts) or reduce --num-requests")
    else:
        print(f"✓ Loaded {len(messages_batch)} real prompts")
    
    # Display statistics
    display_statistics(messages_batch)
    
    # Display prompt previews (unless stats-only)
    if not args.stats_only:
        display_prompt_preview(messages_batch, num_show=args.show)
    
    # Save to files if requested
    if args.save_json:
        save_prompts_json(messages_batch, args.save_json)
    
    if args.save_text:
        save_prompts_text(messages_batch, args.save_text)
    
    # Print summary
    print("\n" + "=" * 80)
    print("✨ Inspection complete!")
    
    if not args.save_json and not args.save_text:
        print("\n💡 Tip: Use --save-json or --save-text to save prompts to a file")
        print("   Example: python inspect_prompts.py --save-json results/prompts.json")
    
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
