"""
Load real human prompts from Hugging Face datasets for benchmarking.

This module provides functions to load realistic prompts from open datasets
like ShareGPT, Dolly, OpenAssistant, etc. instead of synthetic generation.
"""

from typing import List, Dict, Optional
import random


def load_sharegpt_prompts(
    num_prompts: int = 500,
    min_tokens: int = 50,
    max_tokens: int = 1000,
    seed: int = 42
) -> List[List[Dict[str, str]]]:
    """
    Load real instruction prompts from Alpaca dataset.
    
    Args:
        num_prompts: Number of prompts to load
        min_tokens: Minimum estimated tokens per prompt
        max_tokens: Maximum estimated tokens per prompt
        seed: Random seed for reproducibility
        
    Returns:
        List of message lists in OpenAI chat format
    """
    from datasets import load_dataset
    
    print(f"📥 Loading Alpaca dataset...")
    
    # Load Alpaca dataset (high-quality instruction following)
    dataset = load_dataset(
        "tatsu-lab/alpaca",
        split="train"
    )
    
    random.seed(seed)
    
    prompts = []
    seen_prompts = set()
    
    print(f"🔍 Filtering for {num_prompts} prompts between {min_tokens}-{max_tokens} tokens...")
    
    # Shuffle dataset indices
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    for idx in indices:
        if len(prompts) >= num_prompts:
            break
            
        try:
            item = dataset[idx]
            instruction = item.get("instruction", "").strip()
            input_text = item.get("input", "").strip()
            
            # Combine instruction and input if available
            if input_text:
                user_msg = f"{instruction}\n\nInput: {input_text}"
            else:
                user_msg = instruction
            
            if not user_msg or len(user_msg) < 10:
                continue
            
            # Avoid duplicates
            if user_msg in seen_prompts:
                continue
            
            # Estimate tokens (rough: ~0.75 tokens per word)
            estimated_tokens = len(user_msg.split()) * 0.75
            
            if min_tokens <= estimated_tokens <= max_tokens:
                # Format as OpenAI chat messages
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant."
                    },
                    {
                        "role": "user",
                        "content": user_msg
                    }
                ]
                
                prompts.append(messages)
                seen_prompts.add(user_msg)
                
        except Exception as e:
            continue
    
    print(f"✅ Loaded {len(prompts)} real prompts from Alpaca")
    return prompts


def load_dolly_prompts(
    num_prompts: int = 500,
    min_tokens: int = 50,
    max_tokens: int = 1000,
    seed: int = 42
) -> List[List[Dict[str, str]]]:
    """
    Load prompts from Databricks Dolly dataset.
    
    High-quality instruction-following dataset with diverse tasks.
    """
    from datasets import load_dataset
    
    print(f"📥 Loading Dolly dataset...")
    
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    random.seed(seed)
    
    prompts = []
    seen_prompts = set()
    
    print(f"🔍 Filtering for {num_prompts} prompts between {min_tokens}-{max_tokens} tokens...")
    
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    for idx in indices:
        if len(prompts) >= num_prompts:
            break
            
        try:
            item = dataset[idx]
            instruction = item.get("instruction", "").strip()
            context = item.get("context", "").strip()
            
            # Combine instruction and context if available
            if context:
                user_msg = f"{instruction}\n\nContext: {context}"
            else:
                user_msg = instruction
            
            if not user_msg or len(user_msg) < 10:
                continue
            
            # Avoid duplicates
            if user_msg in seen_prompts:
                continue
            
            # Estimate tokens
            estimated_tokens = len(user_msg.split()) * 0.75
            
            if min_tokens <= estimated_tokens <= max_tokens:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant."
                    },
                    {
                        "role": "user",
                        "content": user_msg
                    }
                ]
                
                prompts.append(messages)
                seen_prompts.add(user_msg)
                
        except Exception as e:
            continue
    
    print(f"✅ Loaded {len(prompts)} real prompts from Dolly")
    return prompts


def load_openassistant_prompts(
    num_prompts: int = 500,
    min_tokens: int = 50,
    max_tokens: int = 1000,
    seed: int = 42
) -> List[List[Dict[str, str]]]:
    """
    Load prompts from OpenAssistant Conversations dataset.
    
    High-quality conversational dataset with human feedback.
    """
    from datasets import load_dataset
    
    print(f"📥 Loading OpenAssistant dataset...")
    
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    
    random.seed(seed)
    
    prompts = []
    seen_prompts = set()
    
    print(f"🔍 Filtering for {num_prompts} prompts between {min_tokens}-{max_tokens} tokens...")
    
    # Filter for initial prompts (not replies)
    initial_prompts = [
        item for item in dataset 
        if item.get("role") == "prompter" and item.get("parent_id") is None
    ]
    
    indices = list(range(len(initial_prompts)))
    random.shuffle(indices)
    
    for idx in indices:
        if len(prompts) >= num_prompts:
            break
            
        try:
            item = initial_prompts[idx]
            user_msg = item.get("text", "").strip()
            
            if not user_msg or len(user_msg) < 10:
                continue
            
            # Avoid duplicates
            if user_msg in seen_prompts:
                continue
            
            # Estimate tokens
            estimated_tokens = len(user_msg.split()) * 0.75
            
            if min_tokens <= estimated_tokens <= max_tokens:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant."
                    },
                    {
                        "role": "user",
                        "content": user_msg
                    }
                ]
                
                prompts.append(messages)
                seen_prompts.add(user_msg)
                
        except Exception as e:
            continue
    
    print(f"✅ Loaded {len(prompts)} real prompts from OpenAssistant")
    return prompts


def load_real_prompts(
    num_prompts: int = 500,
    dataset_name: str = "sharegpt",
    min_tokens: int = 50,
    max_tokens: int = 1000,
    seed: int = 42
) -> List[List[Dict[str, str]]]:
    """
    Load real prompts from specified dataset.
    
    Args:
        num_prompts: Number of prompts to load
        dataset_name: Dataset to use ("sharegpt", "dolly", "openassistant")
        min_tokens: Minimum estimated tokens per prompt
        max_tokens: Maximum estimated tokens per prompt
        seed: Random seed for reproducibility
        
    Returns:
        List of message lists in OpenAI chat format
    """
    loaders = {
        "sharegpt": load_sharegpt_prompts,
        "dolly": load_dolly_prompts,
        "openassistant": load_openassistant_prompts,
    }
    
    if dataset_name.lower() not in loaders:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Choose from: {', '.join(loaders.keys())}"
        )
    
    loader = loaders[dataset_name.lower()]
    return loader(
        num_prompts=num_prompts,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        seed=seed
    )


if __name__ == "__main__":
    # Quick test
    import sys
    
    dataset = sys.argv[1] if len(sys.argv) > 1 else "sharegpt"
    
    print(f"\n{'='*80}")
    print(f"Testing {dataset} dataset loader")
    print(f"{'='*80}\n")
    
    prompts = load_real_prompts(
        num_prompts=5,
        dataset_name=dataset,
        min_tokens=50,
        max_tokens=500
    )
    
    print(f"\n{'='*80}")
    print(f"Sample prompts:")
    print(f"{'='*80}\n")
    
    for i, messages in enumerate(prompts, 1):
        print(f"\nPrompt #{i}:")
        print("-" * 80)
        for msg in messages:
            print(f"  {msg['role']}: {msg['content'][:200]}...")
        print("-" * 80)
