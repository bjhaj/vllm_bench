#!/usr/bin/env python3
"""
Inspect model responses for RAG prompts.

Loads prompts from JSONL and sends them to vLLM server to see actual outputs.
Useful for debugging and quality analysis.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# Load environment variables
load_dotenv()

console = Console()


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.0
) -> Dict:
    """Send a single request and return the full response."""
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False  # Non-streaming for easier inspection
    }
    
    try:
        response = await client.post(
            f"{base_url}/chat/completions",
            json=payload,
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def display_prompt_and_response(
    prompt_idx: int,
    messages: List[Dict[str, str]],
    response_data: Dict,
    token_stats: Dict = None
):
    """Display a prompt and its response in a nice format."""
    
    console.print(f"\n{'='*80}")
    console.print(f"[bold cyan]Prompt #{prompt_idx}[/bold cyan]")
    console.print(f"{'='*80}\n")
    
    # Display token stats if available
    if token_stats:
        console.print("[yellow]Token Statistics:[/yellow]")
        console.print(f"  Total prefill: {token_stats.get('total_prefill_tokens', 'N/A')}")
        console.print(f"  Context: {token_stats.get('wiki_context_tokens', 'N/A')}")
        console.print(f"  History: {token_stats.get('history_tokens', 'N/A')}\n")
    
    # Display the conversation
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            console.print(Panel(
                content,
                title="[bold blue]System[/bold blue]",
                border_style="blue"
            ))
        elif role == "user":
            # Truncate long context for readability
            if len(content) > 500:
                display_content = content[:250] + "\n\n[... truncated ...]\n\n" + content[-250:]
            else:
                display_content = content
                
            console.print(Panel(
                display_content,
                title="[bold green]User[/bold green]",
                border_style="green"
            ))
        elif role == "assistant":
            console.print(Panel(
                content,
                title="[bold magenta]Assistant (History)[/bold magenta]",
                border_style="magenta"
            ))
    
    # Display the model's response
    if "error" in response_data:
        console.print(Panel(
            f"[red]Error: {response_data['error']}[/red]",
            title="[bold red]Model Response (Error)[/bold red]",
            border_style="red"
        ))
    else:
        try:
            response_text = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage", {})
            
            console.print(Panel(
                response_text,
                title=f"[bold cyan]Model Response[/bold cyan] (tokens: {usage.get('completion_tokens', 'N/A')})",
                border_style="cyan"
            ))
            
            # Display usage statistics
            console.print("\n[yellow]API Usage:[/yellow]")
            console.print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            console.print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            console.print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")
        except (KeyError, IndexError) as e:
            console.print(Panel(
                f"[red]Failed to parse response: {e}[/red]\n\n{json.dumps(response_data, indent=2)}",
                title="[bold red]Raw Response[/bold red]",
                border_style="red"
            ))


async def inspect_prompts(
    prompt_file: Path,
    num_prompts: int = 5,
    start_idx: int = 0,
    base_url: str = "http://localhost:8000/v1",
    model: str = None,
    max_tokens: int = 256,
    temperature: float = 0.0
):
    """Load prompts and display model responses."""
    
    if model is None:
        model = os.getenv("MODEL", "facebook/opt-1.3b")
    
    console.print(f"[bold green]Inspecting Model Responses[/bold green]")
    console.print(f"Model: {model}")
    console.print(f"Base URL: {base_url}")
    console.print(f"Prompt file: {prompt_file}")
    console.print(f"Prompts to inspect: {num_prompts} (starting at index {start_idx})\n")
    
    # Load prompts
    if not prompt_file.exists():
        console.print(f"[red]Error: Prompt file not found: {prompt_file}[/red]")
        return
    
    prompts = []
    with open(prompt_file, 'r') as f:
        for line in f:
            prompts.append(json.loads(line))
    
    console.print(f"[green]✓ Loaded {len(prompts)} total prompts[/green]\n")
    
    # Select prompts to inspect
    end_idx = min(start_idx + num_prompts, len(prompts))
    selected_prompts = prompts[start_idx:end_idx]
    
    # Send requests
    async with httpx.AsyncClient() as client:
        for i, prompt_data in enumerate(selected_prompts, start=start_idx):
            messages = prompt_data["messages"]
            token_stats = {
                "total_prefill_tokens": prompt_data.get("total_prefill_tokens"),
                "wiki_context_tokens": prompt_data.get("wiki_context_tokens"),
                "history_tokens": prompt_data.get("history_tokens")
            }
            
            console.print(f"[cyan]Sending request {i+1}/{end_idx}...[/cyan]")
            
            response_data = await send_request(
                client,
                base_url,
                model,
                messages,
                max_tokens,
                temperature
            )
            
            display_prompt_and_response(i, messages, response_data, token_stats)
            
            # Pause between requests
            if i < end_idx - 1:
                await asyncio.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect model responses for RAG prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect first 5 prompts
  python inspect_responses.py
  
  # Inspect prompts 10-20
  python inspect_responses.py --start 10 --num 10
  
  # Use custom prompt file
  python inspect_responses.py --file data/rag_prompts_test.jsonl
  
  # Higher temperature for more creative responses
  python inspect_responses.py --temperature 0.7
        """
    )
    
    parser.add_argument(
        "--file", "-f",
        type=Path,
        default=Path(__file__).parent.parent.parent / "data/rag_prompts_1500.jsonl",
        help="Path to JSONL prompt file (default: data/rag_prompts_1500.jsonl)"
    )
    
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=5,
        help="Number of prompts to inspect (default: 5)"
    )
    
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        help="Starting index (default: 0)"
    )
    
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM server base URL (default: http://localhost:8000/v1)"
    )
    
    parser.add_argument(
        "--model",
        help="Model name (defaults to MODEL from .env)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(inspect_prompts(
        prompt_file=args.file,
        num_prompts=args.num,
        start_idx=args.start,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    ))


if __name__ == "__main__":
    main()
