"""
Async load generator for vLLM benchmarking.

Sends concurrent requests to OpenAI-compatible API and measures:
- TTFT (Time To First Token): latency until first streamed chunk
- Total latency: end-to-end request time
- Token counts: prompt and completion tokens
"""

import asyncio
import csv
import sys
import time
import random
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

import httpx
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()


console = Console()


@dataclass
class RequestResult:
    """Result of a single benchmark request."""
    run_id: str
    qid: int
    prefill_tokens: int  # Target/estimated prefill tokens (from config)
    max_new_tokens: int
    ttft_ms: float
    latency_ms: float
    prompt_tokens: int  # Actual prompt tokens (from API response)
    completion_tokens: int  # Actual completion tokens (from API response)
    actual_prefill_tokens: int  # Actual tokenized prefill count (from JSONL token_stats)
    error: str
    timestamp: str


class AsyncLoadGenerator:
    """Async load generator for vLLM benchmarking."""
    
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        timeout: Optional[float] = None
    ):
        """
        Initialize the load generator.
        
        Args:
            base_url: Base URL for the API (e.g., http://localhost:8000/v1)
            model: Model name to use
            temperature: Sampling temperature
            timeout: Request timeout in seconds (None = no timeout, let server dictate)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        # Use very high timeout or None to let server dictate timing
        timeout_config = httpx.Timeout(
            timeout=self.timeout if self.timeout else None,
            connect=30.0,  # Keep reasonable connect timeout
            read=self.timeout if self.timeout else None,
            write=30.0,
            pool=30.0
        )
        self.client = httpx.AsyncClient(timeout=timeout_config)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()
    
    async def send_chat_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        qid: int,
        stream: bool = True
    ) -> RequestResult:
        """
        Send a single chat completion request and measure timings.
        
        This method is fully resilient and ALWAYS returns a RequestResult,
        even on complete failures. Every request produces a result row.
        
        Args:
            messages: Chat messages in OpenAI format
            max_tokens: Maximum tokens to generate
            qid: Query ID for tracking
            stream: Whether to stream the response
            
        Returns:
            RequestResult with timing and token information (never raises)
        """
        request_start = time.perf_counter()
        timestamp = datetime.utcnow().isoformat()
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }
        
        ttft_ms = -1.0
        latency_ms = -1.0
        prompt_tokens = 0
        completion_tokens = 0
        error = ""
        
        # Wrap entire request in try/except to guarantee a result
        try:
            url = f"{self.base_url}/chat/completions"
            
            if stream:
                # Streaming request with robust SSE parsing
                first_chunk_time = None
                chunk_count = 0
                last_chunk_data = None
                
                try:
                    async with self.client.stream("POST", url, json=payload) as response:
                        response.raise_for_status()
                        
                        async for line in response.aiter_lines():
                            line = line.strip()
                            
                            # Skip empty lines
                            if not line:
                                continue
                            
                            # Check for stream end marker
                            if line == "data: [DONE]":
                                break
                            
                            # Process SSE data lines
                            if line.startswith("data: "):
                                data_content = line[6:]  # Remove "data: " prefix
                                
                                # Try to parse JSON chunk
                                try:
                                    import json
                                    chunk_data = json.loads(data_content)
                                    
                                    # Record TTFT on first valid JSON chunk
                                    if first_chunk_time is None:
                                        first_chunk_time = time.perf_counter()
                                        ttft_ms = (first_chunk_time - request_start) * 1000
                                    
                                    chunk_count += 1
                                    
                                    # Keep last chunk for usage extraction
                                    last_chunk_data = chunk_data
                                    
                                except json.JSONDecodeError as e:
                                    # Log but continue - some chunks might be malformed
                                    # Don't fail the entire request for one bad chunk
                                    if chunk_count == 0:
                                        # If we can't parse the very first chunk, that's more serious
                                        error = f"JSON decode error on first chunk: {str(e)[:50]}"
                                    continue
                                except Exception as e:
                                    # Unexpected error parsing chunk - log but continue
                                    continue
                    
                    # Extract usage from last chunk if available
                    if last_chunk_data and "usage" in last_chunk_data:
                        try:
                            usage = last_chunk_data["usage"]
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)
                        except (KeyError, TypeError):
                            # Usage field exists but malformed - not critical
                            pass
                    
                    request_end = time.perf_counter()
                    latency_ms = (request_end - request_start) * 1000
                    
                    # If TTFT wasn't recorded, something went wrong
                    if ttft_ms < 0:
                        if chunk_count == 0:
                            error = "No valid chunks received"
                        else:
                            error = "TTFT not recorded despite chunks"
                
                except httpx.HTTPStatusError as e:
                    # Re-raise to be caught by outer handler
                    raise
                except Exception as e:
                    # Catch any streaming-specific errors
                    error = f"Stream error: {type(e).__name__}: {str(e)[:100]}"
                    request_end = time.perf_counter()
                    latency_ms = (request_end - request_start) * 1000
            
            else:
                # Non-streaming request
                response = await self.client.post(url, json=payload)
                response.raise_for_status()
                
                request_end = time.perf_counter()
                latency_ms = (request_end - request_start) * 1000
                ttft_ms = latency_ms  # For non-streaming, TTFT = total latency
                
                result = response.json()
                if "usage" in result:
                    prompt_tokens = result["usage"].get("prompt_tokens", 0)
                    completion_tokens = result["usage"].get("completion_tokens", 0)
        
        except httpx.HTTPStatusError as e:
            # HTTP error - extract response details safely
            try:
                response_text = e.response.text[:100] if e.response else "No response"
                error = f"HTTP {e.response.status_code}: {response_text}"
            except Exception:
                error = f"HTTP error: {str(e)[:100]}"
            latency_ms = (time.perf_counter() - request_start) * 1000
        
        except httpx.TimeoutException:
            # Request timeout
            error = "Request timeout"
            latency_ms = (time.perf_counter() - request_start) * 1000
        
        except httpx.RequestError as e:
            # Network/connection error
            error = f"Request error: {type(e).__name__}: {str(e)[:100]}"
            latency_ms = (time.perf_counter() - request_start) * 1000
        
        except Exception as e:
            # Catch-all for any other unexpected errors
            # This ensures we ALWAYS return a result, never crash
            error = f"{type(e).__name__}: {str(e)[:100]}"
            latency_ms = (time.perf_counter() - request_start) * 1000
            # Log to stderr for debugging but don't crash
            import sys
            print(f"[WARNING] Request {qid} failed: {error}", file=sys.stderr)
        
        # GUARANTEE: We always return a result, even if everything failed
        return RequestResult(
            run_id="",  # Will be set by caller
            qid=qid,
            prefill_tokens=0,  # Will be set by caller
            max_new_tokens=max_tokens,
            ttft_ms=ttft_ms,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            actual_prefill_tokens=0,  # Will be set by caller from token_stats
            error=error,
            timestamp=timestamp,
        )
    
    async def run_concurrent_requests(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_tokens: int,
        concurrency: int,
        stream: bool = True,
        jitter_ms: float = 50.0
    ) -> List[RequestResult]:
        """
        Run concurrent requests with specified concurrency level.
        
        Distributes requests evenly across concurrent tasks using a semaphore.
        Adds small random jitter to avoid thundering herd effects.
        
        All requests are guaranteed to complete and return a result,
        even if they fail. Errors are recorded in the result and never
        crash the benchmark run.
        
        Args:
            messages_batch: List of chat message lists
            max_tokens: Maximum tokens to generate per request
            concurrency: Number of concurrent requests
            stream: Whether to stream responses
            jitter_ms: Maximum random delay in milliseconds to add (0 to jitter_ms)
            
        Returns:
            List of RequestResults (one per request, guaranteed)
        """
        results = []
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(qid: int, messages: List[Dict[str, str]]):
            """Wrapper that ensures we always return a result."""
            async with semaphore:
                # Add small random jitter to avoid thundering herd
                if jitter_ms > 0:
                    jitter_seconds = random.uniform(0, jitter_ms / 1000.0)
                    await asyncio.sleep(jitter_seconds)
                
                try:
                    return await self.send_chat_request(messages, max_tokens, qid, stream)
                except Exception as e:
                    # Ultimate safety net - if send_chat_request somehow doesn't
                    # catch an exception, we catch it here
                    return RequestResult(
                        run_id="",
                        qid=qid,
                        prefill_tokens=0,
                        max_new_tokens=max_tokens,
                        ttft_ms=-1.0,
                        latency_ms=-1.0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        actual_prefill_tokens=0,
                        error=f"Unexpected error: {type(e).__name__}: {str(e)[:100]}",
                        timestamp=datetime.utcnow().isoformat(),
                    )
        
        # Create all tasks
        tasks = [
            bounded_request(qid, messages)
            for qid, messages in enumerate(messages_batch)
        ]
        
        # Execute with progress bar
        with tqdm(total=len(tasks), desc=f"Concurrency {concurrency}", unit="req") as pbar:
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    results.append(result)
                except Exception as e:
                    # Final failsafe - should never reach here due to bounded_request wrapper
                    console.print(f"[red]Critical error in task execution: {e}[/red]")
                finally:
                    pbar.update(1)
        
        return results


def load_scenarios(yaml_path: str) -> Dict[str, Any]:
    """
    Load benchmark scenarios from YAML file.
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        Dictionary with 'defaults' and 'scenarios' keys
        
    Raises:
        SystemExit: If YAML is invalid or missing required fields
    """
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[red]Error: Scenarios file not found: {yaml_path}[/red]")
        sys.exit(1)
    except yaml.YAMLError as e:
        console.print(f"[red]Error: Invalid YAML file: {e}[/red]")
        sys.exit(1)
    
    # Validate structure
    if not isinstance(config, dict):
        console.print(f"[red]Error: YAML root must be a dictionary[/red]")
        sys.exit(1)
    
    if 'scenarios' not in config:
        console.print(f"[red]Error: YAML must contain 'scenarios' key[/red]")
        sys.exit(1)
    
    scenarios = config.get('scenarios', {})
    if not scenarios:
        console.print(f"[red]Error: No scenarios defined in YAML[/red]")
        sys.exit(1)
    
    # Validate each scenario has required fields
    required_fields = ['prefill_tokens', 'max_new_tokens', 'concurrencies', 'num_requests']
    
    for scenario_name, scenario_config in scenarios.items():
        if not isinstance(scenario_config, dict):
            console.print(f"[red]Error: Scenario '{scenario_name}' must be a dictionary[/red]")
            sys.exit(1)
        
        missing_fields = [field for field in required_fields if field not in scenario_config]
        if missing_fields:
            console.print(
                f"[red]Error: Scenario '{scenario_name}' missing required fields: "
                f"{', '.join(missing_fields)}[/red]"
            )
            console.print(f"[yellow]Required fields: {', '.join(required_fields)}[/yellow]")
            sys.exit(1)
        
        # Validate field types
        if not isinstance(scenario_config['prefill_tokens'], int) or scenario_config['prefill_tokens'] <= 0:
            console.print(f"[red]Error: Scenario '{scenario_name}': prefill_tokens must be a positive integer[/red]")
            sys.exit(1)
        
        if not isinstance(scenario_config['max_new_tokens'], int) or scenario_config['max_new_tokens'] <= 0:
            console.print(f"[red]Error: Scenario '{scenario_name}': max_new_tokens must be a positive integer[/red]")
            sys.exit(1)
        
        if not isinstance(scenario_config['concurrencies'], list) or not scenario_config['concurrencies']:
            console.print(f"[red]Error: Scenario '{scenario_name}': concurrencies must be a non-empty list[/red]")
            sys.exit(1)
        
        if not isinstance(scenario_config['num_requests'], int) or scenario_config['num_requests'] <= 0:
            console.print(f"[red]Error: Scenario '{scenario_name}': num_requests must be a positive integer[/red]")
            sys.exit(1)
    
    return config


def generate_run_id(
    scenario_name: str,
    concurrency: int,
    prefill_tokens: int,
    max_new_tokens: int
) -> str:
    """
    Generate a structured run ID encoding all key parameters.
    
    Format: <scenario>-cc<concurrency>-pref<prefill>-max<max_new>-<timestamp>
    Example: rag_medium-cc64-pref1536-max256-20250423_143052
    
    Args:
        scenario_name: Name of the benchmark scenario
        concurrency: Concurrency level
        prefill_tokens: Prefill token count
        max_new_tokens: Max new token count
        
    Returns:
        Formatted run ID string
    """
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return f"{scenario_name}-cc{concurrency}-pref{prefill_tokens}-max{max_new_tokens}-{timestamp}"


def save_results_csv(results: List[RequestResult], output_path: Path, append: bool = False):
    """
    Save results to CSV file with stable header.
    
    This function GUARANTEES a valid CSV is written, even if results list is empty.
    
    Args:
        results: List of results to save
        output_path: Path to output CSV file
        append: If True, append to existing file; if False, overwrite
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "run_id", "qid", "prefill_tokens", "max_new_tokens",
        "ttft_ms", "latency_ms", "prompt_tokens", "completion_tokens",
        "actual_prefill_tokens", "error", "timestamp"
    ]
    
    # Check if file exists and has content
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    
    mode = 'a' if (append and file_exists) else 'w'
    
    with open(output_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header only if creating new file or overwriting
        if mode == 'w':
            writer.writeheader()
        
        # Write all results (even if empty list, we get header)
        for result in results:
            writer.writerow(asdict(result))
    
    action = "appended to" if (mode == 'a') else "saved to"
    result_count = len(results)
    console.print(f"[green]✓[/green] {result_count} results {action} {output_path}")


async def run_scenario_benchmark(
    scenario_name: str,
    scenario_config: Dict[str, Any],
    defaults: Dict[str, Any],
    output_dir: Path,
    output_pattern: str = "{run_id}.csv",
    enable_telemetry: bool = False
):
    """
    Run benchmark for a single scenario across all concurrency levels.
    
    Args:
        scenario_name: Name of the scenario
        scenario_config: Scenario configuration
        defaults: Default configuration values
        output_dir: Directory to save results
        output_pattern: Output filename pattern (supports {run_id})
        enable_telemetry: Whether to enable GPU telemetry monitoring
    """
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]Scenario: {scenario_name}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"Description: {scenario_config.get('description', 'N/A')}")
    console.print(f"Prefill tokens: {scenario_config['prefill_tokens']}")
    console.print(f"Max new tokens: {scenario_config['max_new_tokens']}")
    console.print(f"Concurrency levels: {scenario_config['concurrencies']}")
    console.print(f"Requests per level: {scenario_config['num_requests']}")
    
    # Extract configuration
    base_url = defaults.get('base_url', 'http://localhost:8000/v1')
    # Load model from environment variable, fallback to scenarios.yaml, then default
    model = os.getenv('MODEL', defaults.get('model', 'facebook/opt-1.3b'))
    temperature = defaults.get('temperature', 0.0)
    stream = defaults.get('stream', True)
    
    prefill_tokens = scenario_config['prefill_tokens']
    max_new_tokens = scenario_config['max_new_tokens']
    concurrencies = scenario_config['concurrencies']
    num_requests = scenario_config['num_requests']
    
    # Initialize telemetry if enabled
    telemetry_monitor = None
    if enable_telemetry:
        try:
            from telemetry import GPUTelemetryMonitor
            telemetry_dir = output_dir / "telemetry"
            telemetry_monitor = GPUTelemetryMonitor(
                output_dir=telemetry_dir,
                sample_interval_ms=500.0
            )
            if not telemetry_monitor.nvml_initialized:
                console.print("[yellow]Warning: GPU telemetry unavailable, continuing without it[/yellow]")
                telemetry_monitor = None
        except ImportError:
            console.print("[yellow]Warning: telemetry.py not found, skipping GPU monitoring[/yellow]")
            telemetry_monitor = None
    
    # Determine if this is a RAG scenario with JSONL prompts
    use_rag = scenario_config.get("rag", False)
    
    # Load ALL available prompts once (outside the concurrency loop)
    all_messages = []
    all_token_stats = []
    
    if use_rag:
        from load_jsonl_prompts import load_prompts_from_jsonl
        from pathlib import Path
        
        prompt_file = Path(__file__).parent / scenario_config.get("prompt_file", "../data/rag_prompts.jsonl")
        
        if not prompt_file.exists():
            console.print(f"[red]Error: Prompt file not found: {prompt_file}[/red]")
            console.print(f"[yellow]Generate it with: python bench/generate_rag_prompts.py -n 1500 -t {scenario_config.get('target_prefill_tokens', 1700)}[/yellow]")
            return
        
        # Load ALL prompts from file
        all_messages, all_token_stats = load_prompts_from_jsonl(
            prompt_file,
            num_prompts=None  # Load all available
        )
        
        console.print(f"[cyan]📚 Loaded {len(all_messages)} total unique prompts from {prompt_file.name}[/cyan]")
    else:
        # Legacy: load from Dolly dataset
        from load_real_prompts import load_real_prompts
        
        console.print(f"[cyan]📚 Loading prompts from Dolly dataset...[/cyan]")
        
        # Load maximum possible prompts
        all_messages = load_real_prompts(
            num_prompts=10000,  # Try to load many
            dataset_name="dolly",
            min_tokens=50,
            max_tokens=max(500, prefill_tokens * 2),
            seed=42
        )
        all_token_stats = None
        
        console.print(f"[cyan]📚 Loaded {len(all_messages)} total unique prompts from Dolly dataset[/cyan]")
    
    # Track which prompts have been used across concurrency levels
    used_prompt_indices = set()
    
    # Run benchmarks for each concurrency level
    async with AsyncLoadGenerator(base_url, model, temperature) as generator:
        for concurrency in concurrencies:
            # Calculate actual num_requests for this concurrency level
            if "requests_per_user" in scenario_config:
                # Scaled workload: total requests = concurrency × requests_per_user
                actual_num_requests = concurrency * scenario_config["requests_per_user"]
                console.print(f"\n[yellow]Scaled workload: {concurrency} users × {scenario_config['requests_per_user']} req/user = {actual_num_requests} total requests[/yellow]")
            else:
                # Fixed workload: use num_requests as-is
                actual_num_requests = num_requests
                console.print(f"\n[yellow]Fixed workload: {actual_num_requests} total requests[/yellow]")
            
            # Check if we have enough unique prompts
            available_prompts = len(all_messages) - len(used_prompt_indices)
            if available_prompts < actual_num_requests:
                console.print(f"[red]❌ Error: Not enough unique prompts![/red]")
                console.print(f"[red]   Need: {actual_num_requests} prompts[/red]")
                console.print(f"[red]   Available: {available_prompts} unused prompts (total: {len(all_messages)})[/red]")
                console.print(f"[yellow]💡 Generate more prompts with:[/yellow]")
                console.print(f"[yellow]   python bench/generate_rag_prompts.py -n {len(all_messages) * 2} -t {scenario_config.get('target_prefill_tokens', 1700)}[/yellow]")
                return
            
            # Select NEW unused prompts
            available_indices = [i for i in range(len(all_messages)) if i not in used_prompt_indices]
            
            # Randomly sample from available indices
            selected_indices = random.sample(available_indices, actual_num_requests)
            
            # Extract selected prompts
            messages_batch = [all_messages[i] for i in selected_indices]
            if all_token_stats:
                token_stats_batch = [all_token_stats[i] for i in selected_indices]
            else:
                token_stats_batch = None
            
            # Mark these prompts as used
            used_prompt_indices.update(selected_indices)
            
            console.print(f"[green]✅ Selected {len(messages_batch)} NEW unique prompts[/green]")
            console.print(f"[green]   Total used: {len(used_prompt_indices)}/{len(all_messages)} ({len(used_prompt_indices)/len(all_messages)*100:.1f}%)[/green]")
            
            if token_stats_batch:
                avg_tokens = sum(s["total_prefill_tokens"] for s in token_stats_batch) / len(token_stats_batch)
                console.print(f"[green]   Average tokens: {avg_tokens:.1f}[/green]")
            
            console.print(f"\n[bold yellow]Running with concurrency: {concurrency}[/bold yellow]")
            
            # Generate structured run ID
            run_id = generate_run_id(
                scenario_name=scenario_name,
                concurrency=concurrency,
                prefill_tokens=prefill_tokens,
                max_new_tokens=max_new_tokens
            )
            
            console.print(f"Run ID: [cyan]{run_id}[/cyan]")
            
            # Start telemetry monitoring
            if telemetry_monitor:
                await telemetry_monitor.start()
            
            start_time = time.time()
            results = await generator.run_concurrent_requests(
                messages_batch=messages_batch,
                max_tokens=max_new_tokens,
                concurrency=concurrency,
                stream=stream,
                jitter_ms=50.0  # 50ms max jitter to avoid thundering herd
            )
            elapsed = time.time() - start_time
            
            # Stop telemetry and save
            if telemetry_monitor:
                await telemetry_monitor.stop()
                telemetry_monitor.save_csv(run_id)
                telemetry_monitor.clear()
            
            # Set run_id, prefill_tokens, and actual_prefill_tokens for all results
            for i, result in enumerate(results):
                result.run_id = run_id
                result.prefill_tokens = prefill_tokens
                # Set actual prefill tokens from token_stats if available
                if token_stats_batch and i < len(token_stats_batch):
                    result.actual_prefill_tokens = token_stats_batch[i]["total_prefill_tokens"]
                else:
                    result.actual_prefill_tokens = prefill_tokens  # Fallback to estimate
            
            # Generate output filename from pattern
            output_filename = output_pattern.format(run_id=run_id)
            output_file = output_dir / output_filename
            
            # Save results (overwrite mode for individual run files)
            save_results_csv(results, output_file, append=False)
            
            # Also append to a consolidated scenario file
            consolidated_file = output_dir / f"{scenario_name}_all_runs.csv"
            save_results_csv(results, consolidated_file, append=True)
            
            # Print summary
            successful = [r for r in results if not r.error]
            failed = [r for r in results if r.error]
            
            # Compute concise statistics
            total_time_sec = elapsed
            throughput = len(successful) / elapsed if elapsed > 0 else 0
            
            console.print(f"\n[bold]{'='*60}[/bold]")
            console.print(f"[bold]Run Summary: {run_id}[/bold]")
            console.print(f"[bold]{'='*60}[/bold]")
            console.print(f"  Total requests:     {len(results)}")
            console.print(f"  Successful:         {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
            console.print(f"  Failed:             {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
            console.print(f"  Total time:         {total_time_sec:.2f}s")
            console.print(f"  Throughput:         {throughput:.2f} req/s")
            
            if successful:
                valid_ttft = [r.ttft_ms for r in successful if r.ttft_ms > 0]
                valid_latency = [r.latency_ms for r in successful if r.latency_ms > 0]
                valid_tokens = [r.completion_tokens for r in successful if r.completion_tokens > 0]
                
                if valid_ttft:
                    avg_ttft = sum(valid_ttft) / len(valid_ttft)
                    p50_ttft = sorted(valid_ttft)[len(valid_ttft)//2]
                    p95_ttft = sorted(valid_ttft)[int(len(valid_ttft)*0.95)]
                    console.print(f"  TTFT:               p50={p50_ttft:.0f}ms, p95={p95_ttft:.0f}ms, avg={avg_ttft:.0f}ms")
                
                if valid_latency:
                    avg_latency = sum(valid_latency) / len(valid_latency)
                    p50_latency = sorted(valid_latency)[len(valid_latency)//2]
                    p95_latency = sorted(valid_latency)[int(len(valid_latency)*0.95)]
                    console.print(f"  Latency:            p50={p50_latency:.0f}ms, p95={p95_latency:.0f}ms, avg={avg_latency:.0f}ms")
                
                if valid_tokens and valid_latency:
                    # Calculate tokens/s
                    tokens_per_sec = [
                        (r.completion_tokens / (r.latency_ms / 1000.0))
                        for r in successful
                        if r.completion_tokens > 0 and r.latency_ms > 0
                    ]
                    if tokens_per_sec:
                        avg_tok_per_sec = sum(tokens_per_sec) / len(tokens_per_sec)
                        console.print(f"  Tokens/s:           {avg_tok_per_sec:.1f} avg")
            
            if failed:
                # Show error breakdown
                error_counts = {}
                for r in failed:
                    error_type = r.error.split(':')[0] if ':' in r.error else r.error[:30]
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
                console.print(f"\n  [yellow]Error breakdown:[/yellow]")
                for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                    console.print(f"    - {error_type}: {count}")
            
            console.print(f"[bold]{'='*60}[/bold]\n")


async def main():
    """Main entry point for benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="vLLM async load generator and benchmarking tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all scenarios
  python bench.py --scenarios bench/scenarios.yaml --output-dir results

  # Run a single scenario
  python bench.py --scenarios bench/scenarios.yaml --name short_chat

  # Custom output pattern
  python bench.py --scenarios bench/scenarios.yaml --out "runs/{run_id}.csv"

  # Run specific scenario with custom output
  python bench.py --name rag_medium --out "experiments/rag/{run_id}.csv"
        """
    )
    parser.add_argument(
        "--scenarios",
        default="bench/scenarios.yaml",
        help="Path to scenarios YAML file (default: bench/scenarios.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--name",
        dest="scenario_name",
        help="Run only specific scenario by name (default: run all)"
    )
    parser.add_argument(
        "--out",
        dest="output_pattern",
        default="{run_id}.csv",
        help="Output filename pattern, supports {run_id} placeholder (default: {run_id}.csv)"
    )
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help="Enable GPU telemetry monitoring (requires nvidia-ml-py3)"
    )
    
    args = parser.parse_args()
    
    # Validate output pattern
    if '{run_id}' not in args.output_pattern:
        console.print("[yellow]Warning: Output pattern doesn't contain {run_id}, appending it[/yellow]")
        args.output_pattern = f"{args.output_pattern}_{{run_id}}.csv"
    
    # Load scenarios with validation
    console.print(f"[cyan]Loading scenarios from {args.scenarios}...[/cyan]")
    config = load_scenarios(args.scenarios)
    defaults = config.get('defaults', {})
    scenarios = config.get('scenarios', {})
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter scenarios if specific name requested
    if args.scenario_name:
        if args.scenario_name not in scenarios:
            console.print(f"[red]Error: Scenario '{args.scenario_name}' not found[/red]")
            console.print(f"[yellow]Available scenarios: {', '.join(scenarios.keys())}[/yellow]")
            sys.exit(1)
        scenarios = {args.scenario_name: scenarios[args.scenario_name]}
        console.print(f"[cyan]Running single scenario: {args.scenario_name}[/cyan]")
    else:
        console.print(f"[cyan]Running all scenarios: {', '.join(scenarios.keys())}[/cyan]")
    
    console.print(f"[bold green]Starting vLLM Benchmark[/bold green]")
    model = os.getenv('MODEL', defaults.get('model', 'facebook/opt-1.3b'))
    console.print(f"Model: {model}")
    console.print(f"Base URL: {defaults.get('base_url', 'http://localhost:8000/v1')}")
    console.print(f"Output directory: {output_dir.absolute()}")
    console.print(f"Output pattern: {args.output_pattern}")
    console.print(f"GPU telemetry: {'enabled' if args.telemetry else 'disabled'}")
    
    # Run each scenario
    for scenario_name, scenario_config in scenarios.items():
        await run_scenario_benchmark(
            scenario_name=scenario_name,
            scenario_config=scenario_config,
            defaults=defaults,
            output_dir=output_dir,
            output_pattern=args.output_pattern,
            enable_telemetry=args.telemetry
        )
    
    console.print(f"\n[bold green]✓ All benchmarks complete![/bold green]")
    console.print(f"Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())
