"""
Benchmark results summarizer.

Reads CSV results from benchmark runs and produces statistical summaries:
- Percentiles (p50, p95, p99) for TTFT and latency
- Throughput metrics (tokens/s)
- Success rates and error analysis
- Outputs to CSV and Markdown formats
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class RunSummary:
    """Statistical summary of a benchmark run."""
    run_id: str
    scenario: str
    concurrency: int
    prefill_tokens: int
    max_new_tokens: int
    
    # Request counts
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    
    # TTFT metrics (ms)
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    
    # Latency metrics (ms)
    latency_p50: float
    latency_p95: float
    latency_p99: float
    
    # Throughput metrics
    tokens_per_sec_p50: float
    tokens_per_sec_p95: float
    
    # Token statistics
    avg_prompt_tokens: float
    avg_completion_tokens: float
    
    # Error summary
    error_types: str


def parse_run_id(run_id: str) -> Dict[str, Any]:
    """
    Parse structured run ID to extract parameters.
    
    Expected format: <scenario>-cc<concurrency>-pref<prefill>-max<max_new>-<timestamp>
    Example: rag_medium-cc64-pref1536-max256-20250423_143052
    
    Returns:
        Dictionary with parsed fields, or best-effort defaults
    """
    try:
        parts = run_id.split('-')
        
        # Find indices of key markers
        cc_idx = next((i for i, p in enumerate(parts) if p.startswith('cc')), None)
        pref_idx = next((i for i, p in enumerate(parts) if p.startswith('pref')), None)
        max_idx = next((i for i, p in enumerate(parts) if p.startswith('max')), None)
        
        if cc_idx is not None and pref_idx is not None and max_idx is not None:
            # Reconstruct scenario name (everything before cc)
            scenario = '-'.join(parts[:cc_idx])
            concurrency = int(parts[cc_idx][2:])
            prefill = int(parts[pref_idx][4:])
            max_new = int(parts[max_idx][3:])
            
            return {
                'scenario': scenario,
                'concurrency': concurrency,
                'prefill_tokens': prefill,
                'max_new_tokens': max_new
            }
    except (ValueError, IndexError):
        pass
    
    # Fallback: use run_id as scenario
    return {
        'scenario': run_id,
        'concurrency': 0,
        'prefill_tokens': 0,
        'max_new_tokens': 0
    }


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load benchmark results from CSV file."""
    return pd.read_csv(csv_path)


def calculate_throughput(row: pd.Series) -> float:
    """
    Calculate tokens per second for a single request.
    
    Args:
        row: DataFrame row with completion_tokens and latency_ms
        
    Returns:
        Tokens per second, or NaN if invalid
    """
    if pd.isna(row['latency_ms']) or row['latency_ms'] <= 0:
        return np.nan
    if pd.isna(row['completion_tokens']) or row['completion_tokens'] <= 0:
        return np.nan
    
    latency_sec = row['latency_ms'] / 1000.0
    return row['completion_tokens'] / latency_sec


def summarize_run(df: pd.DataFrame) -> RunSummary:
    """
    Generate statistical summary for a single run.
    
    Args:
        df: DataFrame with benchmark results for one run
        
    Returns:
        RunSummary with computed statistics
    """
    # Assume all rows have the same run_id and parameters
    run_id = df['run_id'].iloc[0] if len(df) > 0 else "unknown"
    parsed = parse_run_id(run_id)
    
    # Filter successful requests (no errors)
    successful = df[df['error'].isna() | (df['error'] == '')]
    failed = df[~(df['error'].isna() | (df['error'] == ''))]
    
    total_requests = len(df)
    successful_requests = len(successful)
    failed_requests = len(failed)
    success_rate = successful_requests / total_requests if total_requests > 0 else 0.0
    
    # TTFT percentiles (successful requests only, ignore invalid values)
    valid_ttft = successful[successful['ttft_ms'] > 0]['ttft_ms']
    ttft_p50 = np.percentile(valid_ttft, 50) if len(valid_ttft) > 0 else np.nan
    ttft_p95 = np.percentile(valid_ttft, 95) if len(valid_ttft) > 0 else np.nan
    ttft_p99 = np.percentile(valid_ttft, 99) if len(valid_ttft) > 0 else np.nan
    
    # Latency percentiles (successful requests only, ignore invalid values)
    valid_latency = successful[successful['latency_ms'] > 0]['latency_ms']
    latency_p50 = np.percentile(valid_latency, 50) if len(valid_latency) > 0 else np.nan
    latency_p95 = np.percentile(valid_latency, 95) if len(valid_latency) > 0 else np.nan
    latency_p99 = np.percentile(valid_latency, 99) if len(valid_latency) > 0 else np.nan
    
    # Throughput (tokens/s)
    successful['tokens_per_sec'] = successful.apply(calculate_throughput, axis=1)
    valid_throughput = successful['tokens_per_sec'].dropna()
    tokens_per_sec_p50 = np.percentile(valid_throughput, 50) if len(valid_throughput) > 0 else np.nan
    tokens_per_sec_p95 = np.percentile(valid_throughput, 95) if len(valid_throughput) > 0 else np.nan
    
    # Token statistics
    avg_prompt_tokens = successful['prompt_tokens'].mean() if len(successful) > 0 else np.nan
    avg_completion_tokens = successful['completion_tokens'].mean() if len(successful) > 0 else np.nan
    
    # Error types
    if len(failed) > 0:
        error_counts = failed['error'].value_counts()
        # Take top 3 error types
        top_errors = error_counts.head(3)
        error_types = '; '.join([f"{err[:30]}({cnt})" for err, cnt in top_errors.items()])
    else:
        error_types = "None"
    
    return RunSummary(
        run_id=run_id,
        scenario=parsed['scenario'],
        concurrency=parsed['concurrency'],
        prefill_tokens=parsed['prefill_tokens'],
        max_new_tokens=parsed['max_new_tokens'],
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        success_rate=success_rate,
        ttft_p50=ttft_p50,
        ttft_p95=ttft_p95,
        ttft_p99=ttft_p99,
        latency_p50=latency_p50,
        latency_p95=latency_p95,
        latency_p99=latency_p99,
        tokens_per_sec_p50=tokens_per_sec_p50,
        tokens_per_sec_p95=tokens_per_sec_p95,
        avg_prompt_tokens=avg_prompt_tokens,
        avg_completion_tokens=avg_completion_tokens,
        error_types=error_types
    )


def save_summary_csv(summaries: List[RunSummary], output_path: Path):
    """Save summaries to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'run_id', 'scenario', 'concurrency', 'prefill_tokens', 'max_new_tokens',
            'total_requests', 'successful_requests', 'failed_requests', 'success_rate',
            'ttft_p50_ms', 'ttft_p95_ms', 'ttft_p99_ms',
            'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms',
            'tokens_per_sec_p50', 'tokens_per_sec_p95',
            'avg_prompt_tokens', 'avg_completion_tokens',
            'error_types'
        ])
        
        # Data rows
        for s in summaries:
            writer.writerow([
                s.run_id, s.scenario, s.concurrency, s.prefill_tokens, s.max_new_tokens,
                s.total_requests, s.successful_requests, s.failed_requests, f"{s.success_rate:.2%}",
                f"{s.ttft_p50:.2f}", f"{s.ttft_p95:.2f}", f"{s.ttft_p99:.2f}",
                f"{s.latency_p50:.2f}", f"{s.latency_p95:.2f}", f"{s.latency_p99:.2f}",
                f"{s.tokens_per_sec_p50:.2f}", f"{s.tokens_per_sec_p95:.2f}",
                f"{s.avg_prompt_tokens:.1f}", f"{s.avg_completion_tokens:.1f}",
                s.error_types
            ])
    
    console.print(f"[green]✓[/green] Summary CSV saved to {output_path}")


def save_summary_markdown(summaries: List[RunSummary], output_path: Path):
    """Save summaries to Markdown file with compact table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# vLLM Benchmark Summary\n\n")
        
        # Group by scenario
        by_scenario = defaultdict(list)
        for s in summaries:
            by_scenario[s.scenario].append(s)
        
        for scenario, runs in by_scenario.items():
            f.write(f"## {scenario}\n\n")
            
            # Compact table
            f.write("| CC | Prefill | MaxNew | Success | TTFT p50/p95/p99 (ms) | Latency p50/p95/p99 (ms) | Tok/s p50/p95 |\n")
            f.write("|---:|--------:|-------:|--------:|----------------------:|-------------------------:|--------------:|\n")
            
            for s in sorted(runs, key=lambda x: x.concurrency):
                f.write(
                    f"| {s.concurrency} "
                    f"| {s.prefill_tokens} "
                    f"| {s.max_new_tokens} "
                    f"| {s.success_rate:.1%} "
                    f"| {s.ttft_p50:.0f}/{s.ttft_p95:.0f}/{s.ttft_p99:.0f} "
                    f"| {s.latency_p50:.0f}/{s.latency_p95:.0f}/{s.latency_p99:.0f} "
                    f"| {s.tokens_per_sec_p50:.1f}/{s.tokens_per_sec_p95:.1f} |\n"
                )
            
            f.write("\n")
            
            # Error summary if any failures
            failed_runs = [s for s in runs if s.failed_requests > 0]
            if failed_runs:
                f.write("**Errors:**\n\n")
                for s in failed_runs:
                    f.write(f"- CC={s.concurrency}: {s.failed_requests} failed - {s.error_types}\n")
                f.write("\n")
    
    console.print(f"[green]✓[/green] Summary Markdown saved to {output_path}")


def main():
    """Main entry point for summarizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Summarize vLLM benchmark results")
    parser.add_argument(
        'input_csvs',
        nargs='+',
        help='One or more CSV files to summarize (or directories containing CSVs)'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Directory to save summary files (default: results)'
    )
    parser.add_argument(
        '--csv-output',
        default='summary.csv',
        help='Output CSV filename (default: summary.csv)'
    )
    parser.add_argument(
        '--md-output',
        default='summary.md',
        help='Output Markdown filename (default: summary.md)'
    )
    
    args = parser.parse_args()
    
    # Collect all CSV files
    csv_files = []
    for path_str in args.input_csvs:
        path = Path(path_str)
        if path.is_dir():
            # Add all CSVs in directory
            csv_files.extend(path.glob('*.csv'))
        elif path.is_file() and path.suffix == '.csv':
            csv_files.append(path)
        else:
            console.print(f"[yellow]Warning: Skipping {path_str} (not a CSV file or directory)[/yellow]")
    
    if not csv_files:
        console.print("[red]Error: No CSV files found[/red]")
        sys.exit(1)
    
    console.print(f"[cyan]Processing {len(csv_files)} CSV file(s)...[/cyan]")
    
    # Process each CSV
    summaries = []
    for csv_path in csv_files:
        try:
            console.print(f"  Reading {csv_path.name}...")
            df = load_results(csv_path)
            
            # Group by run_id (in case CSV contains multiple runs)
            for run_id, run_df in df.groupby('run_id'):
                summary = summarize_run(run_df)
                summaries.append(summary)
                console.print(f"    ✓ {run_id}: {summary.successful_requests}/{summary.total_requests} successful")
        
        except Exception as e:
            console.print(f"[red]  Error processing {csv_path.name}: {e}[/red]")
            continue
    
    if not summaries:
        console.print("[red]Error: No summaries generated[/red]")
        sys.exit(1)
    
    # Save outputs
    output_dir = Path(args.output_dir)
    csv_output = output_dir / args.csv_output
    md_output = output_dir / args.md_output
    
    save_summary_csv(summaries, csv_output)
    save_summary_markdown(summaries, md_output)
    
    # Display summary table in terminal
    console.print("\n[bold cyan]Summary:[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Scenario", style="cyan")
    table.add_column("CC", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("TTFT p50", justify="right")
    table.add_column("Latency p50", justify="right")
    table.add_column("Tok/s p50", justify="right")
    
    for s in sorted(summaries, key=lambda x: (x.scenario, x.concurrency)):
        table.add_row(
            s.scenario,
            str(s.concurrency),
            f"{s.success_rate:.1%}",
            f"{s.ttft_p50:.0f}ms",
            f"{s.latency_p50:.0f}ms",
            f"{s.tokens_per_sec_p50:.1f}"
        )
    
    console.print(table)
    console.print(f"\n[bold green]✓ Summary complete![/bold green]")


if __name__ == "__main__":
    main()
