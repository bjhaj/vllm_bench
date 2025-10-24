"""
Plotting utilities for benchmark results.

Generates comprehensive plots for benchmark analysis:
- CDF (Cumulative Distribution Function) plots for TTFT, Latency, Throughput
- Token distributions (prompt tokens, completion tokens)
- Time series plots (throughput over time, requests over time)
- Success rate analysis

Outputs publication-quality plots to results/plots/.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from rich.console import Console

console = Console()


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load benchmark results from CSV file."""
    return pd.read_csv(csv_path)


def parse_run_id(run_id: str) -> dict:
    """
    Parse run ID to extract key parameters for plot titles.
    
    Returns dict with scenario, concurrency, prefill_tokens, max_new_tokens
    """
    try:
        parts = run_id.split('-')
        cc_idx = next((i for i, p in enumerate(parts) if p.startswith('cc')), None)
        pref_idx = next((i for i, p in enumerate(parts) if p.startswith('pref')), None)
        max_idx = next((i for i, p in enumerate(parts) if p.startswith('max')), None)
        
        if cc_idx is not None and pref_idx is not None and max_idx is not None:
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
    
    return {
        'scenario': run_id,
        'concurrency': 0,
        'prefill_tokens': 0,
        'max_new_tokens': 0
    }


def create_cdf_plot(
    data: np.ndarray,
    title: str,
    xlabel: str,
    output_path: Path,
    color: str = 'steelblue',
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Create a CDF plot for the given data.
    
    Args:
        data: Array of values to plot
        title: Plot title
        xlabel: X-axis label
        output_path: Path to save the figure
        color: Line color
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Sort data for CDF
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # Compute CDF values (0 to 1)
    cdf = np.arange(1, n + 1) / n
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot CDF
    ax.plot(sorted_data, cdf, color=color, linewidth=2, label='CDF')
    
    # Add percentile markers
    percentiles = [50, 95, 99]
    percentile_values = np.percentile(sorted_data, percentiles)
    
    for p, val in zip(percentiles, percentile_values):
        ax.axvline(val, color='red', linestyle='--', alpha=0.6, linewidth=1)
        ax.text(
            val, 0.02, f'p{p}={val:.1f}',
            rotation=90, verticalalignment='bottom',
            fontsize=9, color='red'
        )
    
    # Styling
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    # Add statistics box
    stats_text = (
        f'n={n}\n'
        f'mean={np.mean(sorted_data):.1f}\n'
        f'std={np.std(sorted_data):.1f}\n'
        f'min={np.min(sorted_data):.1f}\n'
        f'max={np.max(sorted_data):.1f}'
    )
    ax.text(
        0.98, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    console.print(f"[green]✓[/green] Saved plot to {output_path}")
    
    return fig


def create_histogram_plot(
    data: np.ndarray,
    title: str,
    xlabel: str,
    output_path: Path,
    color: str = 'steelblue',
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Create a histogram with distribution for the given data.
    
    Args:
        data: Array of values to plot
        title: Plot title
        xlabel: X-axis label
        output_path: Path to save the figure
        color: Bar color
        bins: Number of histogram bins
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    counts, bin_edges, patches = ax.hist(
        data, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5
    )
    
    # Add percentile markers
    percentiles = [50, 95, 99]
    percentile_values = np.percentile(data, percentiles)
    colors_p = ['green', 'orange', 'red']
    
    for p, val, col in zip(percentiles, percentile_values, colors_p):
        ax.axvline(val, color=col, linestyle='--', alpha=0.8, linewidth=2, label=f'p{p}={val:.1f}')
    
    # Styling
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='upper right', fontsize=9)
    
    # Add statistics box
    stats_text = (
        f'n={len(data)}\n'
        f'mean={np.mean(data):.1f}\n'
        f'std={np.std(data):.1f}\n'
        f'min={np.min(data):.1f}\n'
        f'max={np.max(data):.1f}'
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    console.print(f"[green]✓[/green] Saved plot to {output_path}")
    
    return fig


def create_scatter_plot(
    x_data: np.ndarray,
    y_data: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    color: str = 'steelblue',
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Create a scatter plot with trend line.
    
    Args:
        x_data: X-axis values
        y_data: Y-axis values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_path: Path to save the figure
        color: Point color
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    ax.scatter(x_data, y_data, c=color, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    # Add moving average
    window = max(5, len(x_data) // 20)
    if len(y_data) >= window:
        moving_avg = pd.Series(y_data).rolling(window=window, center=True).mean()
        ax.plot(x_data, moving_avg, color='red', linewidth=2, label=f'Moving Avg (window={window})')
        ax.legend(loc='best', fontsize=9)
    
    # Styling
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    console.print(f"[green]✓[/green] Saved plot to {output_path}")
    
    return fig


def plot_run_cdfs(
    csv_path: Path,
    output_dir: Path,
    show: bool = False
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Generate CDF plots for TTFT and latency from a benchmark run CSV.
    
    Args:
        csv_path: Path to input CSV file
        output_dir: Directory to save plots
        show: Whether to display plots interactively
        
    Returns:
        Tuple of (ttft_plot_path, latency_plot_path)
    """
    console.print(f"[cyan]Loading results from {csv_path.name}...[/cyan]")
    
    # Load data
    df = load_results(csv_path)
    
    # Filter successful requests
    successful = df[df['error'].isna() | (df['error'] == '')]
    
    if len(successful) == 0:
        console.print("[red]Error: No successful requests found in CSV[/red]")
        return None, None
    
    # Get run information
    run_id = df['run_id'].iloc[0] if len(df) > 0 else "unknown"
    parsed = parse_run_id(run_id)
    
    # Extract valid TTFT and latency data
    valid_ttft = successful[successful['ttft_ms'] > 0]['ttft_ms'].values
    valid_latency = successful[successful['latency_ms'] > 0]['latency_ms'].values
    
    console.print(f"[cyan]Run ID: {run_id}[/cyan]")
    console.print(f"  Scenario: {parsed['scenario']}")
    console.print(f"  Concurrency: {parsed['concurrency']}")
    console.print(f"  Prefill: {parsed['prefill_tokens']} tokens")
    console.print(f"  Max new: {parsed['max_new_tokens']} tokens")
    console.print(f"  Valid TTFT samples: {len(valid_ttft)}")
    console.print(f"  Valid latency samples: {len(valid_latency)}")
    
    ttft_plot_path = None
    latency_plot_path = None
    
    # Create TTFT CDF plot
    if len(valid_ttft) > 0:
        title = (
            f"TTFT CDF - {parsed['scenario']}\n"
            f"Concurrency: {parsed['concurrency']} | "
            f"Prefill: {parsed['prefill_tokens']} | "
            f"Max New: {parsed['max_new_tokens']}"
        )
        
        ttft_plot_path = output_dir / f"{run_id}_ttft_cdf.png"
        create_cdf_plot(
            data=valid_ttft,
            title=title,
            xlabel='Time to First Token (ms)',
            output_path=ttft_plot_path,
            color='steelblue'
        )
    else:
        console.print("[yellow]Warning: No valid TTFT data to plot[/yellow]")
    
    # Create Latency CDF plot
    if len(valid_latency) > 0:
        title = (
            f"Latency CDF - {parsed['scenario']}\n"
            f"Concurrency: {parsed['concurrency']} | "
            f"Prefill: {parsed['prefill_tokens']} | "
            f"Max New: {parsed['max_new_tokens']}"
        )
        
        latency_plot_path = output_dir / f"{run_id}_latency_cdf.png"
        create_cdf_plot(
            data=valid_latency,
            title=title,
            xlabel='Total Latency (ms)',
            output_path=latency_plot_path,
            color='forestgreen'
        )
    else:
        console.print("[yellow]Warning: No valid latency data to plot[/yellow]")
    
    # NEW: Create Throughput CDF plot (tokens/second)
    throughput_plot_path = None
    if len(successful) > 0 and 'completion_tokens' in successful.columns:
        # Calculate throughput: completion_tokens / (latency_ms / 1000)
        valid_throughput_df = successful[
            (successful['completion_tokens'] > 0) & 
            (successful['latency_ms'] > 0)
        ].copy()
        
        if len(valid_throughput_df) > 0:
            valid_throughput_df['throughput'] = (
                valid_throughput_df['completion_tokens'] / 
                (valid_throughput_df['latency_ms'] / 1000)
            )
            valid_throughput = valid_throughput_df['throughput'].values
            
            title = (
                f"Throughput CDF - {parsed['scenario']}\n"
                f"Concurrency: {parsed['concurrency']} | "
                f"Prefill: {parsed['prefill_tokens']} | "
                f"Max New: {parsed['max_new_tokens']}"
            )
            
            throughput_plot_path = output_dir / f"{run_id}_throughput_cdf.png"
            create_cdf_plot(
                data=valid_throughput,
                title=title,
                xlabel='Throughput (tokens/second)',
                output_path=throughput_plot_path,
                color='purple'
            )
            console.print(f"  Valid throughput samples: {len(valid_throughput)}")
    
    # NEW: Create Prompt Tokens histogram
    prompt_tokens_plot_path = None
    if len(successful) > 0 and 'prompt_tokens' in successful.columns:
        valid_prompt_tokens = successful[successful['prompt_tokens'] > 0]['prompt_tokens'].values
        
        if len(valid_prompt_tokens) > 0:
            title = (
                f"Prompt Tokens Distribution - {parsed['scenario']}\n"
                f"Concurrency: {parsed['concurrency']}"
            )
            
            prompt_tokens_plot_path = output_dir / f"{run_id}_prompt_tokens_hist.png"
            create_histogram_plot(
                data=valid_prompt_tokens,
                title=title,
                xlabel='Prompt Tokens',
                output_path=prompt_tokens_plot_path,
                color='coral',
                bins=30
            )
            console.print(f"  Valid prompt token samples: {len(valid_prompt_tokens)}")
    
    # NEW: Create Completion Tokens histogram
    completion_tokens_plot_path = None
    if len(successful) > 0 and 'completion_tokens' in successful.columns:
        valid_completion_tokens = successful[successful['completion_tokens'] > 0]['completion_tokens'].values
        
        if len(valid_completion_tokens) > 0:
            title = (
                f"Completion Tokens Distribution - {parsed['scenario']}\n"
                f"Concurrency: {parsed['concurrency']}"
            )
            
            completion_tokens_plot_path = output_dir / f"{run_id}_completion_tokens_hist.png"
            create_histogram_plot(
                data=valid_completion_tokens,
                title=title,
                xlabel='Completion Tokens',
                output_path=completion_tokens_plot_path,
                color='lightgreen',
                bins=30
            )
            console.print(f"  Valid completion token samples: {len(valid_completion_tokens)}")
    
    # NEW: Create Throughput over time scatter plot
    throughput_time_plot_path = None
    if len(successful) > 0 and 'timestamp' in successful.columns and 'completion_tokens' in successful.columns:
        valid_time_df = successful[
            (successful['completion_tokens'] > 0) & 
            (successful['latency_ms'] > 0) &
            (successful['timestamp'].notna())
        ].copy()
        
        if len(valid_time_df) > 0:
            # Parse timestamps and calculate relative time
            valid_time_df['timestamp_dt'] = pd.to_datetime(valid_time_df['timestamp'])
            start_time = valid_time_df['timestamp_dt'].min()
            valid_time_df['relative_time'] = (
                (valid_time_df['timestamp_dt'] - start_time).dt.total_seconds()
            )
            
            # Calculate throughput
            valid_time_df['throughput'] = (
                valid_time_df['completion_tokens'] / 
                (valid_time_df['latency_ms'] / 1000)
            )
            
            title = (
                f"Throughput Over Time - {parsed['scenario']}\n"
                f"Concurrency: {parsed['concurrency']}"
            )
            
            throughput_time_plot_path = output_dir / f"{run_id}_throughput_over_time.png"
            create_scatter_plot(
                x_data=valid_time_df['relative_time'].values,
                y_data=valid_time_df['throughput'].values,
                title=title,
                xlabel='Time (seconds from start)',
                ylabel='Throughput (tokens/second)',
                output_path=throughput_time_plot_path,
                color='darkviolet'
            )
            console.print(f"  Throughput time series samples: {len(valid_time_df)}")
    
    # NEW: Create TTFT over time scatter plot
    ttft_time_plot_path = None
    if len(successful) > 0 and 'timestamp' in successful.columns:
        valid_ttft_time_df = successful[
            (successful['ttft_ms'] > 0) &
            (successful['timestamp'].notna())
        ].copy()
        
        if len(valid_ttft_time_df) > 0:
            # Parse timestamps and calculate relative time
            valid_ttft_time_df['timestamp_dt'] = pd.to_datetime(valid_ttft_time_df['timestamp'])
            start_time = valid_ttft_time_df['timestamp_dt'].min()
            valid_ttft_time_df['relative_time'] = (
                (valid_ttft_time_df['timestamp_dt'] - start_time).dt.total_seconds()
            )
            
            title = (
                f"TTFT Over Time - {parsed['scenario']}\n"
                f"Concurrency: {parsed['concurrency']}"
            )
            
            ttft_time_plot_path = output_dir / f"{run_id}_ttft_over_time.png"
            create_scatter_plot(
                x_data=valid_ttft_time_df['relative_time'].values,
                y_data=valid_ttft_time_df['ttft_ms'].values,
                title=title,
                xlabel='Time (seconds from start)',
                ylabel='Time to First Token (ms)',
                output_path=ttft_time_plot_path,
                color='dodgerblue'
            )
            console.print(f"  TTFT time series samples: {len(valid_ttft_time_df)}")
    
    # Display plots if requested
    if show:
        plt.show()
    else:
        plt.close('all')
    
    return (
        ttft_plot_path, 
        latency_plot_path,
        throughput_plot_path,
        prompt_tokens_plot_path,
        completion_tokens_plot_path,
        throughput_time_plot_path,
        ttft_time_plot_path
    )


def find_most_recent_csv(results_dir: Path = Path("results")) -> Optional[Path]:
    """
    Find the most recent CSV file in the results directory.
    
    Args:
        results_dir: Directory to search for CSV files
        
    Returns:
        Path to most recent CSV file, or None if no CSVs found
    """
    if not results_dir.exists():
        return None
    
    # Find all CSV files
    csv_files = list(results_dir.glob("*.csv"))
    
    if not csv_files:
        return None
    
    # Sort by modification time (most recent first)
    csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return csv_files[0]


def main():
    """Main entry point for plotter."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate comprehensive plots from vLLM benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use most recent CSV (auto-detected)
  python3 plot.py
  
  # Specify a CSV file
  python3 plot.py results/rag_realistic-cc32-pref1500-max256-20251024_065547.csv
  
  # Custom output directory
  python3 plot.py --output-dir my_plots/
  
  # Show plots interactively
  python3 plot.py --show
        """
    )
    parser.add_argument(
        'input_csv',
        nargs='?',  # Make it optional
        default=None,
        help='CSV file containing benchmark results (default: most recent in results/)'
    )
    parser.add_argument(
        '--output-dir',
        default='results/plots',
        help='Directory to save plots (default: results/plots)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively'
    )
    
    args = parser.parse_args()
    
    # Auto-detect most recent CSV if not provided
    if args.input_csv is None:
        console.print("[cyan]No input CSV specified, searching for most recent...[/cyan]")
        csv_path = find_most_recent_csv()
        
        if csv_path is None:
            console.print("[red]Error: No CSV files found in results/ directory[/red]")
            console.print("[yellow]Run a benchmark first or specify a CSV file explicitly[/yellow]")
            sys.exit(1)
        
        console.print(f"[green]✓ Found most recent CSV: {csv_path.name}[/green]\n")
    else:
        csv_path = Path(args.input_csv)
    
    if not csv_path.exists():
        console.print(f"[red]Error: File not found: {csv_path}[/red]")
        sys.exit(1)
    
    if not csv_path.suffix == '.csv':
        console.print(f"[red]Error: File must be a CSV: {csv_path}[/red]")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    console.print("[bold cyan]Generating comprehensive benchmark plots...[/bold cyan]\n")
    
    plot_paths = plot_run_cdfs(
        csv_path=csv_path,
        output_dir=output_dir,
        show=args.show
    )
    
    (ttft_path, latency_path, throughput_path, prompt_tokens_path, 
     completion_tokens_path, throughput_time_path, ttft_time_path) = plot_paths
    
    # Report generated plots
    plots_generated = [p for p in plot_paths if p is not None]
    
    if plots_generated:
        console.print(f"\n[bold green]✓ {len(plots_generated)} plots generated successfully![/bold green]")
        
        if ttft_path:
            console.print(f"  📊 TTFT CDF: {ttft_path.name}")
        if latency_path:
            console.print(f"  📊 Latency CDF: {latency_path.name}")
        if throughput_path:
            console.print(f"  📊 Throughput CDF: {throughput_path.name}")
        if prompt_tokens_path:
            console.print(f"  📊 Prompt Tokens Histogram: {prompt_tokens_path.name}")
        if completion_tokens_path:
            console.print(f"  📊 Completion Tokens Histogram: {completion_tokens_path.name}")
        if throughput_time_path:
            console.print(f"  📊 Throughput Over Time: {throughput_time_path.name}")
        if ttft_time_path:
            console.print(f"  📊 TTFT Over Time: {ttft_time_path.name}")
        
        console.print(f"\n[cyan]All plots saved to: {output_dir}/[/cyan]")
    else:
        console.print("[yellow]No plots generated[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
