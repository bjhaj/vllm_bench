"""
Plotting utilities for benchmark results.

Generates CDF (Cumulative Distribution Function) plots for:
- TTFT (Time To First Token)
- Latency (Total request time)

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
    
    # Display plots if requested
    if show:
        plt.show()
    else:
        plt.close('all')
    
    return ttft_plot_path, latency_plot_path


def main():
    """Main entry point for plotter."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CDF plots from vLLM benchmark results")
    parser.add_argument(
        'input_csv',
        help='CSV file containing benchmark results'
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
    
    csv_path = Path(args.input_csv)
    
    if not csv_path.exists():
        console.print(f"[red]Error: File not found: {csv_path}[/red]")
        sys.exit(1)
    
    if not csv_path.suffix == '.csv':
        console.print(f"[red]Error: File must be a CSV: {csv_path}[/red]")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    console.print("[bold cyan]Generating CDF plots...[/bold cyan]\n")
    
    ttft_path, latency_path = plot_run_cdfs(
        csv_path=csv_path,
        output_dir=output_dir,
        show=args.show
    )
    
    if ttft_path or latency_path:
        console.print(f"\n[bold green]✓ Plots generated successfully![/bold green]")
        if ttft_path:
            console.print(f"  TTFT: {ttft_path}")
        if latency_path:
            console.print(f"  Latency: {latency_path}")
    else:
        console.print("[yellow]No plots generated[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
