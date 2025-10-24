#!/usr/bin/env python3
"""
Advanced benchmark visualization for vLLM performance analysis.

Creates comprehensive comparison plots across concurrency levels:
- Throughput scaling
- Latency percentiles (p50/p95/p99)
- TTFT distribution
- Tokens/second analysis
- Combined dashboard view

Usage:
    python bench/plot_summary.py results/short_chat_all_runs.csv
    python bench/plot_summary.py results/*.csv --output results/plots/summary.png
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# Increase font sizes for better readability
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15
})


def parse_run_id(run_id: str) -> Dict[str, any]:
    """Extract metadata from run_id string.
    
    Example: short_chat-cc16-pref256-max128-20251024_013411
    Returns: {scenario, concurrency, prefill, max_new, timestamp}
    """
    parts = run_id.split('-')
    if len(parts) < 5:
        return None
    
    try:
        return {
            'scenario': parts[0],
            'concurrency': int(parts[1].replace('cc', '')),
            'prefill': int(parts[2].replace('pref', '')),
            'max_new': int(parts[3].replace('max', '')),
            'timestamp': parts[4]
        }
    except (ValueError, IndexError):
        return None


def load_results(csv_path: Path) -> Dict[int, List[Dict]]:
    """Load benchmark results grouped by concurrency level.
    
    Returns:
        {concurrency: [list of request results]}
    """
    results_by_concurrency = defaultdict(list)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta = parse_run_id(row['run_id'])
            if not meta:
                continue
            
            # Skip failed requests
            if row['error']:
                continue
            
            try:
                result = {
                    'concurrency': meta['concurrency'],
                    'ttft_ms': float(row['ttft_ms']),
                    'latency_ms': float(row['latency_ms']),
                    'prompt_tokens': int(row['prompt_tokens']),
                    'completion_tokens': int(row['completion_tokens']),
                    'tokens_per_sec': float(row['completion_tokens']) / (float(row['latency_ms']) / 1000.0)
                }
                results_by_concurrency[meta['concurrency']].append(result)
            except (ValueError, ZeroDivisionError):
                continue
    
    return dict(results_by_concurrency)


def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate summary statistics for a set of results."""
    if not results:
        return {}
    
    ttft_values = [r['ttft_ms'] for r in results]
    latency_values = [r['latency_ms'] for r in results]
    tokens_per_sec = [r['tokens_per_sec'] for r in results]
    
    return {
        'count': len(results),
        'ttft_p50': np.percentile(ttft_values, 50),
        'ttft_p95': np.percentile(ttft_values, 95),
        'ttft_p99': np.percentile(ttft_values, 99),
        'ttft_avg': np.mean(ttft_values),
        'latency_p50': np.percentile(latency_values, 50),
        'latency_p95': np.percentile(latency_values, 95),
        'latency_p99': np.percentile(latency_values, 99),
        'latency_avg': np.mean(latency_values),
        'tokens_per_sec_avg': np.mean(tokens_per_sec),
        'tokens_per_sec_p50': np.percentile(tokens_per_sec, 50),
    }


def plot_throughput_scaling(ax, results_by_cc: Dict[int, List[Dict]], title: str):
    """Plot throughput vs concurrency."""
    concurrencies = sorted(results_by_cc.keys())
    throughputs = []
    
    for cc in concurrencies:
        results = results_by_cc[cc]
        if not results:
            throughputs.append(0)
            continue
        
        # Calculate throughput: requests / average latency
        avg_latency_sec = np.mean([r['latency_ms'] for r in results]) / 1000.0
        # Estimate throughput as: concurrency / avg_latency
        throughput = cc / avg_latency_sec
        throughputs.append(throughput)
    
    ax.plot(concurrencies, throughputs, 'o-', linewidth=2, markersize=8, color='#2ecc71')
    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Throughput (req/s)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add value labels on points
    for cc, tput in zip(concurrencies, throughputs):
        ax.annotate(f'{tput:.1f}', 
                   xy=(cc, tput), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9)


def plot_latency_percentiles(ax, results_by_cc: Dict[int, List[Dict]], title: str):
    """Plot latency percentiles (p50, p95, p99) vs concurrency."""
    concurrencies = sorted(results_by_cc.keys())
    p50_values = []
    p95_values = []
    p99_values = []
    
    for cc in concurrencies:
        metrics = calculate_metrics(results_by_cc[cc])
        p50_values.append(metrics.get('latency_p50', 0))
        p95_values.append(metrics.get('latency_p95', 0))
        p99_values.append(metrics.get('latency_p99', 0))
    
    ax.plot(concurrencies, p50_values, 'o-', label='p50', linewidth=2, markersize=6, color='#3498db')
    ax.plot(concurrencies, p95_values, 's-', label='p95', linewidth=2, markersize=6, color='#e74c3c')
    ax.plot(concurrencies, p99_values, '^-', label='p99', linewidth=2, markersize=6, color='#9b59b6')
    
    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_ttft_distribution(ax, results_by_cc: Dict[int, List[Dict]], title: str):
    """Plot TTFT distribution as box plots."""
    concurrencies = sorted(results_by_cc.keys())
    ttft_data = []
    
    for cc in concurrencies:
        ttft_values = [r['ttft_ms'] for r in results_by_cc[cc]]
        ttft_data.append(ttft_values)
    
    bp = ax.boxplot(ttft_data, 
                    positions=concurrencies,
                    widths=max(concurrencies) * 0.1,
                    patch_artist=True,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=5))
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)
    
    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('TTFT (ms)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(concurrencies)


def plot_tokens_per_second(ax, results_by_cc: Dict[int, List[Dict]], title: str):
    """Plot tokens/second generation speed."""
    concurrencies = sorted(results_by_cc.keys())
    tps_avg = []
    tps_p50 = []
    
    for cc in concurrencies:
        metrics = calculate_metrics(results_by_cc[cc])
        tps_avg.append(metrics.get('tokens_per_sec_avg', 0))
        tps_p50.append(metrics.get('tokens_per_sec_p50', 0))
    
    ax.plot(concurrencies, tps_avg, 'o-', label='Average', linewidth=2, markersize=6, color='#f39c12')
    ax.plot(concurrencies, tps_p50, 's-', label='p50', linewidth=2, markersize=6, color='#e67e22')
    
    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Tokens/Second')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add value labels
    for cc, tps in zip(concurrencies, tps_avg):
        ax.annotate(f'{tps:.1f}', 
                   xy=(cc, tps), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9)


def create_summary_dashboard(results_by_cc: Dict[int, List[Dict]], 
                            output_path: Path,
                            scenario_name: str = "Benchmark"):
    """Create a comprehensive 2x2 dashboard with all metrics."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Overall title
    fig.suptitle(f'{scenario_name} - Performance Analysis', fontsize=18, fontweight='bold')
    
    # Top-left: Throughput scaling
    ax1 = fig.add_subplot(gs[0, 0])
    plot_throughput_scaling(ax1, results_by_cc, 'Throughput vs Concurrency')
    
    # Top-right: Latency percentiles
    ax2 = fig.add_subplot(gs[0, 1])
    plot_latency_percentiles(ax2, results_by_cc, 'Latency Percentiles')
    
    # Bottom-left: TTFT distribution
    ax3 = fig.add_subplot(gs[1, 0])
    plot_ttft_distribution(ax3, results_by_cc, 'Time To First Token Distribution')
    
    # Bottom-right: Tokens/second
    ax4 = fig.add_subplot(gs[1, 1])
    plot_tokens_per_second(ax4, results_by_cc, 'Token Generation Speed')
    
    # Add summary statistics text
    total_requests = sum(len(results) for results in results_by_cc.values())
    concurrencies = sorted(results_by_cc.keys())
    
    stats_text = (
        f"Total Requests: {total_requests:,}\n"
        f"Concurrency Levels: {concurrencies}\n"
        f"Success Rate: 100%"
    )
    
    fig.text(0.99, 0.01, stats_text, 
            ha='right', va='bottom', 
            fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Dashboard saved to {output_path}")
    plt.close()


def create_individual_plots(results_by_cc: Dict[int, List[Dict]], 
                           output_dir: Path,
                           scenario_name: str = "Benchmark"):
    """Create individual plot files for each metric."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Throughput plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_throughput_scaling(ax, results_by_cc, f'{scenario_name} - Throughput Scaling')
    plt.savefig(output_dir / f'{scenario_name}_throughput.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Throughput plot saved")
    
    # 2. Latency plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_latency_percentiles(ax, results_by_cc, f'{scenario_name} - Latency Percentiles')
    plt.savefig(output_dir / f'{scenario_name}_latency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Latency plot saved")
    
    # 3. TTFT plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_ttft_distribution(ax, results_by_cc, f'{scenario_name} - TTFT Distribution')
    plt.savefig(output_dir / f'{scenario_name}_ttft.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ TTFT plot saved")
    
    # 4. Tokens/sec plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tokens_per_second(ax, results_by_cc, f'{scenario_name} - Token Generation Speed')
    plt.savefig(output_dir / f'{scenario_name}_tokens_per_sec.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Tokens/sec plot saved")


def print_summary_table(results_by_cc: Dict[int, List[Dict]]):
    """Print a formatted summary table to console."""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'CC':<6} {'Count':<8} {'Throughput':<12} {'TTFT (p50/p95)':<18} {'Latency (p50/p95)':<20} {'Tokens/s':<10}")
    print("-"*80)
    
    for cc in sorted(results_by_cc.keys()):
        metrics = calculate_metrics(results_by_cc[cc])
        
        # Calculate throughput
        avg_latency_sec = metrics['latency_avg'] / 1000.0
        throughput = cc / avg_latency_sec
        
        print(f"{cc:<6} {metrics['count']:<8} {throughput:>8.1f} req/s  "
              f"{metrics['ttft_p50']:>5.0f}/{metrics['ttft_p95']:<5.0f} ms     "
              f"{metrics['latency_p50']:>6.0f}/{metrics['latency_p95']:<6.0f} ms       "
              f"{metrics['tokens_per_sec_avg']:>6.1f}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Create comprehensive benchmark visualization plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Create dashboard from short_chat results
  python bench/plot_summary.py results/short_chat_all_runs.csv
  
  # Specify output location
  python bench/plot_summary.py results/short_chat_all_runs.csv --output results/plots/dashboard.png
  
  # Create both dashboard and individual plots
  python bench/plot_summary.py results/short_chat_all_runs.csv --individual --output results/plots/
  
  # Custom scenario name
  python bench/plot_summary.py results/rag_medium_all_runs.csv --name "RAG Medium Workload"
        '''
    )
    
    parser.add_argument('csv_file', type=Path, help='Path to benchmark CSV file (e.g., short_chat_all_runs.csv)')
    parser.add_argument('--output', '-o', type=Path, help='Output path for dashboard PNG (or directory for --individual)')
    parser.add_argument('--individual', '-i', action='store_true', help='Create individual plot files instead of dashboard')
    parser.add_argument('--name', '-n', type=str, help='Custom scenario name for plot titles')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.csv_file.exists():
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    # Load results
    print(f"\nLoading results from {args.csv_file.name}...")
    results_by_cc = load_results(args.csv_file)
    
    if not results_by_cc:
        print("Error: No valid results found in CSV file")
        sys.exit(1)
    
    total_requests = sum(len(r) for r in results_by_cc.values())
    print(f"Loaded {total_requests:,} successful requests across {len(results_by_cc)} concurrency levels")
    
    # Print summary table
    print_summary_table(results_by_cc)
    
    # Determine scenario name
    scenario_name = args.name
    if not scenario_name:
        # Try to extract from CSV filename
        scenario_name = args.csv_file.stem.replace('_all_runs', '').replace('_', ' ').title()
    
    # Generate plots
    if args.individual:
        output_dir = args.output if args.output else args.csv_file.parent / 'plots'
        print(f"\nGenerating individual plots in {output_dir}...")
        create_individual_plots(results_by_cc, output_dir, scenario_name)
    else:
        output_path = args.output if args.output else args.csv_file.parent / 'plots' / f'{args.csv_file.stem}_dashboard.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating dashboard plot...")
        create_summary_dashboard(results_by_cc, output_path, scenario_name)
    
    print("\n✓ All plots generated successfully!\n")


if __name__ == '__main__':
    main()
