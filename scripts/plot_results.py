"""Plot results visualization"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(results_path: str):
    """Load results from JSONL file"""
    results = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def plot_cost_accuracy(results: list, output_path: str):
    """Plot cost-accuracy curve
    
    X: Token Cost (total_tokens)
    Y: Accuracy (overall_f1 for JSON, hit_at_1 for needle)
    Four curves for four codecs
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    tasks = ["json_restore", "needle"]
    task_titles = ["JSON Restore", "Needle-in-Haystack"]
    
    for task_idx, (task, title) in enumerate(zip(tasks, task_titles)):
        ax = axes[task_idx]
        
        # Group by codec
        codec_data = defaultdict(lambda: {"costs": [], "accs": []})
        
        for r in results:
            if r.get("task") != task:
                continue
            
            codec = r.get("codec", "unknown")
            cost = r.get("token_cost", {}).get("total_tokens", 0)
            
            if task == "json_restore":
                acc = r.get("metrics", {}).get("overall_f1", 0.0)
            else:
                acc = r.get("metrics", {}).get("hit_at_1", 0.0)
            
            if cost > 0 and acc >= 0:
                codec_data[codec]["costs"].append(cost)
                codec_data[codec]["accs"].append(acc)
        
        # Plot curves
        codec_order = ["text", "render", "codebook", "codebook_external"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        
        for codec, color in zip(codec_order, colors):
            if codec in codec_data:
                costs = codec_data[codec]["costs"]
                accs = codec_data[codec]["accs"]
                
                # Sort by cost
                sorted_data = sorted(zip(costs, accs))
                costs_sorted, accs_sorted = zip(*sorted_data) if sorted_data else ([], [])
                
                if costs_sorted:
                    ax.plot(costs_sorted, accs_sorted, marker='o', label=codec, color=color, linewidth=2, markersize=4)
        
        ax.set_xlabel("Token Cost", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cost-accuracy plot to {output_path}")


def plot_ratio_accuracy(results: list, output_path: str):
    """Plot compression ratio vs accuracy
    
    X: Compression Ratio (orig_text_tokens / total_tokens)
    Y: Accuracy
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    tasks = ["json_restore", "needle"]
    task_titles = ["JSON Restore", "Needle-in-Haystack"]
    
    for task_idx, (task, title) in enumerate(zip(tasks, task_titles)):
        ax = axes[task_idx]
        
        codec_data = defaultdict(lambda: {"ratios": [], "accs": []})
        
        for r in results:
            if r.get("task") != task:
                continue
            
            codec = r.get("codec", "unknown")
            ratio = r.get("compression_ratio")
            
            if ratio is None or ratio <= 0:
                continue
            
            if task == "json_restore":
                acc = r.get("metrics", {}).get("overall_f1", 0.0)
            else:
                acc = r.get("metrics", {}).get("hit_at_1", 0.0)
            
            if acc >= 0:
                codec_data[codec]["ratios"].append(ratio)
                codec_data[codec]["accs"].append(acc)
        
        # Plot
        codec_order = ["text", "render", "codebook", "codebook_external"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        
        for codec, color in zip(codec_order, colors):
            if codec in codec_data:
                ratios = codec_data[codec]["ratios"]
                accs = codec_data[codec]["accs"]
                
                sorted_data = sorted(zip(ratios, accs))
                ratios_sorted, accs_sorted = zip(*sorted_data) if sorted_data else ([], [])
                
                if ratios_sorted:
                    ax.plot(ratios_sorted, accs_sorted, marker='o', label=codec, color=color, linewidth=2, markersize=4)
        
        ax.set_xlabel("Compression Ratio", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ratio-accuracy plot to {output_path}")


def plot_robustness(results: list, output_path: str):
    """Plot robustness curve
    
    X: Augment Level (clean/light/heavy)
    Y: Accuracy
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    tasks = ["json_restore", "needle"]
    task_titles = ["JSON Restore", "Needle-in-Haystack"]
    augment_levels = ["clean", "light", "heavy"]
    
    for task_idx, (task, title) in enumerate(zip(tasks, task_titles)):
        ax = axes[task_idx]
        
        codec_data = defaultdict(lambda: {level: [] for level in augment_levels})
        
        for r in results:
            if r.get("task") != task:
                continue
            
            codec = r.get("codec", "unknown")
            level = r.get("augment_level", "clean")
            
            if task == "json_restore":
                acc = r.get("metrics", {}).get("overall_f1", 0.0)
            else:
                acc = r.get("metrics", {}).get("hit_at_1", 0.0)
            
            if level in augment_levels and acc >= 0:
                codec_data[codec][level].append(acc)
        
        # Plot
        codec_order = ["text", "render", "codebook", "codebook_external"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        x_pos = np.arange(len(augment_levels))
        
        for codec, color in zip(codec_order, colors):
            if codec in codec_data:
                means = []
                stds = []
                for level in augment_levels:
                    accs = codec_data[codec][level]
                    if accs:
                        means.append(np.mean(accs))
                        stds.append(np.std(accs))
                    else:
                        means.append(0)
                        stds.append(0)
                
                ax.plot(x_pos, means, marker='o', label=codec, color=color, linewidth=2, markersize=6)
                ax.errorbar(x_pos, means, yerr=stds, color=color, alpha=0.3, capsize=3)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(augment_levels)
        ax.set_xlabel("Augment Level", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved robustness plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL results file")
    parser.add_argument("--output", type=str, default="outputs/plots", help="Output directory")
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.input)
    print(f"Loaded {len(results)} results")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_cost_accuracy(results, str(output_dir / "cost-acc.png"))
    plot_ratio_accuracy(results, str(output_dir / "ratio-acc.png"))
    plot_robustness(results, str(output_dir / "robustness.png"))
    
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
