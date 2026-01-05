#!/usr/bin/env python
"""
Plot training metrics from JSON file.

Usage:
    python plot_metrics.py --metrics_file ./saved_models/fastformer_final_test_metrics.json
    
Or with custom output:
    python plot_metrics.py --metrics_file ./saved_models/fastformer_final_test_metrics.json --output metrics_plot.png
"""

import json
import argparse
import matplotlib.pyplot as plt
import os


def plot_metrics(metrics_file, output_file=None):
    """
    Plot training and dev metrics from JSON file.
    
    Args:
        metrics_file: Path to the metrics JSON file
        output_file: Path to save the plot (if None, will show interactively)
    """
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    epochs = metrics['epochs']
    
    if len(epochs) == 0:
        print("No data to plot!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics Over Epochs', fontsize=16, fontweight='bold')
    
    # Plot 1: Train Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, metrics['train_loss'], 'b-o', linewidth=2, markersize=8, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Train Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, metrics['train_acc'], 'g-o', linewidth=2, markersize=8, label='Train Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Accuracy', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Dev AUC & MRR
    ax3 = axes[1, 0]
    ax3.plot(epochs, metrics['dev_auc'], 'r-o', linewidth=2, markersize=8, label='Dev AUC')
    ax3.plot(epochs, metrics['dev_mrr'], 'm-s', linewidth=2, markersize=8, label='Dev MRR')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Dev AUC & MRR', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Dev nDCG
    ax4 = axes[1, 1]
    ax4.plot(epochs, metrics['dev_ndcg5'], 'c-o', linewidth=2, markersize=8, label='Dev nDCG@5')
    ax4.plot(epochs, metrics['dev_ndcg10'], 'y-s', linewidth=2, markersize=8, label='Dev nDCG@10')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('nDCG Score', fontsize=12)
    ax4.set_title('Dev nDCG@5 & nDCG@10', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    # Print summary table
    print("\n" + "=" * 70)
    print("TRAINING METRICS SUMMARY")
    print("=" * 70)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Dev AUC':<10} {'Dev MRR':<10} {'nDCG@5':<10} {'nDCG@10':<10}")
    print("-" * 70)
    for i, ep in enumerate(epochs):
        print(f"{ep:<8} {metrics['train_loss'][i]:<12.4f} {metrics['train_acc'][i]:<12.4f} "
              f"{metrics['dev_auc'][i]:<10.4f} {metrics['dev_mrr'][i]:<10.4f} "
              f"{metrics['dev_ndcg5'][i]:<10.4f} {metrics['dev_ndcg10'][i]:<10.4f}")
    print("=" * 70)
    
    # Best epoch
    best_epoch_idx = max(range(len(metrics['dev_auc'])), key=lambda i: metrics['dev_auc'][i])
    print(f"\nBest Epoch: {epochs[best_epoch_idx]} (Dev AUC: {metrics['dev_auc'][best_epoch_idx]:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--metrics_file', type=str, required=True,
                        help='Path to metrics JSON file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for the plot (e.g., metrics.png). If not provided, shows interactively.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.metrics_file):
        print(f"Error: File {args.metrics_file} not found!")
        return
    
    plot_metrics(args.metrics_file, args.output)


if __name__ == "__main__":
    main()
