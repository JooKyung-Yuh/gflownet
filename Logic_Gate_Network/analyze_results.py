"""
Results Analysis Dashboard
===========================

Analyze and visualize GFlowNet training results and model performance.

Usage:
    python analyze_results.py --model experiments/trained_model.pt
"""

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

from lgn.network import LGNState
from gflownet.policy_network_gnn import LGNGNNPolicy
from gflownet.action_space import LGNActionSpace
from reward.reward_fn import RewardFunction
from data.generator import RealDataGenerator, FakeDataGenerator
from rules.rule_1 import Rule1_NoConsecutive1s


def sample_lgn_greedy(policy, action_space, num_inputs, max_gates):
    """Sample LGN using greedy policy."""
    lgn = LGNState(num_inputs=num_inputs, max_gates=max_gates)

    with torch.no_grad():
        for step in range(max_gates + 10):
            valid_actions = action_space.get_valid_actions(lgn)
            if not valid_actions:
                break

            gate_actions = [a for a in valid_actions if 'gate_type' in a]
            if not gate_actions:
                break

            action_q, stop_q = policy.forward_policy(lgn, gate_actions)

            if action_q.max() > stop_q:
                best_idx = action_q.argmax().item()
                best_action = gate_actions[best_idx]
                lgn.add_gate(best_action['gate_type'], best_action['input_indices'])
            else:
                break

    return lgn


def analyze_model(model_path, num_inputs, max_gates, node_emb_dim=64, num_conv_steps=2, num_samples=50, data_samples=20):
    """
    Comprehensive analysis of trained model.

    Returns:
        dict: Analysis results with metrics
    """
    print("="*80)
    print("LGN GFlowNet Results Analysis")
    print("="*80)

    # Load model
    print(f"\nLoading model from {model_path}...")
    policy = LGNGNNPolicy(num_inputs=num_inputs, max_gates=max_gates,
                         node_emb_dim=node_emb_dim, num_conv_steps=num_conv_steps)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    print("✅ Model loaded")

    # Generate test data
    print(f"\nGenerating test data...")
    rule = Rule1_NoConsecutive1s(dimension=num_inputs)
    real_gen = RealDataGenerator()
    fake_gen = FakeDataGenerator()
    real_samples = real_gen.generate(rule, count=data_samples)
    fake_samples = fake_gen.generate(rule, real_samples, count=data_samples)
    print(f"✅ Generated {len(real_samples)} real, {len(fake_samples)} fake samples")

    # Sample LGNs
    print(f"\nSampling {num_samples} LGNs for analysis...")
    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)
    reward_fn = RewardFunction()

    results = {
        'num_gates': [],
        'log_rewards': [],
        'rewards': [],
        # FN/FP analysis
        'fn_rate': [],              # False Negative Rate
        'fp_rate': [],              # False Positive Rate
        'real_rejection_rate': [],  # Real data rejection rate
        'fake_rejection_rate': [],  # Fake data rejection rate
        # Gate type counts (AND/OR/NOT)
        'and_count': [],
        'or_count': [],
        'not_count': [],
        'other_count': [],
        'gate_types': {}
    }

    for i in range(num_samples):
        lgn = sample_lgn_greedy(policy, action_space, num_inputs, max_gates)

        # Metrics
        log_reward = reward_fn.compute_reward(lgn, real_samples, fake_samples)
        reward = np.exp(log_reward)
        real_errors = reward_fn.compute_real_error_count(lgn, real_samples)
        fake_accepts = reward_fn.compute_fake_acceptance_count(lgn, fake_samples)

        # FN/FP rates
        fn_rate = real_errors / len(real_samples)
        fp_rate = fake_accepts / len(fake_samples)
        # Rejection rates
        real_rejection_rate = fn_rate
        fake_rejection_rate = 1.0 - fp_rate

        results['num_gates'].append(lgn.get_num_gates())
        results['log_rewards'].append(log_reward)
        results['rewards'].append(reward)
        results['fn_rate'].append(fn_rate)
        results['fp_rate'].append(fp_rate)
        results['real_rejection_rate'].append(real_rejection_rate)
        results['fake_rejection_rate'].append(fake_rejection_rate)

        # Count gate types (AND/OR/NOT separately)
        and_cnt, or_cnt, not_cnt, other_cnt = 0, 0, 0, 0
        for gate in lgn.gates:
            gate_type = gate.gate_type.name
            results['gate_types'][gate_type] = results['gate_types'].get(gate_type, 0) + 1
            if gate_type == 'AND':
                and_cnt += 1
            elif gate_type == 'OR':
                or_cnt += 1
            elif gate_type == 'NOT':
                not_cnt += 1
            else:
                other_cnt += 1

        results['and_count'].append(and_cnt)
        results['or_count'].append(or_cnt)
        results['not_count'].append(not_cnt)
        results['other_count'].append(other_cnt)

        if (i+1) % 10 == 0:
            print(f"  Sampled {i+1}/{num_samples} LGNs...")

    print("✅ Sampling complete")

    return results


def plot_analysis(results, save_dir):
    """
    Create comprehensive analysis plots.

    Included graphs:
    1. Reward Distribution
    2. Log-Reward Distribution
    3. FN Rate (False Negative Rate)
    4. FP Rate (False Positive Rate)
    5. Rejection Rates Comparison (Real vs Fake)
    6. Gate Type Distribution (AND/OR/NOT)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots (2x3 layout)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Reward Distribution
    ax1 = axes[0, 0]
    ax1.hist(results['rewards'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax1.axvline(np.mean(results['rewards']), color='red', linestyle='--',
                label=f'Mean: {np.mean(results["rewards"]):.4f}')
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Reward Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Log-Reward Distribution
    ax2 = axes[0, 1]
    ax2.hist(results['log_rewards'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(np.mean(results['log_rewards']), color='red', linestyle='--',
                label=f'Mean: {np.mean(results["log_rewards"]):.2f}')
    ax2.set_xlabel('Log-Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Log-Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. FN Rate Distribution (False Negative: Real → Reject)
    ax3 = axes[0, 2]
    ax3.hist(results['fn_rate'], bins=20, alpha=0.7, color='salmon', edgecolor='black')
    mean_fn = np.mean(results['fn_rate'])
    ax3.axvline(mean_fn, color='darkred', linestyle='--', linewidth=2,
                label=f'Mean FN Rate: {mean_fn:.2%}')
    ax3.set_xlabel('False Negative Rate')
    ax3.set_ylabel('Frequency')
    ax3.set_title('FN Rate (Real→Reject, Should be LOW)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. FP Rate Distribution (False Positive: Fake → Accept)
    ax4 = axes[1, 0]
    ax4.hist(results['fp_rate'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    mean_fp = np.mean(results['fp_rate'])
    ax4.axvline(mean_fp, color='darkblue', linestyle='--', linewidth=2,
                label=f'Mean FP Rate: {mean_fp:.2%}')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('Frequency')
    ax4.set_title('FP Rate (Fake→Accept, Should be LOW)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Rejection Rates Comparison (Real vs Fake)
    ax5 = axes[1, 1]
    sample_indices = list(range(len(results['real_rejection_rate'])))
    ax5.scatter(sample_indices, results['real_rejection_rate'], alpha=0.6, color='red',
                label=f'Real Rejection (Mean: {np.mean(results["real_rejection_rate"]):.2%})', s=20)
    ax5.scatter(sample_indices, results['fake_rejection_rate'], alpha=0.6, color='blue',
                label=f'Fake Rejection (Mean: {np.mean(results["fake_rejection_rate"]):.2%})', s=20)
    ax5.axhline(np.mean(results['real_rejection_rate']), color='darkred', linestyle='--', alpha=0.7)
    ax5.axhline(np.mean(results['fake_rejection_rate']), color='darkblue', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Sample Index')
    ax5.set_ylabel('Rejection Rate')
    ax5.set_title('Rejection Rates (Red↓ Blue↑ is Good)')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1.05)

    # 6. Gate Type Distribution (AND/OR/NOT)
    ax6 = axes[1, 2]
    gate_labels = ['AND', 'OR', 'NOT', 'Other']
    gate_totals = [
        sum(results['and_count']),
        sum(results['or_count']),
        sum(results['not_count']),
        sum(results['other_count'])
    ]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9E9E9E']
    bars = ax6.bar(gate_labels, gate_totals, color=colors, edgecolor='black')
    ax6.set_xlabel('Gate Type')
    ax6.set_ylabel('Total Count')
    ax6.set_title('Gate Type Distribution')
    ax6.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, gate_totals):
        if val > 0:
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    save_path = save_dir / 'analysis_dashboard.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Dashboard saved to {save_path}")
    plt.close(fig)

    return fig


def analyze_results(sampled_lgns, real_samples, fake_samples, save_dir=None):
    """
    Analyze sampled LGNs and create visualization dashboard.

    Args:
        sampled_lgns: List of sampled LGNState objects
        real_samples: Real data samples
        fake_samples: Fake data samples
        save_dir: Optional directory to save plots

    Returns:
        matplotlib.figure.Figure: The analysis dashboard figure
    """
    # Compute results
    reward_fn = RewardFunction()

    results = {
        'num_gates': [],
        'rewards': [],
        'log_rewards': [],
        'real_accuracy': [],
        'fake_accuracy': [],
        'overall_accuracy': [],
        'gate_types': {}
    }

    for lgn in sampled_lgns:
        # Metrics
        results['num_gates'].append(lgn.get_num_gates())

        # Compute reward
        log_reward = reward_fn.compute_reward(lgn, real_samples, fake_samples)
        results['log_rewards'].append(log_reward)
        results['rewards'].append(np.exp(log_reward))

        # Compute accuracies
        real_acc = reward_fn._compute_accuracy(lgn, real_samples)
        fake_acc = reward_fn._compute_accuracy(lgn, fake_samples)
        overall_acc = (real_acc + fake_acc) / 2

        results['real_accuracy'].append(real_acc)
        results['fake_accuracy'].append(fake_acc)
        results['overall_accuracy'].append(overall_acc)

        # Count gate types
        for gate in lgn.gates:
            gate_type = gate.gate_type.name
            results['gate_types'][gate_type] = results['gate_types'].get(gate_type, 0) + 1

    # Create plot
    if save_dir is None:
        save_dir = Path('temp_analysis')

    return plot_analysis(results, save_dir)


def print_statistics(results):
    """Print comprehensive statistics."""
    print("\n" + "="*80)
    print("Statistical Summary")
    print("="*80)

    print(f"\n{'Metric':<30} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-"*80)

    # Core metrics
    metrics = {
        'Number of Gates': results['num_gates'],
        'Reward': results['rewards'],
        'Log-Reward': results['log_rewards'],
    }

    # Add FN/FP metrics if available
    if 'fn_rate' in results:
        metrics['False Negative Rate'] = results['fn_rate']
        metrics['False Positive Rate'] = results['fp_rate']
        metrics['Real Rejection Rate'] = results['real_rejection_rate']
        metrics['Fake Rejection Rate'] = results['fake_rejection_rate']
    # Fallback to old accuracy metrics
    elif 'real_accuracy' in results:
        metrics['Real Accuracy'] = results['real_accuracy']
        metrics['Fake Accuracy'] = results['fake_accuracy']
        metrics['Overall Accuracy'] = results['overall_accuracy']

    for name, data in metrics.items():
        mean_val = np.mean(data)
        std_val = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)

        if 'Rate' in name or 'Accuracy' in name:
            print(f"{name:<30} {mean_val:>13.2%} {std_val:>13.2%} {min_val:>13.2%} {max_val:>13.2%}")
        elif 'Reward' in name and 'Log' not in name:
            print(f"{name:<30} {mean_val:>13.6f} {std_val:>13.6f} {min_val:>13.6f} {max_val:>13.6f}")
        else:
            print(f"{name:<30} {mean_val:>13.2f} {std_val:>13.2f} {min_val:>13.2f} {max_val:>13.2f}")

    # Gate Type Distribution (AND/OR/NOT focus)
    print("\n" + "="*80)
    print("Gate Type Distribution")
    print("="*80)

    # New format: AND/OR/NOT counts
    if 'and_count' in results:
        total_and = sum(results['and_count'])
        total_or = sum(results['or_count'])
        total_not = sum(results['not_count'])
        total_other = sum(results['other_count'])
        total_gates = total_and + total_or + total_not + total_other

        if total_gates > 0:
            print(f"  {'AND':<10} {total_and:>5} gates ({100*total_and/total_gates:>5.1f}%)")
            print(f"  {'OR':<10} {total_or:>5} gates ({100*total_or/total_gates:>5.1f}%)")
            print(f"  {'NOT':<10} {total_not:>5} gates ({100*total_not/total_gates:>5.1f}%)")
            print(f"  {'Other':<10} {total_other:>5} gates ({100*total_other/total_gates:>5.1f}%)")
        else:
            print("  No gates sampled")
    # Fallback to old format
    elif results['gate_types']:
        total_gates = sum(results['gate_types'].values())
        for gate_type, count in sorted(results['gate_types'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_gates) * 100
            print(f"  {gate_type:<10} {count:>5} gates ({percentage:>5.1f}%)")
    else:
        print("  No gates sampled")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze GFlowNet training results')
    parser.add_argument('--model', type=str, default='experiments/trained_model.pt',
                        help='Path to trained model')
    parser.add_argument('--num-inputs', type=int, default=6,
                        help='Number of inputs')
    parser.add_argument('--max-gates', type=int, default=3,
                        help='Maximum gates')
    parser.add_argument('--node-emb-dim', type=int, default=64,
                        help='Node embedding dimension (must match training)')
    parser.add_argument('--num-conv-steps', type=int, default=2,
                        help='Number of GNN convolution steps (must match training)')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of LGNs to sample for analysis')
    parser.add_argument('--data-samples', type=int, default=20,
                        help='Number of data samples for evaluation')
    parser.add_argument('--save-dir', type=str, default='experiments/analysis',
                        help='Directory to save results')

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print("\nPlease train a model first:")
        print("  python train_with_real_data.py --num-inputs 6 --max-gates 3 --iterations 30")
        return

    # Analyze
    results = analyze_model(
        model_path=args.model,
        num_inputs=args.num_inputs,
        max_gates=args.max_gates,
        node_emb_dim=args.node_emb_dim,
        num_conv_steps=args.num_conv_steps,
        num_samples=args.num_samples,
        data_samples=args.data_samples
    )

    # Print statistics
    print_statistics(results)

    # Plot results
    print(f"\nGenerating analysis dashboard...")
    plot_analysis(results, save_dir=args.save_dir)

    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
