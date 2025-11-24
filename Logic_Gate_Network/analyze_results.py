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
        'real_accuracy': [],
        'fake_accuracy': [],
        'overall_accuracy': [],
        'gate_types': {}
    }

    for i in range(num_samples):
        lgn = sample_lgn_greedy(policy, action_space, num_inputs, max_gates)

        # Metrics
        log_reward = reward_fn.compute_reward(lgn, real_samples, fake_samples)
        reward = np.exp(log_reward)
        real_errors = reward_fn.compute_real_error_count(lgn, real_samples)
        fake_accepts = reward_fn.compute_fake_acceptance_count(lgn, fake_samples)
        real_acc = 1.0 - (real_errors / len(real_samples))
        fake_acc = 1.0 - (fake_accepts / len(fake_samples))
        overall_acc = (real_acc + fake_acc) / 2.0

        results['num_gates'].append(lgn.get_num_gates())
        results['log_rewards'].append(log_reward)
        results['rewards'].append(reward)
        results['real_accuracy'].append(real_acc)
        results['fake_accuracy'].append(fake_acc)
        results['overall_accuracy'].append(overall_acc)

        # Count gate types
        for gate in lgn.gates:
            gate_type = gate.gate_type.name
            results['gate_types'][gate_type] = results['gate_types'].get(gate_type, 0) + 1

        if (i+1) % 10 == 0:
            print(f"  Sampled {i+1}/{num_samples} LGNs...")

    print("✅ Sampling complete")

    return results


def plot_analysis(results, save_dir):
    """Create comprehensive analysis plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Distribution of Number of Gates
    ax1 = plt.subplot(3, 3, 1)
    plt.hist(results['num_gates'], bins=range(0, max(results['num_gates'])+2), alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Gates')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gate Count')
    plt.grid(True, alpha=0.3)

    # 2. Reward Distribution
    ax2 = plt.subplot(3, 3, 2)
    plt.hist(results['rewards'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.grid(True, alpha=0.3)

    # 3. Log-Reward Distribution
    ax3 = plt.subplot(3, 3, 3)
    plt.hist(results['log_rewards'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Log-Reward')
    plt.ylabel('Frequency')
    plt.title('Log-Reward Distribution')
    plt.grid(True, alpha=0.3)

    # 4. Real Accuracy Distribution
    ax4 = plt.subplot(3, 3, 4)
    plt.hist(results['real_accuracy'], bins=30, alpha=0.7, color='gold', edgecolor='black')
    plt.xlabel('Real Data Accuracy')
    plt.ylabel('Frequency')
    plt.title('Real Data Accuracy Distribution')
    plt.grid(True, alpha=0.3)
    plt.axvline(np.mean(results['real_accuracy']), color='red', linestyle='--', label=f'Mean: {np.mean(results["real_accuracy"]):.2%}')
    plt.legend()

    # 5. Fake Accuracy Distribution
    ax5 = plt.subplot(3, 3, 5)
    plt.hist(results['fake_accuracy'], bins=30, alpha=0.7, color='plum', edgecolor='black')
    plt.xlabel('Fake Data Accuracy (Rejection Rate)')
    plt.ylabel('Frequency')
    plt.title('Fake Data Accuracy Distribution')
    plt.grid(True, alpha=0.3)
    plt.axvline(np.mean(results['fake_accuracy']), color='red', linestyle='--', label=f'Mean: {np.mean(results["fake_accuracy"]):.2%}')
    plt.legend()

    # 6. Overall Accuracy
    ax6 = plt.subplot(3, 3, 6)
    plt.hist(results['overall_accuracy'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    plt.xlabel('Overall Accuracy')
    plt.ylabel('Frequency')
    plt.title('Overall Accuracy Distribution')
    plt.grid(True, alpha=0.3)
    plt.axvline(np.mean(results['overall_accuracy']), color='red', linestyle='--', label=f'Mean: {np.mean(results["overall_accuracy"]):.2%}')
    plt.legend()

    # 7. Accuracy vs Num Gates
    ax7 = plt.subplot(3, 3, 7)
    plt.scatter(results['num_gates'], results['overall_accuracy'], alpha=0.6, color='steelblue')
    plt.xlabel('Number of Gates')
    plt.ylabel('Overall Accuracy')
    plt.title('Accuracy vs Complexity')
    plt.grid(True, alpha=0.3)

    # 8. Gate Type Distribution
    ax8 = plt.subplot(3, 3, 8)
    if results['gate_types']:
        gate_names = list(results['gate_types'].keys())
        gate_counts = list(results['gate_types'].values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(gate_names)))
        plt.bar(gate_names, gate_counts, color=colors, edgecolor='black')
        plt.xlabel('Gate Type')
        plt.ylabel('Total Count')
        plt.title('Gate Type Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

    # 9. Reward vs Accuracy
    ax9 = plt.subplot(3, 3, 9)
    plt.scatter(results['overall_accuracy'], results['rewards'], alpha=0.6, color='coral')
    plt.xlabel('Overall Accuracy')
    plt.ylabel('Reward')
    plt.title('Reward vs Accuracy')
    plt.grid(True, alpha=0.3)

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

    metrics = {
        'Number of Gates': results['num_gates'],
        'Reward': results['rewards'],
        'Log-Reward': results['log_rewards'],
        'Real Accuracy': results['real_accuracy'],
        'Fake Accuracy': results['fake_accuracy'],
        'Overall Accuracy': results['overall_accuracy']
    }

    for name, data in metrics.items():
        mean_val = np.mean(data)
        std_val = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)

        if 'Accuracy' in name:
            print(f"{name:<30} {mean_val:>13.2%} {std_val:>13.2%} {min_val:>13.2%} {max_val:>13.2%}")
        elif 'Reward' in name and 'Log' not in name:
            print(f"{name:<30} {mean_val:>13.6f} {std_val:>13.6f} {min_val:>13.6f} {max_val:>13.6f}")
        else:
            print(f"{name:<30} {mean_val:>13.2f} {std_val:>13.2f} {min_val:>13.2f} {max_val:>13.2f}")

    print("\n" + "="*80)
    print("Gate Type Distribution")
    print("="*80)

    if results['gate_types']:
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
