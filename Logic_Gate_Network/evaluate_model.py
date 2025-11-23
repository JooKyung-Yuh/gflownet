"""
Evaluate Trained GFlowNet Model
================================

Evaluate a trained LGN GFlowNet model on test data and generate performance metrics.

Usage:
    python evaluate_model.py --model experiments/trained_model.pt --num-samples 10
"""

import torch
import argparse
import numpy as np
from pathlib import Path

from lgn.network import LGNState
from gflownet.policy_network_gnn import LGNGNNPolicy
from gflownet.action_space import LGNActionSpace
from reward.reward_fn import RewardFunction
from data.generator import RealDataGenerator, FakeDataGenerator
from data.dataset import LGNDataset
from rules.rule_1 import Rule1_NoConsecutive1s


def sample_lgn_from_policy(policy, action_space, num_inputs, max_gates, device='cpu'):
    """
    Sample an LGN from the trained policy using greedy selection.

    Args:
        policy: Trained LGNGNNPolicy
        action_space: LGNActionSpace
        num_inputs: Number of inputs
        max_gates: Maximum number of gates
        device: Device to run on

    Returns:
        LGNState: Sampled LGN
    """
    lgn = LGNState(num_inputs=num_inputs, max_gates=max_gates)

    for step in range(max_gates + 10):  # Safety limit
        valid_actions = action_space.get_valid_actions(lgn)

        if not valid_actions:
            break

        # Separate gate actions from stop action
        gate_actions = [a for a in valid_actions if 'gate_type' in a]

        if not gate_actions:
            break

        # Get Q-values for all actions
        action_q, stop_q = policy.forward_policy(lgn, gate_actions)

        # Greedy selection: choose action with highest Q-value
        if action_q.max() > stop_q:
            best_idx = action_q.argmax().item()
            best_action = gate_actions[best_idx]

            # Apply action
            lgn.add_gate(best_action['gate_type'], best_action['input_indices'])
        else:
            # Stop
            break

    return lgn


def evaluate_model(model_path, num_inputs, max_gates, node_emb_dim=128, num_conv_steps=3, num_samples=10, data_samples=50):
    """
    Evaluate a trained model by sampling LGNs and computing metrics.

    Args:
        model_path: Path to saved model
        num_inputs: Number of inputs
        max_gates: Maximum gates
        node_emb_dim: Node embedding dimension (must match training)
        num_conv_steps: Number of GNN convolution steps (must match training)
        num_samples: Number of LGNs to sample
        data_samples: Number of data samples for evaluation

    Returns:
        dict: Evaluation metrics
    """
    print("="*80)
    print("LGN GFlowNet Model Evaluation")
    print("="*80)

    # Load model
    print(f"\nLoading model from {model_path}...")
    policy = LGNGNNPolicy(num_inputs=num_inputs, max_gates=max_gates,
                         node_emb_dim=node_emb_dim, num_conv_steps=num_conv_steps)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    print("✅ Model loaded successfully")

    # Generate test data
    print(f"\nGenerating test data (dimension={num_inputs})...")
    rule = Rule1_NoConsecutive1s(dimension=num_inputs)
    real_gen = RealDataGenerator()
    fake_gen = FakeDataGenerator()

    real_samples = real_gen.generate(rule, count=data_samples)
    fake_samples = fake_gen.generate(rule, real_samples, count=data_samples)
    print(f"✅ Generated {len(real_samples)} real, {len(fake_samples)} fake samples")

    # Create components
    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)
    reward_fn = RewardFunction()

    # Sample LGNs from policy
    print(f"\nSampling {num_samples} LGNs from trained policy...")
    sampled_lgns = []

    with torch.no_grad():
        for i in range(num_samples):
            lgn = sample_lgn_from_policy(policy, action_space, num_inputs, max_gates)
            sampled_lgns.append(lgn)
            print(f"  LGN {i+1}: {lgn.get_num_gates()} gates")

    # Compute metrics for each LGN
    print(f"\nEvaluating sampled LGNs...")
    metrics = {
        'num_gates': [],
        'log_rewards': [],
        'rewards': [],
        'real_errors': [],
        'fake_acceptances': [],
        'real_accuracy': [],
        'fake_accuracy': []
    }

    for i, lgn in enumerate(sampled_lgns):
        # Compute reward
        log_reward = reward_fn.compute_reward(lgn, real_samples, fake_samples)
        reward = np.exp(log_reward)

        # Get error/acceptance counts
        real_errors = reward_fn.compute_real_error_count(lgn, real_samples)
        fake_accepts = reward_fn.compute_fake_acceptance_count(lgn, fake_samples)

        # Compute accuracies
        real_acc = 1.0 - (real_errors / len(real_samples))
        fake_acc = 1.0 - (fake_accepts / len(fake_samples))

        metrics['num_gates'].append(lgn.get_num_gates())
        metrics['log_rewards'].append(log_reward)
        metrics['rewards'].append(reward)
        metrics['real_errors'].append(real_errors)
        metrics['fake_acceptances'].append(fake_accepts)
        metrics['real_accuracy'].append(real_acc)
        metrics['fake_accuracy'].append(fake_acc)

        print(f"  LGN {i+1}: Gates={lgn.get_num_gates()}, "
              f"Real_Acc={real_acc:.2%}, Fake_Acc={fake_acc:.2%}, Reward={reward:.2e}")

    # Compute statistics
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)

    print(f"\nNumber of Gates:")
    print(f"  Mean: {np.mean(metrics['num_gates']):.2f}")
    print(f"  Std:  {np.std(metrics['num_gates']):.2f}")
    print(f"  Min:  {np.min(metrics['num_gates'])}")
    print(f"  Max:  {np.max(metrics['num_gates'])}")

    print(f"\nReal Data Accuracy:")
    print(f"  Mean: {np.mean(metrics['real_accuracy']):.2%}")
    print(f"  Std:  {np.std(metrics['real_accuracy']):.2%}")
    print(f"  Min:  {np.min(metrics['real_accuracy']):.2%}")
    print(f"  Max:  {np.max(metrics['real_accuracy']):.2%}")

    print(f"\nFake Data Accuracy (Rejection Rate):")
    print(f"  Mean: {np.mean(metrics['fake_accuracy']):.2%}")
    print(f"  Std:  {np.std(metrics['fake_accuracy']):.2%}")
    print(f"  Min:  {np.min(metrics['fake_accuracy']):.2%}")
    print(f"  Max:  {np.max(metrics['fake_accuracy']):.2%}")

    print(f"\nRewards:")
    print(f"  Mean: {np.mean(metrics['rewards']):.6f}")
    print(f"  Std:  {np.std(metrics['rewards']):.6f}")

    print(f"\nLog-Rewards:")
    print(f"  Mean: {np.mean(metrics['log_rewards']):.4f}")
    print(f"  Std:  {np.std(metrics['log_rewards']):.4f}")

    # Find best LGN
    best_idx = np.argmax(metrics['rewards'])
    best_lgn = sampled_lgns[best_idx]

    print(f"\nBest LGN (highest reward):")
    print(f"  Index: {best_idx}")
    print(f"  Gates: {best_lgn.get_num_gates()}")
    print(f"  Real Accuracy: {metrics['real_accuracy'][best_idx]:.2%}")
    print(f"  Fake Accuracy: {metrics['fake_accuracy'][best_idx]:.2%}")
    print(f"  Reward: {metrics['rewards'][best_idx]:.6f}")

    print("\n" + "="*80)

    return metrics, sampled_lgns, best_lgn


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained LGN GFlowNet model')
    parser.add_argument('--model', type=str, default='experiments/trained_model.pt',
                        help='Path to trained model')
    parser.add_argument('--num-inputs', type=int, default=6,
                        help='Number of inputs (must match training)')
    parser.add_argument('--max-gates', type=int, default=3,
                        help='Maximum gates (must match training)')
    parser.add_argument('--node-emb-dim', type=int, default=64,
                        help='Node embedding dimension (must match training)')
    parser.add_argument('--num-conv-steps', type=int, default=2,
                        help='Number of GNN convolution steps (must match training)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of LGNs to sample for evaluation')
    parser.add_argument('--data-samples', type=int, default=50,
                        help='Number of data samples for evaluation')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("\nPlease train a model first:")
        print("  python train_with_real_data.py --num-inputs 6 --max-gates 3 --iterations 30")
        return

    # Evaluate
    metrics, sampled_lgns, best_lgn = evaluate_model(
        model_path=args.model,
        num_inputs=args.num_inputs,
        max_gates=args.max_gates,
        node_emb_dim=args.node_emb_dim,
        num_conv_steps=args.num_conv_steps,
        num_samples=args.num_samples,
        data_samples=args.data_samples
    )

    print(f"\n✅ Evaluation complete!")
    print(f"   - Sampled {len(sampled_lgns)} LGNs")
    print(f"   - Best LGN has {best_lgn.get_num_gates()} gates")
    print(f"\nTo visualize the best LGN:")
    print(f"  python visualize_lgn.py --model {args.model} --num-inputs {args.num_inputs} --max-gates {args.max_gates}")


if __name__ == "__main__":
    main()
