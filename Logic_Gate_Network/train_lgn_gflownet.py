"""
Logic Gate Network GFlowNet Training Script
============================================

Train GNN policy network with GFlowNet to generate logic gate networks.

Usage:
    python -m pytest train_lgn_gflownet.py -v -s

Or run as standalone (requires package installation):
    python train_lgn_gflownet.py
"""

import torch
import argparse
from pathlib import Path


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from lgn.network import LGNState
    from lgn.gates import GateType
    from gflownet.policy_network_gnn import LGNGNNPolicy
    from gflownet.training import LGNTrainer
    from gflownet.lgn_mdp import LGNMDP
    from gflownet.action_space import LGNActionSpace
    from reward.reward_fn import RewardFunction

    # Configuration
    parser = argparse.ArgumentParser(description='Train LGN GFlowNet')
    parser.add_argument('--num-inputs', type=int, default=10, help='Number of input features')
    parser.add_argument('--max-gates', type=int, default=15, help='Maximum number of gates')
    parser.add_argument('--iterations', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--node-emb-dim', type=int, default=128, help='Node embedding dimension')
    parser.add_argument('--num-conv-steps', type=int, default=3, help='Number of GNN conv steps')
    parser.add_argument('--print-every', type=int, default=10, help='Print frequency')

    args = parser.parse_args()

    print("=" * 80)
    print("Logic Gate Network GFlowNet - Training")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  num_inputs: {args.num_inputs}")
    print(f"  max_gates: {args.max_gates}")
    print(f"  iterations: {args.iterations}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning_rate: {args.lr}")
    print(f"  device: {args.device}")
    print(f"  node_emb_dim: {args.node_emb_dim}")
    print(f"  num_conv_steps: {args.num_conv_steps}")

    # Create components
    print("\n[1/4] Creating components...")

    print("  - Policy Network (GNN)...")
    policy = LGNGNNPolicy(
        num_inputs=args.num_inputs,
        max_gates=args.max_gates,
        node_emb_dim=args.node_emb_dim,
        num_conv_steps=args.num_conv_steps
    )

    print("  - MDP...")
    mdp = LGNMDP(num_inputs=args.num_inputs, max_gates=args.max_gates)

    print("  - Action Space...")
    action_space = LGNActionSpace(num_inputs=args.num_inputs, max_gates=args.max_gates)

    print("  - Reward Function...")
    reward_fn = RewardFunction()

    # Create optimizer and trainer
    print("\n[2/4] Creating optimizer and trainer...")
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    trainer = LGNTrainer(
        policy=policy,
        mdp=mdp,
        action_space=action_space,
        reward_fn=lambda lgn: reward_fn.compute_reward(lgn, [], []),
        optimizer=optimizer,
        device=torch.device(args.device),
        balanced_loss=True,
        leaf_coef=10.0
    )

    # Test trajectory sampling
    print("\n[3/4] Testing trajectory sampling...")
    trajectory = trainer.sample_trajectory()
    print(f"  ✅ Trajectory sampled: {len(trajectory)} transitions")

    # Run training
    print("\n[4/4] Running training...")
    print(f"  Training for {args.iterations} iterations with batch_size={args.batch_size}")
    print()

    metrics = trainer.train(
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        print_every=args.print_every
    )

    print("\n" + "=" * 80)
    print("Training completed successfully! ✅")
    print("=" * 80)

    # Print final metrics
    print("\nFinal Metrics:")
    print(f"  Loss: {metrics['loss'][-1]:.4f}")
    print(f"  Terminal Loss: {metrics['terminal_loss'][-1]:.4f}")
    print(f"  Flow Loss: {metrics['flow_loss'][-1]:.4f}")
    print(f"  Mean Reward: {metrics['mean_reward'][-1]:.4f}")

    # Print trajectory statistics
    traj_lengths = metrics['trajectory_lengths']
    print(f"\nTrajectory Statistics:")
    print(f"  Mean length: {sum(traj_lengths) / len(traj_lengths):.2f}")
    print(f"  Min length: {min(traj_lengths)}")
    print(f"  Max length: {max(traj_lengths)}")

    # Print loss trend
    print(f"\nLoss Trend:")
    print(f"  Initial loss: {metrics['loss'][0]:.4f}")
    print(f"  Final loss: {metrics['loss'][-1]:.4f}")
    print(f"  Change: {metrics['loss'][-1] - metrics['loss'][0]:.4f}")

    print("\n✅ Training complete!")
    return metrics


def test_full_training():
    """
    Pytest wrapper for full training.
    Run with: pytest train_lgn_gflownet.py -v -s
    """
    import sys
    # Simulate command line args for pytest
    sys.argv = ['train_lgn_gflownet.py', '--iterations', '50', '--print-every', '5']
    metrics = main()

    # Assertions
    assert len(metrics['loss']) == 50
    assert all(isinstance(loss, (int, float)) for loss in metrics['loss'])
    print("\n✅ Test passed!")


if __name__ == "__main__":
    main()
