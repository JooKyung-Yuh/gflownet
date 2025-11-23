"""
Simple Training Test - Verify full training pipeline works
Run with: pytest tests/test_simple_training.py -v -s
"""

import torch
from lgn.network import LGNState
from lgn.gates import GateType
from gflownet.policy_network_gnn import LGNGNNPolicy
from gflownet.training import LGNTrainer
from gflownet.lgn_mdp import LGNMDP
from gflownet.action_space import LGNActionSpace
from reward.reward_fn import RewardFunction


def test_simple_training():
    """
    Simple training test - 10 iterations to verify pipeline works.
    """
    print("\n" + "=" * 80)
    print("Logic Gate Network GFlowNet - Simple Training Test")
    print("=" * 80)

    # Configuration
    num_inputs = 10
    max_gates = 15
    num_iterations = 10
    batch_size = 4
    learning_rate = 1e-3
    device = 'cpu'

    print(f"\nConfiguration:")
    print(f"  num_inputs: {num_inputs}")
    print(f"  max_gates: {max_gates}")
    print(f"  iterations: {num_iterations}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  device: {device}")

    # 1. Create components
    print("\n[1/4] Creating components...")
    print("  - Policy Network (GNN)...")
    policy = LGNGNNPolicy(
        num_inputs=num_inputs,
        max_gates=max_gates,
        node_emb_dim=128,
        num_conv_steps=3
    )

    print("  - MDP...")
    mdp = LGNMDP(num_inputs=num_inputs, max_gates=max_gates)

    print("  - Action Space...")
    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)

    print("  - Reward Function...")
    reward_fn = RewardFunction()

    # 2. Create optimizer and trainer
    print("\n[2/4] Creating optimizer and trainer...")
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    trainer = LGNTrainer(
        policy=policy,
        mdp=mdp,
        action_space=action_space,
        reward_fn=lambda lgn: reward_fn.compute_reward(lgn, [], []),
        optimizer=optimizer,
        device=torch.device(device),
        balanced_loss=True,
        leaf_coef=10.0
    )

    # 3. Test trajectory sampling
    print("\n[3/4] Testing trajectory sampling...")
    print("  Sampling a single trajectory...")
    trajectory = trainer.sample_trajectory()
    print(f"  ✅ Trajectory sampled: {len(trajectory)} transitions")
    assert len(trajectory) > 0, "Trajectory should have at least one transition"

    # 4. Run training
    print("\n[4/4] Running training...")
    print(f"  Training for {num_iterations} iterations with batch_size={batch_size}")
    print()

    metrics = trainer.train(
        num_iterations=num_iterations,
        batch_size=batch_size,
        print_every=2
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

    print("\n✅ All systems operational! Ready for full experiments.")

    # Assertions
    assert len(metrics['loss']) == num_iterations
    assert all(isinstance(loss, (int, float)) for loss in metrics['loss'])
    assert len(traj_lengths) > 0
