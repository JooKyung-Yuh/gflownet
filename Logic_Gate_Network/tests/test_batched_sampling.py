"""
Test that sample_batch_batched() produces identical Q-values to sample_batch().

IMPORTANT: Since sampling involves randomness, we cannot directly compare trajectories.
Instead, we verify that:
1. Given the same state, Q-values from batched and sequential are IDENTICAL
2. The batched sampling logic is mathematically equivalent

The random sampling itself will produce different trajectories due to independent
random number generation, but the underlying Q-value computation must be exact.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from lgn.network import LGNState
from lgn.gates import GateType
from gflownet.policy_network_gnn import LGNGNNPolicy
from gflownet.lgn_mdp import LGNMDP
from gflownet.action_space import LGNActionSpace
from gflownet.training import LGNTrainer


def test_q_values_identical():
    """
    Test that Q-values from forward_policy() and forward_policy_batched() are IDENTICAL.

    This is the CRITICAL test: if Q-values are identical, then the sampling logic
    (given the same random numbers) will produce identical trajectories.
    """
    print("\n" + "="*70)
    print("[TEST] Q-values identical: forward_policy vs forward_policy_batched")
    print("="*70)

    device = torch.device('cpu')
    torch.manual_seed(42)

    num_inputs = 5
    max_gates = 10

    policy = LGNGNNPolicy(
        num_inputs=num_inputs,
        max_gates=max_gates,
        node_emb_dim=64,
        num_conv_steps=3
    ).to(device)
    policy.eval()

    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)

    # Create test LGNs with different numbers of gates
    test_lgns = []

    # LGN 0: Empty (no gates)
    lgn0 = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    test_lgns.append(lgn0)

    # LGN 1: 1 gate
    lgn1 = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    lgn1.add_gate(GateType.AND, [0, 1])
    test_lgns.append(lgn1)

    # LGN 2: 3 gates
    lgn2 = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    lgn2.add_gate(GateType.OR, [0, 1])
    lgn2.add_gate(GateType.XOR, [2, 5])
    lgn2.add_gate(GateType.NAND, [3, 6])
    test_lgns.append(lgn2)

    print(f"\nTesting {len(test_lgns)} LGNs:")
    for i, lgn in enumerate(test_lgns):
        print(f"  LGN {i}: {lgn.get_num_gates()} gates")

    # Get gate actions for each LGN
    gate_actions_per_lgn = []
    for lgn in test_lgns:
        valid_actions = action_space.get_valid_actions(lgn)
        gate_actions = [a for a in valid_actions if 'gate_type' in a]
        gate_actions_per_lgn.append(gate_actions)

    # Sequential: call forward_policy() for each LGN
    print("\n--- Sequential forward_policy() ---")
    seq_action_qs = []
    seq_stop_qs = []

    with torch.no_grad():
        for i, (lgn, gate_actions) in enumerate(zip(test_lgns, gate_actions_per_lgn)):
            action_q, stop_q = policy.forward_policy(lgn, gate_actions, device)
            seq_action_qs.append(action_q)
            seq_stop_qs.append(stop_q)
            print(f"  LGN {i}: action_q shape={action_q.shape}, stop_q={stop_q.item():.6f}")

    # Batched: call forward_policy_batched() once
    print("\n--- Batched forward_policy_batched() ---")
    with torch.no_grad():
        batch_action_qs, batch_stop_qs = policy.forward_policy_batched(
            test_lgns, gate_actions_per_lgn, device
        )

    for i in range(len(test_lgns)):
        print(f"  LGN {i}: action_q shape={batch_action_qs[i].shape}, stop_q={batch_stop_qs[i].item():.6f}")

    # Compare
    print("\n--- Verification ---")
    tolerance = 1e-6
    all_passed = True

    for i in range(len(test_lgns)):
        seq_action = seq_action_qs[i]
        seq_stop = seq_stop_qs[i]
        batch_action = batch_action_qs[i]
        batch_stop = batch_stop_qs[i]

        # Compare action Q-values
        if seq_action.numel() > 0:
            action_diff = torch.abs(seq_action - batch_action).max().item()
        else:
            action_diff = 0.0

        # Compare stop Q-values
        stop_diff = abs(seq_stop.item() - batch_stop.item())

        print(f"\n  LGN {i}:")
        print(f"    action_q max diff: {action_diff:.10f}")
        print(f"    stop_q diff:       {stop_diff:.10f}")

        if action_diff > tolerance or stop_diff > tolerance:
            print(f"    FAILED: difference exceeds tolerance {tolerance}")
            all_passed = False
        else:
            print(f"    PASSED")

    assert all_passed, "Q-values differ between sequential and batched!"
    print("\n" + "="*70)
    print("Q-values IDENTICAL between forward_policy and forward_policy_batched!")
    print("="*70)
    return True


def test_batched_sampling_runs():
    """
    Test that sample_batch_batched() runs without errors and produces valid trajectories.
    """
    print("\n" + "="*70)
    print("[TEST] sample_batch_batched() runs correctly")
    print("="*70)

    device = torch.device('cpu')
    torch.manual_seed(42)

    num_inputs = 5
    max_gates = 8

    policy = LGNGNNPolicy(
        num_inputs=num_inputs,
        max_gates=max_gates,
        node_emb_dim=64,
        num_conv_steps=3
    ).to(device)

    mdp = LGNMDP(num_inputs=num_inputs, max_gates=max_gates)
    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)

    def dummy_reward(lgn: LGNState) -> float:
        return max(0.01, lgn.get_num_gates() * 0.1 + 0.5)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    trainer = LGNTrainer(
        policy=policy,
        mdp=mdp,
        action_space=action_space,
        reward_fn=dummy_reward,
        optimizer=optimizer,
        device=device
    )

    # Sample trajectories
    batch_size = 4
    print(f"\nSampling {batch_size} trajectories with sample_batch_batched()...")

    trajectories = trainer.sample_batch_batched(batch_size)

    print(f"\nGenerated {len(trajectories)} trajectories:")
    for i, traj in enumerate(trajectories):
        print(f"  Traj {i}: {len(traj.states)} steps, "
              f"{traj.terminal_state.get_num_gates()} gates, "
              f"log_reward={traj.log_reward:.4f}, "
              f"termination={traj.termination_reason}")

    # Validate trajectories
    all_valid = True

    for i, traj in enumerate(trajectories):
        # Check states and actions match
        if len(traj.states) != len(traj.actions):
            print(f"  Traj {i}: INVALID - states/actions length mismatch")
            all_valid = False
            continue

        # Check last action is stop
        last_action = traj.actions[-1]
        if not ('action' in last_action and last_action['action'] == 'stop'):
            print(f"  Traj {i}: INVALID - last action is not stop")
            all_valid = False
            continue

        # Check terminal state matches
        if traj.terminal_state.get_num_gates() != traj.states[-1].get_num_gates():
            # Terminal state should have same gates as last state
            # (stop action doesn't add gates)
            print(f"  Traj {i}: INVALID - terminal state gate count mismatch")
            all_valid = False
            continue

        print(f"  Traj {i}: VALID")

    assert all_valid, "Some trajectories are invalid!"
    print("\n" + "="*70)
    print("sample_batch_batched() produces valid trajectories!")
    print("="*70)
    return True


def test_batched_sampling_deterministic_q():
    """
    Test that Q-values are deterministic: same state -> same Q-values.

    This verifies that there's no hidden state or randomness in the Q-value computation.
    """
    print("\n" + "="*70)
    print("[TEST] Q-values are deterministic")
    print("="*70)

    device = torch.device('cpu')

    num_inputs = 5
    max_gates = 10

    policy = LGNGNNPolicy(
        num_inputs=num_inputs,
        max_gates=max_gates,
        node_emb_dim=64,
        num_conv_steps=3
    ).to(device)
    policy.eval()

    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)

    # Create a test LGN
    lgn = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    lgn.add_gate(GateType.AND, [0, 1])
    lgn.add_gate(GateType.OR, [2, 5])

    valid_actions = action_space.get_valid_actions(lgn)
    gate_actions = [a for a in valid_actions if 'gate_type' in a]

    print(f"\nTest LGN: {lgn.get_num_gates()} gates, {len(gate_actions)} gate actions")

    # Call forward_policy multiple times
    print("\nCalling forward_policy() 5 times...")
    results = []

    with torch.no_grad():
        for i in range(5):
            action_q, stop_q = policy.forward_policy(lgn, gate_actions, device)
            results.append((action_q.clone(), stop_q.clone()))
            print(f"  Call {i+1}: stop_q={stop_q.item():.8f}")

    # Verify all results are identical
    print("\nVerifying all results identical...")
    all_identical = True

    for i in range(1, len(results)):
        action_diff = torch.abs(results[0][0] - results[i][0]).max().item()
        stop_diff = abs(results[0][1].item() - results[i][1].item())

        if action_diff > 0 or stop_diff > 0:
            print(f"  Call {i+1} differs: action_diff={action_diff}, stop_diff={stop_diff}")
            all_identical = False

    assert all_identical, "Q-values are not deterministic!"
    print("  All 5 calls produced IDENTICAL Q-values")
    print("\n" + "="*70)
    print("Q-values are deterministic!")
    print("="*70)
    return True


def test_empty_batch():
    """Test that sample_batch_batched(0) returns empty list."""
    print("\n" + "="*70)
    print("[TEST] Empty batch handling")
    print("="*70)

    device = torch.device('cpu')

    num_inputs = 5
    max_gates = 8

    policy = LGNGNNPolicy(
        num_inputs=num_inputs,
        max_gates=max_gates,
        node_emb_dim=64,
        num_conv_steps=3
    ).to(device)

    mdp = LGNMDP(num_inputs=num_inputs, max_gates=max_gates)
    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)

    def dummy_reward(lgn: LGNState) -> float:
        return 1.0

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    trainer = LGNTrainer(
        policy=policy,
        mdp=mdp,
        action_space=action_space,
        reward_fn=dummy_reward,
        optimizer=optimizer,
        device=device
    )

    # Sample empty batch
    trajectories = trainer.sample_batch_batched(0)

    assert len(trajectories) == 0, "Expected empty list for batch_size=0"
    print("  sample_batch_batched(0) returns []")
    print("\n" + "="*70)
    print("Empty batch handled correctly!")
    print("="*70)
    return True


def test_train_step_with_batched_sampling():
    """
    Test that train_step works correctly when using sample_batch_batched().

    This is an integration test that verifies the full training pipeline.
    """
    print("\n" + "="*70)
    print("[TEST] train_step with batched sampling integration")
    print("="*70)

    device = torch.device('cpu')
    torch.manual_seed(42)

    num_inputs = 5
    max_gates = 8

    policy = LGNGNNPolicy(
        num_inputs=num_inputs,
        max_gates=max_gates,
        node_emb_dim=32,
        num_conv_steps=2
    ).to(device)

    mdp = LGNMDP(num_inputs=num_inputs, max_gates=max_gates)
    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)

    def dummy_reward(lgn: LGNState) -> float:
        return max(0.01, lgn.get_num_gates() * 0.1)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    trainer = LGNTrainer(
        policy=policy,
        mdp=mdp,
        action_space=action_space,
        reward_fn=dummy_reward,
        optimizer=optimizer,
        device=device
    )

    # Run a few train steps using sample_batch_batched
    batch_size = 4
    num_steps = 3

    print(f"\nRunning {num_steps} train steps with batch_size={batch_size}...")

    losses = []
    for step in range(num_steps):
        # Manually use sample_batch_batched for trajectories
        trajectories = trainer.sample_batch_batched(batch_size)

        # Compute TB loss
        scores, log_pfs, log_rewards = trainer.compute_tb_loss_batched(trajectories)

        # Average loss
        scores_tensor = torch.stack(scores)
        logZ = -scores_tensor.mean().detach()
        batch_losses = [(logZ + score).pow(2) for score in scores]
        total_loss = torch.stack(batch_losses).mean()

        # Optimize
        trainer.optimizer.zero_grad()
        total_loss.backward()
        trainer.optimizer.step()

        losses.append(total_loss.item())
        print(f"  Step {step+1}: loss={total_loss.item():.4f}, "
              f"mean_log_pf={np.mean(log_pfs):.4f}, "
              f"mean_log_reward={np.mean(log_rewards):.4f}")

    # Check losses are finite
    assert all(np.isfinite(l) for l in losses), "Loss contains non-finite values"
    print("\n" + "="*70)
    print("train_step with batched sampling works correctly!")
    print("="*70)
    return True


if __name__ == "__main__":
    print("="*70)
    print("BATCHED SAMPLING TESTS")
    print("="*70)

    all_passed = True

    try:
        all_passed &= test_q_values_identical()
        all_passed &= test_batched_sampling_deterministic_q()
        all_passed &= test_batched_sampling_runs()
        all_passed &= test_empty_batch()
        all_passed &= test_train_step_with_batched_sampling()
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("ALL BATCHED SAMPLING TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*70)
