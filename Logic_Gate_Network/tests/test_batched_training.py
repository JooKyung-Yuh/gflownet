"""
Unit tests for batched TB loss computation in training.py.

Tests verify that compute_tb_loss_batched() produces identical results
to the sequential compute_tb_loss() method.
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
from gflownet.training import LGNTrainer, TBTrajectory


def create_test_trajectory(num_inputs: int, max_gates: int, seed: int) -> TBTrajectory:
    """Create a deterministic test trajectory for comparison."""
    np.random.seed(seed)

    lgn = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    states = [lgn.copy()]
    actions = []

    # Add some gates based on seed
    num_gates_to_add = (seed % 3) + 1  # 1, 2, or 3 gates
    gate_types = [GateType.AND, GateType.OR, GateType.XOR, GateType.NAND]

    for i in range(num_gates_to_add):
        gate_type = gate_types[i % len(gate_types)]
        available_nodes = list(range(lgn.num_inputs + lgn.get_num_gates()))

        if gate_type == GateType.NOT:
            input_indices = (available_nodes[i % len(available_nodes)],)
        else:
            idx1 = i % len(available_nodes)
            idx2 = (i + 1) % len(available_nodes)
            if idx1 == idx2:
                idx2 = (idx2 + 1) % len(available_nodes)
            input_indices = (available_nodes[idx1], available_nodes[idx2])

        action = {'gate_type': gate_type, 'input_indices': input_indices}
        actions.append(action)
        lgn.add_gate(gate_type, input_indices)
        states.append(lgn.copy())

    # Add stop action
    actions.append({'action': 'stop'})

    # Compute fake reward
    log_reward = -1.0 - seed * 0.1

    return TBTrajectory(
        states=states[:-1],  # Exclude final state (after stop)
        actions=actions,
        log_reward=log_reward,
        terminal_state=lgn.copy(),
        termination_reason='stop_action'
    )


def test_compute_tb_loss_batched_vs_sequential():
    """Test that compute_tb_loss_batched produces same results as sequential compute_tb_loss."""
    print("\n[Test 1] compute_tb_loss_batched vs sequential test...")

    device = torch.device('cpu')
    num_inputs = 5
    max_gates = 10

    # Create policy and trainer
    policy = LGNGNNPolicy(
        num_inputs=num_inputs,
        max_gates=max_gates,
        node_emb_dim=64,
        num_conv_steps=2
    ).to(device)
    policy.eval()

    mdp = LGNMDP(num_inputs=num_inputs, max_gates=max_gates)
    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)

    def dummy_reward(lgn: LGNState) -> float:
        return np.exp(lgn.get_num_gates() * 0.1)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    trainer = LGNTrainer(
        policy=policy,
        mdp=mdp,
        action_space=action_space,
        reward_fn=dummy_reward,
        optimizer=optimizer,
        device=device
    )

    # Create test trajectories
    trajectories = [create_test_trajectory(num_inputs, max_gates, seed=i) for i in range(5)]

    # Sequential computation
    sequential_scores = []
    sequential_log_pfs = []
    sequential_log_rewards = []

    for traj in trajectories:
        score, log_pf, log_reward = trainer.compute_tb_loss(traj)
        sequential_scores.append(score)
        sequential_log_pfs.append(log_pf)
        sequential_log_rewards.append(log_reward)

    # Batched computation
    batched_scores, batched_log_pfs, batched_log_rewards = trainer.compute_tb_loss_batched(trajectories)

    # Compare results
    tolerance = 1e-5
    all_passed = True

    print(f"  Comparing {len(trajectories)} trajectories...")

    for i in range(len(trajectories)):
        score_diff = torch.abs(sequential_scores[i] - batched_scores[i]).item()
        log_pf_diff = abs(sequential_log_pfs[i] - batched_log_pfs[i])
        log_reward_diff = abs(sequential_log_rewards[i] - batched_log_rewards[i])

        if score_diff > tolerance:
            print(f"    Trajectory {i}: score differs by {score_diff:.8f}")
            all_passed = False
        if log_pf_diff > tolerance:
            print(f"    Trajectory {i}: log_pf differs by {log_pf_diff:.8f}")
            all_passed = False
        if log_reward_diff > tolerance:
            print(f"    Trajectory {i}: log_reward differs by {log_reward_diff:.8f}")
            all_passed = False

        if score_diff <= tolerance and log_pf_diff <= tolerance and log_reward_diff <= tolerance:
            print(f"    Trajectory {i}: ✓ all values within tolerance")

    assert all_passed, "Some values exceed tolerance"
    print("  ✅ compute_tb_loss_batched produces identical results!")
    return True


def test_compute_tb_loss_batched_with_varying_lengths():
    """Test batched loss with trajectories of different lengths."""
    print("\n[Test 2] compute_tb_loss_batched with varying trajectory lengths...")

    device = torch.device('cpu')
    num_inputs = 5
    max_gates = 10

    policy = LGNGNNPolicy(
        num_inputs=num_inputs,
        max_gates=max_gates,
        node_emb_dim=64,
        num_conv_steps=2
    ).to(device)
    policy.eval()

    mdp = LGNMDP(num_inputs=num_inputs, max_gates=max_gates)
    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)

    def dummy_reward(lgn: LGNState) -> float:
        return np.exp(lgn.get_num_gates() * 0.1)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    trainer = LGNTrainer(
        policy=policy,
        mdp=mdp,
        action_space=action_space,
        reward_fn=dummy_reward,
        optimizer=optimizer,
        device=device
    )

    # Create trajectories with explicitly different lengths
    # Trajectory 1: 1 gate + stop (2 steps)
    lgn1 = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    states1 = [lgn1.copy()]
    lgn1.add_gate(GateType.AND, [0, 1])
    states1.append(lgn1.copy())  # State after adding gate, before stop

    traj1 = TBTrajectory(
        states=states1,
        actions=[
            {'gate_type': GateType.AND, 'input_indices': (0, 1)},
            {'action': 'stop'},
        ],
        log_reward=-1.0,
        terminal_state=lgn1.copy(),
        termination_reason='stop_action'
    )

    # Trajectory 2: 3 gates + stop (4 steps)
    lgn2 = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    states2 = [lgn2.copy()]
    lgn2.add_gate(GateType.OR, [0, 1])
    states2.append(lgn2.copy())
    lgn2.add_gate(GateType.XOR, [2, 5])
    states2.append(lgn2.copy())
    lgn2.add_gate(GateType.NAND, [3, 6])
    states2.append(lgn2.copy())  # State after adding 3rd gate, before stop

    traj2 = TBTrajectory(
        states=states2,
        actions=[
            {'gate_type': GateType.OR, 'input_indices': (0, 1)},
            {'gate_type': GateType.XOR, 'input_indices': (2, 5)},
            {'gate_type': GateType.NAND, 'input_indices': (3, 6)},
            {'action': 'stop'},
        ],
        log_reward=-2.0,
        terminal_state=lgn2.copy(),
        termination_reason='stop_action'
    )

    trajectories = [traj1, traj2]

    # Sequential
    sequential_scores = []
    sequential_log_pfs = []

    for traj in trajectories:
        score, log_pf, _ = trainer.compute_tb_loss(traj)
        sequential_scores.append(score)
        sequential_log_pfs.append(log_pf)

    # Batched
    batched_scores, batched_log_pfs, _ = trainer.compute_tb_loss_batched(trajectories)

    tolerance = 1e-5
    all_passed = True

    for i in range(len(trajectories)):
        score_diff = torch.abs(sequential_scores[i] - batched_scores[i]).item()
        log_pf_diff = abs(sequential_log_pfs[i] - batched_log_pfs[i])

        traj_len = len(trajectories[i].states)
        if score_diff > tolerance or log_pf_diff > tolerance:
            print(f"    Trajectory {i} (len={traj_len}): FAILED - score_diff={score_diff:.8f}, log_pf_diff={log_pf_diff:.8f}")
            all_passed = False
        else:
            print(f"    Trajectory {i} (len={traj_len}): ✓ within tolerance")

    assert all_passed, "Some values exceed tolerance"
    print("  ✅ Varying length trajectories handled correctly!")
    return True


def test_train_step_runs_without_error():
    """Test that train_step with batched loss runs without errors."""
    print("\n[Test 3] train_step with batched loss integration test...")

    device = torch.device('cpu')
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

    # Run a few train steps
    batch_size = 4
    num_steps = 3

    losses = []
    for step in range(num_steps):
        loss, metrics = trainer.train_step(batch_size)
        losses.append(loss)
        print(f"    Step {step+1}: loss={loss:.4f}, log_pf={metrics['log_pf']:.4f}, log_reward={metrics['log_reward']:.4f}")

    # Check that training ran successfully
    assert all(np.isfinite(l) for l in losses), "Loss contains non-finite values"
    assert len(losses) == num_steps, "Wrong number of training steps"

    print("  ✅ train_step with batched loss runs successfully!")
    return True


def test_empty_trajectories_batch():
    """Test handling of edge cases."""
    print("\n[Test 4] Empty trajectories batch test...")

    device = torch.device('cpu')
    num_inputs = 5
    max_gates = 10

    policy = LGNGNNPolicy(
        num_inputs=num_inputs,
        max_gates=max_gates,
        node_emb_dim=64,
        num_conv_steps=2
    ).to(device)
    policy.eval()

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

    # Empty batch
    scores, log_pfs, log_rewards = trainer.compute_tb_loss_batched([])

    assert len(scores) == 0, "Expected empty scores list"
    assert len(log_pfs) == 0, "Expected empty log_pfs list"
    assert len(log_rewards) == 0, "Expected empty log_rewards list"

    print("  ✅ Empty batch handled correctly!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Batched Training Unit Tests")
    print("=" * 60)

    all_passed = True

    try:
        all_passed &= test_compute_tb_loss_batched_vs_sequential()
        all_passed &= test_compute_tb_loss_batched_with_varying_lengths()
        all_passed &= test_train_step_runs_without_error()
        all_passed &= test_empty_trajectories_batch()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✅")
    else:
        print("Some tests failed! ❌")
    print("=" * 60)
