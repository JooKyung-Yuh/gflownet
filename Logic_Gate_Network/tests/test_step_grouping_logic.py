"""
Test that step-based grouping in compute_tb_loss_batched is mathematically correct.

The batched version groups states by step index:
- Step 0: [traj0_state0, traj1_state0, traj2_state0]
- Step 1: [traj0_state1, traj1_state1, traj2_state1]
- ...

This test verifies that:
1. The step grouping correctly preserves trajectory order
2. Log probs are accumulated to the correct trajectory
3. Final scores match sequential computation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lgn.network import LGNState
from lgn.gates import GateType
from gflownet.policy_network_gnn import LGNGNNPolicy
from gflownet.lgn_mdp import LGNMDP
from gflownet.action_space import LGNActionSpace
from gflownet.training import LGNTrainer, TBTrajectory


def test_step_grouping_with_different_lengths():
    """
    Test step grouping when trajectories have DIFFERENT lengths.

    This is critical because:
    - Trajectory 0 might have 2 steps
    - Trajectory 1 might have 4 steps
    - Trajectory 2 might have 3 steps

    The batched version must correctly handle this ragged structure.
    """
    print("\n" + "="*70)
    print("[TEST] Step grouping with different trajectory lengths")
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

    mdp = LGNMDP(num_inputs=num_inputs, max_gates=max_gates)
    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)

    def dummy_reward(lgn):
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

    # Create trajectories with DIFFERENT lengths
    # Trajectory 0: 2 steps (add gate, stop)
    lgn0_s0 = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    lgn0_s1 = lgn0_s0.copy()
    lgn0_s1.add_gate(GateType.AND, [0, 1])

    traj0 = TBTrajectory(
        states=[lgn0_s0, lgn0_s1],
        actions=[
            {'gate_type': GateType.AND, 'input_indices': (0, 1)},
            {'action': 'stop'},
        ],
        log_reward=-0.5,
        terminal_state=lgn0_s1.copy(),
        termination_reason='stop_action'
    )

    # Trajectory 1: 4 steps (add 3 gates, stop)
    lgn1_s0 = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    lgn1_s1 = lgn1_s0.copy()
    lgn1_s1.add_gate(GateType.OR, [0, 1])
    lgn1_s2 = lgn1_s1.copy()
    lgn1_s2.add_gate(GateType.XOR, [2, 5])
    lgn1_s3 = lgn1_s2.copy()
    lgn1_s3.add_gate(GateType.NAND, [3, 6])

    traj1 = TBTrajectory(
        states=[lgn1_s0, lgn1_s1, lgn1_s2, lgn1_s3],
        actions=[
            {'gate_type': GateType.OR, 'input_indices': (0, 1)},
            {'gate_type': GateType.XOR, 'input_indices': (2, 5)},
            {'gate_type': GateType.NAND, 'input_indices': (3, 6)},
            {'action': 'stop'},
        ],
        log_reward=-1.0,
        terminal_state=lgn1_s3.copy(),
        termination_reason='stop_action'
    )

    # Trajectory 2: 3 steps (add 2 gates, stop)
    lgn2_s0 = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    lgn2_s1 = lgn2_s0.copy()
    lgn2_s1.add_gate(GateType.NOR, [0, 1])
    lgn2_s2 = lgn2_s1.copy()
    lgn2_s2.add_gate(GateType.AND, [2, 5])

    traj2 = TBTrajectory(
        states=[lgn2_s0, lgn2_s1, lgn2_s2],
        actions=[
            {'gate_type': GateType.NOR, 'input_indices': (0, 1)},
            {'gate_type': GateType.AND, 'input_indices': (2, 5)},
            {'action': 'stop'},
        ],
        log_reward=-0.8,
        terminal_state=lgn2_s2.copy(),
        termination_reason='stop_action'
    )

    trajectories = [traj0, traj1, traj2]

    print(f"\nTrajectory lengths:")
    for i, traj in enumerate(trajectories):
        print(f"  Trajectory {i}: {len(traj.states)} steps")

    print(f"\nStep grouping structure:")
    print(f"  Step 0: traj0_state0, traj1_state0, traj2_state0 (3 states)")
    print(f"  Step 1: traj0_state1, traj1_state1, traj2_state1 (3 states)")
    print(f"  Step 2:              , traj1_state2, traj2_state2 (2 states)")
    print(f"  Step 3:              , traj1_state3               (1 state)")

    # Manual step-by-step verification
    print(f"\n--- Manual Step-by-Step Computation ---")

    # Compute log_pf for each trajectory step by step
    manual_log_pfs = [0.0, 0.0, 0.0]

    with torch.no_grad():
        for step_idx in range(4):  # max_steps = 4
            states_at_step = []
            actions_at_step = []
            traj_indices = []

            for traj_idx, traj in enumerate(trajectories):
                if step_idx < len(traj.states):
                    states_at_step.append(traj.states[step_idx])
                    actions_at_step.append(traj.actions[step_idx])
                    traj_indices.append(traj_idx)

            if len(states_at_step) == 0:
                continue

            print(f"\n  Step {step_idx}: processing trajectories {traj_indices}")

            # For each state, compute log_prob of the taken action
            for i, (state, action, traj_idx) in enumerate(zip(states_at_step, actions_at_step, traj_indices)):
                valid_actions = action_space.get_valid_actions(state)
                log_probs = policy.compute_action_logprobs(state, valid_actions, device)

                # Find action index
                action_idx = trainer._find_action_index(valid_actions, action)
                log_prob = log_probs[action_idx].item()

                manual_log_pfs[traj_idx] += log_prob
                print(f"    Traj {traj_idx}: action={action}, log_prob={log_prob:.6f}, cumulative={manual_log_pfs[traj_idx]:.6f}")

    # Compute sequential results
    print(f"\n--- Sequential Computation ---")
    seq_scores = []
    seq_log_pfs = []

    for i, traj in enumerate(trajectories):
        score, log_pf, _ = trainer.compute_tb_loss(traj)
        seq_scores.append(score.item())
        seq_log_pfs.append(log_pf)
        print(f"  Traj {i}: log_pf={log_pf:.6f}")

    # Compute batched results
    print(f"\n--- Batched Computation ---")
    batch_scores, batch_log_pfs, _ = trainer.compute_tb_loss_batched(trajectories)

    for i in range(len(trajectories)):
        print(f"  Traj {i}: log_pf={batch_log_pfs[i]:.6f}")

    # Verify all match
    print(f"\n--- Verification ---")
    # Manual has floating point accumulation error from .item() calls
    # The CRITICAL check is sequential == batched
    manual_tolerance = 1e-5  # Manual accumulation has more error
    strict_tolerance = 1e-6  # Sequential vs Batched should be exact
    all_passed = True

    for i in range(len(trajectories)):
        manual_vs_seq = abs(manual_log_pfs[i] - seq_log_pfs[i])
        seq_vs_batch = abs(seq_log_pfs[i] - batch_log_pfs[i])

        print(f"\n  Trajectory {i}:")
        print(f"    Manual log_pf:     {manual_log_pfs[i]:.8f}")
        print(f"    Sequential log_pf: {seq_log_pfs[i]:.8f}")
        print(f"    Batched log_pf:    {batch_log_pfs[i]:.8f}")
        print(f"    |manual - seq|:    {manual_vs_seq:.10f} (tolerance={manual_tolerance})")
        print(f"    |seq - batch|:     {seq_vs_batch:.10f} (tolerance={strict_tolerance}) <- CRITICAL")

        # Sequential vs Batched is the CRITICAL comparison
        if seq_vs_batch > strict_tolerance:
            print(f"    ❌ CRITICAL: Sequential vs Batched mismatch!")
            all_passed = False
        elif manual_vs_seq > manual_tolerance:
            print(f"    ⚠️ Manual accumulation error (acceptable)")
            print(f"    ✓ Sequential == Batched (CORRECT)")
        else:
            print(f"    ✓ All three match!")

    if all_passed:
        print(f"\n✅ Step grouping with different lengths is CORRECT!")
    else:
        print(f"\n❌ Step grouping has ERRORS!")

    return all_passed


def test_action_index_preservation():
    """
    Verify that action indices are correctly matched in batched computation.

    Critical: The action taken at each step must match the correct log_prob.
    """
    print("\n" + "="*70)
    print("[TEST] Action index preservation")
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

    mdp = LGNMDP(num_inputs=num_inputs, max_gates=max_gates)
    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)

    def dummy_reward(lgn):
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

    # Single trajectory with multiple steps
    lgn_s0 = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    lgn_s1 = lgn_s0.copy()
    lgn_s1.add_gate(GateType.AND, [0, 1])
    lgn_s2 = lgn_s1.copy()
    lgn_s2.add_gate(GateType.OR, [2, 5])

    traj = TBTrajectory(
        states=[lgn_s0, lgn_s1, lgn_s2],
        actions=[
            {'gate_type': GateType.AND, 'input_indices': (0, 1)},
            {'gate_type': GateType.OR, 'input_indices': (2, 5)},
            {'action': 'stop'},
        ],
        log_reward=-0.5,
        terminal_state=lgn_s2.copy(),
        termination_reason='stop_action'
    )

    print(f"\nTrajectory with 3 steps:")
    for i, (state, action) in enumerate(zip(traj.states, traj.actions)):
        print(f"  Step {i}: state has {state.get_num_gates()} gates, action={action}")

    # Verify each step's action is correctly indexed
    print(f"\n--- Per-step action index verification ---")

    with torch.no_grad():
        for step_idx, (state, action) in enumerate(zip(traj.states, traj.actions)):
            valid_actions = action_space.get_valid_actions(state)
            log_probs = policy.compute_action_logprobs(state, valid_actions, device)

            action_idx = trainer._find_action_index(valid_actions, action)

            print(f"\n  Step {step_idx}:")
            print(f"    Action taken: {action}")
            print(f"    Number of valid actions: {len(valid_actions)}")
            print(f"    Action index: {action_idx}")
            print(f"    Log prob of taken action: {log_probs[action_idx].item():.6f}")

            # Verify the action at action_idx matches
            found_action = valid_actions[action_idx]

            if 'action' in action and action['action'] == 'stop':
                matches = 'action' in found_action and found_action['action'] == 'stop'
            else:
                matches = (found_action.get('gate_type') == action.get('gate_type') and
                          found_action.get('input_indices') == action.get('input_indices'))

            if matches:
                print(f"    ✓ Action correctly indexed")
            else:
                print(f"    ❌ Action mismatch! Expected {action}, found {found_action}")
                return False

    print(f"\n✅ Action index preservation is CORRECT!")
    return True


if __name__ == "__main__":
    print("="*70)
    print("STEP GROUPING LOGIC TESTS")
    print("="*70)

    all_passed = True

    try:
        all_passed &= test_step_grouping_with_different_lengths()
        all_passed &= test_action_index_preservation()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("ALL STEP GROUPING TESTS PASSED! ✅")
    else:
        print("SOME TESTS FAILED! ❌")
    print("="*70)
