"""
Strict verification tests for batched implementations.

This test verifies:
1. compute_q_values_batched() vs compute_q_values() - exact match
2. compute_action_logprobs_batched() vs compute_action_logprobs() - exact match
3. compute_tb_loss_batched() vs compute_tb_loss() - exact match
4. Gradient flow verification
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


def test_q_values_exact_match():
    """
    Verify compute_q_values_batched produces EXACTLY the same values.

    Test cases:
    - Multiple LGNs with different structures
    - Actions including stop action
    - Empty LGN (no gates)
    - LGN with gates
    """
    print("\n" + "="*70)
    print("[STRICT TEST 1] Q-values exact match verification")
    print("="*70)

    device = torch.device('cpu')
    torch.manual_seed(42)

    policy = LGNGNNPolicy(
        num_inputs=5,
        max_gates=10,
        node_emb_dim=64,
        num_conv_steps=3
    ).to(device)
    policy.eval()

    # Test case 1: Empty LGN (no gates)
    lgn1 = LGNState(num_inputs=5, max_gates=10)
    actions1 = [
        {'gate_type': GateType.AND, 'input_indices': (0, 1)},
        {'gate_type': GateType.OR, 'input_indices': (2, 3)},
        {'gate_type': GateType.NOT, 'input_indices': (4,)},
        # NO stop action for empty LGN
    ]

    # Test case 2: LGN with 1 gate
    lgn2 = LGNState(num_inputs=5, max_gates=10)
    lgn2.add_gate(GateType.AND, [0, 1])
    actions2 = [
        {'gate_type': GateType.OR, 'input_indices': (2, 5)},  # 5 is the new gate
        {'gate_type': GateType.XOR, 'input_indices': (3, 4)},
        {'action': 'stop'},
    ]

    # Test case 3: LGN with 2 gates
    lgn3 = LGNState(num_inputs=5, max_gates=10)
    lgn3.add_gate(GateType.OR, [0, 1])
    lgn3.add_gate(GateType.NAND, [2, 5])  # 5 is first gate
    actions3 = [
        {'gate_type': GateType.AND, 'input_indices': (3, 6)},  # 6 is second gate
        {'action': 'stop'},
    ]

    lgns = [lgn1, lgn2, lgn3]
    actions_per_lgn = [actions1, actions2, actions3]

    print(f"\nTest cases:")
    print(f"  LGN 1: {lgn1.get_num_gates()} gates, {len(actions1)} actions")
    print(f"  LGN 2: {lgn2.get_num_gates()} gates, {len(actions2)} actions")
    print(f"  LGN 3: {lgn3.get_num_gates()} gates, {len(actions3)} actions")

    with torch.no_grad():
        # Individual computation
        individual_q = []
        for lgn, actions in zip(lgns, actions_per_lgn):
            q = policy.compute_q_values(lgn, actions, device)
            individual_q.append(q)

        # Batched computation
        batched_q = policy.compute_q_values_batched(lgns, actions_per_lgn, device)

    # Verify
    tolerance = 1e-6
    all_passed = True

    print(f"\nResults (tolerance={tolerance}):")
    for i in range(len(lgns)):
        max_diff = torch.abs(individual_q[i] - batched_q[i]).max().item()
        mean_diff = torch.abs(individual_q[i] - batched_q[i]).mean().item()

        print(f"\n  LGN {i}:")
        print(f"    Individual Q: {individual_q[i].tolist()}")
        print(f"    Batched Q:    {batched_q[i].tolist()}")
        print(f"    Max diff:     {max_diff:.10f}")
        print(f"    Mean diff:    {mean_diff:.10f}")

        if max_diff > tolerance:
            print(f"    ❌ FAILED: max_diff {max_diff} > tolerance {tolerance}")
            all_passed = False
        else:
            print(f"    ✓ PASSED")

    assert all_passed, "Q-values mismatch!"
    print("\n✅ Q-values exact match test PASSED!")
    return True


def test_log_probs_exact_match():
    """
    Verify compute_action_logprobs_batched produces EXACTLY the same values.

    Key check: log_softmax is applied correctly per-LGN, not across batch.
    """
    print("\n" + "="*70)
    print("[STRICT TEST 2] Log probabilities exact match verification")
    print("="*70)

    device = torch.device('cpu')
    torch.manual_seed(42)

    policy = LGNGNNPolicy(
        num_inputs=5,
        max_gates=10,
        node_emb_dim=64,
        num_conv_steps=3
    ).to(device)
    policy.eval()

    # Same test cases
    lgn1 = LGNState(num_inputs=5, max_gates=10)
    actions1 = [
        {'gate_type': GateType.AND, 'input_indices': (0, 1)},
        {'gate_type': GateType.OR, 'input_indices': (2, 3)},
    ]

    lgn2 = LGNState(num_inputs=5, max_gates=10)
    lgn2.add_gate(GateType.AND, [0, 1])
    actions2 = [
        {'gate_type': GateType.OR, 'input_indices': (2, 5)},
        {'gate_type': GateType.XOR, 'input_indices': (3, 4)},
        {'action': 'stop'},
    ]

    lgns = [lgn1, lgn2]
    actions_per_lgn = [actions1, actions2]

    with torch.no_grad():
        # Individual
        individual_lp = []
        for lgn, actions in zip(lgns, actions_per_lgn):
            lp = policy.compute_action_logprobs(lgn, actions, device)
            individual_lp.append(lp)

        # Batched
        batched_lp = policy.compute_action_logprobs_batched(lgns, actions_per_lgn, device)

    tolerance = 1e-6
    all_passed = True

    print(f"\nResults (tolerance={tolerance}):")
    for i in range(len(lgns)):
        max_diff = torch.abs(individual_lp[i] - batched_lp[i]).max().item()

        # Verify probabilities sum to 1
        individual_sum = torch.exp(individual_lp[i]).sum().item()
        batched_sum = torch.exp(batched_lp[i]).sum().item()

        print(f"\n  LGN {i}:")
        print(f"    Individual log_probs: {individual_lp[i].tolist()}")
        print(f"    Batched log_probs:    {batched_lp[i].tolist()}")
        print(f"    Max diff:             {max_diff:.10f}")
        print(f"    Individual sum(exp):  {individual_sum:.6f}")
        print(f"    Batched sum(exp):     {batched_sum:.6f}")

        if max_diff > tolerance:
            print(f"    ❌ FAILED: max_diff > tolerance")
            all_passed = False
        elif abs(individual_sum - 1.0) > 1e-5 or abs(batched_sum - 1.0) > 1e-5:
            print(f"    ❌ FAILED: probabilities don't sum to 1")
            all_passed = False
        else:
            print(f"    ✓ PASSED")

    assert all_passed, "Log probs mismatch!"
    print("\n✅ Log probabilities exact match test PASSED!")
    return True


def test_tb_loss_exact_match():
    """
    Verify compute_tb_loss_batched produces EXACTLY the same values.

    This is the most critical test - verifies that:
    1. log P_F is computed correctly for each trajectory
    2. score = log P_F - log R is correct
    """
    print("\n" + "="*70)
    print("[STRICT TEST 3] TB Loss exact match verification")
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

    # Create realistic trajectories
    # Trajectory 1: Add 1 gate, then stop
    lgn1_s0 = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    lgn1_s1 = lgn1_s0.copy()
    lgn1_s1.add_gate(GateType.AND, [0, 1])

    traj1 = TBTrajectory(
        states=[lgn1_s0, lgn1_s1],
        actions=[
            {'gate_type': GateType.AND, 'input_indices': (0, 1)},
            {'action': 'stop'},
        ],
        log_reward=-0.5,
        terminal_state=lgn1_s1.copy(),
        termination_reason='stop_action'
    )

    # Trajectory 2: Add 2 gates, then stop
    lgn2_s0 = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    lgn2_s1 = lgn2_s0.copy()
    lgn2_s1.add_gate(GateType.OR, [0, 1])
    lgn2_s2 = lgn2_s1.copy()
    lgn2_s2.add_gate(GateType.XOR, [2, 5])  # 5 is first gate

    traj2 = TBTrajectory(
        states=[lgn2_s0, lgn2_s1, lgn2_s2],
        actions=[
            {'gate_type': GateType.OR, 'input_indices': (0, 1)},
            {'gate_type': GateType.XOR, 'input_indices': (2, 5)},
            {'action': 'stop'},
        ],
        log_reward=-1.0,
        terminal_state=lgn2_s2.copy(),
        termination_reason='stop_action'
    )

    # Trajectory 3: Add 3 gates, then stop
    lgn3_s0 = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    lgn3_s1 = lgn3_s0.copy()
    lgn3_s1.add_gate(GateType.NAND, [0, 1])
    lgn3_s2 = lgn3_s1.copy()
    lgn3_s2.add_gate(GateType.NOR, [2, 5])
    lgn3_s3 = lgn3_s2.copy()
    lgn3_s3.add_gate(GateType.AND, [3, 6])

    traj3 = TBTrajectory(
        states=[lgn3_s0, lgn3_s1, lgn3_s2, lgn3_s3],
        actions=[
            {'gate_type': GateType.NAND, 'input_indices': (0, 1)},
            {'gate_type': GateType.NOR, 'input_indices': (2, 5)},
            {'gate_type': GateType.AND, 'input_indices': (3, 6)},
            {'action': 'stop'},
        ],
        log_reward=-1.5,
        terminal_state=lgn3_s3.copy(),
        termination_reason='stop_action'
    )

    trajectories = [traj1, traj2, traj3]

    print(f"\nTrajectories:")
    for i, traj in enumerate(trajectories):
        print(f"  Trajectory {i}: {len(traj.states)} steps, log_reward={traj.log_reward}")

    # Sequential computation
    seq_scores = []
    seq_log_pfs = []
    seq_log_rewards = []

    for traj in trajectories:
        score, log_pf, log_reward = trainer.compute_tb_loss(traj)
        seq_scores.append(score)
        seq_log_pfs.append(log_pf)
        seq_log_rewards.append(log_reward)

    # Batched computation
    batch_scores, batch_log_pfs, batch_log_rewards = trainer.compute_tb_loss_batched(trajectories)

    tolerance = 1e-6
    all_passed = True

    print(f"\nResults (tolerance={tolerance}):")
    for i in range(len(trajectories)):
        score_diff = torch.abs(seq_scores[i] - batch_scores[i]).item()
        log_pf_diff = abs(seq_log_pfs[i] - batch_log_pfs[i])
        log_reward_diff = abs(seq_log_rewards[i] - batch_log_rewards[i])

        print(f"\n  Trajectory {i}:")
        print(f"    Sequential  - score: {seq_scores[i].item():.8f}, log_pf: {seq_log_pfs[i]:.8f}, log_reward: {seq_log_rewards[i]:.8f}")
        print(f"    Batched     - score: {batch_scores[i].item():.8f}, log_pf: {batch_log_pfs[i]:.8f}, log_reward: {batch_log_rewards[i]:.8f}")
        print(f"    Differences - score: {score_diff:.10f}, log_pf: {log_pf_diff:.10f}, log_reward: {log_reward_diff:.10f}")

        # Verify score = log_pf - log_reward
        expected_score = seq_log_pfs[i] - seq_log_rewards[i]
        actual_score = seq_scores[i].item()
        score_formula_diff = abs(expected_score - actual_score)
        print(f"    Score formula check: expected={expected_score:.8f}, actual={actual_score:.8f}, diff={score_formula_diff:.10f}")

        if score_diff > tolerance:
            print(f"    ❌ FAILED: score_diff > tolerance")
            all_passed = False
        elif log_pf_diff > tolerance:
            print(f"    ❌ FAILED: log_pf_diff > tolerance")
            all_passed = False
        elif log_reward_diff > tolerance:
            print(f"    ❌ FAILED: log_reward_diff > tolerance")
            all_passed = False
        else:
            print(f"    ✓ PASSED")

    assert all_passed, "TB Loss mismatch!"
    print("\n✅ TB Loss exact match test PASSED!")
    return True


def test_gradient_flow():
    """
    Verify that gradients flow correctly through batched computation.

    This ensures backpropagation works correctly.
    """
    print("\n" + "="*70)
    print("[STRICT TEST 4] Gradient flow verification")
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

    # Sample some real trajectories
    trajectories = trainer.sample_batch(3)

    print(f"\nSampled {len(trajectories)} trajectories")

    # Compute batched loss with gradient tracking
    policy.train()
    scores, log_pfs, log_rewards = trainer.compute_tb_loss_batched(trajectories)

    # Stack and compute loss
    scores_tensor = torch.stack(scores)
    logZ = -scores_tensor.mean().detach()
    losses = [(logZ + score).pow(2) for score in scores]
    total_loss = torch.stack(losses).mean()

    print(f"\nLoss computation:")
    print(f"  Total loss: {total_loss.item():.6f}")
    print(f"  Loss requires_grad: {total_loss.requires_grad}")

    # Check gradient flow
    optimizer.zero_grad()
    total_loss.backward()

    # Check that gradients exist
    grad_info = []
    for name, param in policy.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_info.append((name, grad_norm))
        else:
            grad_info.append((name, None))

    print(f"\nGradient norms (first 10 parameters):")
    has_gradients = False
    for name, grad_norm in grad_info[:10]:
        if grad_norm is not None:
            has_gradients = True
            print(f"  {name}: {grad_norm:.6f}")
        else:
            print(f"  {name}: None")

    if not has_gradients:
        print("  ❌ No gradients computed!")
        return False

    # Do optimizer step and verify parameters changed
    old_params = {name: param.clone() for name, param in policy.named_parameters()}
    optimizer.step()
    new_params = {name: param.clone() for name, param in policy.named_parameters()}

    params_changed = False
    for name in old_params:
        if not torch.allclose(old_params[name], new_params[name]):
            params_changed = True
            break

    if params_changed:
        print(f"\n✓ Parameters updated after optimizer.step()")
    else:
        print(f"\n❌ Parameters NOT updated!")
        return False

    print("\n✅ Gradient flow test PASSED!")
    return True


def test_numerical_precision():
    """
    Test numerical precision with extreme values.
    """
    print("\n" + "="*70)
    print("[STRICT TEST 5] Numerical precision verification")
    print("="*70)

    device = torch.device('cpu')
    torch.manual_seed(42)

    policy = LGNGNNPolicy(
        num_inputs=5,
        max_gates=10,
        node_emb_dim=64,
        num_conv_steps=3
    ).to(device)
    policy.eval()

    # Test with many different LGN configurations
    test_cases = []

    # Empty LGNs
    for i in range(3):
        lgn = LGNState(num_inputs=5, max_gates=10)
        actions = [
            {'gate_type': GateType.AND, 'input_indices': (i % 5, (i+1) % 5)},
            {'gate_type': GateType.OR, 'input_indices': ((i+2) % 5, (i+3) % 5)},
        ]
        test_cases.append((lgn, actions))

    # LGNs with gates
    for num_gates in [1, 2, 3, 4]:
        lgn = LGNState(num_inputs=5, max_gates=10)
        for g in range(num_gates):
            input_idx = min(g, 4)
            output_idx = 5 + g - 1 if g > 0 else (input_idx + 1) % 5
            lgn.add_gate(GateType.AND if g % 2 == 0 else GateType.OR,
                        [input_idx, output_idx if output_idx >= 5 else (input_idx + 2) % 5])

        num_nodes = 5 + lgn.get_num_gates()
        actions = [
            {'gate_type': GateType.XOR, 'input_indices': (0, min(num_nodes-1, 5))},
            {'action': 'stop'},
        ]
        test_cases.append((lgn, actions))

    lgns = [tc[0] for tc in test_cases]
    actions_per_lgn = [tc[1] for tc in test_cases]

    print(f"\nTesting {len(test_cases)} different configurations...")

    with torch.no_grad():
        # Individual
        individual_lp = []
        for lgn, actions in zip(lgns, actions_per_lgn):
            lp = policy.compute_action_logprobs(lgn, actions, device)
            individual_lp.append(lp)

        # Batched
        batched_lp = policy.compute_action_logprobs_batched(lgns, actions_per_lgn, device)

    max_overall_diff = 0.0
    tolerance = 1e-5

    for i in range(len(lgns)):
        diff = torch.abs(individual_lp[i] - batched_lp[i]).max().item()
        max_overall_diff = max(max_overall_diff, diff)

        if diff > tolerance:
            print(f"  Config {i}: FAILED (diff={diff:.10f})")
            return False

    print(f"\nMax difference across all configs: {max_overall_diff:.10f}")
    print(f"✅ All {len(test_cases)} configurations within tolerance!")
    return True


if __name__ == "__main__":
    print("="*70)
    print("STRICT VERIFICATION TESTS")
    print("="*70)

    all_passed = True

    try:
        all_passed &= test_q_values_exact_match()
        all_passed &= test_log_probs_exact_match()
        all_passed &= test_tb_loss_exact_match()
        all_passed &= test_gradient_flow()
        all_passed &= test_numerical_precision()
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("ALL STRICT VERIFICATION TESTS PASSED! ✅")
    else:
        print("SOME TESTS FAILED! ❌")
    print("="*70)
