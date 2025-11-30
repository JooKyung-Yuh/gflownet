"""
Unit tests for batched GNN forward pass.

Verifies that forward_batched() produces the same results as
calling forward() individually for each LGN.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lgn.network import LGNState
from lgn.gates import GateType
from gflownet.policy_network_gnn import LGNGNNPolicy
from gflownet.lgn_to_graph import lgn_to_graph, batch_lgn_to_batched_graph


def test_batch_lgn_to_batched_graph():
    """Test that batch_lgn_to_batched_graph creates correct batched structure."""
    print("\n[Test 1] batch_lgn_to_batched_graph structure test...")

    device = torch.device('cpu')

    # Create 3 LGNs with different numbers of gates
    lgn1 = LGNState(num_inputs=5, max_gates=10)  # Empty: 5 nodes

    lgn2 = LGNState(num_inputs=5, max_gates=10)
    lgn2.add_gate(GateType.AND, [0, 1])  # 6 nodes

    lgn3 = LGNState(num_inputs=5, max_gates=10)
    lgn3.add_gate(GateType.OR, [0, 1])
    lgn3.add_gate(GateType.XOR, [2, 5])  # 7 nodes

    lgns = [lgn1, lgn2, lgn3]

    # Create batched graph
    batch, num_nodes_list, num_inputs_list, has_edges_list = batch_lgn_to_batched_graph(lgns, device)

    # Verify structure
    assert num_nodes_list == [5, 6, 7], f"Expected [5, 6, 7], got {num_nodes_list}"
    assert num_inputs_list == [5, 5, 5], f"Expected [5, 5, 5], got {num_inputs_list}"
    assert has_edges_list == [False, True, True], f"Expected [False, True, True], got {has_edges_list}"
    assert batch.x.shape[0] == 18, f"Expected 18 total nodes, got {batch.x.shape[0]}"
    assert batch.batch.shape[0] == 18, f"batch.batch should have 18 elements"

    # Verify batch assignments
    assert (batch.batch[:5] == 0).all(), "First 5 nodes should belong to graph 0"
    assert (batch.batch[5:11] == 1).all(), "Next 6 nodes should belong to graph 1"
    assert (batch.batch[11:18] == 2).all(), "Last 7 nodes should belong to graph 2"

    print("  ✅ Batched graph structure is correct!")
    return True


def test_forward_batched_vs_individual():
    """Test that forward_batched produces same results as individual forward calls.

    Note: When mixing empty graphs (no edges) with non-empty graphs in a batch,
    there can be small numerical differences due to how message passing handles
    the batched edge_index. This is expected and acceptable for training.
    """
    print("\n[Test 2] forward_batched vs individual forward consistency test...")

    device = torch.device('cpu')

    # Create policy
    policy = LGNGNNPolicy(
        num_inputs=5,
        max_gates=10,
        node_emb_dim=64,
        num_conv_steps=2
    ).to(device)
    policy.eval()  # Disable dropout etc.

    # Test with LGNs that ALL have edges (same structure type)
    # This ensures exact consistency
    lgn1 = LGNState(num_inputs=5, max_gates=10)
    lgn1.add_gate(GateType.AND, [0, 1])

    lgn2 = LGNState(num_inputs=5, max_gates=10)
    lgn2.add_gate(GateType.OR, [0, 1])

    lgn3 = LGNState(num_inputs=5, max_gates=10)
    lgn3.add_gate(GateType.XOR, [2, 3])
    lgn3.add_gate(GateType.NAND, [0, 5])

    lgns = [lgn1, lgn2, lgn3]

    with torch.no_grad():
        # Individual forward passes
        individual_results = []
        for lgn in lgns:
            graph = lgn_to_graph(lgn, device)
            node_logits, gate_type_logits, stop_logit = policy.forward(graph)
            individual_results.append({
                'node_logits': node_logits,
                'gate_type_logits': gate_type_logits,
                'stop_logit': stop_logit
            })

        # Batched forward pass
        batch, num_nodes_list, _, has_edges_list = batch_lgn_to_batched_graph(lgns, device)
        node_logits_list, gate_type_logits_batch, stop_logits_batch = policy.forward_batched(
            batch, num_nodes_list, has_edges_list
        )

    # Compare results - strict tolerance for graphs with edges
    tolerance = 1e-5

    for i, (individual, batched_node_logits) in enumerate(zip(individual_results, node_logits_list)):
        # Compare node logits
        diff_node = torch.abs(individual['node_logits'] - batched_node_logits).max().item()
        assert diff_node < tolerance, f"Graph {i}: node_logits differ by {diff_node}"

        # Compare gate type logits
        diff_gate = torch.abs(individual['gate_type_logits'] - gate_type_logits_batch[i]).max().item()
        assert diff_gate < tolerance, f"Graph {i}: gate_type_logits differ by {diff_gate}"

        # Compare stop logits
        diff_stop = torch.abs(individual['stop_logit'] - stop_logits_batch[i]).item()
        assert diff_stop < tolerance, f"Graph {i}: stop_logit differs by {diff_stop}"

    print("  ✅ forward_batched produces identical results to individual forward!")
    return True


def test_forward_batched_mixed_edges():
    """Test forward_batched with mixed edge counts (empty + non-empty graphs).

    When batch contains graphs with and without edges, small numerical differences
    can occur due to batched message passing. This test verifies the differences
    are within acceptable bounds for training.
    """
    print("\n[Test 2b] forward_batched with mixed edge counts (relaxed tolerance)...")

    device = torch.device('cpu')

    policy = LGNGNNPolicy(
        num_inputs=5,
        max_gates=10,
        node_emb_dim=64,
        num_conv_steps=2
    ).to(device)
    policy.eval()

    # Mix of empty and non-empty graphs
    lgn1 = LGNState(num_inputs=5, max_gates=10)  # Empty - 0 edges

    lgn2 = LGNState(num_inputs=5, max_gates=10)
    lgn2.add_gate(GateType.AND, [0, 1])  # Has edges

    lgn3 = LGNState(num_inputs=5, max_gates=10)
    lgn3.add_gate(GateType.OR, [0, 1])
    lgn3.add_gate(GateType.XOR, [2, 5])  # Has edges

    lgns = [lgn1, lgn2, lgn3]

    with torch.no_grad():
        individual_results = []
        for lgn in lgns:
            graph = lgn_to_graph(lgn, device)
            node_logits, gate_type_logits, stop_logit = policy.forward(graph)
            individual_results.append({
                'node_logits': node_logits,
                'gate_type_logits': gate_type_logits,
                'stop_logit': stop_logit
            })

        batch, num_nodes_list, _, has_edges_list = batch_lgn_to_batched_graph(lgns, device)
        node_logits_list, gate_type_logits_batch, stop_logits_batch = policy.forward_batched(
            batch, num_nodes_list, has_edges_list
        )

    # Strict tolerance - now that empty graphs are handled correctly,
    # results should be identical to individual forward passes
    tolerance = 1e-5

    all_close = True
    for i, (individual, batched_node_logits) in enumerate(zip(individual_results, node_logits_list)):
        diff_node = torch.abs(individual['node_logits'] - batched_node_logits).max().item()
        diff_gate = torch.abs(individual['gate_type_logits'] - gate_type_logits_batch[i]).max().item()
        diff_stop = torch.abs(individual['stop_logit'] - stop_logits_batch[i]).item()

        has_edges = len(lgns[i].gates) > 0

        if diff_node > tolerance or diff_gate > tolerance or diff_stop > tolerance:
            print(f"    Graph {i} (edges={has_edges}): node_diff={diff_node:.6f}, gate_diff={diff_gate:.6f}, stop_diff={diff_stop:.6f}")
            all_close = False
        else:
            print(f"    Graph {i} (edges={has_edges}): ✓ within tolerance")

    assert all_close, "Some graphs exceed tolerance"
    print("  ✅ Mixed-edge batch results are within acceptable tolerance!")
    return True


def test_forward_policy_batched_vs_individual():
    """Test that forward_policy_batched produces same Q-values as individual calls."""
    print("\n[Test 3] forward_policy_batched vs individual forward_policy consistency test...")

    device = torch.device('cpu')

    # Create policy
    policy = LGNGNNPolicy(
        num_inputs=5,
        max_gates=10,
        node_emb_dim=64,
        num_conv_steps=2
    ).to(device)
    policy.eval()

    # Create LGNs
    lgn1 = LGNState(num_inputs=5, max_gates=10)
    lgn1.add_gate(GateType.AND, [0, 1])

    lgn2 = LGNState(num_inputs=5, max_gates=10)
    lgn2.add_gate(GateType.OR, [0, 1])
    lgn2.add_gate(GateType.XOR, [2, 5])

    lgns = [lgn1, lgn2]

    # Define gate actions for each LGN
    gate_actions_1 = [
        {'gate_type': GateType.OR, 'input_indices': (0, 2)},
        {'gate_type': GateType.AND, 'input_indices': (3, 5)},
    ]
    gate_actions_2 = [
        {'gate_type': GateType.NAND, 'input_indices': (1, 6)},
        {'gate_type': GateType.XOR, 'input_indices': (0, 5)},
        {'gate_type': GateType.NOT, 'input_indices': (4,)},
    ]
    gate_actions_per_lgn = [gate_actions_1, gate_actions_2]

    with torch.no_grad():
        # Individual calls
        individual_action_q = []
        individual_stop_q = []
        for lgn, actions in zip(lgns, gate_actions_per_lgn):
            action_q, stop_q = policy.forward_policy(lgn, actions, device)
            individual_action_q.append(action_q)
            individual_stop_q.append(stop_q)

        # Batched call
        batched_action_q_list, batched_stop_q = policy.forward_policy_batched(
            lgns, gate_actions_per_lgn, device
        )

    # Compare
    tolerance = 1e-5

    for i in range(len(lgns)):
        # Action Q-values
        diff_action = torch.abs(individual_action_q[i] - batched_action_q_list[i]).max().item()
        assert diff_action < tolerance, f"LGN {i}: action Q-values differ by {diff_action}"

        # Stop Q-values
        diff_stop = torch.abs(individual_stop_q[i] - batched_stop_q[i]).item()
        assert diff_stop < tolerance, f"LGN {i}: stop Q-value differs by {diff_stop}"

    print("  ✅ forward_policy_batched produces identical Q-values!")
    return True


def test_empty_batch():
    """Test handling of empty batch."""
    print("\n[Test 4] Empty batch handling test...")

    device = torch.device('cpu')

    # Create policy
    policy = LGNGNNPolicy(
        num_inputs=5,
        max_gates=10,
        node_emb_dim=64,
        num_conv_steps=2
    ).to(device)

    # Empty lists
    lgns = []
    gate_actions_per_lgn = []

    with torch.no_grad():
        action_q_list, stop_q = policy.forward_policy_batched(lgns, gate_actions_per_lgn, device)

    assert len(action_q_list) == 0, "Should return empty action_q_list"
    assert stop_q.shape[0] == 0, "Should return empty stop_q"

    print("  ✅ Empty batch handled correctly!")
    return True


def test_single_lgn_batch():
    """Test batch with single LGN."""
    print("\n[Test 5] Single LGN batch test...")

    device = torch.device('cpu')

    policy = LGNGNNPolicy(
        num_inputs=5,
        max_gates=10,
        node_emb_dim=64,
        num_conv_steps=2
    ).to(device)
    policy.eval()

    lgn = LGNState(num_inputs=5, max_gates=10)
    lgn.add_gate(GateType.AND, [0, 1])

    gate_actions = [
        {'gate_type': GateType.OR, 'input_indices': (0, 2)},
    ]

    with torch.no_grad():
        # Individual
        individual_action_q, individual_stop_q = policy.forward_policy(lgn, gate_actions, device)

        # Batched (single element)
        batched_action_q_list, batched_stop_q = policy.forward_policy_batched(
            [lgn], [gate_actions], device
        )

    tolerance = 1e-5

    diff_action = torch.abs(individual_action_q - batched_action_q_list[0]).max().item()
    assert diff_action < tolerance, f"Action Q-values differ by {diff_action}"

    diff_stop = torch.abs(individual_stop_q - batched_stop_q[0]).item()
    assert diff_stop < tolerance, f"Stop Q-value differs by {diff_stop}"

    print("  ✅ Single LGN batch works correctly!")
    return True


def test_compute_action_logprobs_batched():
    """Test that compute_action_logprobs_batched produces same results as individual calls."""
    print("\n[Test 6] compute_action_logprobs_batched vs individual test...")

    device = torch.device('cpu')

    policy = LGNGNNPolicy(
        num_inputs=5,
        max_gates=10,
        node_emb_dim=64,
        num_conv_steps=2
    ).to(device)
    policy.eval()

    # Create LGNs with different structures
    lgn1 = LGNState(num_inputs=5, max_gates=10)
    lgn1.add_gate(GateType.AND, [0, 1])

    lgn2 = LGNState(num_inputs=5, max_gates=10)
    lgn2.add_gate(GateType.OR, [0, 1])
    lgn2.add_gate(GateType.XOR, [2, 5])

    lgn3 = LGNState(num_inputs=5, max_gates=10)  # Empty (only inputs)

    lgns = [lgn1, lgn2, lgn3]

    # Define actions for each LGN (including stop action)
    actions_1 = [
        {'gate_type': GateType.OR, 'input_indices': (0, 2)},
        {'gate_type': GateType.AND, 'input_indices': (3, 5)},
        {'action': 'stop'},
    ]
    actions_2 = [
        {'gate_type': GateType.NAND, 'input_indices': (1, 6)},
        {'gate_type': GateType.XOR, 'input_indices': (0, 5)},
        {'action': 'stop'},
    ]
    actions_3 = [
        {'gate_type': GateType.NOT, 'input_indices': (0,)},
        {'gate_type': GateType.AND, 'input_indices': (1, 2)},
        # No stop action for empty LGN (stop is only valid when gates > 0)
    ]
    actions_per_lgn = [actions_1, actions_2, actions_3]

    with torch.no_grad():
        # Individual calls
        individual_log_probs = []
        for lgn, actions in zip(lgns, actions_per_lgn):
            log_probs = policy.compute_action_logprobs(lgn, actions, device)
            individual_log_probs.append(log_probs)

        # Batched call
        batched_log_probs = policy.compute_action_logprobs_batched(lgns, actions_per_lgn, device)

    # Compare results - strict tolerance
    tolerance = 1e-5
    all_passed = True

    for i in range(len(lgns)):
        diff = torch.abs(individual_log_probs[i] - batched_log_probs[i]).max().item()
        if diff > tolerance:
            print(f"    LGN {i}: log_probs differ by {diff:.8f}")
            all_passed = False
        else:
            print(f"    LGN {i}: ✓ within tolerance (diff={diff:.8f})")

    assert all_passed, "Some log_probs exceed tolerance"
    print("  ✅ compute_action_logprobs_batched produces identical results!")
    return True


def test_compute_q_values_batched():
    """Test that compute_q_values_batched produces same results as individual calls."""
    print("\n[Test 7] compute_q_values_batched vs individual test...")

    device = torch.device('cpu')

    policy = LGNGNNPolicy(
        num_inputs=5,
        max_gates=10,
        node_emb_dim=64,
        num_conv_steps=2
    ).to(device)
    policy.eval()

    # Create LGNs
    lgn1 = LGNState(num_inputs=5, max_gates=10)
    lgn1.add_gate(GateType.AND, [0, 1])

    lgn2 = LGNState(num_inputs=5, max_gates=10)
    lgn2.add_gate(GateType.OR, [0, 1])

    lgns = [lgn1, lgn2]

    # Define actions for each LGN
    actions_1 = [
        {'gate_type': GateType.OR, 'input_indices': (0, 2)},
        {'gate_type': GateType.AND, 'input_indices': (3, 5)},
        {'action': 'stop'},
    ]
    actions_2 = [
        {'gate_type': GateType.NAND, 'input_indices': (1, 5)},
        {'action': 'stop'},
    ]
    actions_per_lgn = [actions_1, actions_2]

    with torch.no_grad():
        # Individual calls
        individual_q_values = []
        for lgn, actions in zip(lgns, actions_per_lgn):
            q_values = policy.compute_q_values(lgn, actions, device)
            individual_q_values.append(q_values)

        # Batched call
        batched_q_values = policy.compute_q_values_batched(lgns, actions_per_lgn, device)

    # Compare results
    tolerance = 1e-5
    all_passed = True

    for i in range(len(lgns)):
        diff = torch.abs(individual_q_values[i] - batched_q_values[i]).max().item()
        if diff > tolerance:
            print(f"    LGN {i}: Q-values differ by {diff:.8f}")
            all_passed = False
        else:
            print(f"    LGN {i}: ✓ within tolerance (diff={diff:.8f})")

    assert all_passed, "Some Q-values exceed tolerance"
    print("  ✅ compute_q_values_batched produces identical results!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Batched Forward Pass Unit Tests")
    print("=" * 60)

    all_passed = True

    try:
        all_passed &= test_batch_lgn_to_batched_graph()
        all_passed &= test_forward_batched_vs_individual()
        all_passed &= test_forward_batched_mixed_edges()
        all_passed &= test_forward_policy_batched_vs_individual()
        all_passed &= test_empty_batch()
        all_passed &= test_single_lgn_batch()
        all_passed &= test_compute_action_logprobs_batched()
        all_passed &= test_compute_q_values_batched()
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
