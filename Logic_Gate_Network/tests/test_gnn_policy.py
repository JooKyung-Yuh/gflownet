"""
Tests for GNN-based Policy Network.

This module tests the Graph Neural Network policy implementation for LGN GFlowNet.
"""

import pytest
import torch
from lgn.network import LGNState
from lgn.gates import GateType, GATE_TYPE_TO_IDX
from gflownet.lgn_to_graph import lgn_to_graph
from gflownet.policy_network_gnn import LGNGNNPolicy
from gflownet.action_space import LGNActionSpace


@pytest.fixture
def device():
    """Test device."""
    return torch.device('cpu')


@pytest.fixture
def policy(device):
    """Create a test policy network."""
    return LGNGNNPolicy(
        num_inputs=5,
        max_gates=10,
        node_emb_dim=32,  # Small for testing
        num_conv_steps=2,
    ).to(device)


@pytest.fixture
def action_space():
    """Create action space."""
    return LGNActionSpace(
        num_inputs=5,
        max_gates=10,
    )


def test_lgn_to_graph_empty():
    """Test converting empty LGN to graph."""
    lgn = LGNState(num_inputs=5, max_gates=10)
    graph = lgn_to_graph(lgn)

    assert graph.x.shape[0] == 5  # 5 input nodes
    assert graph.edge_index.shape[1] == 0  # No edges
    assert graph.num_inputs == 5
    assert graph.num_gates == 0


def test_lgn_to_graph_with_gates():
    """Test converting LGN with gates to graph."""
    lgn = LGNState(num_inputs=5, max_gates=10)

    # Add gate: AND(input_0, input_1)
    lgn.add_gate(GateType.AND, (0, 1))

    # Add gate: OR(input_2, gate_0)
    lgn.add_gate(GateType.OR, (2, 5))  # 5 = num_inputs + 0 (gate_0)

    graph = lgn_to_graph(lgn)

    assert graph.x.shape[0] == 7  # 5 inputs + 2 gates
    assert graph.edge_index.shape[1] == 4  # 4 edges (2 per gate)
    assert graph.num_inputs == 5
    assert graph.num_gates == 2

    # Check node types
    assert graph.x[0] == 0  # Input
    assert graph.x[5] == GATE_TYPE_TO_IDX[GateType.AND] + 1  # Gate 0 (gates are 1-16)
    assert graph.x[6] == GATE_TYPE_TO_IDX[GateType.OR] + 1  # Gate 1 (gates are 1-16)


def test_gnn_forward_empty(policy, device):
    """Test GNN forward pass on empty LGN."""
    lgn = LGNState(num_inputs=5, max_gates=10)
    graph = lgn_to_graph(lgn, device)

    node_logits, gate_type_logits, stop_logit = policy(graph)

    assert node_logits.shape == (5,)  # 5 nodes
    assert gate_type_logits.shape == (16,)  # 16 gate types
    assert stop_logit.shape == ()  # Scalar


def test_gnn_forward_with_gates(policy, device):
    """Test GNN forward pass with gates."""
    lgn = LGNState(num_inputs=5, max_gates=10)
    lgn.add_gate(GateType.AND, (0, 1))
    lgn.add_gate(GateType.OR, (2, 5))

    graph = lgn_to_graph(lgn, device)
    node_logits, gate_type_logits, stop_logit = policy(graph)

    assert node_logits.shape == (7,)  # 5 inputs + 2 gates
    assert gate_type_logits.shape == (16,)
    assert stop_logit.shape == ()


def test_compute_q_values(policy, action_space, device):
    """Test computing Q-values for actions."""
    lgn = LGNState(num_inputs=5, max_gates=10)

    # Get valid actions
    actions = action_space.get_valid_actions(lgn)

    # Compute Q-values
    q_values = policy.compute_q_values(lgn, actions, device)

    assert q_values.shape == (len(actions),)
    assert torch.all(torch.isfinite(q_values))  # No NaN or Inf


def test_compute_q_values_with_stop(policy, device):
    """Test Q-values including stop action."""
    lgn = LGNState(num_inputs=5, max_gates=10)
    lgn.add_gate(GateType.AND, (0, 1))

    # Create actions including stop
    actions = [
        {'gate_type': GateType.OR, 'input_indices': (0, 1)},
        {'action': 'stop'}
    ]

    q_values = policy.compute_q_values(lgn, actions, device)

    assert q_values.shape == (2,)
    assert torch.all(torch.isfinite(q_values))


def test_compute_action_logprobs(policy, action_space, device):
    """Test computing action log probabilities."""
    lgn = LGNState(num_inputs=5, max_gates=10)
    actions = action_space.get_valid_actions(lgn)

    log_probs = policy.compute_action_logprobs(lgn, actions, device)

    assert log_probs.shape == (len(actions),)
    assert torch.all(log_probs <= 0)  # Log probs are negative
    assert torch.isclose(torch.exp(log_probs).sum(), torch.tensor(1.0), atol=1e-5)  # Sum to 1


def test_sum_exp_q_values(policy, action_space, device):
    """Test sum of exp(Q) computation."""
    lgn = LGNState(num_inputs=5, max_gates=10)
    actions = action_space.get_valid_actions(lgn)

    sum_exp_q = policy.sum_exp_q_values(lgn, actions, device)

    assert sum_exp_q.shape == ()  # Scalar
    assert sum_exp_q > 0  # Positive
    assert torch.isfinite(sum_exp_q)


def test_gradients_flow(policy, action_space, device):
    """Test that gradients flow through the network."""
    lgn = LGNState(num_inputs=5, max_gates=10)
    lgn.add_gate(GateType.AND, (0, 1))

    actions = action_space.get_valid_actions(lgn)
    q_values = policy.compute_q_values(lgn, actions, device)

    # Compute loss
    loss = q_values.sum()

    # Backward
    loss.backward()

    # Check gradients exist
    has_grad = False
    for param in policy.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients computed"


def test_policy_consistency(policy, device):
    """Test that forward pass is deterministic."""
    lgn = LGNState(num_inputs=5, max_gates=10)
    lgn.add_gate(GateType.AND, (0, 1))

    graph = lgn_to_graph(lgn, device)

    # Forward twice
    out1 = policy(graph)
    out2 = policy(graph)

    # Should be identical (deterministic)
    assert torch.allclose(out1[0], out2[0])  # node_logits
    assert torch.allclose(out1[1], out2[1])  # gate_type_logits
    assert torch.allclose(out1[2], out2[2])  # stop_logit


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
