"""
Test end-to-end training with GNN policy.

This module tests the complete training pipeline:
- GNN policy network
- LGNMDP for parent transitions
- LGNActionSpace for valid actions
- Reward function
- Training loop with TB loss
"""

import pytest
import torch
from lgn.network import LGNState
from lgn.gates import GateType
from gflownet.policy_network_gnn import LGNGNNPolicy
from gflownet.lgn_mdp import LGNMDP
from gflownet.action_space import LGNActionSpace
from gflownet.training import LGNTrainer


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device('cpu')


@pytest.fixture
def num_inputs():
    """Number of inputs."""
    return 5


@pytest.fixture
def max_gates():
    """Maximum gates."""
    return 10


@pytest.fixture
def policy(num_inputs, max_gates, device):
    """Create GNN policy."""
    return LGNGNNPolicy(
        num_inputs=num_inputs,
        max_gates=max_gates,
        node_emb_dim=32,
        num_conv_steps=2,
    ).to(device)


@pytest.fixture
def mdp(num_inputs, max_gates):
    """Create MDP."""
    return LGNMDP(num_inputs=num_inputs, max_gates=max_gates)


@pytest.fixture
def action_space(num_inputs, max_gates):
    """Create action space."""
    return LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)


@pytest.fixture
def reward_fn():
    """Simple reward function for testing."""
    def _reward(lgn: LGNState) -> float:
        # Reward based on number of gates (simple test reward)
        return float(len(lgn.gates))
    return _reward


@pytest.fixture
def trainer(policy, mdp, action_space, reward_fn, device):
    """Create trainer."""
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    return LGNTrainer(
        policy=policy,
        mdp=mdp,
        action_space=action_space,
        reward_fn=reward_fn,
        optimizer=optimizer,
        device=device,
        balanced_loss=True,
        leaf_coef=10.0,
    )


def test_sample_trajectory(trainer, num_inputs, max_gates):
    """Test trajectory sampling."""
    trajectory = trainer.sample_trajectory()

    # Check trajectory structure
    assert len(trajectory) > 0, "Trajectory should have at least one transition"

    # Check last transition is terminal
    parents, actions, reward, state, done = trajectory[-1]
    assert done == True, "Last transition should be terminal"
    assert reward > 0, "Terminal reward should be positive (number of gates)"

    # Check all transitions before last are non-terminal
    for parents, actions, reward, state, done in trajectory[:-1]:
        assert done == False, "Non-terminal transitions should have done=False"
        assert reward == 0.0, "Non-terminal transitions should have reward=0"

    # Check parents and actions have same length
    for parents, actions, reward, state, done in trajectory:
        assert len(parents) == len(actions), "Parents and actions should match"


def test_sample_batch(trainer):
    """Test batch sampling."""
    batch_size = 4
    p_list, pb, a_list, r, s_list, d = trainer.sample_batch(batch_size)

    # Check batch structure
    assert len(p_list) == len(pb) == len(a_list), "Parent batch dimensions should match"
    assert len(s_list) == len(r) == len(d), "State batch dimensions should match"

    # Check parent batch indices are valid
    assert pb.min() >= 0
    assert pb.max() < len(s_list)

    # Check done flags
    assert d.min() >= 0
    assert d.max() <= 1
    assert (d == 0).any() or (d == 1).any(), "Should have some transitions"


def test_compute_tb_loss(trainer):
    """Test TB loss computation."""
    batch_size = 4
    p_list, pb, a_list, r, s_list, d = trainer.sample_batch(batch_size)

    # Compute loss
    loss, metrics = trainer.compute_tb_loss(p_list, pb, a_list, r, s_list, d)

    # Check loss is scalar
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Check metrics
    assert 'loss' in metrics
    assert 'term_loss' in metrics
    assert 'flow_loss' in metrics
    assert 'mean_reward' in metrics
    assert 'num_terminals' in metrics
    assert 'num_transitions' in metrics

    # Check loss can backprop
    loss.backward()

    # Check gradients exist
    for param in trainer.policy.parameters():
        if param.requires_grad:
            assert param.grad is not None, "Gradients should exist after backward"


def test_train_step(trainer):
    """Test single training step."""
    batch_size = 4

    # Get initial parameters
    initial_params = [p.clone() for p in trainer.policy.parameters()]

    # Train step
    loss, metrics = trainer.train_step(batch_size)

    # Check loss
    assert isinstance(loss, float)
    assert loss >= 0

    # Check metrics
    assert 'loss' in metrics
    assert metrics['loss'] >= 0

    # Check parameters were updated
    param_changed = False
    for initial, current in zip(initial_params, trainer.policy.parameters()):
        if not torch.allclose(initial, current):
            param_changed = True
            break

    assert param_changed, "Parameters should change after training step"


def test_train_multiple_steps(trainer):
    """Test multiple training steps."""
    num_iterations = 5
    batch_size = 4

    all_metrics = trainer.train(
        num_iterations=num_iterations,
        batch_size=batch_size,
        log_every=10,
        verbose=False,
    )

    # Check metrics structure
    assert 'loss' in all_metrics
    assert 'term_loss' in all_metrics
    assert 'flow_loss' in all_metrics
    assert 'mean_reward' in all_metrics

    # Check metrics length
    assert len(all_metrics['loss']) == num_iterations
    assert len(all_metrics['term_loss']) == num_iterations
    assert len(all_metrics['flow_loss']) == num_iterations
    assert len(all_metrics['mean_reward']) == num_iterations


def test_policy_forward_interface(policy, num_inputs, max_gates, device):
    """Test policy forward interface matches training expectations."""
    lgn = LGNState(num_inputs=num_inputs, max_gates=max_gates)
    lgn.add_gate(GateType.AND, (0, 1))

    # Get valid actions
    gate_actions = [
        {'gate_type': GateType.OR, 'input_indices': (0, 1)},
        {'gate_type': GateType.XOR, 'input_indices': (2, 3)},
    ]

    # Test forward_policy method
    action_q, stop_q = policy.forward_policy(lgn, gate_actions, device)

    assert action_q.shape == (len(gate_actions),)
    assert stop_q.dim() == 0, "Stop Q should be scalar"


def test_policy_compute_q_value_for_action(policy, num_inputs, max_gates, device):
    """Test single action Q-value computation."""
    lgn = LGNState(num_inputs=num_inputs, max_gates=max_gates)

    action = {'gate_type': GateType.AND, 'input_indices': (0, 1)}
    q_value = policy.compute_q_value_for_action(lgn, action, device)

    assert q_value.dim() == 0, "Q-value should be scalar"
    assert q_value.requires_grad, "Q-value should have gradients"


def test_policy_sum_exp_q_values(policy, num_inputs, max_gates, device):
    """Test sum of exp Q-values computation."""
    lgn = LGNState(num_inputs=num_inputs, max_gates=max_gates)

    actions = [
        {'gate_type': GateType.AND, 'input_indices': (0, 1)},
        {'gate_type': GateType.OR, 'input_indices': (2, 3)},
        {'action': 'stop'}
    ]

    sum_exp_q = policy.sum_exp_q_values(lgn, actions, device)

    assert sum_exp_q.dim() == 0, "Sum should be scalar"
    assert sum_exp_q.item() > 0, "Sum of exp should be positive"
    assert sum_exp_q.requires_grad, "Should have gradients"


def test_integration_with_real_reward(num_inputs, max_gates, device):
    """Integration test with a more realistic reward function."""

    # Create components
    policy = LGNGNNPolicy(
        num_inputs=num_inputs,
        max_gates=max_gates,
        node_emb_dim=64,
        num_conv_steps=3,
    ).to(device)

    mdp = LGNMDP(num_inputs=num_inputs, max_gates=max_gates)
    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)

    # Reward function: prefer networks with more gates but penalize excessive depth
    def reward_fn(lgn: LGNState) -> float:
        if len(lgn.gates) == 0:
            return 0.0
        num_gates = len(lgn.gates)
        # Simple reward: number of gates with slight penalty for very large networks
        return float(num_gates) - 0.01 * max(0, num_gates - 5)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    trainer = LGNTrainer(
        policy=policy,
        mdp=mdp,
        action_space=action_space,
        reward_fn=reward_fn,
        optimizer=optimizer,
        device=device,
        balanced_loss=True,
        leaf_coef=10.0,
    )

    # Train for a few steps
    all_metrics = trainer.train(
        num_iterations=3,
        batch_size=2,
        log_every=10,
        verbose=False,
    )

    # Verify training ran successfully
    assert len(all_metrics['loss']) == 3
    assert all(loss >= 0 for loss in all_metrics['loss'])
    assert all(reward >= 0 for reward in all_metrics['mean_reward'])
