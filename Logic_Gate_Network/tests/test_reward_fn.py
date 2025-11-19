"""
Unit Tests for Reward Function Module.

This module tests the RewardFunction class, metrics functions, and config loader.

Test Coverage:
--------------
1. RewardFunction.__init__: Parameter validation
2. RewardFunction.compute_real_error_count: Error counting on Real data
3. RewardFunction.compute_fake_acceptance_count: Acceptance counting on Fake data
4. RewardFunction._compute_complexity_penalty: Complexity penalty calculation
5. RewardFunction.compute_reward: Full reward computation (integration)
6. Metrics functions: compute_real_accuracy, compute_fake_rejection_rate, compute_metrics_dict
7. Config loader: load_reward_config, validate_reward_config, get_preset_config

Test Strategy:
--------------
- Unit tests for individual methods (isolated)
- Integration tests for compute_reward (all components together)
- Edge case tests (empty data, perfect/worst performance)
- Error handling tests (invalid parameters, missing files)

Running Tests:
--------------
From project root:
    pytest tests/test_reward_fn.py -v
    pytest tests/test_reward_fn.py -v -k "test_init"  # Run specific tests
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# Import modules to test
from Logic_Gate_Network.reward.reward_fn import RewardFunction
from Logic_Gate_Network.reward.metrics import (
    compute_real_accuracy,
    compute_fake_rejection_rate,
    compute_metrics_dict
)
from Logic_Gate_Network.reward.config_loader import (
    load_reward_config,
    validate_reward_config,
    get_preset_config
)
from Logic_Gate_Network.lgn.network import LGNState
from Logic_Gate_Network.lgn.gates import GateType


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_lgn():
    """Create a simple LGN with 2 gates for testing."""
    lgn = LGNState(num_inputs=10, max_gates=15)
    lgn.add_gate(GateType.AND, [0, 1, 2])  # 3-input AND
    lgn.add_gate(GateType.OR, [3, 10])     # OR using feature 3 and gate 0 output
    return lgn


@pytest.fixture
def single_gate_lgn():
    """Create an LGN with 1 gate for testing."""
    lgn = LGNState(num_inputs=10, max_gates=15)
    lgn.add_gate(GateType.AND, [0, 1])  # 2-input AND
    return lgn


@pytest.fixture
def real_data_perfect():
    """Real data that would be perfectly classified by simple AND gate."""
    # For AND gate on [0,1], these should output 1
    return [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Output: 1 ✓
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Output: 1 ✓
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # Output: 1 ✓
    ]


@pytest.fixture
def real_data_with_errors():
    """Real data with some misclassifications."""
    # Mix of samples that AND[0,1] classifies correctly and incorrectly
    return [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Output: 1 ✓ (correct)
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Output: 0 ✗ (error)
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Output: 0 ✗ (error)
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Output: 1 ✓ (correct)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Output: 0 ✗ (error)
    ]  # 3 errors out of 5 samples


@pytest.fixture
def fake_data_perfect_rejection():
    """Fake data that would be perfectly rejected by AND gate."""
    # For AND gate on [0,1], these should output 0
    return [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Output: 0 ✓
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Output: 0 ✓
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Output: 0 ✓
    ]


@pytest.fixture
def fake_data_with_acceptances():
    """Fake data with some incorrect acceptances."""
    return [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Output: 1 ✗ (accepted, bad)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Output: 0 ✓ (rejected, good)
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Output: 0 ✓ (rejected, good)
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Output: 1 ✗ (accepted, bad)
    ]  # 2 acceptances out of 4 samples


@pytest.fixture
def reward_fn_default():
    """Default RewardFunction instance."""
    return RewardFunction(C=1.0, epsilon=1e-6, lambda_complexity=0.1)


@pytest.fixture
def temp_config_file():
    """Create a temporary YAML config file for testing."""
    config_content = """
experiment:
  name: "test_experiment"
  seed: 42
  version: "1.0"

reward_function:
  C: 2.0
  epsilon: 1.0e-10
  lambda_complexity: 0.05

lgn_network:
  num_inputs: 10
  max_gates: 15

presets:
  test_preset:
    C: 3.0
    epsilon: 1.0e-8
    lambda_complexity: 0.2
    num_inputs: 20
    max_gates: 25
"""
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


# ============================================================================
# RewardFunction.__init__ Tests
# ============================================================================

class TestRewardFunctionInit:
    """Test RewardFunction initialization and parameter validation."""

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        reward_fn = RewardFunction(C=1.0, epsilon=1e-6, lambda_complexity=0.1)

        assert reward_fn.C == 1.0
        assert reward_fn.epsilon == 1e-6
        assert reward_fn.lambda_complexity == 0.1

    def test_init_custom_parameters(self):
        """Test initialization with custom parameter values."""
        reward_fn = RewardFunction(C=2.5, epsilon=1e-10, lambda_complexity=0.5)

        assert reward_fn.C == 2.5
        assert reward_fn.epsilon == 1e-10
        assert reward_fn.lambda_complexity == 0.5

    def test_init_invalid_C_zero(self):
        """Test that C=0 raises ValueError."""
        with pytest.raises(ValueError, match="C must be positive"):
            RewardFunction(C=0.0, epsilon=1e-6, lambda_complexity=0.1)

    def test_init_invalid_C_negative(self):
        """Test that negative C raises ValueError."""
        with pytest.raises(ValueError, match="C must be positive"):
            RewardFunction(C=-1.0, epsilon=1e-6, lambda_complexity=0.1)

    def test_init_invalid_epsilon_zero(self):
        """Test that epsilon=0 raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            RewardFunction(C=1.0, epsilon=0.0, lambda_complexity=0.1)

    def test_init_invalid_epsilon_negative(self):
        """Test that negative epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            RewardFunction(C=1.0, epsilon=-1e-6, lambda_complexity=0.1)

    def test_init_invalid_lambda_negative(self):
        """Test that negative lambda_complexity raises ValueError."""
        with pytest.raises(ValueError, match="lambda_complexity must be non-negative"):
            RewardFunction(C=1.0, epsilon=1e-6, lambda_complexity=-0.1)

    def test_init_lambda_zero_allowed(self):
        """Test that lambda_complexity=0 is allowed (no complexity penalty)."""
        reward_fn = RewardFunction(C=1.0, epsilon=1e-6, lambda_complexity=0.0)
        assert reward_fn.lambda_complexity == 0.0


# ============================================================================
# compute_real_error_count Tests
# ============================================================================

class TestComputeRealErrorCount:
    """Test compute_real_error_count method."""

    def test_perfect_classification(self, reward_fn_default, single_gate_lgn, real_data_perfect):
        """Test when all Real samples are correctly classified (error_count = 0)."""
        error_count = reward_fn_default.compute_real_error_count(single_gate_lgn, real_data_perfect)
        assert error_count == 0

    def test_with_errors(self, reward_fn_default, single_gate_lgn, real_data_with_errors):
        """Test when some Real samples are misclassified."""
        error_count = reward_fn_default.compute_real_error_count(single_gate_lgn, real_data_with_errors)
        assert error_count == 3  # 3 errors out of 5 samples

    def test_all_wrong(self, reward_fn_default, single_gate_lgn):
        """Test when all Real samples are misclassified (worst case)."""
        # All samples output 0 for AND gate
        real_data_all_wrong = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        error_count = reward_fn_default.compute_real_error_count(single_gate_lgn, real_data_all_wrong)
        assert error_count == 3  # All 3 samples are errors

    def test_empty_data_raises_error(self, reward_fn_default, single_gate_lgn):
        """Test that empty real_data raises ValueError."""
        with pytest.raises(ValueError, match="real_data cannot be empty"):
            reward_fn_default.compute_real_error_count(single_gate_lgn, [])

    def test_single_sample(self, reward_fn_default, single_gate_lgn):
        """Test with single sample."""
        real_data = [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]  # Correct
        error_count = reward_fn_default.compute_real_error_count(single_gate_lgn, real_data)
        assert error_count == 0

        real_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # Error
        error_count = reward_fn_default.compute_real_error_count(single_gate_lgn, real_data)
        assert error_count == 1

    def test_return_type(self, reward_fn_default, single_gate_lgn, real_data_with_errors):
        """Test that return type is int."""
        error_count = reward_fn_default.compute_real_error_count(single_gate_lgn, real_data_with_errors)
        assert isinstance(error_count, int)


# ============================================================================
# compute_fake_acceptance_count Tests
# ============================================================================

class TestComputeFakeAcceptanceCount:
    """Test compute_fake_acceptance_count method."""

    def test_perfect_rejection(self, reward_fn_default, single_gate_lgn, fake_data_perfect_rejection):
        """Test when all Fake samples are correctly rejected (acceptance_count = 0)."""
        acceptance_count = reward_fn_default.compute_fake_acceptance_count(single_gate_lgn, fake_data_perfect_rejection)
        assert acceptance_count == 0

    def test_with_acceptances(self, reward_fn_default, single_gate_lgn, fake_data_with_acceptances):
        """Test when some Fake samples are incorrectly accepted."""
        acceptance_count = reward_fn_default.compute_fake_acceptance_count(single_gate_lgn, fake_data_with_acceptances)
        assert acceptance_count == 2  # 2 acceptances out of 4 samples

    def test_all_accepted(self, reward_fn_default, single_gate_lgn):
        """Test when all Fake samples are accepted (worst case)."""
        # All samples output 1 for AND gate
        fake_data_all_accepted = [
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        ]
        acceptance_count = reward_fn_default.compute_fake_acceptance_count(single_gate_lgn, fake_data_all_accepted)
        assert acceptance_count == 3  # All 3 samples accepted

    def test_empty_data_raises_error(self, reward_fn_default, single_gate_lgn):
        """Test that empty fake_data raises ValueError."""
        with pytest.raises(ValueError, match="fake_data cannot be empty"):
            reward_fn_default.compute_fake_acceptance_count(single_gate_lgn, [])

    def test_single_sample(self, reward_fn_default, single_gate_lgn):
        """Test with single sample."""
        fake_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # Rejected
        acceptance_count = reward_fn_default.compute_fake_acceptance_count(single_gate_lgn, fake_data)
        assert acceptance_count == 0

        fake_data = [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]  # Accepted
        acceptance_count = reward_fn_default.compute_fake_acceptance_count(single_gate_lgn, fake_data)
        assert acceptance_count == 1

    def test_return_type(self, reward_fn_default, single_gate_lgn, fake_data_with_acceptances):
        """Test that return type is int."""
        acceptance_count = reward_fn_default.compute_fake_acceptance_count(single_gate_lgn, fake_data_with_acceptances)
        assert isinstance(acceptance_count, int)


# ============================================================================
# _compute_complexity_penalty Tests
# ============================================================================

class TestComputeComplexityPenalty:
    """Test _compute_complexity_penalty method."""

    def test_zero_gates(self, reward_fn_default):
        """Test penalty with zero gates."""
        lgn = LGNState(num_inputs=10, max_gates=15)
        penalty = reward_fn_default._compute_complexity_penalty(lgn)
        assert penalty == 0.0

    def test_one_gate(self, reward_fn_default, single_gate_lgn):
        """Test penalty with one gate."""
        penalty = reward_fn_default._compute_complexity_penalty(single_gate_lgn)
        expected = -0.1 * 1  # lambda=0.1, gates=1
        assert penalty == expected

    def test_multiple_gates(self, reward_fn_default, simple_lgn):
        """Test penalty with multiple gates."""
        penalty = reward_fn_default._compute_complexity_penalty(simple_lgn)
        expected = -0.1 * 2  # lambda=0.1, gates=2
        assert penalty == expected

    def test_different_lambda(self, simple_lgn):
        """Test penalty with different lambda values."""
        reward_fn = RewardFunction(C=1.0, epsilon=1e-6, lambda_complexity=0.5)
        penalty = reward_fn._compute_complexity_penalty(simple_lgn)
        expected = -0.5 * 2  # lambda=0.5, gates=2
        assert penalty == expected

    def test_lambda_zero(self, simple_lgn):
        """Test penalty with lambda=0 (no complexity penalty)."""
        reward_fn = RewardFunction(C=1.0, epsilon=1e-6, lambda_complexity=0.0)
        penalty = reward_fn._compute_complexity_penalty(simple_lgn)
        assert penalty == 0.0

    def test_return_type(self, reward_fn_default, simple_lgn):
        """Test that return type is float."""
        penalty = reward_fn_default._compute_complexity_penalty(simple_lgn)
        assert isinstance(penalty, (int, float))

    def test_penalty_always_non_positive(self, reward_fn_default, simple_lgn):
        """Test that penalty is always non-positive (≤ 0)."""
        penalty = reward_fn_default._compute_complexity_penalty(simple_lgn)
        assert penalty <= 0.0


# ============================================================================
# compute_reward Tests (Integration)
# ============================================================================

class TestComputeReward:
    """Test compute_reward method (full integration)."""

    def test_basic_reward_computation(self, reward_fn_default, single_gate_lgn, real_data_with_errors, fake_data_with_acceptances):
        """Test basic reward computation with all components."""
        reward = reward_fn_default.compute_reward(single_gate_lgn, real_data_with_errors, fake_data_with_acceptances)

        # Verify reward is a float
        assert isinstance(reward, (float, np.floating))

        # Verify reward is non-positive (≤ 0)
        assert reward <= 0.0

        # Manual calculation to verify
        # real_error_count = 3, fake_acceptance_count = 2, num_gates = 1
        # real_term = -1.0 * 3 = -3.0
        # fake_term = -log(2 + 1e-6) ≈ -0.693
        # complexity = -0.1 * 1 = -0.1
        # total ≈ -3.0 - 0.693 - 0.1 = -3.793
        expected_approx = -3.0 - np.log(2 + 1e-6) - 0.1
        assert abs(reward - expected_approx) < 0.01

    def test_perfect_performance(self, reward_fn_default, single_gate_lgn, real_data_perfect, fake_data_perfect_rejection):
        """Test reward with perfect performance (0 errors, 0 acceptances)."""
        reward = reward_fn_default.compute_reward(single_gate_lgn, real_data_perfect, fake_data_perfect_rejection)

        # real_term = 0, fake_term = -log(0 + 1e-6) ≈ +13.8 (POSITIVE!), complexity = -0.1
        # Note: When fake_acceptance_count=0, -log(epsilon) becomes positive!
        expected_approx = 0.0 - np.log(1e-6) - 0.1
        assert abs(reward - expected_approx) < 0.1

        # With perfect fake rejection, reward can be positive
        assert reward > 0.0

        # Reward should be better (higher) than non-perfect case
        real_data_imperfect = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 5  # Will have errors
        fake_data_imperfect = [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]] * 4  # Will have acceptances
        reward_imperfect = reward_fn_default.compute_reward(single_gate_lgn, real_data_imperfect, fake_data_imperfect)
        assert reward > reward_imperfect

    def test_worst_performance(self, reward_fn_default, single_gate_lgn):
        """Test reward with worst performance (all errors, all acceptances)."""
        real_data_worst = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 10
        fake_data_worst = [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]] * 10

        reward = reward_fn_default.compute_reward(single_gate_lgn, real_data_worst, fake_data_worst)

        # real_term = -10, fake_term = -log(10 + 1e-6) ≈ -2.3, complexity = -0.1
        expected_approx = -10.0 - np.log(10 + 1e-6) - 0.1
        assert abs(reward - expected_approx) < 0.1

    def test_empty_real_data_raises_error(self, reward_fn_default, single_gate_lgn, fake_data_with_acceptances):
        """Test that empty real_data raises ValueError."""
        with pytest.raises(ValueError, match="real_data cannot be empty"):
            reward_fn_default.compute_reward(single_gate_lgn, [], fake_data_with_acceptances)

    def test_empty_fake_data_raises_error(self, reward_fn_default, single_gate_lgn, real_data_with_errors):
        """Test that empty fake_data raises ValueError."""
        with pytest.raises(ValueError, match="fake_data cannot be empty"):
            reward_fn_default.compute_reward(single_gate_lgn, real_data_with_errors, [])

    def test_different_C_values(self, single_gate_lgn, real_data_with_errors, fake_data_with_acceptances):
        """Test that different C values affect reward."""
        reward_fn_C1 = RewardFunction(C=1.0, epsilon=1e-6, lambda_complexity=0.1)
        reward_fn_C2 = RewardFunction(C=2.0, epsilon=1e-6, lambda_complexity=0.1)

        reward_C1 = reward_fn_C1.compute_reward(single_gate_lgn, real_data_with_errors, fake_data_with_acceptances)
        reward_C2 = reward_fn_C2.compute_reward(single_gate_lgn, real_data_with_errors, fake_data_with_acceptances)

        # Higher C should penalize real errors more, so reward_C2 < reward_C1
        assert reward_C2 < reward_C1

    def test_different_epsilon_values(self, single_gate_lgn, real_data_perfect, fake_data_perfect_rejection):
        """Test that different epsilon values affect reward when fake_acceptance = 0."""
        reward_fn_eps1 = RewardFunction(C=1.0, epsilon=1e-6, lambda_complexity=0.1)
        reward_fn_eps2 = RewardFunction(C=1.0, epsilon=1e-10, lambda_complexity=0.1)

        reward_eps1 = reward_fn_eps1.compute_reward(single_gate_lgn, real_data_perfect, fake_data_perfect_rejection)
        reward_eps2 = reward_fn_eps2.compute_reward(single_gate_lgn, real_data_perfect, fake_data_perfect_rejection)

        # When fake_acceptance=0: fake_term = -log(epsilon)
        # Smaller epsilon → larger -log(ε) → HIGHER (more positive) reward
        assert reward_eps2 > reward_eps1  # eps2 (1e-10) gives higher reward than eps1 (1e-6)


# ============================================================================
# Metrics Tests
# ============================================================================

class TestMetrics:
    """Test metrics functions."""

    def test_compute_real_accuracy_perfect(self, reward_fn_default, single_gate_lgn, real_data_perfect):
        """Test real accuracy with perfect performance."""
        accuracy = compute_real_accuracy(reward_fn_default, single_gate_lgn, real_data_perfect)
        assert accuracy == 1.0

    def test_compute_real_accuracy_with_errors(self, reward_fn_default, single_gate_lgn, real_data_with_errors):
        """Test real accuracy with some errors."""
        accuracy = compute_real_accuracy(reward_fn_default, single_gate_lgn, real_data_with_errors)
        # 3 errors out of 5 → accuracy = 2/5 = 0.4
        assert abs(accuracy - 0.4) < 0.01

    def test_compute_fake_rejection_rate_perfect(self, reward_fn_default, single_gate_lgn, fake_data_perfect_rejection):
        """Test fake rejection rate with perfect rejection."""
        rejection_rate = compute_fake_rejection_rate(reward_fn_default, single_gate_lgn, fake_data_perfect_rejection)
        assert rejection_rate == 1.0

    def test_compute_fake_rejection_rate_with_acceptances(self, reward_fn_default, single_gate_lgn, fake_data_with_acceptances):
        """Test fake rejection rate with some acceptances."""
        rejection_rate = compute_fake_rejection_rate(reward_fn_default, single_gate_lgn, fake_data_with_acceptances)
        # 2 acceptances out of 4 → rejection = 2/4 = 0.5
        assert abs(rejection_rate - 0.5) < 0.01

    def test_compute_metrics_dict_structure(self, reward_fn_default, single_gate_lgn, real_data_with_errors, fake_data_with_acceptances):
        """Test that compute_metrics_dict returns correct structure."""
        metrics = compute_metrics_dict(reward_fn_default, single_gate_lgn, real_data_with_errors, fake_data_with_acceptances)

        # Check all required keys exist
        assert "real_accuracy" in metrics
        assert "fake_rejection_rate" in metrics
        assert "real_error_count" in metrics
        assert "fake_acceptance_count" in metrics
        assert "num_gates" in metrics
        assert "reward" in metrics

        # Check types
        assert isinstance(metrics["real_accuracy"], float)
        assert isinstance(metrics["fake_rejection_rate"], float)
        assert isinstance(metrics["real_error_count"], int)
        assert isinstance(metrics["fake_acceptance_count"], int)
        assert isinstance(metrics["num_gates"], int)
        assert isinstance(metrics["reward"], (float, np.floating))

    def test_compute_metrics_dict_values(self, reward_fn_default, single_gate_lgn, real_data_with_errors, fake_data_with_acceptances):
        """Test that compute_metrics_dict returns correct values."""
        metrics = compute_metrics_dict(reward_fn_default, single_gate_lgn, real_data_with_errors, fake_data_with_acceptances)

        # Verify values
        assert metrics["real_error_count"] == 3
        assert metrics["fake_acceptance_count"] == 2
        assert metrics["num_gates"] == 1
        assert abs(metrics["real_accuracy"] - 0.4) < 0.01
        assert abs(metrics["fake_rejection_rate"] - 0.5) < 0.01
        assert metrics["reward"] <= 0.0


# ============================================================================
# Config Loader Tests
# ============================================================================

class TestConfigLoader:
    """Test config_loader functions."""

    def test_load_valid_config(self, temp_config_file):
        """Test loading a valid config file."""
        config = load_reward_config(temp_config_file)

        # Check structure
        assert "reward_function" in config
        assert "lgn_network" in config

        # Check values
        assert config["reward_function"]["C"] == 2.0
        assert config["reward_function"]["epsilon"] == 1e-10
        assert config["reward_function"]["lambda_complexity"] == 0.05
        assert config["lgn_network"]["num_inputs"] == 10
        assert config["lgn_network"]["max_gates"] == 15

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_reward_config("/nonexistent/path/config.yaml")

    def test_validate_valid_config(self, temp_config_file):
        """Test validating a valid config."""
        config = load_reward_config(temp_config_file, validate=False)
        # Should not raise
        validate_reward_config(config)

    def test_validate_missing_section_raises_error(self):
        """Test that missing required section raises ValueError."""
        invalid_config = {
            "reward_function": {"C": 1.0, "epsilon": 1e-6, "lambda_complexity": 0.1}
            # Missing lgn_network section
        }
        with pytest.raises(ValueError, match="Missing required section"):
            validate_reward_config(invalid_config)

    def test_validate_missing_field_raises_error(self):
        """Test that missing required field raises ValueError."""
        invalid_config = {
            "reward_function": {"C": 1.0, "epsilon": 1e-6},  # Missing lambda_complexity
            "lgn_network": {"num_inputs": 10, "max_gates": 15}
        }
        with pytest.raises(ValueError, match="Missing required field"):
            validate_reward_config(invalid_config)

    def test_validate_invalid_C_raises_error(self):
        """Test that invalid C value raises ValueError."""
        invalid_config = {
            "reward_function": {"C": -1.0, "epsilon": 1e-6, "lambda_complexity": 0.1},
            "lgn_network": {"num_inputs": 10, "max_gates": 15}
        }
        with pytest.raises(ValueError, match="C must be positive"):
            validate_reward_config(invalid_config)

    def test_validate_invalid_epsilon_raises_error(self):
        """Test that invalid epsilon value raises ValueError."""
        invalid_config = {
            "reward_function": {"C": 1.0, "epsilon": 0.0, "lambda_complexity": 0.1},
            "lgn_network": {"num_inputs": 10, "max_gates": 15}
        }
        with pytest.raises(ValueError, match="epsilon must be positive"):
            validate_reward_config(invalid_config)

    def test_validate_invalid_lambda_raises_error(self):
        """Test that invalid lambda_complexity value raises ValueError."""
        invalid_config = {
            "reward_function": {"C": 1.0, "epsilon": 1e-6, "lambda_complexity": -0.1},
            "lgn_network": {"num_inputs": 10, "max_gates": 15}
        }
        with pytest.raises(ValueError, match="lambda_complexity must be non-negative"):
            validate_reward_config(invalid_config)

    def test_get_preset_config(self, temp_config_file):
        """Test loading a preset configuration."""
        preset = get_preset_config("test_preset", temp_config_file)

        assert preset["C"] == 3.0
        assert preset["epsilon"] == 1e-8
        assert preset["lambda_complexity"] == 0.2
        assert preset["num_inputs"] == 20
        assert preset["max_gates"] == 25

    def test_get_nonexistent_preset_raises_error(self, temp_config_file):
        """Test that requesting nonexistent preset raises ValueError."""
        with pytest.raises(ValueError, match="Preset .* not found"):
            get_preset_config("nonexistent_preset", temp_config_file)


# ============================================================================
# Edge Cases and Stress Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_dataset(self, reward_fn_default, single_gate_lgn):
        """Test with large dataset (1000 samples)."""
        real_data = [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]] * 1000
        fake_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 1000

        reward = reward_fn_default.compute_reward(single_gate_lgn, real_data, fake_data)

        # Should complete without errors
        assert isinstance(reward, (float, np.floating))
        # Note: Reward can be positive with perfect fake rejection
        assert np.isfinite(reward)

    def test_very_small_epsilon(self, single_gate_lgn, real_data_perfect, fake_data_perfect_rejection):
        """Test with very small epsilon (1e-15)."""
        reward_fn = RewardFunction(C=1.0, epsilon=1e-15, lambda_complexity=0.1)
        reward = reward_fn.compute_reward(single_gate_lgn, real_data_perfect, fake_data_perfect_rejection)

        # Should not overflow or underflow
        assert np.isfinite(reward)

    def test_very_large_C(self, single_gate_lgn, real_data_with_errors, fake_data_with_acceptances):
        """Test with very large C value (100.0)."""
        reward_fn = RewardFunction(C=100.0, epsilon=1e-6, lambda_complexity=0.1)
        reward = reward_fn.compute_reward(single_gate_lgn, real_data_with_errors, fake_data_with_acceptances)

        # Should not overflow
        assert np.isfinite(reward)

    def test_complex_lgn(self, reward_fn_default):
        """Test with complex LGN (many gates)."""
        lgn = LGNState(num_inputs=10, max_gates=15)
        for i in range(5):
            lgn.add_gate(GateType.AND, [i, i+1])

        real_data = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * 10
        fake_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 10

        reward = reward_fn_default.compute_reward(lgn, real_data, fake_data)

        # Get metrics to verify
        metrics = compute_metrics_dict(reward_fn_default, lgn, real_data, fake_data)
        assert metrics["num_gates"] == 5

        # Complexity penalty should be present (5 gates * -0.1 = -0.5)
        # But reward can still be positive if fake rejection is perfect
        assert np.isfinite(reward)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
