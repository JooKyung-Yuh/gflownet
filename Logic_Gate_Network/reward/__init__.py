"""
Reward Function Module for Logic Gate Network GFlowNet.

This module provides the complete reward function implementation for evaluating
Logic Gate Networks on Real and Fake data classification tasks.

Key Components:
---------------
1. RewardFunction: Main reward computation class
2. Metrics: Performance evaluation functions (accuracy, rejection rate, etc.)
3. Config Loader: YAML configuration file loader for hyperparameters

Quick Start:
------------
# Option 1: Using configuration file (Recommended)
>>> from Logic_Gate_Network.reward import load_reward_config, RewardFunction
>>> from Logic_Gate_Network.lgn import LGNState
>>>
>>> config = load_reward_config("configs/reward_config.yaml")
>>> reward_fn = RewardFunction(
...     C=config['reward_function']['C'],
...     epsilon=config['reward_function']['epsilon'],
...     lambda_complexity=config['reward_function']['lambda_complexity']
... )
>>> lgn = LGNState(
...     num_inputs=config['lgn_network']['num_inputs'],
...     max_gates=config['lgn_network']['max_gates']
... )

# Option 2: Direct initialization
>>> from Logic_Gate_Network.reward import RewardFunction
>>> from Logic_Gate_Network.lgn import LGNState, GateType
>>>
>>> reward_fn = RewardFunction(C=1.0, epsilon=1e-6, lambda_complexity=0.1)
>>> lgn = LGNState(num_inputs=10, max_gates=15)

# Compute reward
>>> reward = reward_fn.compute_reward(lgn, real_data, fake_data)

# Compute metrics
>>> from Logic_Gate_Network.reward import compute_metrics_dict
>>> metrics = compute_metrics_dict(reward_fn, lgn, real_data, fake_data)
>>> print(f"Real Accuracy: {metrics['real_accuracy']:.2%}")
>>> print(f"Fake Rejection: {metrics['fake_rejection_rate']:.2%}")

Exported Components:
--------------------
From reward_fn.py:
  - RewardFunction: Main reward function class

From metrics.py:
  - compute_real_accuracy: Compute Real data accuracy (0-1)
  - compute_fake_rejection_rate: Compute Fake data rejection rate (0-1)
  - compute_metrics_dict: Compute all metrics as dictionary

From config_loader.py:
  - load_reward_config: Load YAML configuration file
  - validate_reward_config: Validate configuration values
  - get_preset_config: Load preset configuration

Configuration:
--------------
See configs/reward_config.yaml for:
  - Detailed hyperparameter documentation
  - Preset configurations (baseline, high_accuracy, etc.)
  - Parameter tuning guidelines
  - Example values and typical ranges

Theory:
-------
The reward function implements:

  log R(F) = -C * sum_i (1-F(X_i)) - log(sum_j F(X_j) + epsilon) + Omega(F)

Where:
  - sum_i (1-F(X_i)) = Raw count of misclassified Real samples
  - sum_j F(X_j) = Raw count of accepted Fake samples
  - Omega(F) = -lambda * (number of gates)
  - C = Balance parameter
  - epsilon = Numerical stability constant

See notion.md and GFlowNet-LogicGates-Final.md for full theoretical derivation.
"""

# Import main classes and functions for easy access
from .reward_fn import RewardFunction
from .metrics import (
    compute_real_accuracy,
    compute_fake_rejection_rate,
    compute_metrics_dict
)
from .config_loader import (
    load_reward_config,
    validate_reward_config,
    get_preset_config
)

# Define public API
__all__ = [
    # Core reward function
    'RewardFunction',

    # Metrics functions
    'compute_real_accuracy',
    'compute_fake_rejection_rate',
    'compute_metrics_dict',

    # Configuration utilities
    'load_reward_config',
    'validate_reward_config',
    'get_preset_config',
]
