"""
Evaluation Metrics for Logic Gate Network (LGN) Performance.

This module provides evaluation metrics to assess the performance of Logic Gate Networks
on Real and Fake data classification tasks. These metrics complement the reward function
by providing intuitive, normalized performance indicators.

The metrics are designed for:
- Monitoring training progress
- Evaluating final model performance
- Comparing different LGN architectures
- Debugging and analysis

Key Metrics:
-----------
1. Real Accuracy: Proportion of Real samples correctly classified as 1 (Accept)
2. Fake Rejection Rate: Proportion of Fake samples correctly classified as 0 (Reject)
3. Metrics Dictionary: Comprehensive summary of all metrics in one dict

Usage:
------
>>> from Logic_Gate_Network.reward.metrics import compute_real_accuracy
>>> from Logic_Gate_Network.reward.reward_fn import RewardFunction
>>> from Logic_Gate_Network.lgn import LGNState, GateType
>>>
>>> # Setup
>>> reward_fn = RewardFunction()
>>> lgn = LGNState(num_inputs=10, max_gates=15)
>>> lgn.add_gate(GateType.AND, [0, 1, 2])
>>>
>>> # Evaluate
>>> real_data = [[1,0,1,0,...], ...]  # 1000 samples
>>> accuracy = compute_real_accuracy(reward_fn, lgn, real_data)
>>> print(f"Real Accuracy: {accuracy:.2%}")  # e.g., 95.00%
"""

from ..lgn import LGNState
from .reward_fn import RewardFunction


def compute_real_accuracy(reward_fn: RewardFunction, lgn_state: LGNState, real_data: list[list[int]]) -> float:
    """
    Compute the accuracy of the LGN on Real (rule-compliant) data samples.

    This metric measures what proportion of Real samples are correctly classified
    as 1 (Accept) by the Logic Gate Network. It provides an intuitive, normalized
    performance indicator in the range [0.0, 1.0].

    Formula:
    --------
    accuracy = (N - error_count) / N
             = correct_count / total_count
             = 1 - (error_count / N)

    Where:
      N = Total number of Real samples
      error_count = Number of Real samples misclassified as 0

    Args:
        reward_fn (RewardFunction):
            RewardFunction instance to access the compute_real_error_count() method.

        lgn_state (LGNState):
            The Logic Gate Network to evaluate.

        real_data (list[list[int]]):
            List of Real data samples (rule-compliant binary vectors).
            Each sample is a list of binary values (0s and 1s).
            All samples should ideally be classified as 1 (Accept).

    Returns:
        float: Accuracy in the range [0.0, 1.0]
            - 1.0 = Perfect accuracy (all Real samples � 1)
            - 0.0 = Worst accuracy (all Real samples � 0)
            - 0.95 = 95% accuracy (950 correct out of 1000 samples)

    Raises:
        ValueError: If real_data is empty (propagated from reward_fn.compute_real_error_count()).

    Example:
        >>> # Setup
        >>> reward_fn = RewardFunction(C=1.0, epsilon=1e-6, lambda_complexity=0.1)
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.add_gate(GateType.AND, [0, 1, 2])
        >>> lgn.add_gate(GateType.OR, [3, 10])
        >>>
        >>> # Prepare data
        >>> real_data = [[1,0,1,0,1,0,1,0,1,0], ...]  # 1000 samples
        >>>
        >>> # Compute accuracy
        >>> accuracy = compute_real_accuracy(reward_fn, lgn, real_data)
        >>> print(f"Real Accuracy: {accuracy:.2%}")
        >>> # Example output: Real Accuracy: 95.30%
        >>>
        >>> # If 47 errors out of 1000:
        >>> # accuracy = (1000 - 47) / 1000 = 0.953 = 95.3%

    Notes:
        - This is a normalized metric (0-1 range) unlike the raw count from compute_real_error_count().
        - Higher accuracy � Better performance.
        - Accuracy = 1.0 - (error_rate).
        - This metric is more intuitive than raw counts for reporting and visualization.
        - Complements the reward function by providing a clear, interpretable performance indicator.

    See Also:
        - compute_fake_rejection_rate(): Complementary metric for Fake data
        - compute_metrics_dict(): Get all metrics together
        - RewardFunction.compute_real_error_count(): Underlying raw count computation
    """
    # Get raw error count using RewardFunction's public method
    error_count = reward_fn.compute_real_error_count(lgn_state, real_data)

    # Calculate total number of samples
    total_count = len(real_data)

    # Calculate accuracy: correct_count / total_count
    # correct_count = total_count - error_count
    accuracy = (total_count - error_count) / total_count

    return accuracy


def compute_fake_rejection_rate(reward_fn: RewardFunction, lgn_state: LGNState, fake_data: list[list[int]]) -> float:
    """
    Compute the rejection rate of the LGN on Fake (rule-violating) data samples.

    This metric measures what proportion of Fake samples are correctly classified
    as 0 (Reject) by the Logic Gate Network. It provides an intuitive, normalized
    performance indicator in the range [0.0, 1.0].

    Formula:
    --------
    rejection_rate = (M - acceptance_count) / M
                   = correct_rejection_count / total_count
                   = 1 - (acceptance_count / M)

    Where:
      M = Total number of Fake samples
      acceptance_count = Number of Fake samples incorrectly accepted as 1

    Args:
        reward_fn (RewardFunction):
            RewardFunction instance to access the compute_fake_acceptance_count() method.

        lgn_state (LGNState):
            The Logic Gate Network to evaluate.

        fake_data (list[list[int]]):
            List of Fake data samples (rule-violating binary vectors).
            Each sample is a list of binary values (0s and 1s).
            All samples should ideally be classified as 0 (Reject).

    Returns:
        float: Rejection rate in the range [0.0, 1.0]
            - 1.0 = Perfect rejection (all Fake samples → 0)
            - 0.0 = Worst rejection (all Fake samples → 1)
            - 0.92 = 92% rejection rate (920 correctly rejected out of 1000 samples)

    Raises:
        ValueError: If fake_data is empty (propagated from reward_fn.compute_fake_acceptance_count()).

    Example:
        >>> # Setup
        >>> reward_fn = RewardFunction(C=1.0, epsilon=1e-6, lambda_complexity=0.1)
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.add_gate(GateType.NAND, [0, 1])
        >>> lgn.add_gate(GateType.XOR, [2, 10])
        >>>
        >>> # Prepare data
        >>> fake_data = [[1,1,0,0,0,0,0,0,0,0], ...]  # 1000 samples
        >>>
        >>> # Compute rejection rate
        >>> rejection_rate = compute_fake_rejection_rate(reward_fn, lgn, fake_data)
        >>> print(f"Fake Rejection Rate: {rejection_rate:.2%}")
        >>> # Example output: Fake Rejection Rate: 91.50%
        >>>
        >>> # If 85 acceptances out of 1000:
        >>> # rejection_rate = (1000 - 85) / 1000 = 0.915 = 91.5%

    Notes:
        - This is a normalized metric (0-1 range) unlike the raw count from compute_fake_acceptance_count().
        - Higher rejection rate → Better performance.
        - Rejection rate = 1.0 - (acceptance_rate).
        - This metric is complementary to Real accuracy: both measure correctness but on different data types.
        - A good LGN should have high Real accuracy AND high Fake rejection rate.

    See Also:
        - compute_real_accuracy(): Complementary metric for Real data
        - compute_metrics_dict(): Get all metrics together
        - RewardFunction.compute_fake_acceptance_count(): Underlying raw count computation
    """
    # Get raw acceptance count using RewardFunction's public method
    acceptance_count = reward_fn.compute_fake_acceptance_count(lgn_state, fake_data)

    # Calculate total number of samples
    total_count = len(fake_data)

    # Calculate rejection rate: correct_rejection_count / total_count
    # correct_rejection_count = total_count - acceptance_count
    rejection_rate = (total_count - acceptance_count) / total_count

    return rejection_rate


def compute_metrics_dict(reward_fn: RewardFunction, lgn_state: LGNState, real_data: list[list[int]], fake_data: list[list[int]]) -> dict:
    """
    Compute all evaluation metrics for an LGN and return them as a dictionary.

    This convenience function aggregates all performance metrics into a single
    dictionary for easy logging, visualization, and analysis. It computes both
    raw counts and normalized rates for comprehensive evaluation.

    Metrics Included:
    -----------------
    1. real_accuracy: Proportion of Real samples correctly classified (0-1)
    2. fake_rejection_rate: Proportion of Fake samples correctly rejected (0-1)
    3. real_error_count: Raw count of misclassified Real samples (integer)
    4. fake_acceptance_count: Raw count of incorrectly accepted Fake samples (integer)
    5. num_gates: Number of gates in the LGN (integer)
    6. reward: Log-reward value from the reward function (float, ≤ 0)

    Args:
        reward_fn (RewardFunction):
            RewardFunction instance for computing counts and reward.

        lgn_state (LGNState):
            The Logic Gate Network to evaluate.

        real_data (list[list[int]]):
            List of Real data samples (rule-compliant binary vectors).

        fake_data (list[list[int]]):
            List of Fake data samples (rule-violating binary vectors).

    Returns:
        dict: Dictionary containing all metrics with the following keys:
            - "real_accuracy" (float): Real data accuracy [0.0, 1.0]
            - "fake_rejection_rate" (float): Fake data rejection rate [0.0, 1.0]
            - "real_error_count" (int): Number of Real errors
            - "fake_acceptance_count" (int): Number of Fake acceptances
            - "num_gates" (int): Number of gates in the network
            - "reward" (float): Log-reward value (≤ 0)

    Raises:
        ValueError: If real_data or fake_data is empty.

    Example:
        >>> # Setup
        >>> reward_fn = RewardFunction(C=1.0, epsilon=1e-6, lambda_complexity=0.1)
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.add_gate(GateType.AND, [0, 1, 2])
        >>> lgn.add_gate(GateType.OR, [3, 10])
        >>>
        >>> # Prepare data
        >>> real_data = [[1,0,1,0,1,0,1,0,1,0], ...]  # 1000 samples
        >>> fake_data = [[0,0,0,0,0,0,0,0,0,0], ...]  # 1000 samples
        >>>
        >>> # Compute all metrics
        >>> metrics = compute_metrics_dict(reward_fn, lgn, real_data, fake_data)
        >>> print(metrics)
        >>> # Example output:
        >>> # {
        >>> #     'real_accuracy': 0.953,
        >>> #     'fake_rejection_rate': 0.915,
        >>> #     'real_error_count': 47,
        >>> #     'fake_acceptance_count': 85,
        >>> #     'num_gates': 2,
        >>> #     'reward': -14.234
        >>> # }
        >>>
        >>> # Easy logging
        >>> print(f"Real Acc: {metrics['real_accuracy']:.2%}, "
        ...       f"Fake Rej: {metrics['fake_rejection_rate']:.2%}, "
        ...       f"Gates: {metrics['num_gates']}, "
        ...       f"Reward: {metrics['reward']:.2f}")
        >>> # Output: Real Acc: 95.30%, Fake Rej: 91.50%, Gates: 2, Reward: -14.23

    Notes:
        - This function is optimized to avoid redundant computations.
        - Raw counts are computed once and reused for both metrics and reward.
        - Perfect for tracking training progress, creating plots, and reporting results.
        - All metrics are computed from the same LGN state for consistency.
        - The dictionary format makes it easy to log to TensorBoard, Weights & Biases, etc.

    Usage Patterns:
        # Training loop monitoring
        >>> for epoch in range(num_epochs):
        ...     metrics = compute_metrics_dict(reward_fn, lgn, real_data, fake_data)
        ...     logger.log(metrics, step=epoch)

        # Model comparison
        >>> models = [lgn1, lgn2, lgn3]
        >>> results = [compute_metrics_dict(reward_fn, m, real_data, fake_data) for m in models]
        >>> best_model = max(results, key=lambda x: x['reward'])

    See Also:
        - compute_real_accuracy(): Individual metric for Real data
        - compute_fake_rejection_rate(): Individual metric for Fake data
        - RewardFunction.compute_reward(): Main reward computation
    """
    # Compute raw counts (reuse for efficiency)
    real_error_count = reward_fn.compute_real_error_count(lgn_state, real_data)
    fake_acceptance_count = reward_fn.compute_fake_acceptance_count(lgn_state, fake_data)

    # Compute normalized metrics using existing functions
    real_accuracy = compute_real_accuracy(reward_fn, lgn_state, real_data)
    fake_rejection_rate = compute_fake_rejection_rate(reward_fn, lgn_state, fake_data)

    # Get network complexity
    num_gates = lgn_state.get_num_gates()

    # Compute reward
    reward = reward_fn.compute_reward(lgn_state, real_data, fake_data)

    # Aggregate all metrics into a dictionary
    metrics_dict = {
        "real_accuracy": real_accuracy,
        "fake_rejection_rate": fake_rejection_rate,
        "real_error_count": real_error_count,
        "fake_acceptance_count": fake_acceptance_count,
        "num_gates": num_gates,
        "reward": reward
    }

    return metrics_dict
