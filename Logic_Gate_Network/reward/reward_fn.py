from  ..lgn import LGNState, LGNEvaluator

import numpy as np

class RewardFunction:
  """
  Reward Function for Logic Gate Network (LGN) evaluation in GFlowNet training.

  This class implements the reward function that evaluates how well an LGN
  classifies Real and Fake data, with a complexity penalty for network size.

  Mathematical Formulation (Form 2):
  ----------------------------------
  log R(F) = -C · E_real - log(E_fake + ε) + Ω(F)

  Where:
    E_real = Real error rate = (# misclassified real samples) / (total real samples)
    E_fake = Fake acceptance rate = (# accepted fake samples) / (total fake samples)
    Ω(F) = Complexity penalty = -λ · (number of gates in LGN)
    ε = Small constant for numerical stability (prevents log(0))

  Components:
  -----------
  1. Real Data Term: -C · E_real
    - Penalizes misclassification of Real samples
    - Lower error rate → higher reward
    - C controls the weight of this term

  2. Fake Data Term: -log(E_fake + ε)
    - Penalizes acceptance of Fake samples
    - Lower acceptance rate → higher reward
    - ε prevents log(0) when perfectly rejecting all fakes

  3. Complexity Penalty: Ω(F) = -λ · (# gates)
    - Encourages simpler networks (fewer gates)
    - λ controls the strength of this penalty
    - Prevents overly complex solutions

  Attributes:
  -----------
  C : float
      Balance parameter between Real and Fake data objectives.
      Typical range: [0.1, 10.0]
      Default: 1.0

  epsilon : float
      Numerical stability constant to prevent log(0).
      Should be very small (e.g., 1e-6 or 1e-10).
      Default: 1e-6

  lambda_complexity : float
      Weight for complexity penalty (gate count).
      Higher values encourage simpler networks.
      Typical range: [0.01, 1.0]
      Default: 0.1

  Methods:
  --------
  compute_reward(lgn_state, real_data, fake_data)
      Computes the reward for a given LGN on Real and Fake datasets.

  Example:
  --------
  >>> reward_fn = RewardFunction(C=1.0, epsilon=1e-6, lambda_complexity=0.1)
  >>> lgn = LGNState(num_inputs=10, max_gates=15)
  >>> lgn.add_gate(GateType.AND, [0, 1, 2])
  >>> real_data = [[1,0,1,...], [0,1,0,...], ...]
  >>> fake_data = [[1,1,0,...], [0,0,1,...], ...]
  >>> reward = reward_fn.compute_reward(lgn, real_data, fake_data)
  >>> print(f"Reward: {reward}")

  Notes:
  ------
  - This reward function uses normalized error rates (E_real, E_fake ∈ [0, 1])
    to ensure scale-invariance across different dataset sizes.
  - The epsilon parameter is critical for numerical stability when the LGN
    perfectly rejects all fake data (E_fake = 0).
  - The complexity penalty helps prevent overfitting and encourages
    interpretable, compact network structures.

  References:
  -----------
  - GFlowNet: Bengio et al. (2021) "Flow Network based Generative Models"
  - Reward function design: Critical-Issues-Solutions.md (Option B)
  """
  
  def __init__(self, C=1.0, epsilon=1e-6, lambda_complexity=0.1) -> None:
    """
    Initialize the Reward Function with hyperparameters.
    
    Args:
        C (float, optional): 
            Real/Fake balance parameter. Controls the weight of the 
            Real data error term relative to the Fake data term.
            Default: 1.0
        
        epsilon (float, optional): 
            Numerical stability constant. Prevents log(0) when computing
            the Fake data term. Should be very small (e.g., 1e-6).
            Default: 1e-6
        
        lambda_complexity (float, optional): 
            Complexity penalty weight. Controls how much simpler networks
            are preferred over complex ones. Higher values penalize 
            gate count more heavily.
            Default: 0.1
    
    Raises:
        ValueError: If C <= 0 (must be positive)
        ValueError: If epsilon <= 0 (must be positive)
        ValueError: If lambda_complexity < 0 (cannot be negative)
    
    Example:
        >>> # Default hyperparameters
        >>> reward_fn = RewardFunction()
        >>> 
        >>> # Custom hyperparameters
        >>> reward_fn = RewardFunction(C=2.0, epsilon=1e-10, lambda_complexity=0.5)
    """
    # Validate hyperparameters
    if C <= 0:
        raise ValueError(f"C must be positive, got {C}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if lambda_complexity < 0:
        raise ValueError(f"lambda_complexity must be non-negative, got {lambda_complexity}")
      
    # Store hyperparameters
    self.C = C
    self.epsilon = epsilon
    self.lambda_complexity = lambda_complexity
  
  def _compute_real_error_rate(self, lgn_state:LGNState, real_data:list[list[int]]) -> float:
    """
    Compute the error rate on Real (rule-compliant) data samples.
    
    This method evaluates how accurately the LGN classifies Real data samples.
    Real samples should be classified as 1 (Accept) by the LGN. Any sample
    classified as 0 (Reject) is counted as a misclassification error.
    
    Args:
        lgn_state (LGNState): 
            The Logic Gate Network to evaluate.
        
        real_data (list[list[int]]): 
            List of Real data samples (rule-compliant binary vectors).
            Each sample is a list of binary values (0s and 1s).
            All samples should ideally be classified as 1 (Accept).
    
    Returns:
        float: Real data error rate in range [0.0, 1.0]
            - 0.0 = Perfect classification (all Real samples → 1)
            - 1.0 = Worst classification (all Real samples → 0)
            - 0.1 = 10% of Real samples misclassified
    
    Raises:
        ValueError: If real_data is empty.
    
    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.add_gate(GateType.AND, [0, 1, 2])
        >>> real_data = [[1,0,1,0,...], [0,1,0,1,...], ...]  # 1000 samples
        >>> 
        >>> reward_fn = RewardFunction()
        >>> error_rate = reward_fn._compute_real_error_rate(lgn, real_data)
        >>> print(f"Error rate: {error_rate}")  # e.g., 0.05 (5% error)
    
    Notes:
        - This is a private helper method (prefix _) used internally by compute_reward().
        - Error rate = (# samples classified as 0) / (total # samples)
        - Lower error rate → higher reward in the reward function.
    """
    # Validate input: real_data must not be empty
    if not real_data:
      raise ValueError("real_data cannot be empty")

    # Evaluate LGN on all Real data samples using batch evaluation
    evaluator = LGNEvaluator()
    outputs = evaluator.evaluate_batch(lgn_state, real_data)
    # outputs: list of LGN predictions [1, 0, 1, 1, 0, ...] for each sample
    
    # Count correct classifications (Real samples should output 1)
    # error_rate = 1 - (# correct / total)
    #            = (# incorrect / total)
    error_rate:float =  1 - outputs.count(1)/len(outputs)
    
    return error_rate

  def _compute_fake_acceptance_rate(self, lgn_state:LGNState, fake_data:list[list[int]]) -> float:
    """
    Compute the acceptance rate on Fake (rule-violating) data samples.
    
    This method evaluates how well the LGN rejects Fake data samples.
    Fake samples should be classified as 0 (Reject) by the LGN. Any sample
    classified as 1 (Accept) is counted as an incorrect acceptance.
    
    Args:
        lgn_state (LGNState): 
            The Logic Gate Network to evaluate.
        
        fake_data (list[list[int]]): 
            List of Fake data samples (rule-violating binary vectors).
            Each sample is a list of binary values (0s and 1s).
            All samples should ideally be classified as 0 (Reject).
    
    Returns:
        float: Fake data acceptance rate in range [0.0, 1.0]
            - 0.0 = Perfect rejection (all Fake samples → 0)
            - 1.0 = Worst rejection (all Fake samples → 1)
            - 0.05 = 5% of Fake samples incorrectly accepted
    
    Raises:
        ValueError: If fake_data is empty.
    
    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.add_gate(GateType.NAND, [0, 1])
        >>> fake_data = [[1,1,0,0,...], [0,0,1,1,...], ...]  # 1000 samples
        >>> 
        >>> reward_fn = RewardFunction()
        >>> acceptance_rate = reward_fn._compute_fake_acceptance_rate(lgn, fake_data)
        >>> print(f"Acceptance rate: {acceptance_rate}")  # e.g., 0.03 (3% accepted)
    
    Notes:
        - This is a private helper method (prefix _) used internally by compute_reward().
        - Acceptance rate = (# samples classified as 1) / (total # samples)
        - Lower acceptance rate → higher reward in the reward function.
        - Opposite of Real data: Real wants 1, Fake wants 0.
    """
    # Validate input: fake_data must not be empty
    if not fake_data:
      raise ValueError("fake_data cannot be empty")

    # Evaluate LGN on all Fake data samples using batch evaluation
    evaluator = LGNEvaluator()
    outputs = evaluator.evaluate_batch(lgn_state, fake_data)
    # outputs: list of LGN predictions [1, 0, 1, 1, 0, ...] for each sample
    
    # Count incorrect acceptances (Fake samples should output 0, not 1)
    # acceptance_rate = (# samples classified as 1) / total
    #                 = (# incorrectly accepted) / total
    acceptance_rate:float =  outputs.count(1)/len(outputs)
    
    return acceptance_rate