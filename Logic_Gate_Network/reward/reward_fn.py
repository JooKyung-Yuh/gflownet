from lgn import LGNState, LGNEvaluator

import numpy as np

class RewardFunction:
  """
  Reward Function for Logic Gate Network (LGN) evaluation in GFlowNet training.

  This class implements the reward function that evaluates how well an LGN
  classifies Real and Fake data, with a complexity penalty for network size.

  Mathematical Formulation (notion.md):
  -------------------------------------
  log R(F) = -C·∑ᵢ₌₁ᴺ (1-F(X⁽ⁱ⁾)) - log(∑ⱼ₌₁ᴹ F(X⁽⁻ʲ⁾) + ε) + Ω(F)

  Where:
    ∑ᵢ₌₁ᴺ (1-F(X⁽ⁱ⁾)) = Raw count of misclassified Real samples
    ∑ⱼ₌₁ᴹ F(X⁽⁻ʲ⁾) = Raw count of accepted Fake samples
    Ω(F) = Complexity penalty = -λ · (number of gates in LGN)
    ε = Small constant for numerical stability (prevents log(0))
    C = Balance parameter between Real and Fake data objectives

  Components:
  -----------
  1. Real Data Term: -C · ∑(1-F(X⁽ⁱ⁾))
    - Penalizes misclassification of Real samples using RAW COUNT
    - Each misclassified Real sample contributes 1 to the sum
    - Lower error count → higher reward
    - C controls the weight of this term

  2. Fake Data Term: -log(∑F(X⁽⁻ʲ⁾) + ε)
    - Penalizes acceptance of Fake samples using RAW COUNT
    - Each accepted Fake sample contributes 1 to the sum
    - Lower acceptance count → higher reward
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
  >>> real_data = [[1,0,1,...], [0,1,0,...], ...]  # N samples
  >>> fake_data = [[1,1,0,...], [0,0,1,...], ...]  # M samples
  >>> reward = reward_fn.compute_reward(lgn, real_data, fake_data)
  >>> print(f"Reward: {reward}")

  Notes:
  ------
  - This reward function uses RAW COUNTS (not normalized rates) following notion.md.
  - The first term counts the actual number of misclassified Real samples.
  - The second term counts the actual number of accepted Fake samples.
  - The epsilon parameter is critical for numerical stability when the LGN
    perfectly rejects all fake data (count = 0).
  - The complexity penalty helps prevent overfitting and encourages
    interpretable, compact network structures.

  References:
  -----------
  - GFlowNet: Bengio et al. (2021) "Flow Network based Generative Models"
  - Reward function design: notion.md (Bayesian Logic Gate Networks)
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
  
  def compute_real_error_count(self, lgn_state:LGNState, real_data:list[list[int]]) -> int:
    """
    Compute the raw count of errors on Real (rule-compliant) data samples.

    This method evaluates how accurately the LGN classifies Real data samples
    and returns the RAW COUNT of misclassified samples (not a normalized rate).
    Real samples should be classified as 1 (Accept) by the LGN. Any sample
    classified as 0 (Reject) is counted as a misclassification error.

    This corresponds to the term ∑ᵢ₌₁ᴺ (1-F(X⁽ⁱ⁾)) in the notion.md formula.

    Args:
        lgn_state (LGNState):
            The Logic Gate Network to evaluate.

        real_data (list[list[int]]):
            List of Real data samples (rule-compliant binary vectors).
            Each sample is a list of binary values (0s and 1s).
            All samples should ideally be classified as 1 (Accept).

    Returns:
        int: Raw count of misclassified Real samples
            - 0 = Perfect classification (all Real samples → 1)
            - N = Worst classification (all Real samples → 0, where N = len(real_data))
            - 50 = 50 out of N Real samples misclassified

    Raises:
        ValueError: If real_data is empty.

    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.add_gate(GateType.AND, [0, 1, 2])
        >>> real_data = [[1,0,1,0,...], [0,1,0,1,...], ...]  # 1000 samples
        >>>
        >>> reward_fn = RewardFunction()
        >>> error_count = reward_fn.compute_real_error_count(lgn, real_data)
        >>> print(f"Error count: {error_count}")  # e.g., 50 (50 errors out of 1000)

    Notes:
        - This is a public helper method that can be used by other modules (e.g., metrics.py).
        - Error count = # samples classified as 0 = N - (# samples classified as 1)
        - Returns RAW COUNT (integer), not normalized rate.
        - Lower error count → higher reward in the reward function.
    """
    # Validate input: real_data must not be empty
    if not real_data:
      raise ValueError("real_data cannot be empty")

    # Evaluate LGN on all Real data samples using batch evaluation
    # Uses evaluate_final_batch which ANDs all root gates' outputs
    evaluator = LGNEvaluator()
    outputs = evaluator.evaluate_final_batch(lgn_state, real_data)
    # outputs: list of LGN predictions [1, 0, 1, 1, 0, ...] for each sample

    # Count misclassifications (Real samples should output 1, not 0)
    # error_count = total samples - correct classifications
    #             = N - (# samples classified as 1)
    error_count:int = len(real_data) - outputs.count(1)

    return error_count

  def compute_fake_acceptance_count(self, lgn_state:LGNState, fake_data:list[list[int]]) -> int:
    """
    Compute the raw count of accepted Fake (rule-violating) data samples.

    This method evaluates how well the LGN rejects Fake data samples
    and returns the RAW COUNT of incorrectly accepted samples (not a normalized rate).
    Fake samples should be classified as 0 (Reject) by the LGN. Any sample
    classified as 1 (Accept) is counted as an incorrect acceptance.

    This corresponds to the term ∑ⱼ₌₁ᴹ F(X⁽⁻ʲ⁾) in the notion.md formula.

    Args:
        lgn_state (LGNState):
            The Logic Gate Network to evaluate.

        fake_data (list[list[int]]):
            List of Fake data samples (rule-violating binary vectors).
            Each sample is a list of binary values (0s and 1s).
            All samples should ideally be classified as 0 (Reject).

    Returns:
        int: Raw count of accepted Fake samples
            - 0 = Perfect rejection (all Fake samples → 0)
            - M = Worst rejection (all Fake samples → 1, where M = len(fake_data))
            - 30 = 30 out of M Fake samples incorrectly accepted

    Raises:
        ValueError: If fake_data is empty.

    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.add_gate(GateType.NAND, [0, 1])
        >>> fake_data = [[1,1,0,0,...], [0,0,1,1,...], ...]  # 1000 samples
        >>>
        >>> reward_fn = RewardFunction()
        >>> acceptance_count = reward_fn.compute_fake_acceptance_count(lgn, fake_data)
        >>> print(f"Acceptance count: {acceptance_count}")  # e.g., 30 (30 accepted out of 1000)

    Notes:
        - This is a public helper method that can be used by other modules (e.g., metrics.py).
        - Acceptance count = # samples classified as 1
        - Returns RAW COUNT (integer), not normalized rate.
        - Lower acceptance count → higher reward in the reward function.
        - Opposite of Real data: Real wants 1, Fake wants 0.
    """
    # Validate input: fake_data must not be empty
    if not fake_data:
      raise ValueError("fake_data cannot be empty")

    # Evaluate LGN on all Fake data samples using batch evaluation
    # Uses evaluate_final_batch which ANDs all root gates' outputs
    evaluator = LGNEvaluator()
    outputs = evaluator.evaluate_final_batch(lgn_state, fake_data)
    # outputs: list of LGN predictions [1, 0, 1, 1, 0, ...] for each sample

    # Count incorrect acceptances (Fake samples should output 0, not 1)
    # acceptance_count = # samples classified as 1
    acceptance_count:int = outputs.count(1)

    return acceptance_count
  
  def _compute_complexity_penalty(self, lgn_state:LGNState) -> float:
    """
    Compute the complexity penalty for the Logic Gate Network.

    This method implements the complexity penalty term Ω(F) from the reward function,
    which encourages simpler networks by penalizing the total number of gates.
    The penalty is linearly proportional to the gate count, weighted by lambda_complexity.

    This corresponds to the term Ω(F) = -λ · (number of gates) in the notion.md formula.

    Mathematical Formulation:
    -------------------------
    Ω(F) = -λ · (number of gates in LGN)

    Where:
      λ = self.lambda_complexity (complexity penalty weight)
      number of gates = Total count of gates in the network

    Args:
        lgn_state (LGNState):
            The Logic Gate Network whose complexity is being evaluated.

    Returns:
        float: Complexity penalty value (always non-positive)
            - 0.0 = No penalty (empty network with 0 gates, or lambda=0)
            - -0.5 = Small penalty (e.g., 5 gates with λ=0.1)
            - -5.0 = Large penalty (e.g., 50 gates with λ=0.1)

    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.add_gate(GateType.AND, [0, 1, 2])
        >>> lgn.add_gate(GateType.OR, [3, 10])
        >>> lgn.add_gate(GateType.XOR, [4, 11])
        >>> # Network now has 3 gates
        >>>
        >>> reward_fn = RewardFunction(lambda_complexity=0.1)
        >>> penalty = reward_fn._compute_complexity_penalty(lgn)
        >>> print(f"Penalty: {penalty}")  # -0.3 (= -0.1 * 3)

    Notes:
        - This is a private helper method (prefix _) used internally by compute_reward().
        - The penalty is always non-positive (≤ 0).
        - Higher lambda_complexity → stronger preference for simpler networks.
        - A network with 0 gates has 0 penalty.
        - The penalty scales linearly with gate count (not quadratic or exponential).
        - This term balances the accuracy terms (real error, fake acceptance) to
          prevent overfitting and encourage interpretable network structures.

    Design Rationale:
        - Linear penalty is simple and effective for balancing accuracy vs complexity.
        - Prevents the GFlowNet from building unnecessarily large networks.
        - Encourages sparse, interpretable solutions that are easier to analyze.
        - The lambda parameter allows tuning the accuracy-complexity trade-off.
    """
    return -1 * self.lambda_complexity * lgn_state.get_num_gates()
  
    
  def compute_log_reward(self, lgn_state:LGNState, real_data:list[list[int]], fake_data:list[list[int]], return_details:bool=False, mode:str='log'):
    """
    Compute the log-reward for a Logic Gate Network on Real and Fake datasets.

    This method always returns log R(F) for use in TB Loss:
        Loss = (logZ + log P_F(τ) - log R(x))²

    Two modes control how the fake term is computed:

    mode='log' (default, original notion.md):
        log R(F) = -C·∑(1-F(X)) - log(∑F(X⁻) + ε) + Ω(F)
        Fake term uses log → gradient diminishes for large fake_accept.

    mode='diff':
        log R(F) = TP - FP + Ω(F)
                 = ∑F(X⁺) - ∑F(X⁻) + Ω(F)
        Direct difference between correct accepts and incorrect accepts.
        Interpretation: R(F) = exp(TP - FP + Ω) is always positive.
        This provides balanced gradient for both TP and FP.

    Args:
        lgn_state (LGNState):
            The Logic Gate Network to evaluate.

        real_data (list[list[int]]):
            List of Real data samples (rule-compliant binary vectors).

        fake_data (list[list[int]]):
            List of Fake data samples (rule-violating binary vectors).

        return_details (bool):
            If True, return dict with breakdown of reward components.

        mode (str):
            'log' (default): Original formula with log fake term.
            'diff': TP - FP formula (balanced gradient).

    Returns:
        float: log R(F) value for TB Loss
    """
    # Step 1: Compute raw counts using helper methods
    real_error_count = self.compute_real_error_count(lgn_state, real_data)
    fake_acceptance_count = self.compute_fake_acceptance_count(lgn_state, fake_data)
    complexity_penalty = self._compute_complexity_penalty(lgn_state)

    # Compute TP and FP for clarity
    # TP = Real samples correctly accepted = num_real - real_error_count
    # FP = Fake samples incorrectly accepted = fake_acceptance_count
    tp = len(real_data) - real_error_count
    fp = fake_acceptance_count

    if mode == 'log':
      # log R(F) = -C·real_err - log(fake_acc + ε) + Ω(F)
      real_term = -self.C * real_error_count
      fake_term = -np.log(fake_acceptance_count + self.epsilon)
      log_reward = real_term + fake_term + complexity_penalty
    elif mode == 'diff':
      # log R(F) = (TP - FP) * C
      # C acts as reward tempering: C < 1 smooths reward distribution, C > 1 sharpens
      # R(F) = exp((TP - FP) * C) is always positive
      real_term = tp   # ∑F(X⁺) = TP
      fake_term = -fp  # -∑F(X⁻) = -FP
      log_reward = (real_term + fake_term) * self.C
    else:
      raise ValueError(f"Invalid mode: {mode}. Must be 'log' or 'diff'.")

    if return_details:
      return {
        'log_reward': log_reward,
        'real_error_count': real_error_count,
        'fake_acceptance_count': fake_acceptance_count,
        'tp': tp,
        'fp': fp,
        'real_term': real_term,
        'fake_term': fake_term,
        'complexity': complexity_penalty,
        'num_gates': lgn_state.get_num_gates(),
        'num_real': len(real_data),
        'num_fake': len(fake_data),
        'mode': mode,
      }

    return log_reward

  # Backward compatibility alias
  def compute_reward(self, lgn_state:LGNState, real_data:list[list[int]], fake_data:list[list[int]], return_details:bool=False, mode:str='log'):
    """Alias for compute_log_reward() for backward compatibility."""
    return self.compute_log_reward(lgn_state, real_data, fake_data, return_details, mode)
    
    
    