from typing import Any
from itertools import combinations
from lgn.network import LGNState
from lgn.gates import GateType

class LGNActionSpace:
  """
  Action Space for Logic Gate Network generation in GFlowNet.

  This class defines the forward policy action space for GFlowNet training.
  Given a current LGNState, it generates all valid actions (gate additions)
  that can be taken to expand the network while respecting DAG constraints.

  Role in GFlowNet:
  -----------------
  - Defines forward actions: From state s, what gates can we add?
  - Enables forward sampling: Policy network samples from valid actions
  - Complements LGNMDP: LGNMDP handles backward (parent_transitions), this handles forward
  - Used in policy network training and trajectory generation

  Design Philosophy:
  ------------------
  - **Stateless**: Does not store LGNState internally; takes state as method parameter
  - **Forward-focused**: Primary role is generating valid forward actions
  - **DAG-aware**: Only generates actions that maintain DAG structure
  - **Simple constraints**: Only DAG constraint (no recency buffer)

  Action Types:
  -------------
  1. **Gate Addition Actions**: Add a new gate to the network
     - Format: {'gate_type': GateType, 'input_indices': tuple}
     - Must satisfy DAG constraint (inputs reference only past nodes)
     - Must satisfy arity constraint (correct number of inputs per gate type)

  2. **Stop Action**: Terminate trajectory and evaluate network
     - Format: {'action': 'stop'}
     - Available when network can be evaluated (has at least one gate)

  Comparison to LGNMDP:
  ---------------------
  - LGNMDP.parent_transitions(): Given state s, find all states that lead to s (backward)
  - LGNActionSpace.get_valid_actions(): Given state s, find all actions from s (forward)
  - Both use same action dict format for consistency

  Attributes:
  -----------
  num_inputs : int
      Number of input features to the Logic Gate Network (e.g., 10 for binary vectors).
      Must match the data dimensionality and LGNState configuration.

  max_gates : int
      Maximum number of gates allowed in the network (e.g., 15).
      Acts as hard constraint to prevent unbounded network growth.

  Example:
  --------
  >>> from Logic_Gate_Network.gflownet import LGNActionSpace
  >>> from Logic_Gate_Network.lgn import LGNState, GateType
  >>>
  >>> # Initialize action space
  >>> action_space = LGNActionSpace(num_inputs=10, max_gates=15)
  >>>
  >>> # Get valid actions from empty state
  >>> lgn = LGNState(num_inputs=10, max_gates=15)
  >>> actions = action_space.get_valid_actions(lgn)
  >>> print(f"Found {len(actions)} valid actions from empty state")
  >>>
  >>> # Add a gate and get new valid actions
  >>> lgn.add_gate(GateType.AND, [0, 1])
  >>> actions = action_space.get_valid_actions(lgn)
  >>> print(f"Found {len(actions)} valid actions after adding 1 gate")

  See Also:
  ---------
  - LGNMDP: The MDP wrapper providing parent_transitions() for backward sampling
  - LGNState: The state representation for Logic Gate Networks
  - GateType: Enum defining logic gate types
  - Forward policy: Uses this class to sample actions during trajectory generation
  """
  def __init__(self, num_inputs: int, max_gates: int, max_and_arity: int = 0) -> None:
    """
    Initialize the Logic Gate Network Action Space.

    Parameters:
    -----------
    num_inputs : int
        Number of input features to the Logic Gate Network.
        Must be >= 1 and match the dimensionality of training/evaluation data.
        Example: For 10-bit binary vectors, num_inputs = 10.

    max_gates : int
        Maximum number of gates allowed in the network.
        Must be >= 1. Acts as hard constraint to prevent unbounded network growth.
        Example: max_gates = 15 allows networks with up to 15 logic gates.

    max_and_arity : int
        Maximum number of inputs for AND gates. Default is 0 (no limit).
        This prevents action space explosion since AND gates support variable arity.
        - 0: No limit (original behavior, can be very slow with large networks)
        - 2: AND gates behave like other binary gates (fastest)
        - 4: Balanced speed/expressiveness
        Example: max_and_arity=2 limits AND to 2 inputs like OR, XOR, etc.

    Returns:
    --------
    None

    Notes:
    ------
    - This class is **stateless**: It does not store any LGNState internally.
    - LGNState objects are passed as parameters to methods (e.g., get_valid_actions).
    - num_inputs and max_gates are stored only for validation and action generation.
    - These values should match the configuration used in LGNState and LGNMDP.

    Example:
    --------
    >>> action_space = LGNActionSpace(num_inputs=10, max_gates=15, max_and_arity=2)
    >>> print(f"Action space configured for {action_space.num_inputs} inputs")
    """
    self.num_inputs = num_inputs
    self.max_gates = max_gates
    self.max_and_arity = max_and_arity

  def get_valid_actions(self, lgn_state: LGNState) -> list[dict[str, Any]]:
    """
    Get all valid actions that can be taken from the current state.

    This method generates all possible gate addition actions that:
    1. Respect DAG constraints (only reference past nodes)
    2. Respect arity constraints (correct number of inputs per gate type)
    3. Do not exceed max_gates limit

    Additionally, if the network is non-empty, a 'stop' action is included.

    Parameters:
    -----------
    lgn_state : LGNState
        The current Logic Gate Network state from which to generate actions.

    Returns:
    --------
    list[dict[str, Any]]
        List of valid action dictionaries. Each dict contains:
        - For gate actions: {'gate_type': GateType, 'input_indices': tuple}
        - For stop action: {'action': 'stop'}

    Examples:
    ---------
    **Case 1: Empty LGN**
    >>> lgn = LGNState(num_inputs=10, max_gates=15)
    >>> actions = action_space.get_valid_actions(lgn)
    >>> # Many actions: all 2-input gate combinations from 10 inputs
    >>> # No stop action (network is empty)

    **Case 2: LGN with gates**
    >>> lgn = LGNState(num_inputs=10, max_gates=15)
    >>> lgn.add_gate(GateType.AND, [0, 1])
    >>> actions = action_space.get_valid_actions(lgn)
    >>> # Actions include: gates with inputs from {0..9, 10} where 10 is gate output
    >>> # Plus: stop action

    **Case 3: Max gates reached**
    >>> lgn = LGNState(num_inputs=10, max_gates=2)
    >>> lgn.add_gate(GateType.AND, [0, 1])
    >>> lgn.add_gate(GateType.OR, [2, 10])
    >>> actions = action_space.get_valid_actions(lgn)
    >>> # Only stop action (max_gates reached)

    Notes:
    ------
    - This is the forward counterpart to LGNMDP.parent_transitions()
    - DAG constraint: input indices must be < (num_inputs + num_current_gates)
    - Stop action only available if network has at least one gate
    """
    # =========================================================================
    # Step 1: Max Gates Constraint Check
    # =========================================================================
    # Check if the network has reached the maximum number of gates allowed.
    # This uses the same logic as LGNState.is_terminal() (network.py:191)
    # to maintain consistency across the codebase.
    #
    # Logic: If len(gates) >= max_gates, no more gates can be added.
    # - If network is non-empty: Only stop action is available
    # - If network is empty: No actions available (edge case, shouldn't occur)
    #
    # Consistency with is_terminal():
    # - is_terminal() checks: max_gates_reached = (len(self.gates) >= self.max_gates)
    # - We use identical comparison (>=) to ensure consistent behavior
    if len(lgn_state.gates) >= self.max_gates:
      # Network has reached max capacity - can only stop (if non-empty)
      if len(lgn_state.gates) > 0:
        # Network has gates: return stop action only
        return [{'action': 'stop'}]
      else:
        # Edge case: empty network at max gates (shouldn't happen in practice)
        # Cannot add gates (max reached) and cannot stop (network empty)
        return []

    # =========================================================================
    # Step 2: Calculate Available Node Indices (DAG Constraint)
    # =========================================================================
    # Calculate which node indices can be used as inputs for a new gate.
    # This enforces the DAG (Directed Acyclic Graph) constraint by ensuring
    # that new gates can only reference nodes that already exist.
    #
    # Available indices structure:
    # - Indices 0 to (num_inputs - 1): Original input features
    # - Indices num_inputs to (num_inputs + num_current_gates - 1): Gate outputs
    #
    # Example with num_inputs=10, and 2 gates already added:
    # - Available indices: [0, 1, 2, ..., 9, 10, 11]
    # - Index 10 = output of gate 0
    # - Index 11 = output of gate 1
    #
    # DAG guarantee: Since we can only reference indices < next_gate_index,
    # cycles are impossible (would require referencing future gates).
    num_current_gates = len(lgn_state.gates)
    next_gate_index = self.num_inputs + num_current_gates

    # All available node indices that can be used as gate inputs
    # Range: [0, next_gate_index) = [0, num_inputs + num_current_gates)
    available_indices = list(range(next_gate_index))

    # =========================================================================
    # Step 3: Generate All Valid Gate Addition Actions
    # =========================================================================
    # For each gate type, generate all valid input combinations that satisfy:
    # 1. Arity constraint: correct number of inputs for the gate type
    # 2. DAG constraint: all inputs reference existing nodes (already satisfied)
    #
    # Gate types by arity:
    # - 1-input gates: NOT, BUFFER (unary operations)
    # - 2-input gates: 13 gate types (binary operations, excluding AND)
    # - Variable-arity gates: AND (2+ inputs, all possible combinations)
    #
    # AND gate special handling:
    # - Supports 2, 3, ..., n inputs where n = len(available_indices)
    # - This provides maximum expressiveness for logic gate networks
    # - Example: With 10 available indices, AND can have C(10,2) + C(10,3) + ... + C(10,10) = 1013 combinations

    actions = []

    # Define gate types by arity for organized generation
    # Unary gates: Require exactly 1 input
    unary_gates = [GateType.NOT, GateType.BUFFER]

    # Binary gates: Require exactly 2 inputs (AND excluded - handled separately)
    binary_gates = [
        GateType.OR, GateType.XOR,
        GateType.NAND, GateType.NOR, GateType.XNOR,
        GateType.IMPLY, GateType.NIMPLY,
        GateType.CONVERSE_IMPLY, GateType.CONVERSE_NIMPLY,
        GateType.FIRST, GateType.SECOND,
        GateType.NFIRST, GateType.NSECOND
    ]

    # Generate actions for 1-input gates (NOT, BUFFER)
    # For each unary gate type, create action for each available index
    for gate_type in unary_gates:
        for idx in available_indices:
            action = {
                'gate_type': gate_type,
                'input_indices': (idx,)  # Single-element tuple
            }
            actions.append(action)

    # Generate actions for 2-input gates (13 types, excluding AND)
    # For each binary gate type, create action for each 2-combination of indices
    # combinations(available_indices, 2) generates all unordered pairs
    # This ensures we don't generate duplicate actions like (0,1) and (1,0)
    for gate_type in binary_gates:
        for input_pair in combinations(available_indices, 2):
            action = {
                'gate_type': gate_type,
                'input_indices': input_pair  # Tuple of (idx1, idx2) where idx1 < idx2
            }
            actions.append(action)

    # Generate actions for variable-arity AND gate (2+ inputs)
    # AND gate is unique: it supports any number of inputs >= 2
    #
    # If max_and_arity is set (> 0), limit the maximum arity to reduce action space.
    # Without limit: Total AND actions = C(n,2) + C(n,3) + ... + C(n,n)
    # With limit k:  Total AND actions = C(n,2) + C(n,3) + ... + C(n,k) (much smaller)
    #
    # Example with n=10:
    # - No limit (max_and_arity=0): 1013 AND actions (C(10,2) + ... + C(10,10))
    # - max_and_arity=2: 45 AND actions (like other binary gates)
    # - max_and_arity=4: 375 AND actions
    num_available = len(available_indices)

    # Determine max arity for AND gates
    if self.max_and_arity > 0:
        max_arity = min(self.max_and_arity, num_available)
    else:
        max_arity = num_available  # No limit

    for arity in range(2, max_arity + 1):
        # For each arity (2, 3, 4, ..., max_arity)
        # Generate all combinations of that size
        for input_combination in combinations(available_indices, arity):
            action = {
                'gate_type': GateType.AND,
                'input_indices': input_combination  # Tuple of indices (sorted)
            }
            actions.append(action)

    # =========================================================================
    # Step 4: Add Stop Action (if network is non-empty)
    # =========================================================================
    # The stop action allows the policy to terminate trajectory generation
    # and evaluate the current network. It is only available if:
    # - The network has at least one gate (non-empty)
    #
    # Rationale: Cannot evaluate an empty network (no computation to perform).
    # This matches the behavior in LGNMDP.parent_transitions() where stop
    # action creates a parent-child relationship with the same state.
    if num_current_gates > 0:
        # Network is non-empty: add stop action
        actions.append({'action': 'stop'})

    # Return all valid actions (gate additions + optional stop)
    return actions