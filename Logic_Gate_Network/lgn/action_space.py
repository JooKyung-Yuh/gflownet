from .network import LGNState
from .gates import GateType, is_valid_arity

from itertools import combinations

class ActionSpace:
  """
  Action Space for Logic Gate Networks (LGN).

  This class defines structurally valid actions that can be taken from a given LGN state.
  It enforces structural constraints to ensure valid network construction.

  Version 5 Key Feature:
    - NO recency constraint (all gates are always referenceable!)
    - ONLY DAG constraint (prevents cycles)
    - Maximum expressiveness for GFlowNet exploration

  Structural Rules:
    1. DAG constraint: No cyclic references (only constraint!)
    2. Arity constraint: Correct number of inputs per gate type
    3. Termination: Check if network can still be expanded

  Usage:
    - Phase 2.4: Define structurally possible actions
    - Phase 4: Used by GFlowNet policy network for action selection

  Attributes:
      lgn_state (LGNState): The current LGN state instance

  Example:
      >>> lgn = LGNState(num_inputs=10, max_gates=15)
      >>> action_space = ActionSpace(lgn)
      >>> valid_actions = action_space.get_valid_actions()
  """
  def __init__(self, lgn_state:LGNState) -> None:
    """
    Initialize ActionSpace for a given LGN state.

    Args:
        lgn_state (LGNState): The LGN state instance to define actions for.

    NOTE (Ver5):
        - NO lookback parameter (removed in Ver5!)
        - All input features are ALWAYS referenceable
        - All gate outputs are ALWAYS referenceable
        - Only DAG validation is performed
    """
    self.lgn_state = lgn_state

  def get_available_inputs(self) -> list[int]:
    """
    Get all referenceable input indices (NO recency constraint!).

    This method returns ALL indices that can be referenced when adding a new gate.
    In Ver5, there is NO recency constraint, so all inputs and gate outputs are
    always available.

    Returns:
        list[int]: List of all referenceable indices:
            - All input features: 0 to num_inputs-1
            - ALL gate outputs: num_inputs to num_inputs+num_gates-1

    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.add_gate(GateType.AND, [0, 1, 2])
        >>> lgn.add_gate(GateType.OR, [3, 10])
        >>> action_space = ActionSpace(lgn)
        >>> action_space.get_available_inputs()
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # All inputs (0-9) + All gates (10-11)
    """
    
    # Get all input feature indices (always available in Ver5)
    input_indices = list(range(self.lgn_state.num_inputs))
    
    # Get all gate output indices (NO recency constraint!)
    gate_indices = list(range(
        self.lgn_state.num_inputs,
        self.lgn_state.num_inputs + self.lgn_state.get_num_gates()
    ))
    
    # Combine both (all inputs and all gates are referenceable)
    available_inputs = input_indices + gate_indices
    
    return available_inputs


  
  def get_valid_actions(self) -> list[dict]:
    """
    Get all structurally valid actions from current LGN state.

    This method generates all possible actions that satisfy structural constraints.
    Each action represents adding a gate with specific inputs to the network.

    Structural Rules (Ver5 - DAG only!):
        1. DAG constraint: No cyclic references (prevents loops)
        2. Arity constraint: Correct input count per gate type
        3. Termination check: Network can still be expanded
        4. NO recency constraint (all gates referenceable!)

    Returns:
        list[dict]: List of valid actions, each with format:
            {
                "gate_type": GateType,
                "inputs": [index1, index2, ...]
            }

    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> action_space = ActionSpace(lgn)
        >>> actions = action_space.get_valid_actions()
        >>> actions[0]
        {'gate_type': <GateType.AND>, 'inputs': [0, 1]}
        >>> actions[1]
        {'gate_type': <GateType.OR>, 'inputs': [2, 3]}
    """
    
    if self.lgn_state.is_terminal():
      return []  # No actions if terminal state
    
    
    available_indices = self.get_available_inputs()    
    valid_actions = []

    for gate_type in GateType:
      # AND gate: variable arity (1 to all available indices)
      if gate_type == GateType.AND:
        # Generate all combinations for this arity
        for arity in range(1, len(available_indices) + 1):
          for combination in combinations(available_indices, arity):
            # Convert tuple to list
            inputs = list(combination)
            # Validate DAG constraint (no cycles)
            if self.lgn_state._is_valid_connection(inputs):
              # Create action dict and add to valid actions
              action = {
                "gate_type": gate_type,
                "inputs": inputs
              }
              valid_actions.append(action)
      
      # Unary gates (NOT, BUFFER): fixed arity of 1
      elif gate_type in [GateType.NOT, GateType.BUFFER]:
        arity = 1
        for combination in combinations(available_indices, arity):
          inputs = list(combination)
          if self.lgn_state._is_valid_connection(inputs):
            action = {
                "gate_type": gate_type,
                "inputs": inputs
              }
            valid_actions.append(action)
      
      # Binary gates (OR, XOR, NAND, etc.): fixed arity of 2
      else:
        arity = 2
        for combination in combinations(available_indices, arity):
          inputs = list(combination)
          if self.lgn_state._is_valid_connection(inputs):
            action = {
                "gate_type": gate_type,
                "inputs": inputs
              }
            valid_actions.append(action)
      
    return valid_actions
            
  def is_valid_action(self, action: dict) -> bool:
    """
    Check if a specific action is structurally valid.

    This method validates a single action against the same structural rules
    used by get_valid_actions().

    Args:
        action (dict): Action to validate with format:
            {
                "gate_type": GateType,
                "inputs": [index1, index2, ...]
            }

    Returns:
        bool: True if action is structurally valid, False otherwise.

    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> action_space = ActionSpace(lgn)
        >>> 
        >>> # Valid action
        >>> action_space.is_valid_action({'gate_type': GateType.AND, 'inputs': [0, 1]})
        True
        >>> 
        >>> # Invalid action (references non-existent gate)
        >>> action_space.is_valid_action({'gate_type': GateType.OR, 'inputs': [0, 50]})
        False
    """
    pass