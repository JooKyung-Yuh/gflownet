from typing import Any
from ..lgn.network import LGNState
from ..lgn.gates import GateType

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
  def __init__(self, num_inputs: int, max_gates: int) -> None:
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
    >>> action_space = LGNActionSpace(num_inputs=10, max_gates=15)
    >>> print(f"Action space configured for {action_space.num_inputs} inputs")
    """
    self.num_inputs = num_inputs
    self.max_gates = max_gates

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
    pass