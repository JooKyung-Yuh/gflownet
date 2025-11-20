from typing import Any
from ..lgn.network import LGNState
from ..lgn.gates import GateType

class LGNMDP:
  """
  MDP (Markov Decision Process) wrapper for Logic Gate Networks in GFlowNet.

  This class defines the state space and transition dynamics for training a GFlowNet
  to generate Logic Gate Networks. It provides the critical parent_transitions() method
  needed for backward trajectory sampling and Trajectory Balance (TB) loss computation.

  Role in GFlowNet:
  -----------------
  - Defines the MDP structure: states (LGNState), actions (gate additions/removal), transitions
  - Enables backward sampling: Given a state s, find all (parent, action) pairs leading to s
  - Used in TB loss: log[sum_parents exp(Q(parent, action))] = log[R(s) + sum_actions exp(Q(s, a))]

  Design Philosophy:
  ------------------
  - **Stateless**: Does not store LGNState internally; takes state as method parameter
  - **Backward-focused**: Primary role is parent_transitions() for backward trajectory
  - **Simple constraints**: Only constraint is max_gates (no recency buffer like molecules)

  Comparison to Reference Projects:
  ----------------------------------
  - Grid: Simple MDP with discrete positions (~10-20 lines for parent_transitions)
  - Molecules: Complex graph MDP with symmetries (~100-150 lines for parent_transitions)
  - LGN: Medium complexity DAG structure (expected ~20-30 lines for parent_transitions)

  Attributes:
  -----------
  num_inputs : int
      Number of input features to the Logic Gate Network (e.g., 10 for binary vectors).
      Must match the data dimensionality used for training and evaluation.

  max_gates : int
      Maximum number of gates allowed in the network (e.g., 15).
      Acts as hard constraint to prevent runaway network growth.

  Example:
  --------
  >>> from Logic_Gate_Network.gflownet import LGNMDP
  >>> from Logic_Gate_Network.lgn import LGNState, GateType
  >>>
  >>> # Initialize MDP
  >>> mdp = LGNMDP(num_inputs=10, max_gates=15)
  >>>
  >>> # Create a state
  >>> lgn = LGNState(num_inputs=10, max_gates=15)
  >>> lgn.add_gate(gate_type=GateType.AND, input_indices=(0, 1))
  >>>
  >>> # Get all parent states (for backward sampling)
  >>> parents, actions = mdp.parent_transitions(lgn, used_stop_action=False)
  >>> print(f"Found {len(parents)} parent states")

  See Also:
  ---------
  - LGNState: The state representation for Logic Gate Networks
  - Trajectory Balance: The GFlowNet training objective requiring parent_transitions
  - grid/envs/grid.py: Reference MDP implementation for simple discrete environment
  - mols/mdp.py: Reference MDP implementation for complex graph environment
  """
  def __init__(self, num_inputs: int, max_gates: int) -> None:
    """
    Initialize the Logic Gate Network MDP.

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
    - LGNState objects are passed as parameters to methods (e.g., parent_transitions).
    - num_inputs and max_gates are stored only for validation and reference.
    - These values should match the configuration used in LGNState initialization.

    Example:
    --------
    >>> mdp = LGNMDP(num_inputs=10, max_gates=15)
    >>> print(f"MDP configured for {mdp.num_inputs} inputs, up to {mdp.max_gates} gates")
    """
    self.num_inputs = num_inputs
    self.max_gates = max_gates
  
  def parent_transitions(
    self,
    lgn_state: LGNState,
    used_stop_action: bool
  ) -> tuple[list[LGNState], list[dict[str, Any]]]:
    """
    Compute all parent states that can transition to the given state.

    This is the **most critical method** for GFlowNet training. It enables backward
    trajectory sampling by finding all (parent_state, action) pairs that lead to
    the current state. This is required for computing the Trajectory Balance (TB) loss.

    Algorithm Overview:
    -------------------
    1. **Stop Action Case**: If used_stop_action=True, return current state with stop action
    2. **Empty LGN Case**: If lgn_state has no gates, return empty lists (no parents exist)
    3. **Gate Removal**: For each gate in lgn_state:
       - Create a parent state by removing that gate
       - Record the action (gate_type, input_indices) that was removed
       - Add (parent_state, action) to the lists
    4. Return all parent states and corresponding actions

    Backward vs Forward:
    --------------------
    - **Forward (action space)**: From state s, what actions can we take? (add gates with valid inputs)
    - **Backward (parent_transitions)**: From state s, how did we get here? (which gate was added last?)
    - Forward has DAG constraints (topological ordering), Backward is always valid (any gate removal)

    Parameters:
    -----------
    lgn_state : LGNState
        The current Logic Gate Network state for which we want to find parent states.
        Can be empty (no gates), single gate, or multiple gates.

    used_stop_action : bool
        Whether this state was reached via a stop action (terminal state).
        - If True: The parent is lgn_state itself (without stop), action is "stop"
        - If False: The parents are all states with one fewer gate

    Returns:
    --------
    tuple[list[LGNState], list[dict[str, Any]]]
        A tuple containing:
        - parent_states (list[LGNState]): All parent states (one per possible transition)
        - actions (list[dict]): Corresponding actions taken from each parent to reach lgn_state

        Each action dict contains:
          - 'gate_type': GateType enum (AND, OR, NAND, NOR, XOR, XNOR)
          - 'input_indices': tuple[int, int] of input indices to the gate
          - (Optional) Other metadata as needed

    Examples:
    ---------
    **Case 1: Terminal state (used stop action)**
    >>> lgn = LGNState(num_inputs=10, max_gates=15)
    >>> lgn.add_gate(GateType.AND, (0, 1))
    >>> parents, actions = mdp.parent_transitions(lgn, used_stop_action=True)
    >>> len(parents)  # 1 parent: the same state without stop
    1
    >>> actions[0]  # The action is "stop"
    {'action': 'stop'}

    **Case 2: Empty LGN (no gates)**
    >>> lgn = LGNState(num_inputs=10, max_gates=15)
    >>> parents, actions = mdp.parent_transitions(lgn, used_stop_action=False)
    >>> len(parents)  # 0 parents: this is the initial state
    0

    **Case 3: Single gate**
    >>> lgn = LGNState(num_inputs=10, max_gates=15)
    >>> lgn.add_gate(GateType.AND, (0, 1))
    >>> parents, actions = mdp.parent_transitions(lgn, used_stop_action=False)
    >>> len(parents)  # 1 parent: empty LGN
    1
    >>> actions[0]  # The action that was taken: add AND gate with inputs (0,1)
    {'gate_type': <GateType.AND: 1>, 'input_indices': (0, 1)}

    **Case 4: Multiple gates**
    >>> lgn = LGNState(num_inputs=10, max_gates=15)
    >>> lgn.add_gate(GateType.AND, (0, 1))  # Gate 0
    >>> lgn.add_gate(GateType.OR, (2, 3))   # Gate 1
    >>> parents, actions = mdp.parent_transitions(lgn, used_stop_action=False)
    >>> len(parents)  # 2 parents: one for each gate removal
    2
    >>> # Parent 0: LGN with only Gate 1 (removed Gate 0)
    >>> # Parent 1: LGN with only Gate 0 (removed Gate 1)

    Trajectory Balance (TB) Loss Usage:
    ------------------------------------
    The TB loss requires computing:
      log[sum_parents exp(Q(parent, action))] = log[R(s) + sum_actions exp(Q(s, a))]

    This method provides the left-hand side (inflow):
    >>> parents, actions = mdp.parent_transitions(s, used_stop_action=False)
    >>> inflow = torch.logsumexp([Q(p, a) for p, a in zip(parents, actions)], dim=0)

    Reference Implementations:
    --------------------------
    - **grid/envs/grid.py**: Simple parent_transitions (~10-20 lines)
      - Discrete position states
      - 4 parent states (up, down, left, right moves)
      - No complex constraints

    - **mols/mdp.py**: Complex parent_transitions (~100-150 lines)
      - Graph-based states with blocks
      - Symmetry handling and reindexing
      - Recency buffer constraints

    - **LGN (this implementation)**: Expected complexity ~20-30 lines
      - DAG-based states with gates
      - No symmetries (gates are distinguishable)
      - No recency constraints (simpler than molecules)

    Implementation Notes:
    ---------------------
    - Must create **deep copies** of LGNState when generating parents
    - Gate removal should properly update DAG structure
    - Action dicts should contain all info needed to reconstruct the transition
    - Order of parents/actions doesn't matter (GFlowNet is order-agnostic)

    See Also:
    ---------
    - LGNState.copy(): Deep copy method for state cloning
    - LGNState.remove_gate(): Gate removal method (to be implemented in Task 4.2)
    - GateType: Enum defining logic gate types
    - Trajectory Balance loss: Uses this method for backward sampling
    """
    pass