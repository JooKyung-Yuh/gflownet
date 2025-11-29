from .gates import GateType, is_valid_arity

from dataclasses import dataclass

@dataclass
class Gate:
    gate_type: GateType
    inputs: list[int]


class LGNState:
  """
  Represents the state of a Logic Gate Network (LGN) in the GFlowNet system.

  This class manages the construction and validation of logic gate networks using
  a hybrid architecture where the AND gate supports variable arity (1 or more inputs)
  and all other 15 gate types require exactly 2 inputs.

  The network is represented as a list of Gate objects. Each Gate contains
  a gate_type and a list of input_indices. Input indices can reference either
  original input features (0 to num_inputs-1) or outputs from previous gates
  (num_inputs onwards).

  Attributes:
      num_inputs (int): Number of input features for the network.
      max_gates (int): Maximum number of gates allowed in the network.
      gates (list): List of Gate objects.

  Example:
      >>> lgn = LGNState(num_inputs=10, max_gates=15)
      >>> lgn.add_gate(GateType.AND, [0, 1, 2])  # AND gate with 3 inputs
      >>> lgn.add_gate(GateType.OR, [3, 10])     # OR gate using input 3 and gate 0's output
  """
  def __init__(self, num_inputs:int=10, max_gates:int=15) -> None:
    """
    Initialize an empty Logic Gate Network state.

    Args:
        num_inputs (int): Number of input features. Default is 10.
        max_gates (int): Maximum number of gates allowed. Default is 15.
    """
    self.num_inputs = num_inputs
    self.max_gates = max_gates
    self.gates: list[Gate] = []
    
  def add_gate(self, gate_type: GateType, input_indices: list[int]) -> None:
    """
    Add a gate to the network with arity and DAG validation.

    This method validates the gate before adding it to the network:
    1. Arity validation: Ensures the gate type supports the number of inputs
    2. DAG validation: Ensures the connection maintains acyclic structure

    Args:
        gate_type (GateType): The type of logic gate to add.
        input_indices (list[int]): List of input indices for the gate.
            Each index can reference either original input features (0 to num_inputs-1)
            or outputs from previous gates (num_inputs onwards).

    Raises:
        ValueError: If the input count doesn't match gate type requirements.
        ValueError: If the connection violates DAG constraints (creates a cycle
            or references non-existent gate outputs).

    Example:
        >>> lgn = LGNState(num_inputs=10)
        >>> lgn.add_gate(GateType.AND, [0, 1, 2])  # Valid: 3-input AND
        >>> lgn.add_gate(GateType.OR, [3, 10])     # Valid: uses input 3 and gate 0's output
    """
    # Arity validation
    if not is_valid_arity(gate_type, len(input_indices)):
      if gate_type == GateType.AND:
        raise ValueError(f"AND gate requires at least 2 inputs, got {len(input_indices)}")
      elif gate_type in [GateType.NOT, GateType.BUFFER]:
        raise ValueError(f"{gate_type.value} gate requires exactly 1 input, got {len(input_indices)}")
      else:
        raise ValueError(f"{gate_type.value} gate requires exactly 2 inputs, got {len(input_indices)}")
    
    # DAG validation
    if not self._is_valid_connection(input_indices):
      raise ValueError(f"Invalid connection: {input_indices}")
    
    #Add gate
    self.gates.append(Gate(gate_type, input_indices))
  
  def _is_valid_connection(self, input_indices: list[int]) -> bool:
    """
    Validate that input indices maintain DAG structure by enforcing temporal ordering.
    
    This method ensures the network remains a Directed Acyclic Graph (DAG) by
    allowing gates to reference only previously added gates or original inputs.
    The temporal constraint (past-only references) guarantees acyclicity:
    a cycle would require time-reversal, which is structurally impossible.
    
    Args:
        input_indices (list[int]): Indices to validate. Each index can reference:
            - Original inputs: 0 to num_inputs-1
            - Previously added gates: num_inputs to num_inputs+len(gates)-1
    
    Returns:
        bool: True if all indices are within valid range (preventing future/self references),
              False otherwise.
    
    Example:
        >>> lgn = LGNState(num_inputs=10)
        >>> lgn._is_valid_connection([0, 1, 2])  # True: references inputs
        >>> lgn.add_gate(GateType.AND, [0, 1, 2])
        >>> lgn._is_valid_connection([3, 10])    # True: 10 is gate 0 output
        >>> lgn._is_valid_connection([5, 11])    # False: 11 doesn't exist yet
    """
    max_valid_index = self.num_inputs + len(self.gates) - 1
    return all(0 <= idx <= max_valid_index for idx in input_indices)
  
  def get_num_gates(self) -> int:
    """
    Get the number of gates currently in the network.
    
    Returns:
        int: The count of gates added to the network.
    
    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.get_num_gates()
        0
        >>> lgn.add_gate(GateType.AND, [0, 1, 2])
        >>> lgn.get_num_gates()
        1
    """
    return len(self.gates)
  
  def get_features_used(self) -> set[int]:
    """
    Get the set of original input features used by the network.

    This method traverses all gates and collects indices that reference
    original input features (indices < num_inputs), ignoring gate outputs.

    Returns:
        set[int]: Set of input feature indices (0 to num_inputs-1) that are
                  referenced by at least one gate in the network.

    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.get_features_used()
        set()
        >>> lgn.add_gate(GateType.AND, [0, 1, 2])
        >>> lgn.get_features_used()
        {0, 1, 2}
        >>> lgn.add_gate(GateType.OR, [3, 10])  # 10 is gate output, not feature
        >>> lgn.get_features_used()
        {0, 1, 2, 3}
    """
    features = set()
    for gate in self.gates:
      for idx in gate.inputs:
        if idx < self.num_inputs:  # Only original input features
          features.add(idx)
    return features

  def get_connected_inputs_to_output(self) -> set[int]:
    """
    Get the set of input features that are connected to the final output.

    This method traces backwards from the last gate to find all input features
    that contribute to the final output. This is different from get_features_used()
    which counts ALL features used by ANY gate, even if they don't connect to output.

    Returns:
        set[int]: Set of input indices (0 to num_inputs-1) connected to final output.

    Example:
        >>> lgn = LGNState(num_inputs=5, max_gates=10)
        >>> lgn.add_gate(GateType.AND, [0, 1])    # Gate 0, output at 5
        >>> lgn.add_gate(GateType.OR, [2, 3])     # Gate 1, output at 6 (disconnected!)
        >>> lgn.add_gate(GateType.NOT, [5])       # Gate 2, output at 7, uses Gate 0
        >>> lgn.get_features_used()
        {0, 1, 2, 3}  # All features used by some gate
        >>> lgn.get_connected_inputs_to_output()
        {0, 1}  # Only features connected to final output (Gate 2 -> Gate 0)
    """
    if len(self.gates) == 0:
      return set()

    # Start from last gate and trace backwards
    connected_nodes = set()
    to_visit = [self.num_inputs + len(self.gates) - 1]  # Last gate output index

    while to_visit:
      node_idx = to_visit.pop()
      if node_idx in connected_nodes:
        continue
      connected_nodes.add(node_idx)

      if node_idx < self.num_inputs:
        # This is an input feature, don't trace further
        continue

      # This is a gate output, trace its inputs
      gate_idx = node_idx - self.num_inputs
      if 0 <= gate_idx < len(self.gates):
        for input_idx in self.gates[gate_idx].inputs:
          to_visit.append(input_idx)

    # Filter to only input features
    return {idx for idx in connected_nodes if idx < self.num_inputs}

  def get_gate_usage_stats(self) -> dict:
    """
    Get statistics about gate usage in the network.

    Returns:
        dict: Contains:
            - 'total_gates': Total number of gates
            - 'connected_gates': Number of gates connected to final output
            - 'connected_inputs': Number of inputs connected to final output
            - 'total_inputs': Total number of inputs
            - 'gate_type_counts': Dict of gate type -> count
    """
    from collections import Counter

    stats = {
      'total_gates': len(self.gates),
      'total_inputs': self.num_inputs,
      'connected_inputs': len(self.get_connected_inputs_to_output()),
      'features_used': len(self.get_features_used()),
      'gate_type_counts': Counter(gate.gate_type.value for gate in self.gates)
    }

    # Count connected gates
    if len(self.gates) == 0:
      stats['connected_gates'] = 0
    else:
      connected_nodes = set()
      to_visit = [self.num_inputs + len(self.gates) - 1]
      while to_visit:
        node_idx = to_visit.pop()
        if node_idx in connected_nodes:
          continue
        connected_nodes.add(node_idx)
        if node_idx >= self.num_inputs:
          gate_idx = node_idx - self.num_inputs
          if 0 <= gate_idx < len(self.gates):
            for input_idx in self.gates[gate_idx].inputs:
              to_visit.append(input_idx)
      stats['connected_gates'] = len([n for n in connected_nodes if n >= self.num_inputs])

    return stats
  
  def is_terminal(self)->bool:
    """
    Check if the network has reached a terminal state.
    
    Two-fold termination condition (OR logic):
    1. All input features are connected to the network
    2. Maximum number of gates has been reached
    
    Returns:
        bool: True if either termination condition is met, False otherwise.
    
    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.is_terminal()
        False
        >>> # Add gates until all 10 features are used
        >>> lgn.add_gate(GateType.AND, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> lgn.is_terminal()
        True  # Condition 1 met
        >>> 
        >>> lgn2 = LGNState(num_inputs=10, max_gates=3)
        >>> lgn2.add_gate(GateType.AND, [0, 1])
        >>> lgn2.add_gate(GateType.OR, [2, 10])
        >>> lgn2.add_gate(GateType.XOR, [3, 11])
        >>> lgn2.is_terminal()
        True  # Condition 2 met (3 gates == max_gates)
    """
    # Condition 1: All features connected
    all_features_connected = (len(self.get_features_used()) == self.num_inputs)
    
    # Condition 2: Max gates reached
    max_gates_reached = (len(self.gates) >= self.max_gates)
    
    # Terminate if EITHER is true
    return all_features_connected or max_gates_reached
    
  
  def to_dict(self) -> dict:
    """
    Serialize the LGN state to a dictionary for logging and storage.

    Returns:
        dict: Dictionary representation with num_inputs, max_gates, and gates.
              Gates are serialized as (gate_type_string, input_indices) tuples.

    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.add_gate(GateType.AND, [0, 1, 2])
        >>> lgn.add_gate(GateType.OR, [3, 10])
        >>> lgn.to_dict()
        {
            'num_inputs': 10,
            'max_gates': 15,
            'gates': [('AND', [0, 1, 2]), ('OR', [3, 10])]
        }
    """
    return {
        'num_inputs': self.num_inputs,
        'max_gates': self.max_gates,
        'gates': [(gate.gate_type.value, gate.inputs) for gate in self.gates]
    }

  def copy(self) -> 'LGNState':
    """
    Create a deep copy of the LGNState.

    This method creates a completely independent duplicate of the current state.
    Modifying the copy will not affect the original state, and vice versa.
    This is essential for parent_transitions() in LGNMDP, where we need to
    create parent states by removing gates without modifying the original.

    Returns:
        LGNState: A new LGNState instance with the same configuration and gates.

    Example:
        >>> lgn = LGNState(num_inputs=10, max_gates=15)
        >>> lgn.add_gate(GateType.AND, [0, 1, 2])
        >>> copy = lgn.copy()
        >>> copy.add_gate(GateType.OR, [3, 10])  # Original unchanged
        >>> lgn.get_num_gates()
        1  # Original has 1 gate
        >>> copy.get_num_gates()
        2  # Copy has 2 gates
    """
    new_lgn = LGNState(num_inputs=self.num_inputs, max_gates=self.max_gates)
    # Deep copy each gate
    for gate in self.gates:
      # inputs can be either list or tuple, handle both
      inputs_copy = list(gate.inputs) if isinstance(gate.inputs, (list, tuple)) else gate.inputs.copy()
      new_lgn.gates.append(Gate(gate.gate_type, inputs_copy))
    return new_lgn

  def remove_gate(self, gate_idx: int) -> None:
    """
    Remove a gate from the network by index and reindex remaining gates.

    This method removes a gate at the specified index from the gates list,
    then updates all remaining gates' input indices to account for the removal.
    This is similar to the molecules implementation where block removal triggers
    reindexing of junction bonds.

    Reindexing Logic:
    -----------------
    When Gate i is removed, its output index (num_inputs + i) becomes invalid.
    All gates that reference indices > (num_inputs + i) must decrement their
    input indices by 1 to account for the shift in gate output positions.

    Example:
        >>> lgn = LGNState(num_inputs=5, max_gates=10)
        >>> lgn.add_gate(GateType.AND, [0, 1])    # Gate 0 -> output at index 5
        >>> lgn.add_gate(GateType.OR, [2, 3])     # Gate 1 -> output at index 6
        >>> lgn.add_gate(GateType.XOR, [4, 6])    # Gate 2 -> output at index 7, uses Gate 1
        >>>
        >>> # Before removal: Gate 2 inputs are [4, 6] (6 is Gate 1's output)
        >>> lgn.remove_gate(0)  # Remove Gate 0
        >>> # After removal: Gate 2 (now Gate 1) inputs are [4, 5] (5 is new index of Gate 1's output)

    Args:
        gate_idx (int): The index of the gate to remove (0-indexed).
            Must be within range [0, len(gates)-1].

    Raises:
        ValueError: If gate_idx is out of valid range.

    Implementation Notes:
    ---------------------
    - This follows the molecules/mdp.py pattern of reindexing after removal
    - Ensures parent states are semantically valid (no dangling references)
    - Critical for maintaining DAG consistency in GFlowNet training
    """
    if not (0 <= gate_idx < len(self.gates)):
      raise ValueError(f"Invalid gate index {gate_idx}. Must be in range [0, {len(self.gates)-1}]")

    # Calculate the output index of the gate being removed
    # This gate's output is at: num_inputs + gate_idx
    removed_output_index = self.num_inputs + gate_idx

    # Remove the gate from the list
    del self.gates[gate_idx]

    # Reindex remaining gates' inputs to account for the removed gate
    # Any input index > removed_output_index needs to be decremented by 1
    # because all subsequent gate outputs shift down by one position
    for gate in self.gates:
      reindexed_inputs = []
      for inp in gate.inputs:
        if inp > removed_output_index:
          # This input references a gate output that came after the removed gate
          # Decrement by 1 since all subsequent gate outputs shift down by 1
          reindexed_inputs.append(inp - 1)
        else:
          # Input references:
          # - Original input features (indices 0 to num_inputs-1), OR
          # - Gates before the removed gate (indices num_inputs to removed_output_index)
          # These indices remain unchanged
          reindexed_inputs.append(inp)

      # Update the gate's inputs with reindexed values
      gate.inputs = reindexed_inputs