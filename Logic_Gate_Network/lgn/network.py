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
        raise ValueError(f"AND gate requires at least 1 inputs, got {len(input_indices)}")
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