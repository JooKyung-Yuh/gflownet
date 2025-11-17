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