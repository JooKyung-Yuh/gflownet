from enum import Enum, IntEnum

class GateType(Enum):
  """
  Enumeration of 16 logic gate types supported in the Logic Gate Network (LGN) system.
  
  This implementation uses a hybrid architecture:
  - AND gate: Supports variable arity
  - All other 15 gates: Fixed arity (exactly 2 inputs)
  
  Available gate types:
  
  Basic Logic Gates:
    - AND: Logical conjunction (output 1 if all inputs are 1)
    - OR: Logical disjunction (output 1 if any input is 1)
    - XOR: Exclusive OR (output 1 if inputs differ)
    - NAND: Negated AND
    - NOR: Negated OR
    - XNOR: Negated XOR (equivalence gate)
  
  Unary Gates:
    - NOT: Logical negation (inverts input)
    - BUFFER: Identity function (passes input unchanged)
  
  Implication Gates:
    - IMPLY: Material implication (A → B)
    - NIMPLY: Negated implication
    - CONVERSE_IMPLY: Converse implication (B → A)
    - CONVERSE_NIMPLY: Negated converse implication
  
  Projection Gates:
    - FIRST: Returns first input
    - SECOND: Returns second input
    - NFIRST: Returns negation of first input
    - NSECOND: Returns negation of second input
  
  Example:
    >>> gate_type = GateType.AND
    >>> gate_type.value
    'AND'
  """
  AND = "AND"
  OR = "OR"
  XOR = "XOR"
  NAND = "NAND"
  NOR = "NOR"
  XNOR = "XNOR"
  NOT = "NOT"
  BUFFER = "BUFFER"
  IMPLY = "IMPLY"
  NIMPLY = "NIMPLY"
  CONVERSE_IMPLY = "CONVERSE_IMPLY"
  CONVERSE_NIMPLY = "CONVERSE_NIMPLY"
  FIRST = "FIRST"
  SECOND = "SECOND"
  NFIRST = "NFIRST"
  NSECOND = "NSECOND"

def is_valid_arity(gate_type: GateType, num_inputs: int) -> bool:
  """
  Check if the number of inputs is valid for the given gate type.

  Args:
      gate_type (GateType): The gate type to validate.
      num_inputs (int): The number of inputs to check.

  Returns:
      bool: True if the input count is valid for the gate type, False otherwise.

  Example:
      >>> is_valid_arity(GateType.AND, 3)
      True
      >>> is_valid_arity(GateType.OR, 3)
      False
      >>> is_valid_arity(GateType.NOT, 1)
      True
  """
  # Validate input count based on gate type
  if gate_type == GateType.AND:
    if not (2 <= num_inputs):
      return False
  elif gate_type in [GateType.NOT, GateType.BUFFER]:
    if num_inputs != 1:
      return False
  else:
    if num_inputs != 2:
      return False
  return True

def apply_gate(gate_type: GateType, inputs: list[int]) -> int:
  """
  Apply logic gate operation to given inputs.

  Args:
      gate_type (GateType): The type of logic gate to apply.
      inputs (list[int]): List of binary inputs (0 or 1).

  Returns:
      int: Binary output (0 or 1).

  Raises:
      ValueError: If input count doesn't match gate requirements.
      ValueError: If input values are not binary (0 or 1).

  Example:
      >>> apply_gate(GateType.AND, [1, 1, 1])
      1
      >>> apply_gate(GateType.OR, [0, 1])
      1
      >>> apply_gate(GateType.NOT, [1])
      0
  """
  # Binary value check
  if not all(x == 0 or x == 1 for x in inputs):
    raise ValueError("All inputs must be binary (0 or 1)")
  
  # Validate arity
  if not is_valid_arity(gate_type, len(inputs)):
    if gate_type == GateType.AND:
      raise ValueError(f"AND gate requires at least 2 inputs, got {len(inputs)}")
    elif gate_type in [GateType.NOT, GateType.BUFFER]:
      raise ValueError(f"{gate_type.value} gate requires exactly 1 input, got {len(inputs)}")
    else:
      raise ValueError(f"{gate_type.value} gate requires exactly 2 inputs, got {len(inputs)}")
  
  
  # Gate operation logic
  if gate_type == GateType.AND:
    return int(all(x == 1 for x in inputs))
  elif gate_type == GateType.OR:
    return int(any(x == 1 for x in inputs))
  elif gate_type == GateType.XOR:
    return inputs[0] ^ inputs[1]
  elif gate_type == GateType.NAND:
    return int(not all(x == 1 for x in inputs))
  elif gate_type == GateType.NOR:
    return int(not any(x == 1 for x in inputs))
  elif gate_type == GateType.XNOR:
    return int(inputs[0] == inputs[1])
  
  # Unary gates
  elif gate_type == GateType.NOT:
    return 1 - inputs[0]
  elif gate_type == GateType.BUFFER:
    return inputs[0]
  
  # Implication gates
  elif gate_type == GateType.IMPLY:
    return int((inputs[0] == 0) or (inputs[1] == 1))
  elif gate_type == GateType.NIMPLY:
    return int((inputs[0] == 1) and (inputs[1] == 0))
  elif gate_type == GateType.CONVERSE_IMPLY:
    return int((inputs[1] == 0) or (inputs[0] == 1))
  elif gate_type == GateType.CONVERSE_NIMPLY:
    return int((inputs[1] == 1) and (inputs[0] == 0))
  
  # Projection gates
  elif gate_type == GateType.FIRST:
    return inputs[0]
  elif gate_type == GateType.SECOND:
    return inputs[1]
  elif gate_type == GateType.NFIRST:
    return 1 - inputs[0]
  elif gate_type == GateType.NSECOND:
    return 1 - inputs[1]


# Centralized gate type to integer index mapping
# This is the SINGLE SOURCE OF TRUTH for gate type indexing
# Use this mapping in all neural network code (policy networks, embeddings, etc.)
GATE_TYPE_TO_IDX = {
    GateType.AND: 0,
    GateType.OR: 1,
    GateType.XOR: 2,
    GateType.NAND: 3,
    GateType.NOR: 4,
    GateType.XNOR: 5,
    GateType.NOT: 6,
    GateType.BUFFER: 7,
    GateType.IMPLY: 8,
    GateType.NIMPLY: 9,
    GateType.CONVERSE_IMPLY: 10,
    GateType.CONVERSE_NIMPLY: 11,
    GateType.FIRST: 12,
    GateType.SECOND: 13,
    GateType.NFIRST: 14,
    GateType.NSECOND: 15,
}


def get_gate_type_index(gate_type: GateType) -> int:
    """
    Get the integer index (0-15) for a gate type.

    This is used for tensor indexing in neural networks.

    Args:
        gate_type: The GateType enum value

    Returns:
        int: Index from 0 to 15

    Example:
        >>> get_gate_type_index(GateType.AND)
        0
        >>> get_gate_type_index(GateType.OR)
        1
    """
    return GATE_TYPE_TO_IDX[gate_type]
