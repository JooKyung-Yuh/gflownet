from enum import Enum

class GateType(Enum):
  """
  Enumeration of 16 logic gate types supported in the Logic Gate Network (LGN) system.
  
  This implementation uses a hybrid architecture:
  - AND gate: Supports variable arity (1 to 5 inputs)
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
  
  
  # Validate input count based on gate type
  if gate_type == GateType.AND:
    if not (1 <= len(inputs) <= 5):
      raise ValueError(f"AND gate requires 1-5 inputs, got {len(inputs)}")
  elif gate_type in [GateType.NOT, GateType.BUFFER]:
    if len(inputs) != 1:
      raise ValueError(f"{gate_type.value} gate requires exactly 1 input, got {len(inputs)}")
  else:
    if len(inputs) != 2:
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
  