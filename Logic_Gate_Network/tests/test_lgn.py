import pytest
from ..lgn import GateType, LGNState, LGNEvaluator, apply_gate, is_valid_arity

def test_gate_operations():
  """
  Test all 16 logic gate operations and arity validation.

  This test validates:
  - Truth tables for all 16 gate types
  - Unary gates (NOT, BUFFER) with arity=1
  - Binary gates (OR, XOR, NAND, etc.) with arity=2
  - AND gate variable arity (arity 1 to 5)
  """
  # ====== Unary Gates (arity = 1) ======
  # NOT Gate
  assert 1 == apply_gate(GateType.NOT, [0])
  assert 0 == apply_gate(GateType.NOT, [1])
  # BUFFER Gate
  assert 0 == apply_gate(GateType.BUFFER, [0])
  assert 1 == apply_gate(GateType.BUFFER, [1])

  # ====== Binary Gates (arity = 2) ======
  # OR Gate
  assert 0 == apply_gate(GateType.OR, [0, 0])
  assert 1 == apply_gate(GateType.OR, [0, 1])
  assert 1 == apply_gate(GateType.OR, [1, 0])
  assert 1 == apply_gate(GateType.OR, [1, 1])
  #XOR Gate
  assert 0 == apply_gate(GateType.XOR, [0, 0])
  assert 1 == apply_gate(GateType.XOR, [0, 1])
  assert 1 == apply_gate(GateType.XOR, [1, 0])
  assert 0 == apply_gate(GateType.XOR, [1, 1])
  # NAND Gate
  assert 1 == apply_gate(GateType.NAND, [0, 0])
  assert 1 == apply_gate(GateType.NAND, [0, 1])
  assert 1 == apply_gate(GateType.NAND, [1, 0])
  assert 0 == apply_gate(GateType.NAND, [1, 1])
  # NOR Gate
  assert 1 == apply_gate(GateType.NOR, [0, 0])
  assert 0 == apply_gate(GateType.NOR, [0, 1])
  assert 0 == apply_gate(GateType.NOR, [1, 0])
  assert 0 == apply_gate(GateType.NOR, [1, 1])
  # XNOR Gate
  assert 1 == apply_gate(GateType.XNOR, [0, 0])
  assert 0 == apply_gate(GateType.XNOR, [0, 1])
  assert 0 == apply_gate(GateType.XNOR, [1, 0])
  assert 1 == apply_gate(GateType.XNOR, [1, 1])
  # IMPLY Gate
  assert 1 == apply_gate(GateType.IMPLY, [0, 0])
  assert 1 == apply_gate(GateType.IMPLY, [0, 1])
  assert 0 == apply_gate(GateType.IMPLY, [1, 0])
  assert 1 == apply_gate(GateType.IMPLY, [1, 1])
  # NIMPLY Gate
  assert 0 == apply_gate(GateType.NIMPLY, [0, 0])
  assert 0 == apply_gate(GateType.NIMPLY, [0, 1])
  assert 1 == apply_gate(GateType.NIMPLY, [1, 0])
  assert 0 == apply_gate(GateType.NIMPLY, [1, 1])
  # CONVERSE_IMPLY Gate
  assert 1 == apply_gate(GateType.CONVERSE_IMPLY, [0, 0])
  assert 0 == apply_gate(GateType.CONVERSE_IMPLY, [0, 1])
  assert 1 == apply_gate(GateType.CONVERSE_IMPLY, [1, 0])
  assert 1 == apply_gate(GateType.CONVERSE_IMPLY, [1, 1])
  # CONVERSE_NIMPLY Gate
  assert 0 == apply_gate(GateType.CONVERSE_NIMPLY, [0, 0])
  assert 1 == apply_gate(GateType.CONVERSE_NIMPLY, [0, 1])
  assert 0 == apply_gate(GateType.CONVERSE_NIMPLY, [1, 0])
  assert 0 == apply_gate(GateType.CONVERSE_NIMPLY, [1, 1])
  # FIRST Gate
  assert 0 == apply_gate(GateType.FIRST, [0, 0])
  assert 0 == apply_gate(GateType.FIRST, [0, 1])
  assert 1 == apply_gate(GateType.FIRST, [1, 0])
  assert 1 == apply_gate(GateType.FIRST, [1, 1])
  # SECOND Gate
  assert 0 == apply_gate(GateType.SECOND, [0, 0])
  assert 1 == apply_gate(GateType.SECOND, [0, 1])
  assert 0 == apply_gate(GateType.SECOND, [1, 0])
  assert 1 == apply_gate(GateType.SECOND, [1, 1])
  # NFIRST Gate
  assert 1 == apply_gate(GateType.NFIRST, [0, 0])
  assert 1 == apply_gate(GateType.NFIRST, [0, 1])
  assert 0 == apply_gate(GateType.NFIRST, [1, 0])
  assert 0 == apply_gate(GateType.NFIRST, [1, 1])
  # NSECOND Gate
  assert 1 == apply_gate(GateType.NSECOND, [0, 0])
  assert 0 == apply_gate(GateType.NSECOND, [0, 1])
  assert 1 == apply_gate(GateType.NSECOND, [1, 0])
  assert 0 == apply_gate(GateType.NSECOND, [1, 1])

  # ====== AND Gate (Variable Arity: arity >= 1) ======
  # Arity 1
  assert 0 == apply_gate(GateType.AND, [0])
  assert 1 == apply_gate(GateType.AND, [1])
  # Arity 2
  assert 0 == apply_gate(GateType.AND, [0, 0])
  assert 0 == apply_gate(GateType.AND, [0, 1])
  assert 0 == apply_gate(GateType.AND, [1, 0])
  assert 1 == apply_gate(GateType.AND, [1, 1])
  # Arity 3
  assert 0 == apply_gate(GateType.AND, [0, 0, 0])
  assert 0 == apply_gate(GateType.AND, [1, 1, 0])
  assert 1 == apply_gate(GateType.AND, [1, 1, 1])
  # Arity 4
  assert 0 == apply_gate(GateType.AND, [1, 1, 1, 0])
  assert 1 == apply_gate(GateType.AND, [1, 1, 1, 1])
  # Arity 5
  assert 1 == apply_gate(GateType.AND, [1, 1, 1, 1, 1])
  
  # ====== Arity Validation Tests ======
  # AND gate (arity >= 1)
  assert True == is_valid_arity(GateType.AND, 1)
  assert True == is_valid_arity(GateType.AND, 3)
  assert True == is_valid_arity(GateType.AND, 10)
  assert False == is_valid_arity(GateType.AND, 0)
  # Unary gates (arity = 1 only)
  assert True == is_valid_arity(GateType.NOT, 1)
  assert False == is_valid_arity(GateType.NOT, 2)
  assert True == is_valid_arity(GateType.BUFFER, 1)
  assert False == is_valid_arity(GateType.BUFFER, 2)
  # Binary gates (arity = 2 only)
  assert True == is_valid_arity(GateType.OR, 2)
  assert False == is_valid_arity(GateType.OR, 1)
  assert True == is_valid_arity(GateType.XOR, 2)
  assert False == is_valid_arity(GateType.XOR, 3)
  
  # ====== Edge Cases (Exception Handling) ======
  # Invalid arity
  with pytest.raises(ValueError):
    apply_gate(GateType.NOT, [0, 1])

  with pytest.raises(ValueError):
    apply_gate(GateType.OR, [0])

  # Invalid input values (non-binary)
  with pytest.raises(ValueError):
    apply_gate(GateType.AND, [2])

  with pytest.raises(ValueError):
    apply_gate(GateType.OR, [0, 5])

def test_lgn_state():
  """
  """
  
  pass

def test_evaluator():
  pass