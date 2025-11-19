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
  Test LGNState initialization, gate management, and state tracking.

  This test validates:
  - Initialization with default and custom parameters
  - Gate addition and counting
  - Feature tracking
  - Terminal state detection
  - DAG validation
  """
  # ====== Initialization Tests ======
  # Default parameters (num_inputs=10, max_gates=15)
  lgn = LGNState()
  assert 10 == lgn.num_inputs # Verify default num_inputs
  assert 15 == lgn.max_gates  # Verify default max_gates
  assert 0 == lgn.get_num_gates() # Verify empty network
  
  # Custom parameters (num_inputs=5, max_gates=10)
  lgn2 = LGNState(num_inputs=5, max_gates=10)
  assert 5 == lgn2.num_inputs
  assert 10 == lgn2.max_gates
  assert 0 == lgn2.get_num_gates()
  
  # ====== Gate Addition Tests ======
  # Add first gate (AND gate with inputs [0, 1])
  lgn.add_gate(GateType.AND, [0, 1])
  assert 1 == lgn.get_num_gates() # Verify gate count increased to 1
  # Add second gate (OR gate with inputs [0, 1])
  lgn.add_gate(GateType.OR, [0, 1])
  assert 2 == lgn.get_num_gates() # Verify gate count increased to 2
  # Add third gate (NOT gate with input [1])
  lgn.add_gate(GateType.NOT, [1])
  assert 3 == lgn.get_num_gates() # Verify gate count increased to 3
  
  
  # ====== Feature Tracking Tests ======
  # Create new network for feature tracking
  lgn_feature_used = LGNState()
  # Add first gate using input features [0, 1, 2]
  lgn_feature_used.add_gate(GateType.AND, [0, 1, 2])
  assert {0, 1, 2} == lgn_feature_used.get_features_used()  # Verify features 0, 1, 2 are tracked
  
  # Add second gate using input feature 3 and gate output 10 (index of first gate's output)
  lgn_feature_used.add_gate(GateType.OR, [3, 10])
  assert {0, 1, 2, 3} == lgn_feature_used.get_features_used()  # Verify gate output (10) is NOT tracked
  
  
  # ====== Terminal State Tests ======
  lgn_terminal = LGNState(num_inputs=3, max_gates=5)
  assert False == lgn_terminal.is_terminal()  # Verify empty network is not terminal

  lgn_terminal.add_gate(GateType.AND, [0, 1, 2])
  assert True == lgn_terminal.is_terminal() # Verify terminal when all features used
  
  # Test Condition 2: Max gates reached
  lgn_max_gates = LGNState(num_inputs=10, max_gates=2)
  lgn_max_gates.add_gate(GateType.AND, [0, 1])
  assert False == lgn_max_gates.is_terminal()  # Not terminal with 1 gate
  lgn_max_gates.add_gate(GateType.OR, [2, 3])
  assert True == lgn_max_gates.is_terminal()  # Verify terminal when max gates reached


  # ====== DAG Validation Tests ======
  lgn_dag = LGNState(num_inputs=5)
  # Valid connection: references input features only
  lgn_dag.add_gate(GateType.AND, [0, 1])
  assert 1 == lgn_dag.get_num_gates()  # Verify gate was added successfully
  
  # Valid connection: references input feature 2 and gate 0's output (index 5)
  lgn_dag.add_gate(GateType.OR, [2, 5])
  assert 2 == lgn_dag.get_num_gates()  # Verify gate was added successfully
  
  # Valid connection: references previous gate outputs (indices 5 and 6)
  lgn_dag.add_gate(GateType.XOR, [5, 6])
  assert 3 == lgn_dag.get_num_gates()  # Verify gate was added successfully
  
  # Invalid connection: references non-existent gate output (index 10)
  with pytest.raises(ValueError):
    lgn_dag.add_gate(GateType.NOT, [10])
  
  # Invalid connection: references out-of-range index (index 100)
  with pytest.raises(ValueError):
    lgn_dag.add_gate(GateType.AND, [0, 100])
  
  
  # ====== Serialization Tests (to_dict()) ======
  lgn_dict = LGNState(num_inputs=3, max_gates=5)
  dict_empty = lgn_dict.to_dict()
  
  assert 3 == dict_empty['num_inputs']  # Verify num_inputs is serialized correctly
  assert 5 == dict_empty['max_gates'] # Verify max_gates is serialized correctly
  assert [] == dict_empty['gates']  # Verify gates list is empty for empty network
  
  lgn_dict.add_gate(GateType.AND, [0, 1])
  lgn_dict.add_gate(GateType.OR, [1, 3])
  dict_with_gates = lgn_dict.to_dict()
  
  assert 2 == len(dict_with_gates['gates']) # Verify 2 gates are serialized
  assert ('AND', [0, 1]) == dict_with_gates['gates'][0] # Verify first gate is serialized correctly
  assert ('OR', [1, 3]) == dict_with_gates['gates'][1]  # Verify second gate is serialized correctly

def test_evaluator():
  """
  Test LGNEvaluator forward pass evaluation.

  This test validates:
  - Single input evaluation (evaluate method)
  - Batch input evaluation (evaluate_batch method)
  - Edge cases (empty network, input dimension mismatch)
  """
  # ====== Single Input Evaluation Tests ======
  lgn_simple = LGNState(num_inputs=3, max_gates=5)
  lgn_simple.add_gate(GateType.AND, [0, 1, 2])
  
  evaluator = LGNEvaluator()
  
  assert 1 == evaluator.evaluate(lgn_simple, [1, 1, 1])  # All 1s: AND returns 1
  assert 0 == evaluator.evaluate(lgn_simple, [1, 1, 0])  # One 0: AND returns 0
  assert 0 == evaluator.evaluate(lgn_simple, [0, 0, 0])  # All 0s: AND returns 0
  
  
  # Test complex network with multiple gates (chained)
  lgn_complex = LGNState(num_inputs=3, max_gates=5)
  lgn_complex.add_gate(GateType.AND, [0, 1])
  lgn_complex.add_gate(GateType.OR, [2, 3])
  lgn_complex.add_gate(GateType.NOT, [4])

  # Input [1,0,1]: Gate0=AND(1,0)=0, Gate1=OR(1,0)=1, Gate2=NOT(1)=0
  result = evaluator.evaluate(lgn_complex, [1, 0, 1])
  assert 0 == result
  
  # Input [1,1,0]: Gate0=AND(1,1)=1, Gate1=OR(0,1)=1, Gate2=NOT(1)=0
  assert 0 == evaluator.evaluate(lgn_complex, [1, 1, 0])
  
  
  # ====== Batch Evaluation Tests ======
  batch_inputs = [
    [1, 0, 1],  # 예상 출력: 0
    [1, 1, 0],  # 예상 출력: 0
    [0, 0, 1],  # Input [0,0,1]: Gate0=AND(0,0)=0, Gate1=OR(1,0)=1, Gate2=NOT(1)=0
  ]
  
  batch_result = evaluator.evaluate_batch(lgn_complex, batch_inputs)
  assert [0, 0, 0] == batch_result  # Verify batch evaluation returns correct outputs for all inputs
  
  
  # ====== Edge Cases Tests ======
  # Input dimension mismatch: too few inputs
  with pytest.raises(AssertionError):
    evaluator.evaluate(lgn_complex, [1, 0])
  # Input dimension mismatch: too many inputs
  with pytest.raises(AssertionError):
    evaluator.evaluate(lgn_complex, [1, 0, 1, 1])
  