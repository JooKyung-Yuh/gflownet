import pytest
from gflownet import LGNMDP
from lgn import LGNState, GateType


def test_lgn_mdp_initialization():
  """
  Test LGNMDP initialization with various configurations.

  Validates:
  - Correct storage of num_inputs and max_gates
  - No internal state storage (stateless design)
  """
  # Test default configuration
  mdp = LGNMDP(num_inputs=10, max_gates=15)
  assert mdp.num_inputs == 10
  assert mdp.max_gates == 15

  # Test custom configuration
  mdp_custom = LGNMDP(num_inputs=5, max_gates=20)
  assert mdp_custom.num_inputs == 5
  assert mdp_custom.max_gates == 20


def test_parent_transitions_stop_action():
  """
  Test parent_transitions() with stop action (terminal state).

  Case: used_stop_action=True
  Expected behavior:
  - Returns exactly 1 parent state (identical to current state)
  - Returns exactly 1 action: {'action': 'stop'}
  - Parent state is independent copy (modifying it doesn't affect original)
  """
  # Setup: Create MDP and a non-empty LGN state
  mdp = LGNMDP(num_inputs=10, max_gates=15)
  lgn = LGNState(num_inputs=10, max_gates=15)

  # Add a gate to make it non-empty
  lgn.add_gate(GateType.AND, [0, 1])

  # Execute: Get parent transitions for terminal state
  parents, actions = mdp.parent_transitions(lgn, used_stop_action=True)

  # Validate: Should return exactly 1 parent with stop action
  assert len(parents) == 1, "Terminal state should have exactly 1 parent"
  assert len(actions) == 1, "Terminal state should have exactly 1 action"

  # Validate: Action should be stop
  assert actions[0] == {'action': 'stop'}, "Action should be stop"

  # Validate: Parent should be a copy (independent)
  parent = parents[0]
  assert parent.get_num_gates() == 1, "Parent should have same gates as original"

  # Modify parent and verify original is unchanged
  parent.add_gate(GateType.OR, [2, 3])
  assert lgn.get_num_gates() == 1, "Original should remain unchanged"
  assert parent.get_num_gates() == 2, "Parent copy should be modified"


def test_parent_transitions_empty_lgn():
  """
  Test parent_transitions() with empty LGN (initial state).

  Case: Empty network (no gates)
  Expected behavior:
  - Returns empty lists for both parents and actions
  - No parent states exist for the initial state
  """
  # Setup: Create MDP and empty LGN state
  mdp = LGNMDP(num_inputs=10, max_gates=15)
  lgn = LGNState(num_inputs=10, max_gates=15)

  # Execute: Get parent transitions for empty state
  parents, actions = mdp.parent_transitions(lgn, used_stop_action=False)

  # Validate: Should return empty lists
  assert len(parents) == 0, "Empty LGN should have no parent states"
  assert len(actions) == 0, "Empty LGN should have no actions"


def test_parent_transitions_single_gate():
  """
  Test parent_transitions() with single gate.

  Case: LGN with exactly 1 gate
  Expected behavior:
  - Returns exactly 1 parent state (empty LGN)
  - Returns exactly 1 action (the gate that was added)
  - Parent state has 0 gates
  """
  # Setup: Create MDP and LGN with one gate
  mdp = LGNMDP(num_inputs=10, max_gates=15)
  lgn = LGNState(num_inputs=10, max_gates=15)

  # Add a single AND gate
  lgn.add_gate(GateType.AND, [0, 1])

  # Execute: Get parent transitions
  parents, actions = mdp.parent_transitions(lgn, used_stop_action=False)

  # Validate: Should return exactly 1 parent
  assert len(parents) == 1, "Single-gate LGN should have exactly 1 parent"
  assert len(actions) == 1, "Single-gate LGN should have exactly 1 action"

  # Validate: Parent should be empty LGN
  parent = parents[0]
  assert parent.get_num_gates() == 0, "Parent should be empty LGN"

  # Validate: Action should match the gate that was removed
  action = actions[0]
  assert action['gate_type'] == GateType.AND, "Action gate_type should match"
  assert action['input_indices'] == (0, 1), "Action input_indices should match"


def test_parent_transitions_multiple_gates():
  """
  Test parent_transitions() with multiple gates.

  Case: LGN with 3 gates
  Expected behavior:
  - Returns exactly 3 parent states (one for each gate removal)
  - Each parent has 2 gates (current has 3, remove 1)
  - Actions correspond to the gates that were added
  """
  # Setup: Create MDP and LGN with three gates
  mdp = LGNMDP(num_inputs=10, max_gates=15)
  lgn = LGNState(num_inputs=10, max_gates=15)

  # Add three gates
  lgn.add_gate(GateType.AND, [0, 1])     # Gate 0
  lgn.add_gate(GateType.OR, [2, 3])      # Gate 1
  lgn.add_gate(GateType.XOR, [4, 10])    # Gate 2 (uses Gate 0's output)

  # Execute: Get parent transitions
  parents, actions = mdp.parent_transitions(lgn, used_stop_action=False)

  # Validate: Should return exactly 3 parents (one for each gate)
  assert len(parents) == 3, "3-gate LGN should have exactly 3 parents"
  assert len(actions) == 3, "3-gate LGN should have exactly 3 actions"

  # Validate: Each parent should have 2 gates
  for parent in parents:
    assert parent.get_num_gates() == 2, "Each parent should have 2 gates"

  # Validate: Actions should match the gates
  expected_actions = [
    {'gate_type': GateType.AND, 'input_indices': (0, 1)},
    {'gate_type': GateType.OR, 'input_indices': (2, 3)},
    {'gate_type': GateType.XOR, 'input_indices': (4, 10)}
  ]

  # Check that all expected actions are present (order may vary)
  for expected_action in expected_actions:
    assert expected_action in actions, f"Action {expected_action} should be in actions list"


def test_parent_transitions_variable_arity_and_gate():
  """
  Test parent_transitions() with variable-arity AND gate.

  Case: LGN with multi-input AND gate
  Expected behavior:
  - Correctly handles AND gates with 1, 2, 3+ inputs
  - Action preserves the exact input_indices tuple
  """
  # Setup: Create MDP and test various AND gate arities
  mdp = LGNMDP(num_inputs=10, max_gates=15)

  # Test Case 1: Single-input AND gate
  lgn1 = LGNState(num_inputs=10, max_gates=15)
  lgn1.add_gate(GateType.AND, [5])  # 1-input AND
  parents1, actions1 = mdp.parent_transitions(lgn1, used_stop_action=False)

  assert len(parents1) == 1, "1-gate LGN should have 1 parent"
  assert actions1[0]['gate_type'] == GateType.AND
  assert actions1[0]['input_indices'] == (5,), "1-input AND should have 1-element tuple"

  # Test Case 2: 3-input AND gate
  lgn2 = LGNState(num_inputs=10, max_gates=15)
  lgn2.add_gate(GateType.AND, [0, 1, 2])  # 3-input AND
  parents2, actions2 = mdp.parent_transitions(lgn2, used_stop_action=False)

  assert len(parents2) == 1, "1-gate LGN should have 1 parent"
  assert actions2[0]['gate_type'] == GateType.AND
  assert actions2[0]['input_indices'] == (0, 1, 2), "3-input AND should have 3-element tuple"

  # Test Case 3: 5-input AND gate
  lgn3 = LGNState(num_inputs=10, max_gates=15)
  lgn3.add_gate(GateType.AND, [1, 2, 3, 4, 5])  # 5-input AND
  parents3, actions3 = mdp.parent_transitions(lgn3, used_stop_action=False)

  assert len(parents3) == 1, "1-gate LGN should have 1 parent"
  assert actions3[0]['gate_type'] == GateType.AND
  assert actions3[0]['input_indices'] == (1, 2, 3, 4, 5), "5-input AND should have 5-element tuple"


def test_parent_transitions_all_gate_types():
  """
  Test parent_transitions() with all 16 gate types.

  Validates that parent_transitions works correctly for:
  - Unary gates: NOT, BUFFER
  - Binary gates: OR, XOR, NAND, NOR, XNOR, IMPLY, etc.
  - Variable-arity gates: AND
  """
  mdp = LGNMDP(num_inputs=10, max_gates=20)

  # List of all gate types with their expected arity
  gate_test_cases = [
    # Unary gates
    (GateType.NOT, [0]),
    (GateType.BUFFER, [1]),
    # Binary gates
    (GateType.OR, [0, 1]),
    (GateType.XOR, [2, 3]),
    (GateType.NAND, [4, 5]),
    (GateType.NOR, [6, 7]),
    (GateType.XNOR, [8, 9]),
    (GateType.IMPLY, [0, 2]),
    (GateType.NIMPLY, [1, 3]),
    (GateType.CONVERSE_IMPLY, [2, 4]),
    (GateType.CONVERSE_NIMPLY, [3, 5]),
    (GateType.FIRST, [4, 6]),
    (GateType.SECOND, [5, 7]),
    (GateType.NFIRST, [6, 8]),
    (GateType.NSECOND, [7, 9]),
    # Variable-arity gate
    (GateType.AND, [0, 1, 2]),
  ]

  # Test each gate type
  for gate_type, inputs in gate_test_cases:
    lgn = LGNState(num_inputs=10, max_gates=20)
    lgn.add_gate(gate_type, inputs)

    # Get parent transitions
    parents, actions = mdp.parent_transitions(lgn, used_stop_action=False)

    # Validate
    assert len(parents) == 1, f"{gate_type.value} should have 1 parent"
    assert len(actions) == 1, f"{gate_type.value} should have 1 action"
    assert actions[0]['gate_type'] == gate_type, f"Gate type should match for {gate_type.value}"
    assert actions[0]['input_indices'] == tuple(inputs), f"Input indices should match for {gate_type.value}"


def test_parent_transitions_independence():
  """
  Test that parent states are independent copies.

  Validates:
  - Modifying a parent state doesn't affect other parents
  - Modifying a parent state doesn't affect the original state
  - Each parent is a true deep copy
  """
  # Setup: Create MDP and LGN with two gates
  mdp = LGNMDP(num_inputs=10, max_gates=15)
  lgn = LGNState(num_inputs=10, max_gates=15)

  lgn.add_gate(GateType.AND, [0, 1])
  lgn.add_gate(GateType.OR, [2, 3])

  # Execute: Get parent transitions
  parents, actions = mdp.parent_transitions(lgn, used_stop_action=False)

  # Validate: Should have 2 parents
  assert len(parents) == 2

  # Modify first parent
  parents[0].add_gate(GateType.XOR, [4, 5])

  # Validate: Original should be unchanged
  assert lgn.get_num_gates() == 2, "Original should still have 2 gates"

  # Validate: Second parent should be unchanged
  assert parents[1].get_num_gates() == 1, "Second parent should still have 1 gate"

  # Validate: First parent should be modified
  assert parents[0].get_num_gates() == 2, "First parent should have 2 gates after modification"


def test_parent_transitions_complex_dag():
  """
  Test parent_transitions() with complex DAG structure.

  Case: Network with gates referencing other gates' outputs
  Expected behavior:
  - Correctly handles gates with mixed inputs (features + gate outputs)
  - Parent states maintain DAG structure
  """
  # Setup: Create MDP and complex LGN
  mdp = LGNMDP(num_inputs=5, max_gates=10)
  lgn = LGNState(num_inputs=5, max_gates=10)

  # Build complex DAG:
  # Gate 0: AND(0, 1) -> output at index 5
  # Gate 1: OR(2, 3) -> output at index 6
  # Gate 2: XOR(5, 6) -> uses outputs of Gate 0 and Gate 1
  # Gate 3: NAND(4, 7) -> uses input 4 and output of Gate 2

  lgn.add_gate(GateType.AND, [0, 1])     # Gate 0 -> index 5
  lgn.add_gate(GateType.OR, [2, 3])      # Gate 1 -> index 6
  lgn.add_gate(GateType.XOR, [5, 6])     # Gate 2 -> index 7 (uses 0, 1)
  lgn.add_gate(GateType.NAND, [4, 7])    # Gate 3 -> index 8 (uses 2)

  # Execute: Get parent transitions
  parents, actions = mdp.parent_transitions(lgn, used_stop_action=False)

  # Validate: Should have 4 parents
  assert len(parents) == 4, "4-gate LGN should have 4 parents"
  assert len(actions) == 4, "4-gate LGN should have 4 actions"

  # Validate: Each parent has 3 gates
  for parent in parents:
    assert parent.get_num_gates() == 3, "Each parent should have 3 gates"

  # Validate: All actions are present
  expected_gate_types = [GateType.AND, GateType.OR, GateType.XOR, GateType.NAND]
  actual_gate_types = [action['gate_type'] for action in actions]

  for expected_type in expected_gate_types:
    assert expected_type in actual_gate_types, f"{expected_type.value} should be in actions"


def test_parent_transitions_max_gates_boundary():
  """
  Test parent_transitions() at max_gates boundary.

  Case: Network at max_gates limit
  Expected behavior:
  - Works correctly even when max_gates is reached
  - Parent states can still be generated
  """
  # Setup: Create MDP with low max_gates
  mdp = LGNMDP(num_inputs=10, max_gates=2)
  lgn = LGNState(num_inputs=10, max_gates=2)

  # Add gates up to max_gates
  lgn.add_gate(GateType.AND, [0, 1])
  lgn.add_gate(GateType.OR, [2, 3])

  # Execute: Get parent transitions
  parents, actions = mdp.parent_transitions(lgn, used_stop_action=False)

  # Validate: Should still work at max_gates
  assert len(parents) == 2, "Should have 2 parents even at max_gates"
  assert len(actions) == 2, "Should have 2 actions even at max_gates"

  # Validate: Each parent should have 1 gate
  for parent in parents:
    assert parent.get_num_gates() == 1, "Each parent should have 1 gate"
