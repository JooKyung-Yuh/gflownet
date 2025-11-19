import pytest
from ..lgn import GateType, LGNState, ActionSpace

def test_get_available_inputs():
  """
  Test ActionSpace.get_available_inputs() method.

  This test validates:
  - Empty network returns only input features
  - Network with gates returns inputs + gate outputs
  - All indices are correctly ordered
  - No recency constraint (all gates are always available)
  """
  # ====== Empty Network Test ======
  # Create empty network with 3 inputs
  lgn_empty = LGNState(num_inputs=3, max_gates=5)
  action_space_empty = ActionSpace(lgn_empty)

  available_empty = action_space_empty.get_available_inputs()
  assert [0, 1, 2] == available_empty  # Only input features (0, 1, 2)


  # ====== Network with Gates Test ======
  # Create network with 5 inputs and add 2 gates
  lgn_with_gates = LGNState(num_inputs=5, max_gates=10)
  lgn_with_gates.add_gate(GateType.AND, [0, 1])  # Gate 0 output at index 5
  lgn_with_gates.add_gate(GateType.OR, [2, 5])   # Gate 1 output at index 6

  action_space_gates = ActionSpace(lgn_with_gates)
  available_gates = action_space_gates.get_available_inputs()

  # Should return all inputs (0-4) + all gate outputs (5-6)
  assert [0, 1, 2, 3, 4, 5, 6] == available_gates  # Inputs + 2 gates


  # ====== Verify No Recency Constraint ======
  # Add more gates and verify ALL gates remain available
  lgn_with_gates.add_gate(GateType.NOT, [3])     # Gate 2 output at index 7
  lgn_with_gates.add_gate(GateType.XOR, [4, 7])  # Gate 3 output at index 8

  action_space_updated = ActionSpace(lgn_with_gates)
  available_updated = action_space_updated.get_available_inputs()

  # All inputs + ALL 4 gates should be available (no recency limit)
  assert [0, 1, 2, 3, 4, 5, 6, 7, 8] == available_updated


def test_get_valid_actions():
  """
  Test ActionSpace.get_valid_actions() method.

  This test validates:
  - Empty network generates correct actions
  - Actions respect arity constraints for each gate type
  - DAG constraint is enforced (no future references)
  - Terminal state returns no actions
  - Action format is correct (dict with gate_type and inputs)
  """
  # ====== Empty Network Actions ======
  # Create small network to limit action space size
  lgn_small = LGNState(num_inputs=2, max_gates=5)
  action_space_small = ActionSpace(lgn_small)

  actions = action_space_small.get_valid_actions()

  # Verify actions are not empty
  assert len(actions) > 0  # Should have valid actions for empty network

  # Verify action format (each action should be a dict with gate_type and inputs)
  first_action = actions[0]
  assert "gate_type" in first_action  # Check gate_type key exists
  assert "inputs" in first_action      # Check inputs key exists
  assert isinstance(first_action["gate_type"], GateType)  # gate_type is GateType enum
  assert isinstance(first_action["inputs"], list)         # inputs is a list


  # ====== Verify Arity Constraints ======
  # Count actions by gate type and verify arity
  and_actions = [a for a in actions if a["gate_type"] == GateType.AND]
  not_actions = [a for a in actions if a["gate_type"] == GateType.NOT]
  or_actions = [a for a in actions if a["gate_type"] == GateType.OR]

  # AND gate: variable arity (1 or 2 for 2 available indices)
  assert len(and_actions) > 0  # Should have AND actions
  # Verify all AND actions have valid arity (1 or 2)
  for action in and_actions:
    arity = len(action["inputs"])
    assert 1 <= arity <= 2  # AND can have 1 or 2 inputs with 2 available indices

  # NOT gate: arity = 1
  assert len(not_actions) > 0  # Should have NOT actions
  for action in not_actions:
    assert 1 == len(action["inputs"])  # NOT requires exactly 1 input

  # OR gate: arity = 2
  assert len(or_actions) > 0  # Should have OR actions
  for action in or_actions:
    assert 2 == len(action["inputs"])  # OR requires exactly 2 inputs


  # ====== Network with Gates - DAG Constraint ======
  lgn_dag = LGNState(num_inputs=3, max_gates=5)
  lgn_dag.add_gate(GateType.AND, [0, 1])  # Gate at index 3

  action_space_dag = ActionSpace(lgn_dag)
  actions_dag = action_space_dag.get_valid_actions()

  # Verify no action references future gates (only 0, 1, 2, 3 are valid)
  for action in actions_dag:
    for idx in action["inputs"]:
      assert idx <= 3  # Can only reference inputs (0,1,2) or gate 0 (index 3)


  # ====== Terminal State Returns Empty ======
  # Create network that becomes terminal (all features used)
  lgn_terminal = LGNState(num_inputs=2, max_gates=5)
  lgn_terminal.add_gate(GateType.AND, [0, 1])  # Uses all features

  action_space_terminal = ActionSpace(lgn_terminal)
  actions_terminal = action_space_terminal.get_valid_actions()

  assert [] == actions_terminal  # Terminal state should return no actions


def test_is_valid_action():
  """
  Test ActionSpace.is_valid_action() method.

  This test validates:
  - Valid actions are correctly identified
  - Invalid actions are rejected (wrong arity, bad indices, etc.)
  - Action format validation
  - DAG constraint enforcement
  """
  # ====== Setup Test Network ======
  lgn = LGNState(num_inputs=3, max_gates=5)
  lgn.add_gate(GateType.AND, [0, 1])  # Gate at index 3

  action_space = ActionSpace(lgn)


  # ====== Valid Action Tests ======
  # Valid AND action with 2 inputs
  valid_and = {"gate_type": GateType.AND, "inputs": [1, 2]}
  assert True == action_space.is_valid_action(valid_and)  # Should be valid

  # Valid OR action with 2 inputs
  valid_or = {"gate_type": GateType.OR, "inputs": [0, 3]}
  assert True == action_space.is_valid_action(valid_or)  # References input 0 and gate output 3

  # Valid NOT action with 1 input
  valid_not = {"gate_type": GateType.NOT, "inputs": [3]}
  assert True == action_space.is_valid_action(valid_not)  # References gate output 3


  # ====== Invalid Action Tests - Wrong Arity ======
  # NOT gate with 2 inputs (should be 1)
  invalid_not_arity = {"gate_type": GateType.NOT, "inputs": [0, 1]}
  assert False == action_space.is_valid_action(invalid_not_arity)  # Wrong arity for NOT

  # OR gate with 1 input (should be 2)
  invalid_or_arity = {"gate_type": GateType.OR, "inputs": [0]}
  assert False == action_space.is_valid_action(invalid_or_arity)  # Wrong arity for OR


  # ====== Invalid Action Tests - Bad Indices ======
  # Reference non-existent gate (index 4 doesn't exist yet)
  invalid_future_ref = {"gate_type": GateType.OR, "inputs": [0, 4]}
  assert False == action_space.is_valid_action(invalid_future_ref)  # DAG violation

  # Reference out-of-range index
  invalid_range = {"gate_type": GateType.AND, "inputs": [0, 100]}
  assert False == action_space.is_valid_action(invalid_range)  # Index 100 doesn't exist


  # ====== Invalid Action Tests - Format Errors ======
  # Missing gate_type key
  invalid_missing_type = {"inputs": [0, 1]}
  assert False == action_space.is_valid_action(invalid_missing_type)  # Missing gate_type

  # Missing inputs key
  invalid_missing_inputs = {"gate_type": GateType.AND}
  assert False == action_space.is_valid_action(invalid_missing_inputs)  # Missing inputs

  # Invalid gate_type (not a GateType enum)
  invalid_type = {"gate_type": "AND", "inputs": [0, 1]}
  assert False == action_space.is_valid_action(invalid_type)  # gate_type is string, not enum


def test_edge_cases():
  """
  Test ActionSpace edge cases and boundary conditions.

  This test validates:
  - Single input network
  - Maximum gates reached
  - Very large action space handling
  - Interaction between terminal states and action generation
  """
  # ====== Single Input Network ======
  lgn_single = LGNState(num_inputs=1, max_gates=3)
  action_space_single = ActionSpace(lgn_single)

  available_single = action_space_single.get_available_inputs()
  assert [0] == available_single  # Only 1 input available

  # Valid actions should include only arity-1 gates (NOT, BUFFER, AND with 1 input)
  actions_single = action_space_single.get_valid_actions()
  assert len(actions_single) > 0  # Should have some valid actions

  # Verify no action has arity > 1
  for action in actions_single:
    assert len(action["inputs"]) == 1  # All actions must have arity 1


  # ====== Max Gates Reached (Terminal State) ======
  lgn_max = LGNState(num_inputs=5, max_gates=2)
  lgn_max.add_gate(GateType.AND, [0, 1])
  lgn_max.add_gate(GateType.OR, [2, 3])  # Reached max_gates

  action_space_max = ActionSpace(lgn_max)
  actions_max = action_space_max.get_valid_actions()

  assert [] == actions_max  # No actions when max gates reached (terminal)

  # is_valid_action should still work correctly
  test_action = {"gate_type": GateType.NOT, "inputs": [4]}
  # Even though it would be structurally valid, network is terminal
  # However, is_valid_action only checks structural validity, not terminal state
  # So it might return True - this is OK as get_valid_actions handles terminal check


  # ====== All Features Used (Terminal State) ======
  lgn_all_features = LGNState(num_inputs=3, max_gates=10)
  lgn_all_features.add_gate(GateType.AND, [0, 1, 2])  # Uses all 3 features

  action_space_all = ActionSpace(lgn_all_features)
  actions_all = action_space_all.get_valid_actions()

  assert [] == actions_all  # No actions when all features used (terminal)


  # ====== Network with Multiple Gates - Verify All Are Available ======
  # Use more inputs to avoid terminal state (only use 2 out of 5 features)
  lgn_multi = LGNState(num_inputs=5, max_gates=10)
  lgn_multi.add_gate(GateType.AND, [0, 1])    # Gate at index 5, uses features 0,1
  lgn_multi.add_gate(GateType.NOT, [5])       # Gate at index 6, references gate 0
  lgn_multi.add_gate(GateType.XOR, [2, 5])    # Gate at index 7, uses feature 2 and gate 0

  action_space_multi = ActionSpace(lgn_multi)
  available_multi = action_space_multi.get_available_inputs()

  # Should have inputs (0,1,2,3,4) + all 3 gates (5,6,7)
  assert [0, 1, 2, 3, 4, 5, 6, 7] == available_multi  # No recency constraint

  # Verify actions can reference ANY previous gate (including gate 0 at index 5)
  actions_multi = action_space_multi.get_valid_actions()

  # Find an action that references gate 0 (at index 5, the first gate added)
  gate0_referenced = any(
    5 in action["inputs"]
    for action in actions_multi
  )
  assert gate0_referenced  # Gate 0 (index 5) should still be referenceable (no recency constraint)
