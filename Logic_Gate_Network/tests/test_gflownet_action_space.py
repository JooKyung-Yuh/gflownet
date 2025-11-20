import pytest
from ..gflownet import LGNActionSpace
from ..lgn import LGNState, GateType


def test_action_space_initialization():
  """
  Test LGNActionSpace initialization with various configurations.

  Validates:
  - Correct storage of num_inputs and max_gates
  - No internal state storage (stateless design)
  """
  # Test default configuration
  action_space = LGNActionSpace(num_inputs=10, max_gates=15)
  assert action_space.num_inputs == 10
  assert action_space.max_gates == 15

  # Test custom configuration
  action_space_custom = LGNActionSpace(num_inputs=5, max_gates=20)
  assert action_space_custom.num_inputs == 5
  assert action_space_custom.max_gates == 20


def test_get_valid_actions_empty_lgn():
  """
  Test get_valid_actions() with empty LGN.

  Case: Empty network (no gates)
  Expected behavior:
  - Returns many gate addition actions
  - NO stop action (network is empty, cannot evaluate)
  - All actions reference only original input features
  """
  # Setup: Create action space and empty LGN
  action_space = LGNActionSpace(num_inputs=3, max_gates=10)
  lgn = LGNState(num_inputs=3, max_gates=10)

  # Execute: Get valid actions
  actions = action_space.get_valid_actions(lgn)

  # Validate: No stop action for empty network
  stop_actions = [a for a in actions if 'action' in a and a['action'] == 'stop']
  assert len(stop_actions) == 0, "Empty network should not have stop action"

  # Validate: All actions are gate additions
  gate_actions = [a for a in actions if 'gate_type' in a]
  assert len(gate_actions) > 0, "Empty network should have gate addition actions"

  # Validate: Expected action count for num_inputs=3:
  # - Unary gates (NOT, BUFFER): 2 types × 3 indices = 6 actions
  # - Binary gates (13 types): 13 types × C(3,2) = 13 × 3 = 39 actions
  # - AND gates (variable arity): C(3,1) + C(3,2) + C(3,3) = 3 + 3 + 1 = 7 actions
  # Total: 6 + 39 + 7 = 52 actions
  expected_total = 52
  assert len(actions) == expected_total, f"Empty LGN with 3 inputs should have {expected_total} actions"


def test_get_valid_actions_with_one_gate():
  """
  Test get_valid_actions() with one gate added.

  Case: Network with 1 gate
  Expected behavior:
  - Returns gate addition actions (more than empty case)
  - INCLUDES stop action (network is non-empty, can evaluate)
  - Actions can reference both inputs and gate output
  """
  # Setup: Create action space and LGN with one gate
  action_space = LGNActionSpace(num_inputs=3, max_gates=10)
  lgn = LGNState(num_inputs=3, max_gates=10)

  # Add one gate: AND(0, 1) -> output at index 3
  lgn.add_gate(GateType.AND, [0, 1])

  # Execute: Get valid actions
  actions = action_space.get_valid_actions(lgn)

  # Validate: Stop action should be present
  stop_actions = [a for a in actions if 'action' in a and a['action'] == 'stop']
  assert len(stop_actions) == 1, "Non-empty network should have exactly 1 stop action"

  # Validate: Gate actions should be present
  gate_actions = [a for a in actions if 'gate_type' in a]
  assert len(gate_actions) > 0, "Network should have gate addition actions"

  # Validate: Actions can reference gate output (index 3)
  # Check for at least one action using index 3
  actions_using_gate_output = [
    a for a in gate_actions
    if 3 in a['input_indices']
  ]
  assert len(actions_using_gate_output) > 0, "Should have actions using gate output (index 3)"

  # Validate: Expected action count for num_inputs=3, num_gates=1:
  # Available indices: [0, 1, 2, 3] (4 indices)
  # - Unary gates: 2 × 4 = 8 actions
  # - Binary gates: 13 × C(4,2) = 13 × 6 = 78 actions
  # - AND gates: C(4,1) + C(4,2) + C(4,3) + C(4,4) = 4 + 6 + 4 + 1 = 15 actions
  # - Stop action: 1 action
  # Total: 8 + 78 + 15 + 1 = 102 actions
  expected_total = 102
  assert len(actions) == expected_total, f"LGN with 3 inputs and 1 gate should have {expected_total} actions"


def test_get_valid_actions_max_gates_reached():
  """
  Test get_valid_actions() when max_gates is reached.

  Case: Network at max_gates limit
  Expected behavior:
  - Returns ONLY stop action
  - NO gate addition actions (max_gates reached)
  """
  # Setup: Create action space with low max_gates
  action_space = LGNActionSpace(num_inputs=10, max_gates=2)
  lgn = LGNState(num_inputs=10, max_gates=2)

  # Add gates up to max_gates
  lgn.add_gate(GateType.AND, [0, 1])
  lgn.add_gate(GateType.OR, [2, 3])

  # Execute: Get valid actions
  actions = action_space.get_valid_actions(lgn)

  # Validate: Only stop action should be present
  assert len(actions) == 1, "At max_gates, only stop action should be available"

  # Validate: The action is stop
  assert actions[0] == {'action': 'stop'}, "Action should be stop"

  # Validate: No gate addition actions
  gate_actions = [a for a in actions if 'gate_type' in a]
  assert len(gate_actions) == 0, "No gate addition actions at max_gates"


def test_get_valid_actions_dag_constraint():
  """
  Test get_valid_actions() respects DAG constraint.

  Case: Network with gates, check that actions only reference past nodes
  Expected behavior:
  - All generated actions reference only existing nodes
  - No actions reference future gates (would violate DAG)
  """
  # Setup: Create action space and LGN with two gates
  action_space = LGNActionSpace(num_inputs=3, max_gates=10)
  lgn = LGNState(num_inputs=3, max_gates=10)

  # Add two gates
  # Gate 0: AND(0, 1) -> output at index 3
  # Gate 1: OR(1, 2) -> output at index 4
  lgn.add_gate(GateType.AND, [0, 1])
  lgn.add_gate(GateType.OR, [1, 2])

  # Execute: Get valid actions
  actions = action_space.get_valid_actions(lgn)

  # Validate: All actions should only reference indices 0-4
  # (3 inputs + 2 gates = 5 nodes, indices 0-4)
  gate_actions = [a for a in actions if 'gate_type' in a]

  for action in gate_actions:
    input_indices = action['input_indices']
    for idx in input_indices:
      assert 0 <= idx <= 4, f"Index {idx} should be in range [0, 4]"

  # Validate: No action should reference index 5 or higher
  invalid_actions = [
    a for a in gate_actions
    if any(idx >= 5 for idx in a['input_indices'])
  ]
  assert len(invalid_actions) == 0, "No actions should reference future nodes"


def test_get_valid_actions_unary_gates():
  """
  Test that unary gates (NOT, BUFFER) are generated correctly.

  Expected behavior:
  - Each available index has 2 unary gate actions (NOT and BUFFER)
  - Input indices are single-element tuples
  """
  # Setup: Create action space with small num_inputs
  action_space = LGNActionSpace(num_inputs=2, max_gates=10)
  lgn = LGNState(num_inputs=2, max_gates=10)

  # Execute: Get valid actions
  actions = action_space.get_valid_actions(lgn)

  # Filter unary gate actions
  not_actions = [a for a in actions if 'gate_type' in a and a['gate_type'] == GateType.NOT]
  buffer_actions = [a for a in actions if 'gate_type' in a and a['gate_type'] == GateType.BUFFER]

  # Validate: Should have 2 NOT actions (one for each input)
  assert len(not_actions) == 2, "Should have 2 NOT actions"

  # Validate: Should have 2 BUFFER actions (one for each input)
  assert len(buffer_actions) == 2, "Should have 2 BUFFER actions"

  # Validate: All unary actions have single-element tuples
  for action in not_actions + buffer_actions:
    assert len(action['input_indices']) == 1, "Unary gates should have 1 input"
    assert isinstance(action['input_indices'], tuple), "input_indices should be tuple"


def test_get_valid_actions_binary_gates():
  """
  Test that binary gates (13 types) are generated correctly.

  Expected behavior:
  - Each 2-combination of indices has 13 binary gate actions
  - AND gate is NOT in binary gates (handled separately)
  - Input indices are 2-element tuples with idx1 < idx2
  """
  # Setup: Create action space with small num_inputs
  action_space = LGNActionSpace(num_inputs=3, max_gates=10)
  lgn = LGNState(num_inputs=3, max_gates=10)

  # Execute: Get valid actions
  actions = action_space.get_valid_actions(lgn)

  # Define binary gate types (excluding AND)
  binary_gate_types = [
    GateType.OR, GateType.XOR,
    GateType.NAND, GateType.NOR, GateType.XNOR,
    GateType.IMPLY, GateType.NIMPLY,
    GateType.CONVERSE_IMPLY, GateType.CONVERSE_NIMPLY,
    GateType.FIRST, GateType.SECOND,
    GateType.NFIRST, GateType.NSECOND
  ]

  # Validate: Each binary gate type should have C(3,2) = 3 actions
  for gate_type in binary_gate_types:
    gate_actions = [a for a in actions if 'gate_type' in a and a['gate_type'] == gate_type]
    assert len(gate_actions) == 3, f"{gate_type.value} should have 3 actions"

    # Validate: All have 2-element tuples
    for action in gate_actions:
      assert len(action['input_indices']) == 2, f"{gate_type.value} should have 2 inputs"

      # Validate: idx1 < idx2 (sorted)
      idx1, idx2 = action['input_indices']
      assert idx1 < idx2, f"Indices should be sorted: {action['input_indices']}"


def test_get_valid_actions_and_gate_variable_arity():
  """
  Test that AND gate with variable arity is generated correctly.

  Expected behavior:
  - AND gates generated for all arities from 1 to num_available_indices
  - Total AND actions = 2^n - 1 (where n = num_available_indices)
  """
  # Setup: Create action space with num_inputs=3
  action_space = LGNActionSpace(num_inputs=3, max_gates=10)
  lgn = LGNState(num_inputs=3, max_gates=10)

  # Execute: Get valid actions
  actions = action_space.get_valid_actions(lgn)

  # Filter AND gate actions
  and_actions = [a for a in actions if 'gate_type' in a and a['gate_type'] == GateType.AND]

  # Validate: Expected number of AND actions
  # With 3 available indices: C(3,1) + C(3,2) + C(3,3) = 3 + 3 + 1 = 7
  # Formula: 2^n - 1 = 2^3 - 1 = 7
  expected_and_count = 7
  assert len(and_actions) == expected_and_count, f"Should have {expected_and_count} AND actions"

  # Validate: AND actions have different arities
  arity_counts = {}
  for action in and_actions:
    arity = len(action['input_indices'])
    arity_counts[arity] = arity_counts.get(arity, 0) + 1

  # Expected arity distribution: {1: 3, 2: 3, 3: 1}
  assert arity_counts[1] == 3, "Should have 3 single-input AND gates"
  assert arity_counts[2] == 3, "Should have 3 two-input AND gates"
  assert arity_counts[3] == 1, "Should have 1 three-input AND gate"


def test_get_valid_actions_and_gate_with_gates():
  """
  Test AND gate variable arity with existing gates.

  Case: Network with 1 gate (4 available indices)
  Expected behavior:
  - AND actions for arities 1, 2, 3, 4
  - Total: 2^4 - 1 = 15 AND actions
  """
  # Setup: Create action space and add one gate
  action_space = LGNActionSpace(num_inputs=3, max_gates=10)
  lgn = LGNState(num_inputs=3, max_gates=10)

  # Add one gate: available indices become [0, 1, 2, 3]
  lgn.add_gate(GateType.OR, [0, 1])

  # Execute: Get valid actions
  actions = action_space.get_valid_actions(lgn)

  # Filter AND actions
  and_actions = [a for a in actions if 'gate_type' in a and a['gate_type'] == GateType.AND]

  # Validate: Expected number of AND actions
  # With 4 available indices: 2^4 - 1 = 15
  expected_and_count = 15
  assert len(and_actions) == expected_and_count, f"Should have {expected_and_count} AND actions"

  # Validate: Arity distribution
  arity_counts = {}
  for action in and_actions:
    arity = len(action['input_indices'])
    arity_counts[arity] = arity_counts.get(arity, 0) + 1

  # Expected: {1: 4, 2: 6, 3: 4, 4: 1} = C(4,1) + C(4,2) + C(4,3) + C(4,4)
  assert arity_counts[1] == 4, "Should have 4 single-input AND gates"
  assert arity_counts[2] == 6, "Should have 6 two-input AND gates"
  assert arity_counts[3] == 4, "Should have 4 three-input AND gates"
  assert arity_counts[4] == 1, "Should have 1 four-input AND gate"


def test_get_valid_actions_all_gate_types_present():
  """
  Test that all 16 gate types are represented in action space.

  Expected behavior:
  - All 16 gate types (NOT, BUFFER, OR, XOR, ... , AND) present
  - Each gate type has appropriate number of actions
  """
  # Setup: Create action space
  action_space = LGNActionSpace(num_inputs=5, max_gates=10)
  lgn = LGNState(num_inputs=5, max_gates=10)

  # Execute: Get valid actions
  actions = action_space.get_valid_actions(lgn)

  # Collect all gate types present in actions
  gate_types_present = set()
  for action in actions:
    if 'gate_type' in action:
      gate_types_present.add(action['gate_type'])

  # Define all 16 gate types
  all_gate_types = [
    GateType.AND, GateType.OR, GateType.XOR,
    GateType.NAND, GateType.NOR, GateType.XNOR,
    GateType.NOT, GateType.BUFFER,
    GateType.IMPLY, GateType.NIMPLY,
    GateType.CONVERSE_IMPLY, GateType.CONVERSE_NIMPLY,
    GateType.FIRST, GateType.SECOND,
    GateType.NFIRST, GateType.NSECOND
  ]

  # Validate: All gate types should be present
  assert len(gate_types_present) == 16, "All 16 gate types should be present"

  for gate_type in all_gate_types:
    assert gate_type in gate_types_present, f"{gate_type.value} should be present in actions"


def test_get_valid_actions_no_duplicates():
  """
  Test that get_valid_actions() doesn't generate duplicate actions.

  Expected behavior:
  - No two actions are identical
  - Each (gate_type, input_indices) pair is unique
  """
  # Setup: Create action space
  action_space = LGNActionSpace(num_inputs=4, max_gates=10)
  lgn = LGNState(num_inputs=4, max_gates=10)

  # Execute: Get valid actions
  actions = action_space.get_valid_actions(lgn)

  # Filter gate actions only (exclude stop action)
  gate_actions = [a for a in actions if 'gate_type' in a]

  # Create set of (gate_type, input_indices) tuples
  action_signatures = set()
  for action in gate_actions:
    signature = (action['gate_type'], action['input_indices'])
    assert signature not in action_signatures, f"Duplicate action found: {signature}"
    action_signatures.add(signature)

  # Validate: Number of unique actions equals total actions
  assert len(action_signatures) == len(gate_actions), "All actions should be unique"


def test_get_valid_actions_action_format():
  """
  Test that actions have correct format.

  Expected behavior:
  - Gate actions: {'gate_type': GateType, 'input_indices': tuple}
  - Stop actions: {'action': 'stop'}
  - input_indices is always a tuple (immutable)
  """
  # Setup: Create action space and LGN with one gate
  action_space = LGNActionSpace(num_inputs=3, max_gates=10)
  lgn = LGNState(num_inputs=3, max_gates=10)
  lgn.add_gate(GateType.AND, [0, 1])

  # Execute: Get valid actions
  actions = action_space.get_valid_actions(lgn)

  # Validate: Each action has correct format
  for action in actions:
    if 'action' in action:
      # Stop action format
      assert action == {'action': 'stop'}, "Stop action should have correct format"
    else:
      # Gate action format
      assert 'gate_type' in action, "Gate action should have 'gate_type'"
      assert 'input_indices' in action, "Gate action should have 'input_indices'"

      # Validate gate_type is GateType enum
      assert isinstance(action['gate_type'], GateType), "gate_type should be GateType enum"

      # Validate input_indices is tuple
      assert isinstance(action['input_indices'], tuple), "input_indices should be tuple"

      # Validate input_indices are integers
      for idx in action['input_indices']:
        assert isinstance(idx, int), f"Index {idx} should be integer"


def test_get_valid_actions_incremental_growth():
  """
  Test that action space grows as gates are added.

  Expected behavior:
  - More gates → more available indices → more actions
  - Action count should increase monotonically (except at max_gates)
  """
  # Setup: Create action space
  action_space = LGNActionSpace(num_inputs=3, max_gates=5)

  # Track action counts as we add gates
  action_counts = []

  # Start with empty LGN
  lgn = LGNState(num_inputs=3, max_gates=5)
  actions = action_space.get_valid_actions(lgn)
  action_counts.append(len(actions))

  # Add gates one by one and track action count
  for i in range(3):
    lgn.add_gate(GateType.OR, [0, 1])  # Add a simple gate
    actions = action_space.get_valid_actions(lgn)
    action_counts.append(len(actions))

  # Validate: Action count should increase with each gate
  for i in range(len(action_counts) - 1):
    assert action_counts[i] < action_counts[i+1], \
      f"Action count should increase: {action_counts[i]} < {action_counts[i+1]}"


def test_get_valid_actions_consistency_with_mdp():
  """
  Test that action format is consistent with LGNMDP.parent_transitions().

  Expected behavior:
  - Action dicts from get_valid_actions() match format from parent_transitions()
  - Both use {'gate_type': GateType, 'input_indices': tuple} format
  """
  # Setup: Create action space and MDP
  from ..gflownet import LGNMDP

  action_space = LGNActionSpace(num_inputs=3, max_gates=10)
  mdp = LGNMDP(num_inputs=3, max_gates=10)

  # Create LGN with one gate
  lgn = LGNState(num_inputs=3, max_gates=10)
  lgn.add_gate(GateType.AND, [0, 1])

  # Get actions from both sources
  forward_actions = action_space.get_valid_actions(lgn)
  _, backward_actions = mdp.parent_transitions(lgn, used_stop_action=False)

  # Filter gate actions only
  forward_gate_actions = [a for a in forward_actions if 'gate_type' in a]

  # Validate: Action format should be identical
  # Both should have 'gate_type' and 'input_indices'
  for action in forward_gate_actions:
    assert 'gate_type' in action, "Forward action should have gate_type"
    assert 'input_indices' in action, "Forward action should have input_indices"
    assert isinstance(action['input_indices'], tuple), "input_indices should be tuple"

  for action in backward_actions:
    assert 'gate_type' in action, "Backward action should have gate_type"
    assert 'input_indices' in action, "Backward action should have input_indices"
    assert isinstance(action['input_indices'], tuple), "input_indices should be tuple"


def test_get_valid_actions_edge_case_single_input():
  """
  Test action space with num_inputs=1 (minimal case).

  Expected behavior:
  - Works correctly with minimal configuration
  - Generates appropriate actions for single input
  """
  # Setup: Create action space with num_inputs=1
  action_space = LGNActionSpace(num_inputs=1, max_gates=5)
  lgn = LGNState(num_inputs=1, max_gates=5)

  # Execute: Get valid actions
  actions = action_space.get_valid_actions(lgn)

  # Validate: Should have some actions
  assert len(actions) > 0, "Single input should have actions"

  # Validate: Should have unary gates
  unary_actions = [a for a in actions if 'gate_type' in a and len(a['input_indices']) == 1]
  assert len(unary_actions) > 0, "Should have unary gate actions"

  # Validate: No binary gates (only 1 input available)
  binary_actions = [a for a in actions if 'gate_type' in a and len(a['input_indices']) == 2]
  assert len(binary_actions) == 0, "Should have no binary gates with 1 input"


def test_get_valid_actions_large_num_inputs():
  """
  Test action space with large num_inputs.

  Expected behavior:
  - Handles large action space correctly
  - Action count follows expected formula
  """
  # Setup: Create action space with larger num_inputs
  action_space = LGNActionSpace(num_inputs=10, max_gates=15)
  lgn = LGNState(num_inputs=10, max_gates=15)

  # Execute: Get valid actions
  actions = action_space.get_valid_actions(lgn)

  # Validate: Should have many actions
  # Expected:
  # - Unary: 2 × 10 = 20
  # - Binary: 13 × C(10,2) = 13 × 45 = 585
  # - AND: 2^10 - 1 = 1023
  # Total: 1628 actions
  expected_total = 1628
  assert len(actions) == expected_total, f"Should have {expected_total} actions"

  # Validate: No stop action (empty network)
  stop_actions = [a for a in actions if 'action' in a and a['action'] == 'stop']
  assert len(stop_actions) == 0, "Empty network should not have stop action"
