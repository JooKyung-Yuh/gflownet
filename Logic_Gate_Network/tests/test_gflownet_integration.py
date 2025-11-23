import pytest
import random
from gflownet import LGNMDP, LGNActionSpace
from lgn import LGNState, GateType


def test_forward_backward_consistency_simple():
    """
    Test that forward and backward transitions are consistent.

    Verifies the fundamental GFlowNet property:
    If we can go from parent -> child via action A,
    then parent_transitions(child) should include (parent, A).
    """
    # Setup
    mdp = LGNMDP(num_inputs=5, max_gates=10)
    action_space = LGNActionSpace(num_inputs=5, max_gates=10)

    # Start with empty state
    parent = LGNState(num_inputs=5, max_gates=10)

    # Get valid actions from empty state
    valid_actions = action_space.get_valid_actions(parent)
    gate_actions = [a for a in valid_actions if 'gate_type' in a]

    # Take one forward action
    action = gate_actions[0]  # Take first valid action
    child = parent.copy()
    child.add_gate(action['gate_type'], list(action['input_indices']))

    # Get parent transitions from child
    parents, backward_actions = mdp.parent_transitions(child, used_stop_action=False)

    # Verify: The parent and action should be in the backward list
    assert len(parents) == 1, "Single-gate child should have exactly 1 parent"
    assert len(backward_actions) == 1, "Single-gate child should have exactly 1 action"

    # Check that the backward action matches the forward action
    assert backward_actions[0]['gate_type'] == action['gate_type']
    assert backward_actions[0]['input_indices'] == action['input_indices']

    # Check that the parent matches
    assert parents[0].get_num_gates() == 0, "Parent should be empty"


def test_forward_backward_consistency_complex():
    """
    Test forward-backward consistency with a complex trajectory.

    Build a network step by step, and at each step verify that
    parent_transitions() correctly identifies all possible parents.
    """
    # Setup
    mdp = LGNMDP(num_inputs=3, max_gates=10)
    action_space = LGNActionSpace(num_inputs=3, max_gates=10)

    # Build a trajectory
    trajectory = []
    current_state = LGNState(num_inputs=3, max_gates=10)

    # Action 1: Add AND(0, 1)
    action1 = {'gate_type': GateType.AND, 'input_indices': (0, 1)}
    state1 = current_state.copy()
    state1.add_gate(action1['gate_type'], list(action1['input_indices']))
    trajectory.append((current_state.copy(), action1, state1.copy()))
    current_state = state1

    # Action 2: Add OR(2, 3) - using gate 0 output
    action2 = {'gate_type': GateType.OR, 'input_indices': (2, 3)}
    state2 = current_state.copy()
    state2.add_gate(action2['gate_type'], list(action2['input_indices']))
    trajectory.append((current_state.copy(), action2, state2.copy()))
    current_state = state2

    # Verify each transition is reversible
    for parent, action, child in trajectory:
        parents, backward_actions = mdp.parent_transitions(child, used_stop_action=False)

        # Check that the action is in backward_actions
        action_found = False
        parent_found = False

        for bp, ba in zip(parents, backward_actions):
            if (ba['gate_type'] == action['gate_type'] and
                ba['input_indices'] == action['input_indices']):
                action_found = True
                # Check that the parent matches
                if bp.get_num_gates() == parent.get_num_gates():
                    parent_found = True

        assert action_found, f"Action {action} should be in backward actions"
        assert parent_found, f"Parent with {parent.get_num_gates()} gates should be found"


def test_all_parents_can_reach_child():
    """
    Test that all parent states returned by parent_transitions() can
    actually reach the child state via forward actions.

    This is the inverse of forward-backward consistency:
    For each (parent, action) pair from parent_transitions(child),
    verify that action is in get_valid_actions(parent).
    """
    # Setup
    mdp = LGNMDP(num_inputs=4, max_gates=10)
    action_space = LGNActionSpace(num_inputs=4, max_gates=10)

    # Create a child state with multiple gates
    child = LGNState(num_inputs=4, max_gates=10)
    child.add_gate(GateType.AND, [0, 1])
    child.add_gate(GateType.OR, [2, 3])
    child.add_gate(GateType.XOR, [1, 4])  # Uses gate 0 output

    # Get all parent states
    parents, backward_actions = mdp.parent_transitions(child, used_stop_action=False)

    # For each parent, verify the action is valid
    for parent, action in zip(parents, backward_actions):
        # Get valid actions from parent
        valid_actions = action_space.get_valid_actions(parent)
        gate_actions = [a for a in valid_actions if 'gate_type' in a]

        # Check that the action is in valid actions
        action_found = False
        for va in gate_actions:
            if (va['gate_type'] == action['gate_type'] and
                va['input_indices'] == action['input_indices']):
                action_found = True
                break

        assert action_found, f"Action {action} should be valid from parent state"

        # Verify that taking this action leads to a valid state
        reconstructed_child = parent.copy()
        reconstructed_child.add_gate(action['gate_type'], list(action['input_indices']))
        assert reconstructed_child.get_num_gates() == child.get_num_gates()


def test_trajectory_reversibility_random():
    """
    Test trajectory reversibility with random trajectories.

    Generate random trajectories, then verify we can walk backward
    from the final state to the initial state using parent_transitions().
    """
    random.seed(42)

    # Setup
    mdp = LGNMDP(num_inputs=5, max_gates=8)
    action_space = LGNActionSpace(num_inputs=5, max_gates=8)

    # Run multiple random trajectories
    for trial in range(10):
        # Generate random trajectory
        trajectory_length = random.randint(3, 7)
        current_state = LGNState(num_inputs=5, max_gates=8)
        trajectory_states = [current_state.copy()]

        for step in range(trajectory_length):
            # Get valid actions
            valid_actions = action_space.get_valid_actions(current_state)
            gate_actions = [a for a in valid_actions if 'gate_type' in a]

            if len(gate_actions) == 0:
                break

            # Take random action
            action = random.choice(gate_actions)
            current_state.add_gate(action['gate_type'], list(action['input_indices']))
            trajectory_states.append(current_state.copy())

        # Now walk backward
        final_state = trajectory_states[-1]
        current = final_state.copy()

        for expected_num_gates in range(len(current.gates) - 1, -1, -1):
            # Get parents
            parents, _ = mdp.parent_transitions(current, used_stop_action=False)

            # Should have at least one parent
            assert len(parents) > 0, f"State with {current.get_num_gates()} gates should have parents"

            # Check that at least one parent has expected_num_gates
            valid_parent_found = False
            for parent in parents:
                if parent.get_num_gates() == expected_num_gates:
                    current = parent.copy()
                    valid_parent_found = True
                    break

            assert valid_parent_found, f"Should find parent with {expected_num_gates} gates"

        # Should reach empty state
        assert current.get_num_gates() == 0, "Should reach empty state at the end"


def test_stop_action_consistency():
    """
    Test that stop action handling is consistent between forward and backward.

    If we take a stop action from a state, then parent_transitions with
    used_stop_action=True should return that state as the parent.
    """
    # Setup
    mdp = LGNMDP(num_inputs=3, max_gates=10)
    action_space = LGNActionSpace(num_inputs=3, max_gates=10)

    # Create a non-empty state
    state = LGNState(num_inputs=3, max_gates=10)
    state.add_gate(GateType.AND, [0, 1])

    # Get valid actions - should include stop
    valid_actions = action_space.get_valid_actions(state)
    stop_actions = [a for a in valid_actions if 'action' in a and a['action'] == 'stop']

    assert len(stop_actions) == 1, "Non-empty state should have stop action"

    # Simulate taking stop action (state remains the same, just marked as terminal)
    terminal_state = state.copy()

    # Get parent transitions with used_stop_action=True
    parents, actions = mdp.parent_transitions(terminal_state, used_stop_action=True)

    # Should return exactly one parent (the same state before stop)
    assert len(parents) == 1, "Terminal state should have exactly 1 parent"
    assert len(actions) == 1, "Terminal state should have exactly 1 action"

    # The action should be stop
    assert actions[0] == {'action': 'stop'}, "Action should be stop"

    # The parent should have the same number of gates
    assert parents[0].get_num_gates() == state.get_num_gates()


def test_action_space_completeness():
    """
    Test that the action space is complete: every gate that can be added
    (according to DAG constraints) should have a corresponding action.
    """
    # Setup
    action_space = LGNActionSpace(num_inputs=3, max_gates=10)

    # Test with empty state
    state = LGNState(num_inputs=3, max_gates=10)
    actions = action_space.get_valid_actions(state)

    # Count expected actions for empty state with 3 inputs:
    # - Unary gates (NOT, BUFFER): 2 types × 3 indices = 6
    # - Binary gates (13 types): 13 types × C(3,2) = 13 × 3 = 39
    # - AND gates: C(3,1) + C(3,2) + C(3,3) = 3 + 3 + 1 = 7
    # Total: 52
    expected_count = 52
    assert len(actions) == expected_count, f"Empty state should have {expected_count} actions"

    # Test after adding one gate
    state.add_gate(GateType.AND, [0, 1])
    actions = action_space.get_valid_actions(state)

    # Now we have 4 available indices [0, 1, 2, 3]
    # - Unary: 2 × 4 = 8
    # - Binary: 13 × C(4,2) = 13 × 6 = 78
    # - AND: 2^4 - 1 = 15
    # - Stop: 1
    # Total: 102
    expected_count = 102
    assert len(actions) == expected_count, f"State with 1 gate should have {expected_count} actions"


def test_no_invalid_forward_backward_cycles():
    """
    Test that we never generate invalid forward-backward cycles.

    Specifically, verify that:
    1. We can't add a gate and then immediately remove a different gate
       and get back to the same state.
    2. The parent-child relationship is deterministic based on gate content.
    """
    # Setup
    mdp = LGNMDP(num_inputs=3, max_gates=10)

    # Create a state with two gates
    state = LGNState(num_inputs=3, max_gates=10)
    state.add_gate(GateType.AND, [0, 1])
    state.add_gate(GateType.OR, [2, 3])

    # Get parent transitions
    parents, actions = mdp.parent_transitions(state, used_stop_action=False)

    # Should have exactly 2 parents (one for each gate removal)
    assert len(parents) == 2, "2-gate state should have 2 parents"

    # Each parent should have exactly 1 gate
    for parent in parents:
        assert parent.get_num_gates() == 1, "Each parent should have 1 gate"

    # The two parents should be different (removing different gates)
    parent0_gate = parents[0].gates[0]
    parent1_gate = parents[1].gates[0]

    # They should have different gate types or inputs
    different = (parent0_gate.gate_type != parent1_gate.gate_type or
                 parent0_gate.inputs != parent1_gate.inputs)

    assert different, "Two parents should be distinct states"


def test_batch_consistency():
    """
    Test that batch operations maintain consistency.

    This simulates what happens during GFlowNet training:
    - Generate multiple trajectories
    - For each state, compute parent transitions
    - Verify consistency across batch
    """
    # Setup
    mdp = LGNMDP(num_inputs=4, max_gates=10)
    action_space = LGNActionSpace(num_inputs=4, max_gates=10)

    # Generate a batch of states
    batch_size = 5
    states = []

    for i in range(batch_size):
        state = LGNState(num_inputs=4, max_gates=10)
        # Add 2-3 gates
        num_gates = 2 + (i % 2)
        state.add_gate(GateType.AND, [0, 1])
        state.add_gate(GateType.OR, [2, 3])
        if num_gates == 3:
            state.add_gate(GateType.XOR, [1, 4])
        states.append(state)

    # For each state, verify parent transitions
    for state in states:
        parents, actions = mdp.parent_transitions(state, used_stop_action=False)

        # Number of parents should equal number of gates
        assert len(parents) == state.get_num_gates()
        assert len(actions) == state.get_num_gates()

        # Each parent should have one fewer gate
        for parent in parents:
            assert parent.get_num_gates() == state.get_num_gates() - 1


def test_edge_case_max_gates():
    """
    Test edge case when max_gates is reached.

    Verify that:
    1. Action space only returns stop action
    2. Parent transitions still work correctly
    """
    # Setup with low max_gates
    mdp = LGNMDP(num_inputs=3, max_gates=2)
    action_space = LGNActionSpace(num_inputs=3, max_gates=2)

    # Build to max_gates
    state = LGNState(num_inputs=3, max_gates=2)
    state.add_gate(GateType.AND, [0, 1])
    state.add_gate(GateType.OR, [2, 3])

    # Get valid actions - should only be stop
    actions = action_space.get_valid_actions(state)
    assert len(actions) == 1, "At max_gates, should only have stop action"
    assert actions[0] == {'action': 'stop'}, "Action should be stop"

    # Parent transitions should still work
    parents, backward_actions = mdp.parent_transitions(state, used_stop_action=False)
    assert len(parents) == 2, "Should have 2 parents"

    # Each parent should have 1 gate
    for parent in parents:
        assert parent.get_num_gates() == 1


def test_deep_copy_isolation():
    """
    Test that copy operations truly isolate states.

    This is critical for parent_transitions() to work correctly.
    """
    # Setup
    mdp = LGNMDP(num_inputs=3, max_gates=10)

    # Create a state
    original = LGNState(num_inputs=3, max_gates=10)
    original.add_gate(GateType.AND, [0, 1])
    original.add_gate(GateType.OR, [2, 3])

    # Get parent transitions
    parents, _ = mdp.parent_transitions(original, used_stop_action=False)

    # Modify one parent
    parents[0].add_gate(GateType.XOR, [1, 2])

    # Original should be unchanged
    assert original.get_num_gates() == 2, "Original should remain unchanged"

    # Other parent should be unchanged
    assert parents[1].get_num_gates() == 1, "Other parent should remain unchanged"

    # Modified parent should have 2 gates
    assert parents[0].get_num_gates() == 2, "Modified parent should have 2 gates"
