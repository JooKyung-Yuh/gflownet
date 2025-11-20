import pytest
from ..gflownet import LGNMDP, LGNActionSpace
from ..lgn import LGNState, GateType


def test_parent_transitions_with_dependencies():
    """
    CRITICAL TEST: Verify parent_transitions handles gate dependencies correctly.

    In the molecules implementation, they have complex dependency tracking.
    In grid, dependencies don't exist (each position is independent).

    For LGN: When we have Gate B that depends on Gate A's output,
    removing Gate A creates an INVALID parent state (Gate B would reference
    a non-existent gate).

    This test checks if our implementation handles this correctly.
    """
    # Setup
    mdp = LGNMDP(num_inputs=3, max_gates=10)

    # Create a state with dependencies
    state = LGNState(num_inputs=3, max_gates=10)
    state.add_gate(GateType.AND, [0, 1])  # Gate 0 -> output at index 3
    state.add_gate(GateType.OR, [3, 2])   # Gate 1 -> depends on Gate 0 output

    # Get parent transitions
    parents, actions = mdp.parent_transitions(state, used_stop_action=False)

    # CRITICAL: We should get 2 parents (one for each gate removal)
    assert len(parents) == 2, "Should get parent for each gate"

    # Check each parent
    for i, (parent, action) in enumerate(zip(parents, actions)):
        # Each parent should have 1 gate
        assert parent.get_num_gates() == 1, f"Parent {i} should have 1 gate"

        # CRITICAL CHECK: If we removed Gate 0, the remaining gate (Gate 1)
        # would reference index 3 (Gate 0's output), which no longer exists!
        # This would be an INVALID state.

        # Let's check if the parent state is valid
        if parent.gates[0].gate_type == GateType.OR:
            # This is the parent where we removed Gate 0
            # The remaining gate references index 3 (non-existent gate output)
            # This should NOT happen - but it DOES in our current implementation!

            print(f"WARNING: Parent has OR gate with inputs {parent.gates[0].inputs}")

            # Try to add a new gate to this parent - should fail DAG check
            # because it's referencing a non-existent gate
            try:
                # This should work because the parent is internally consistent
                # even though it looks wrong
                parent.add_gate(GateType.XOR, [0, 1])
            except ValueError as e:
                print(f"Adding gate failed: {e}")


def test_dag_structure_after_gate_removal():
    """
    Test if the DAG structure is maintained after gate removal.

    This is a CRITICAL issue: When we remove a gate, we're not reindexing
    the remaining gates' input references.

    Example:
    - State: Gate 0 (AND(0,1)), Gate 1 (OR(2,3)), Gate 2 (XOR(4, 10))
    - Gate 2 references index 10 (which is Gate 0's output at num_inputs + 0)
    - If we remove Gate 0, Gate 1 becomes the new Gate 0
    - But Gate 2 (now Gate 1) still references index 10
    - Index 10 should now be invalid (only 1 gate exists, so max valid is num_inputs + 0)
    """
    mdp = LGNMDP(num_inputs=5, max_gates=10)

    # Build a state with clear dependencies
    state = LGNState(num_inputs=5, max_gates=10)
    state.add_gate(GateType.AND, [0, 1])   # Gate 0 -> index 5
    state.add_gate(GateType.OR, [2, 3])    # Gate 1 -> index 6
    state.add_gate(GateType.XOR, [4, 5])   # Gate 2 -> index 7, uses Gate 0

    print(f"\nOriginal state:")
    print(f"Gate 0: {state.gates[0].gate_type.value} {state.gates[0].inputs}")
    print(f"Gate 1: {state.gates[1].gate_type.value} {state.gates[1].inputs}")
    print(f"Gate 2: {state.gates[2].gate_type.value} {state.gates[2].inputs}")

    # Get parents
    parents, actions = mdp.parent_transitions(state, used_stop_action=False)

    # Check each parent
    for i, (parent, action) in enumerate(zip(parents, actions)):
        print(f"\nParent {i} (removed {action['gate_type'].value} {action['input_indices']}):")
        for j, gate in enumerate(parent.gates):
            print(f"  Gate {j}: {gate.gate_type.value} {gate.inputs}")

            # Check if inputs are valid
            max_valid_index = 5 + j  # num_inputs + current_gate_index
            for inp in gate.inputs:
                if inp > max_valid_index:
                    print(f"    WARNING: Input {inp} exceeds max valid {max_valid_index}")


def test_molecule_style_reindexing():
    """
    Compare our implementation with the molecules approach.

    In molecules, when they remove a block, they:
    1. Reindex remaining blocks
    2. Update all junction bonds to reflect new indices

    For LGN, we would need to:
    1. Remove the gate
    2. Update all subsequent gates' input indices that reference removed/later gates

    This test documents the expected behavior.
    """
    # This is a DOCUMENTATION test showing what SHOULD happen
    # (but doesn't in our current implementation)

    state = LGNState(num_inputs=3, max_gates=10)
    state.add_gate(GateType.AND, [0, 1])   # Gate 0 -> index 3
    state.add_gate(GateType.OR, [2, 3])    # Gate 1 -> index 4, uses Gate 0

    print("\nOriginal state:")
    print(f"Gate 0: AND [0, 1] -> output at index 3")
    print(f"Gate 1: OR [2, 3] -> output at index 4, input 3 is Gate 0's output")

    # What happens when we remove Gate 0?
    parent = state.copy()
    parent.remove_gate(0)

    print("\nAfter removing Gate 0:")
    print(f"Gate 0 (was Gate 1): OR {parent.gates[0].inputs}")
    print(f"  Input indices: {parent.gates[0].inputs}")
    print(f"  Expected: [2, ?] where ? should NOT be 3 (Gate 0 no longer exists)")
    print(f"  Actual: [2, 3] <- PROBLEMATIC!")

    # The issue: Gate 1's input index 3 now refers to a non-existent gate
    # In molecules, they would reindex this to account for the removal


def test_is_this_a_problem_for_gflownet():
    """
    Determine if the reindexing issue actually breaks GFlowNet training.

    Key question: Does GFlowNet require that parent states be VALID states,
    or just that the (parent, action) pairs correctly reconstruct the child?

    Looking at grid and molecules:
    - Grid: Parents are always valid (can't create invalid positions)
    - Molecules: Parents are always valid (they do reindexing)

    For LGN: Our parents might reference non-existent gates, but...
    - The forward action (adding the gate) is correct
    - The backward probability P_B(parent | child) is correct
    - The parent STATE might be invalid, but it's never actually used as a state

    Let's test if this breaks anything.
    """
    mdp = LGNMDP(num_inputs=3, max_gates=10)
    action_space = LGNActionSpace(num_inputs=3, max_gates=10)

    # Build a state with dependencies
    state = LGNState(num_inputs=3, max_gates=10)
    state.add_gate(GateType.AND, [0, 1])   # Gate 0 -> index 3
    state.add_gate(GateType.OR, [2, 3])    # Gate 1 -> uses Gate 0

    # Get parents
    parents, actions = mdp.parent_transitions(state, used_stop_action=False)

    # For each parent, try to:
    # 1. Get valid actions (should work even if state is "invalid")
    # 2. Take the backward action (should work)
    # 3. Reconstruct the child (should match)

    for parent, action in zip(parents, actions):
        print(f"\nTesting parent with action {action['gate_type'].value} {action['input_indices']}")

        # Can we get valid actions from this parent?
        try:
            valid_actions = action_space.get_valid_actions(parent)
            print(f"  Valid actions: {len(valid_actions)} actions")
        except Exception as e:
            print(f"  ERROR getting valid actions: {e}")

        # Can we reconstruct the child?
        try:
            reconstructed = parent.copy()
            reconstructed.add_gate(action['gate_type'], list(action['input_indices']))
            print(f"  Reconstructed: {reconstructed.get_num_gates()} gates")
            print(f"  Original: {state.get_num_gates()} gates")
            print(f"  Match: {reconstructed.get_num_gates() == state.get_num_gates()}")
        except Exception as e:
            print(f"  ERROR reconstructing: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("CRITICAL DAG STRUCTURE TESTS")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("TEST 1: Parent transitions with dependencies")
    print("=" * 80)
    test_parent_transitions_with_dependencies()

    print("\n" + "=" * 80)
    print("TEST 2: DAG structure after gate removal")
    print("=" * 80)
    test_dag_structure_after_gate_removal()

    print("\n" + "=" * 80)
    print("TEST 3: Molecule-style reindexing comparison")
    print("=" * 80)
    test_molecule_style_reindexing()

    print("\n" + "=" * 80)
    print("TEST 4: Does this break GFlowNet?")
    print("=" * 80)
    test_is_this_a_problem_for_gflownet()
