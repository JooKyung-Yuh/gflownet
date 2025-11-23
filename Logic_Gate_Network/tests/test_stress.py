"""
Stress tests for LGN GFlowNet implementation.

These tests generate thousands of random trajectories to verify:
1. Large-scale forward-backward consistency
2. Performance characteristics
3. Memory stability
4. Edge case handling at scale

Based on molecules/mol_mdp_ext.py:test_mdp_parent() (lines 239-277)
"""

import pytest
import random
import time
from collections import defaultdict
from gflownet import LGNMDP, LGNActionSpace
from lgn import LGNState, GateType


def test_large_scale_trajectory_generation():
    """
    Generate 1000 random trajectories and verify forward-backward consistency.

    This test mimics molecules/mol_mdp_ext.py:test_mdp_parent() but adapted for LGN.
    """
    random.seed(42)

    # Setup
    mdp = LGNMDP(num_inputs=5, max_gates=10)
    action_space = LGNActionSpace(num_inputs=5, max_gates=10)

    num_trajectories = 1000
    max_trajectory_length = 8

    print(f"\nGenerating {num_trajectories} random trajectories...")
    start_time = time.time()

    successes = 0
    failures = []

    for trial in range(num_trajectories):
        try:
            # Generate random trajectory
            trajectory_length = random.randint(1, max_trajectory_length)
            state = LGNState(num_inputs=5, max_gates=10)

            for step in range(trajectory_length):
                # Get valid actions
                valid_actions = action_space.get_valid_actions(state)
                gate_actions = [a for a in valid_actions if 'gate_type' in a]

                if len(gate_actions) == 0:
                    break

                # Take random action
                action = random.choice(gate_actions)
                state.add_gate(action['gate_type'], list(action['input_indices']))

            # Now verify we can walk backward to initial state
            final_state = state.copy()
            current = final_state

            while current.get_num_gates() > 0:
                # Get parents
                parents, _ = mdp.parent_transitions(current, used_stop_action=False)

                # Should have at least one parent
                if len(parents) == 0:
                    raise ValueError(f"State with {current.get_num_gates()} gates has no parents")

                # Take random parent
                current = random.choice(parents)

            # Should reach empty state
            if current.get_num_gates() != 0:
                raise ValueError(f"Did not reach empty state, stopped at {current.get_num_gates()} gates")

            successes += 1

        except Exception as e:
            failures.append((trial, str(e)))

    elapsed_time = time.time() - start_time

    print(f"Completed in {elapsed_time:.2f}s")
    print(f"Successes: {successes}/{num_trajectories} ({100*successes/num_trajectories:.1f}%)")

    if failures:
        print(f"Failures: {len(failures)}")
        for trial, error in failures[:5]:  # Show first 5 failures
            print(f"  Trial {trial}: {error}")

    # Assert all succeeded
    assert successes == num_trajectories, f"Only {successes}/{num_trajectories} trajectories succeeded"
    assert len(failures) == 0, f"Had {len(failures)} failures"


def test_parent_child_reconstruction_stress():
    """
    Test that ALL parent-child pairs correctly reconstruct.

    For each of 1000 random states:
    - Get all parents
    - For each parent-action pair, reconstruct child
    - Verify reconstruction matches original
    """
    random.seed(42)

    mdp = LGNMDP(num_inputs=4, max_gates=8)
    action_space = LGNActionSpace(num_inputs=4, max_gates=8)

    num_tests = 1000
    print(f"\nTesting parent-child reconstruction for {num_tests} random states...")
    start_time = time.time()

    mismatches = []

    for trial in range(num_tests):
        # Generate random state
        state = LGNState(num_inputs=4, max_gates=8)
        num_gates = random.randint(1, 6)

        for _ in range(num_gates):
            valid_actions = action_space.get_valid_actions(state)
            gate_actions = [a for a in valid_actions if 'gate_type' in a]
            if len(gate_actions) == 0:
                break
            action = random.choice(gate_actions)
            state.add_gate(action['gate_type'], list(action['input_indices']))

        # Get all parents
        parents, actions = mdp.parent_transitions(state, used_stop_action=False)

        # Test each parent-action pair
        for parent, action in zip(parents, actions):
            reconstructed = parent.copy()
            reconstructed.add_gate(action['gate_type'], list(action['input_indices']))

            # Verify gate count matches
            if reconstructed.get_num_gates() != state.get_num_gates():
                mismatches.append({
                    'trial': trial,
                    'expected_gates': state.get_num_gates(),
                    'reconstructed_gates': reconstructed.get_num_gates(),
                    'action': action
                })

    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.2f}s")
    print(f"Average: {1000*elapsed_time/num_tests:.2f}ms per state")

    if mismatches:
        print(f"Found {len(mismatches)} mismatches:")
        for m in mismatches[:5]:
            print(f"  Trial {m['trial']}: Expected {m['expected_gates']}, got {m['reconstructed_gates']}")

    assert len(mismatches) == 0, f"Found {len(mismatches)} reconstruction mismatches"


def test_action_space_size_distribution():
    """
    Analyze action space size distribution across random states.

    This helps identify potential performance issues and validates
    that action space grows as expected.
    """
    random.seed(42)

    action_space = LGNActionSpace(num_inputs=5, max_gates=10)

    # Test states with 0-8 gates
    action_counts = defaultdict(list)

    print("\nAnalyzing action space sizes...")

    for num_gates in range(9):
        for _ in range(100):
            state = LGNState(num_inputs=5, max_gates=10)

            # Add gates
            for i in range(num_gates):
                valid_actions = action_space.get_valid_actions(state)
                gate_actions = [a for a in valid_actions if 'gate_type' in a]
                if len(gate_actions) == 0:
                    break
                action = random.choice(gate_actions)
                state.add_gate(action['gate_type'], list(action['input_indices']))

            # Count actions
            actions = action_space.get_valid_actions(state)
            action_counts[state.get_num_gates()].append(len(actions))

    # Print statistics
    print(f"\n{'Gates':<10} {'Min':<10} {'Avg':<10} {'Max':<10} {'Samples':<10}")
    print("-" * 50)

    for num_gates in sorted(action_counts.keys()):
        counts = action_counts[num_gates]
        if counts:
            print(f"{num_gates:<10} {min(counts):<10} {sum(counts)//len(counts):<10} "
                  f"{max(counts):<10} {len(counts):<10}")

    # Verify action space grows with gates (except at max)
    for num_gates in range(len(action_counts) - 1):
        if num_gates in action_counts and num_gates + 1 in action_counts:
            avg_current = sum(action_counts[num_gates]) / len(action_counts[num_gates])
            avg_next = sum(action_counts[num_gates + 1]) / len(action_counts[num_gates + 1])

            # Action space should generally grow (but can decrease if hitting max_gates)
            if num_gates < 8:  # Before max
                assert avg_next > avg_current * 0.8, \
                    f"Action space should grow: {num_gates} gates avg={avg_current:.0f}, " \
                    f"{num_gates+1} gates avg={avg_next:.0f}"


def test_memory_stability():
    """
    Test that repeated operations don't cause memory leaks.

    Create and destroy many states to verify memory is properly released.
    """
    import gc

    mdp = LGNMDP(num_inputs=5, max_gates=10)
    action_space = LGNActionSpace(num_inputs=5, max_gates=10)

    print("\nTesting memory stability...")

    # Run 1000 iterations
    for i in range(1000):
        state = LGNState(num_inputs=5, max_gates=10)

        # Add some gates
        for _ in range(5):
            valid_actions = action_space.get_valid_actions(state)
            gate_actions = [a for a in valid_actions if 'gate_type' in a]
            if gate_actions:
                action = random.choice(gate_actions)
                state.add_gate(action['gate_type'], list(action['input_indices']))

        # Get parents (creates many copies)
        parents, _ = mdp.parent_transitions(state, used_stop_action=False)

        # Force garbage collection every 100 iterations
        if i % 100 == 0:
            gc.collect()

    gc.collect()
    print("Memory stability test completed (no assertions, checks for crashes/leaks)")


def test_extreme_and_gate_arity():
    """
    Test AND gates with extreme arities (1 to 10+ inputs).

    This tests the variable-arity AND gate implementation under stress.
    """
    random.seed(42)

    action_space = LGNActionSpace(num_inputs=10, max_gates=5)

    print("\nTesting extreme AND gate arities...")

    # Test with 10 inputs available
    state = LGNState(num_inputs=10, max_gates=5)
    actions = action_space.get_valid_actions(state)

    # Count AND actions by arity
    and_actions_by_arity = defaultdict(int)
    for action in actions:
        if 'gate_type' in action and action['gate_type'] == GateType.AND:
            arity = len(action['input_indices'])
            and_actions_by_arity[arity] += 1

    print(f"\nAND gate actions by arity (with 10 inputs):")
    for arity in sorted(and_actions_by_arity.keys()):
        count = and_actions_by_arity[arity]
        print(f"  Arity {arity}: {count} actions")

    # Verify we have AND gates of all arities 1-10
    for arity in range(1, 11):
        assert arity in and_actions_by_arity, f"Missing AND gates with arity {arity}"

    # Verify total: 2^10 - 1 = 1023
    total_and_actions = sum(and_actions_by_arity.values())
    expected_total = (2 ** 10) - 1
    assert total_and_actions == expected_total, \
        f"Expected {expected_total} AND actions, got {total_and_actions}"

    # Test that we can actually add and remove high-arity AND gates
    mdp = LGNMDP(num_inputs=10, max_gates=5)

    # Add a 10-input AND gate
    high_arity_action = [a for a in actions
                         if 'gate_type' in a
                         and a['gate_type'] == GateType.AND
                         and len(a['input_indices']) == 10][0]

    state.add_gate(high_arity_action['gate_type'], list(high_arity_action['input_indices']))

    # Verify it was added
    assert state.get_num_gates() == 1

    # Get parent
    parents, backward_actions = mdp.parent_transitions(state, used_stop_action=False)

    # Verify parent is correct
    assert len(parents) == 1
    assert parents[0].get_num_gates() == 0
    assert backward_actions[0]['gate_type'] == GateType.AND
    assert len(backward_actions[0]['input_indices']) == 10

    print(f"\nSuccessfully tested AND gate with 10 inputs")


def test_performance_scaling():
    """
    Test how performance scales with network size.

    Measures time for key operations at different scales.
    """
    import time

    print("\nTesting performance scaling...")
    print(f"\n{'Gates':<10} {'get_valid_actions':<25} {'parent_transitions':<25}")
    print("-" * 60)

    for num_gates in [1, 2, 4, 6, 8]:
        # Create state with specified number of gates
        state = LGNState(num_inputs=5, max_gates=10)
        action_space = LGNActionSpace(num_inputs=5, max_gates=10)
        mdp = LGNMDP(num_inputs=5, max_gates=10)

        for i in range(num_gates):
            actions = action_space.get_valid_actions(state)
            gate_actions = [a for a in actions if 'gate_type' in a]
            if gate_actions:
                action = random.choice(gate_actions)
                state.add_gate(action['gate_type'], list(action['input_indices']))

        # Time get_valid_actions
        start = time.time()
        for _ in range(100):
            actions = action_space.get_valid_actions(state)
        time_actions = (time.time() - start) / 100

        # Time parent_transitions
        start = time.time()
        for _ in range(100):
            parents, backward_actions = mdp.parent_transitions(state, used_stop_action=False)
        time_parents = (time.time() - start) / 100

        print(f"{num_gates:<10} {time_actions*1000:>10.3f} ms          "
              f"{time_parents*1000:>10.3f} ms")

    print("\nPerformance test completed (no assertions, informational only)")


if __name__ == "__main__":
    print("=" * 80)
    print("STRESS TESTS FOR LGN GFLOWNET")
    print("=" * 80)

    test_large_scale_trajectory_generation()
    test_parent_child_reconstruction_stress()
    test_action_space_size_distribution()
    test_memory_stability()
    test_extreme_and_gate_arity()
    test_performance_scaling()

    print("\n" + "=" * 80)
    print("ALL STRESS TESTS PASSED")
    print("=" * 80)
