"""Debug: Check action space size"""
import time
from lgn.network import LGNState
from gflownet.action_space import LGNActionSpace

print("Testing action space size...")

# Create initial state
lgn = LGNState(num_inputs=10, max_gates=15)
action_space = LGNActionSpace(num_inputs=10, max_gates=15)

print(f"\nInitial state: {len(lgn.gates)} gates, {lgn.num_inputs} inputs")
print("Getting valid actions...")
start = time.time()
actions = action_space.get_valid_actions(lgn)
end = time.time()

print(f"\n✅ Found {len(actions)} valid actions")
print(f"⏱️  Time taken: {end - start:.3f} seconds")

if len(actions) > 1000:
    print(f"\n⚠️  WARNING: Action space is very large ({len(actions):,} actions)")
    print("This may cause performance issues during training!")
