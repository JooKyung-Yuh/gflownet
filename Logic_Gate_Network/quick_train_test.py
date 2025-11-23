"""
Quick Training Test - Minimal parameters for demonstration
===========================================================
"""
import torch
from lgn.network import LGNState
from gflownet.policy_network_gnn import LGNGNNPolicy
from gflownet.training import LGNTrainer
from gflownet.lgn_mdp import LGNMDP
from gflownet.action_space import LGNActionSpace
from reward.reward_fn import RewardFunction
from data.generator import RealDataGenerator, FakeDataGenerator
from data.dataset import LGNDataset
from rules.rule_1 import Rule1_NoConsecutive1s

print("Quick Training Test with Minimal Parameters")
print("=" * 60)

# Minimal parameters
num_inputs = 4
max_gates = 3
iterations = 5
batch_size = 2

print(f"num_inputs={num_inputs}, max_gates={max_gates}")

# Check action space size
lgn = LGNState(num_inputs=num_inputs, max_gates=max_gates)
action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)
actions = action_space.get_valid_actions(lgn)
print(f"Action space: {len(actions)} actions ✅\n")

# Generate minimal data
rule = Rule1_NoConsecutive1s()
real_gen = RealDataGenerator()
fake_gen = FakeDataGenerator()
real_samples = real_gen.generate(rule, count=10)
fake_samples = fake_gen.generate(rule, real_samples, count=10)
dataset = LGNDataset(real_samples, fake_samples)
dataset.split_train_test(test_ratio=0.2, random_seed=42)
train_real, train_fake = dataset.get_train()

print(f"Data: {len(train_real)} real, {len(train_fake)} fake ✅\n")

# Create policy and trainer
policy = LGNGNNPolicy(num_inputs=num_inputs, max_gates=max_gates, node_emb_dim=32, num_conv_steps=1)
mdp = LGNMDP(num_inputs=num_inputs, max_gates=max_gates)
reward_fn = RewardFunction()

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
trainer = LGNTrainer(
    policy=policy,
    mdp=mdp,
    action_space=action_space,
    reward_fn=lambda lgn: reward_fn.compute_reward(lgn, train_real, train_fake),
    optimizer=optimizer,
    device=torch.device('cpu')
)

print("Training...")
metrics = trainer.train(num_iterations=iterations, batch_size=batch_size, log_every=1, verbose=True)

print("\n" + "=" * 60)
print(f"✅ SUCCESS! Training completed")
print(f"Final Loss: {metrics['loss'][-1]:.4f}")
print(f"Mean Reward: {metrics['mean_reward'][-1]:.4f}")
