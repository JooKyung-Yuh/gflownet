"""
LGN GFlowNet Training with Real Data
====================================

Train GNN policy network with real/fake data from generator.

Usage:
    python train_with_real_data.py --iterations 100 --batch-size 8
"""

import torch
import argparse
import numpy as np
from pathlib import Path
import pickle
import hashlib

from lgn.network import LGNState
from lgn.gates import GateType
from gflownet.policy_network_gnn import LGNGNNPolicy
from gflownet.training import LGNTrainer
from gflownet.lgn_mdp import LGNMDP
from gflownet.action_space import LGNActionSpace
from reward.reward_fn import RewardFunction
from data.generator import RealDataGenerator, FakeDataGenerator
from data.dataset import LGNDataset
from rules.rule_1 import Rule1_NoConsecutive1s


def get_cache_path(rule_name: str, dimension: int, count: int) -> Path:
    """
    Generate cache file path based on data generation parameters.

    Args:
        rule_name: Name of the rule (e.g., 'Rule1_NoConsecutive1s')
        dimension: Input dimension
        count: Number of samples

    Returns:
        Path to cache file
    """
    cache_dir = Path("experiments/cached_data")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create unique filename based on parameters
    cache_key = f"{rule_name}_dim{dimension}_n{count}"
    cache_file = cache_dir / f"{cache_key}.pkl"

    return cache_file


def load_cached_data(cache_path: Path):
    """
    Load cached data from disk.

    Args:
        cache_path: Path to cache file

    Returns:
        Tuple of (real_samples, fake_samples) if cache exists, None otherwise
    """
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data['real_samples'], data['fake_samples']
    except Exception as e:
        print(f"  ⚠️  Failed to load cache: {e}")
        return None


def save_cached_data(cache_path: Path, real_samples, fake_samples):
    """
    Save generated data to disk cache.

    Args:
        cache_path: Path to cache file
        real_samples: Real data samples
        fake_samples: Fake data samples
    """
    try:
        data = {
            'real_samples': real_samples,
            'fake_samples': fake_samples
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  ✅ Data cached to {cache_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to save cache: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train LGN GFlowNet with real data')
    parser.add_argument('--num-inputs', type=int, default=10, help='Number of input features')
    parser.add_argument('--max-gates', type=int, default=15, help='Maximum number of gates')
    parser.add_argument('--iterations', type=int, default=100, help='Training iterations')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--node-emb-dim', type=int, default=128)
    parser.add_argument('--num-conv-steps', type=int, default=3)
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--data-samples', type=int, default=1000, help='Number of data samples to generate')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--no-cache', action='store_true', help='Force regeneration of data (ignore cache)')

    args = parser.parse_args()

    print("=" * 80)
    print("Logic Gate Network GFlowNet - Training with Real Data")
    print("=" * 80)

    # Configuration
    print(f"\nConfiguration:")
    print(f"  num_inputs: {args.num_inputs}")
    print(f"  max_gates: {args.max_gates}")
    print(f"  iterations: {args.iterations}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning_rate: {args.lr}")
    print(f"  data_samples: {args.data_samples}")
    print(f"  test_ratio: {args.test_ratio}")

    # Step 1: Load or generate training data
    print(f"\n[1/5] Loading/Generating training data...")
    rule = Rule1_NoConsecutive1s(dimension=args.num_inputs)

    # Check cache
    cache_path = get_cache_path(
        rule_name="Rule1_NoConsecutive1s",
        dimension=args.num_inputs,
        count=args.data_samples
    )

    real_samples = None
    fake_samples = None

    # Try to load from cache if --no-cache not specified
    if not args.no_cache:
        print(f"  Checking cache at {cache_path}...")
        cached_data = load_cached_data(cache_path)
        if cached_data is not None:
            real_samples, fake_samples = cached_data
            print(f"  ✅ Loaded {len(real_samples)} real samples from cache")
            print(f"  ✅ Loaded {len(fake_samples)} fake samples from cache")

    # Generate data if not cached or --no-cache specified
    if real_samples is None or fake_samples is None:
        if args.no_cache:
            print(f"  --no-cache specified, regenerating data...")
        else:
            print(f"  Cache not found, generating new data...")

        print(f"  Generating {args.data_samples} real samples...")
        real_generator = RealDataGenerator()
        real_samples = real_generator.generate(rule, count=args.data_samples)
        print(f"  ✅ Generated {len(real_samples)} real samples")

        print(f"  Generating {args.data_samples} fake samples...")
        fake_generator = FakeDataGenerator()
        fake_samples = fake_generator.generate(rule, real_samples, count=args.data_samples)
        print(f"  ✅ Generated {len(fake_samples)} fake samples")

        # Save to cache
        if not args.no_cache:
            save_cached_data(cache_path, real_samples, fake_samples)

    # Step 2: Split train/test
    print(f"\n[2/5] Splitting train/test...")
    dataset = LGNDataset(real_samples, fake_samples)
    dataset.split_train_test(test_ratio=args.test_ratio, random_seed=42)

    train_real, train_fake = dataset.get_train()
    test_real, test_fake = dataset.get_test()
    print(f"  Train: {len(train_real)} real, {len(train_fake)} fake")
    print(f"  Test:  {len(test_real)} real, {len(test_fake)} fake")

    # Step 3: Create GFlowNet components
    print(f"\n[3/5] Creating GFlowNet components...")

    policy = LGNGNNPolicy(
        num_inputs=args.num_inputs,
        max_gates=args.max_gates,
        node_emb_dim=args.node_emb_dim,
        num_conv_steps=args.num_conv_steps
    )
    print(f"  ✅ Policy network created")

    mdp = LGNMDP(num_inputs=args.num_inputs, max_gates=args.max_gates)
    action_space = LGNActionSpace(num_inputs=args.num_inputs, max_gates=args.max_gates)
    print(f"  ✅ MDP and Action Space created")

    # Create reward function connected to real data
    reward_fn = RewardFunction()

    def compute_reward(lgn: LGNState) -> float:
        """
        Compute reward based on real/fake classification accuracy.

        Note: RewardFunction returns log-reward (which can be negative),
        but GFlowNet requires non-negative rewards for TB loss computation.
        We exponentiate to get R = exp(log_reward) which is always positive.
        """
        log_reward = reward_fn.compute_reward(lgn, train_real, train_fake)
        import numpy as np
        return np.exp(log_reward)  # Convert log-reward to actual reward

    print(f"  ✅ Reward function created")

    # Step 4: Create trainer
    print(f"\n[4/5] Creating trainer...")
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    trainer = LGNTrainer(
        policy=policy,
        mdp=mdp,
        action_space=action_space,
        reward_fn=compute_reward,
        optimizer=optimizer,
        device=torch.device(args.device),
        balanced_loss=True,
        leaf_coef=10.0
    )
    print(f"  ✅ Trainer created")

    # Step 5: Train!
    print(f"\n[5/5] Training...")
    print(f"  Running {args.iterations} iterations with batch_size={args.batch_size}")
    print()

    metrics = trainer.train(
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        log_every=args.print_every,
        verbose=True
    )

    print("\n" + "=" * 80)
    print("Training Completed! ✅")
    print("=" * 80)

    # Print results
    print("\nFinal Metrics:")
    print(f"  Loss: {metrics['loss'][-1]:.4f}")
    print(f"  Terminal Loss: {metrics['term_loss'][-1]:.4f}")
    print(f"  Flow Loss: {metrics['flow_loss'][-1]:.4f}")
    print(f"  Mean Reward: {metrics['mean_reward'][-1]:.4f}")

    # Loss trend
    print(f"\nLoss Trend:")
    print(f"  Initial loss: {metrics['loss'][0]:.4f}")
    print(f"  Final loss: {metrics['loss'][-1]:.4f}")
    print(f"  Improvement: {metrics['loss'][0] - metrics['loss'][-1]:.4f}")

    print(f"\n✅ Training complete! Model ready for evaluation.")

    # Save model
    model_path = Path("experiments/trained_model.pt")
    model_path.parent.mkdir(exist_ok=True)
    torch.save(policy.state_dict(), model_path)
    print(f"\n✅ Model saved to {model_path}")

    return metrics, policy


if __name__ == "__main__":
    main()
