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
from datetime import datetime

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

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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
    parser.add_argument('--iterations', type=int, default=10000, help='Training iterations (gradient steps, recommended: 10000+)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (recommended: 10-100)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use (auto=best available, mps=Apple Silicon GPU)')
    parser.add_argument('--node-emb-dim', type=int, default=128)
    parser.add_argument('--num-conv-steps', type=int, default=3)
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--data-samples', type=int, default=1000, help='Number of data samples to generate')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--no-cache', action='store_true', help='Force regeneration of data (ignore cache)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases experiment tracking')
    parser.add_argument('--wandb-project', type=str, default='lgn-gflownet', help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='Wandb run name (auto-generated if not specified)')
    parser.add_argument('--eval-every', type=int, default=50, help='Evaluate FN/FP rates every N iterations')
    parser.add_argument('--eval-samples', type=int, default=5, help='Number of LGNs to sample for evaluation')
    parser.add_argument('--max-and-arity', type=int, default=2,
                        help='Max inputs for AND gates (0=no limit, 2=fast, 4=balanced). Default: 2')

    args = parser.parse_args()

    # Wandb is enabled by default (unless --no-wandb is specified)
    use_wandb = not args.no_wandb and WANDB_AVAILABLE

    # Check wandb availability
    if not args.no_wandb and not WANDB_AVAILABLE:
        print("⚠️  Warning: wandb is not installed. Experiment tracking disabled.")
        print("   Install with: pip install wandb")
        print("   Or use --no-wandb to suppress this warning.\n")
        use_wandb = False

    print("=" * 80)
    print("Logic Gate Network GFlowNet - Training with Real Data")
    print("=" * 80)

    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # Configuration
    print(f"\nConfiguration:")
    print(f"  num_inputs: {args.num_inputs}")
    print(f"  max_gates: {args.max_gates}")
    print(f"  iterations: {args.iterations}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning_rate: {args.lr}")
    print(f"  data_samples: {args.data_samples}")
    print(f"  test_ratio: {args.test_ratio}")
    print(f"  device: {device}")
    print(f"  max_and_arity: {args.max_and_arity} {'(no limit)' if args.max_and_arity == 0 else ''}")

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.num_inputs}in_{args.max_gates}g_{args.iterations}it"

    # Initialize wandb if enabled
    if use_wandb:
        run_name = args.wandb_run_name or run_id
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
        print(f"\n✅ Wandb initialized: {wandb.run.name}")
        print(f"   Dashboard: {wandb.run.url}")
    else:
        print(f"\n⚠️  Wandb disabled (use without --no-wandb to enable)")

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
    ).to(device)
    print(f"  ✅ Policy network created (on {device})")

    mdp = LGNMDP(num_inputs=args.num_inputs, max_gates=args.max_gates)
    action_space = LGNActionSpace(
        num_inputs=args.num_inputs,
        max_gates=args.max_gates,
        max_and_arity=args.max_and_arity
    )
    print(f"  ✅ MDP and Action Space created (max_and_arity={args.max_and_arity})")

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
        device=device,
        balanced_loss=True,
        leaf_coef=10.0
    )
    print(f"  ✅ Trainer created")

    # Create evaluation function for FN/FP trend tracking
    def create_eval_fn(policy, mdp, action_space, test_real, test_fake, num_samples):
        """Create evaluation function closure for periodic FN/FP evaluation."""
        from lgn import LGNEvaluator
        evaluator = LGNEvaluator()

        def eval_fn():
            """Sample LGNs and compute FN/FP rates on test data."""
            fn_rates = []
            fp_rates = []
            real_rejection_rates = []
            fake_rejection_rates = []

            with torch.no_grad():
                for _ in range(num_samples):
                    # Sample one LGN using greedy policy (argmax)
                    lgn = LGNState(num_inputs=mdp.num_inputs, max_gates=mdp.max_gates)

                    while not lgn.is_terminal():
                        actions = action_space.get_valid_actions(lgn)
                        gate_actions = [a for a in actions if 'gate_type' in a]

                        action_q, stop_q = policy.forward_policy(lgn, gate_actions)

                        # Greedy: take argmax
                        if len(gate_actions) > 0:
                            all_q = torch.cat([action_q, stop_q.unsqueeze(0)])
                            all_actions = gate_actions + [{'action': 'stop'}]
                        else:
                            all_q = stop_q.unsqueeze(0)
                            all_actions = [{'action': 'stop'}]

                        best_idx = all_q.argmax().item()
                        best_action = all_actions[best_idx]

                        if 'action' in best_action and best_action['action'] == 'stop':
                            break
                        else:
                            lgn.add_gate(best_action['gate_type'], best_action['input_indices'])

                    # Evaluate on test data
                    real_outputs = evaluator.evaluate_batch(lgn, test_real)
                    fake_outputs = evaluator.evaluate_batch(lgn, test_fake)

                    # FN rate: Real samples incorrectly rejected (output 0)
                    fn_rate = (len(test_real) - real_outputs.count(1)) / len(test_real)
                    fn_rates.append(fn_rate)

                    # FP rate: Fake samples incorrectly accepted (output 1)
                    fp_rate = fake_outputs.count(1) / len(test_fake)
                    fp_rates.append(fp_rate)

                    # Rejection rates (for trend tracking)
                    real_rejection_rate = real_outputs.count(0) / len(test_real)
                    fake_rejection_rate = fake_outputs.count(0) / len(test_fake)
                    real_rejection_rates.append(real_rejection_rate)
                    fake_rejection_rates.append(fake_rejection_rate)

            return {
                'fn_rate': np.mean(fn_rates),
                'fp_rate': np.mean(fp_rates),
                'real_rejection_rate': np.mean(real_rejection_rates),
                'fake_rejection_rate': np.mean(fake_rejection_rates),
            }

        return eval_fn

    eval_fn = create_eval_fn(
        policy=policy,
        mdp=mdp,
        action_space=action_space,
        test_real=test_real,
        test_fake=test_fake,
        num_samples=args.eval_samples
    )
    print(f"  ✅ Evaluation function created (eval_every={args.eval_every}, samples={args.eval_samples})")

    # Step 5: Train!
    print(f"\n[5/5] Training...")
    print(f"  Running {args.iterations} iterations with batch_size={args.batch_size}")
    print()

    metrics = trainer.train(
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        log_every=args.print_every,
        verbose=True,
        wandb_log=use_wandb,  # Real-time wandb logging
        eval_fn=eval_fn,  # FN/FP trend tracking
        eval_every=args.eval_every
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

    # Log final summary to wandb (real-time logging already done during training)
    if use_wandb:
        wandb.log({
            "final/loss": metrics['loss'][-1],
            "final/terminal_loss": metrics['term_loss'][-1],
            "final/flow_loss": metrics['flow_loss'][-1],
            "final/mean_reward": metrics['mean_reward'][-1],
            "final/loss_improvement": metrics['loss'][0] - metrics['loss'][-1],
        })

    # Save model with timestamp
    models_dir = Path("experiments/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_filename = f"model_{run_id}.pt"
    model_path = models_dir / model_filename
    torch.save(policy.state_dict(), model_path)
    print(f"\n✅ Model saved to {model_path}")

    # Also save as 'latest' for convenience
    latest_path = Path("experiments/trained_model.pt")
    torch.save(policy.state_dict(), latest_path)
    print(f"✅ Latest model link: {latest_path}")

    # Save model to wandb
    if use_wandb:
        wandb.save(str(model_path))
        print(f"✅ Model saved to wandb")

    # Save convergence graphs
    print(f"\n📊 Generating convergence graphs...")
    import matplotlib.pyplot as plt

    graphs_dir = Path("experiments/graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    iterations_x = list(range(len(metrics['loss'])))

    # 1. Total Loss Curve
    ax1 = axes[0, 0]
    ax1.plot(iterations_x, metrics['loss'], 'b-', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve (Should Decrease)')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 2. Terminal Loss (TB Loss for terminal states)
    ax2 = axes[0, 1]
    ax2.plot(iterations_x, metrics['term_loss'], 'r-', linewidth=1, alpha=0.7, label='Terminal Loss')
    ax2.plot(iterations_x, metrics['flow_loss'], 'g-', linewidth=1, alpha=0.7, label='Flow Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Trajectory Balance Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # 3. Reward Curve
    ax3 = axes[1, 0]
    ax3.plot(iterations_x, metrics['mean_reward'], 'purple', linewidth=1, alpha=0.7)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Mean Reward')
    ax3.set_title('Reward Curve (Should Increase)')
    ax3.grid(True, alpha=0.3)

    # 4. Smoothed versions (moving average)
    ax4 = axes[1, 1]
    window = min(100, len(metrics['loss']) // 10) if len(metrics['loss']) > 10 else 1
    if window > 1:
        loss_smooth = np.convolve(metrics['loss'], np.ones(window)/window, mode='valid')
        reward_smooth = np.convolve(metrics['mean_reward'], np.ones(window)/window, mode='valid')
        smooth_x = list(range(len(loss_smooth)))
        ax4.plot(smooth_x, loss_smooth, 'b-', linewidth=2, label='Loss (smoothed)')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(smooth_x, reward_smooth, 'r-', linewidth=2, label='Reward (smoothed)')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Loss', color='blue')
        ax4_twin.set_ylabel('Reward', color='red')
        ax4.set_title(f'Smoothed Curves (window={window})')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Not enough data for smoothing', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Smoothed Curves')

    plt.tight_layout()
    convergence_path = graphs_dir / f"convergence_{run_id}.png"
    plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Convergence graphs saved to {convergence_path}")

    # Log convergence image to wandb
    if use_wandb:
        wandb.log({"convergence_graphs": wandb.Image(str(convergence_path))})

    # Finish wandb run
    if use_wandb:
        wandb.finish()

    return metrics, policy


if __name__ == "__main__":
    main()
