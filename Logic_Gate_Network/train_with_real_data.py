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
from data.generator import RealDataGenerator, FakeDataGenerator, save_to_csv, save_to_json
from data.dataset import LGNDataset
from rules.rule_1 import Rule1_NoConsecutive1s
from visualize_lgn import visualize_lgn

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
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for policy (TB paper: 5e-4)')
    parser.add_argument('--lr-logz', type=float, default=5e-3, help='Learning rate for logZ (TB paper: 5e-3)')
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
    parser.add_argument('--eval-every', type=int, default=10, help='Evaluate FN/FP rates every N iterations')
    parser.add_argument('--eval-samples', type=int, default=5, help='Number of LGNs to sample for evaluation')
    parser.add_argument('--max-and-arity', type=int, default=2,
                        help='Max inputs for AND gates (0=no limit, 2=fast, 4=balanced). Default: 2')
    parser.add_argument('--visualize-every', type=int, default=100,
                        help='Visualize best LGN circuit every N iterations (0 to disable). Default: 100')
    parser.add_argument('--clip-grad', type=float, default=10.0,
                        help='Gradient clipping value (0 to disable). Default: 10.0')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for Boltzmann sampling (higher=more exploration). Default: 1.0')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Print accuracy every N steps (0 to disable). Default: 100')
    parser.add_argument('--profile', type=int, default=0,
                        help='Enable PyTorch profiler for N iterations (0 to disable). Outputs to profile_trace.json')
    parser.add_argument('--profile-detailed', action='store_true',
                        help='Include stack traces in profiler output (slower but more detailed)')

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

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.num_inputs}in_{args.max_gates}g_{args.iterations}it"

    # Initialize wandb if enabled (BEFORE printing config so it shows in wandb)
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
        run_name = run_id
        print(f"\n⚠️  Wandb disabled (use without --no-wandb to enable)")

    # Configuration (printed AFTER wandb init so it appears in wandb logs)
    print(f"\nConfiguration:")
    print(f"  num_inputs: {args.num_inputs}")
    print(f"  max_gates: {args.max_gates}")
    print(f"  iterations: {args.iterations}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning_rate (policy): {args.lr}")
    print(f"  learning_rate (logZ): {args.lr_logz}")
    print(f"  data_samples: {args.data_samples}")
    print(f"  test_ratio: {args.test_ratio}")
    print(f"  device: {device}")
    print(f"  max_and_arity: {args.max_and_arity} {'(no limit)' if args.max_and_arity == 0 else ''}")
    print(f"  visualize_every: {args.visualize_every} {'(disabled)' if args.visualize_every == 0 else ''}")
    print(f"  clip_grad: {args.clip_grad} {'(disabled)' if args.clip_grad == 0 else ''}")
    print(f"  temperature: {args.temperature}")
    print(f"  log_interval: {args.log_interval} {'(disabled)' if args.log_interval == 0 else ''}")

    # Save experiment parameters to file (ALL parser arguments)
    params_dir = Path("experiments/params")
    params_dir.mkdir(parents=True, exist_ok=True)

    params_filename = f"params_{run_name}.md"
    params_path = params_dir / params_filename

    # Get all arguments as dictionary
    all_args = vars(args)

    with open(params_path, 'w') as f:
        f.write(f"# Experiment Parameters\n\n")
        f.write(f"**Run Name:** {run_name}\n")
        f.write(f"**Timestamp:** {timestamp}\n")
        f.write(f"**Device (resolved):** {device}\n\n")
        f.write(f"## All Parameters\n\n")
        f.write(f"| Parameter | Value |\n")
        f.write(f"|-----------|-------|\n")
        for key, value in sorted(all_args.items()):
            f.write(f"| {key} | {value} |\n")

    # Also save as JSON for programmatic access
    import json
    params_json_path = params_dir / f"params_{run_name}.json"
    with open(params_json_path, 'w') as f:
        save_dict = {
            'run_name': run_name,
            'timestamp': timestamp,
            'device_resolved': str(device),
            **all_args
        }
        json.dump(save_dict, f, indent=2)

    print(f"\n✅ Parameters saved to {params_path}")
    print(f"✅ Parameters saved to {params_json_path}")

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

        # Save to CSV and JSON for inspection
        save_to_csv(real_samples, "real")
        save_to_csv(fake_samples, "fake")
        save_to_json(real_samples, rule, "real")
        save_to_json(fake_samples, rule, "fake")
        print(f"  ✅ Saved data to data/csv/ and data/json/")

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
    # C=3.0: Weight real errors more heavily to prevent "reject everything" strategy
    reward_fn = RewardFunction(C=3.0)

    def compute_log_reward(lgn: LGNState) -> float:
        """
        Compute log-reward based on real/fake classification accuracy.

        Returns log R(F) directly as defined in notion.md Eq. (2):
            log R(F) = -C·∑(1-F(X⁽ⁱ⁾)) - log(∑F(X⁽⁻ʲ⁾) + ε) + Ω(F)

        Note: Returns log-reward directly (can be negative).
        This is used in TB Loss: Loss = (logZ + log P_F(τ) - log R(x))²
        """
        return reward_fn.compute_reward(lgn, train_real, train_fake)

    def compute_reward_details(lgn: LGNState) -> dict:
        """Compute reward with detailed breakdown for wandb logging."""
        return reward_fn.compute_reward(lgn, train_real, train_fake, return_details=True)

    print(f"  ✅ Reward function created")

    # Step 4: Create trainer
    print(f"\n[4/5] Creating trainer...")

    # Create trainer first (to get logZ parameter)
    # Note: optimizer will be set after trainer creation
    trainer = LGNTrainer(
        policy=policy,
        mdp=mdp,
        action_space=action_space,
        log_reward_fn=compute_log_reward,
        reward_fn_details=compute_reward_details,  # For wandb logging
        optimizer=None,  # Will be set below
        device=device,
        init_logZ=0.0,  # Start with Z=1
        clip_grad=args.clip_grad,
        temperature=args.temperature,
    )

    # Create optimizer with separate learning rates for policy and logZ
    # TB Paper recommends: policy lr = 5e-4, logZ lr = 5e-3 (10x higher)
    optimizer = torch.optim.Adam([
        {'params': policy.parameters(), 'lr': args.lr},
        {'params': [trainer.logZ], 'lr': args.lr_logz}
    ])
    trainer.optimizer = optimizer
    print(f"  ✅ Trainer created (TB Loss with logZ parameter)")
    print(f"     Policy lr: {args.lr}, logZ lr: {args.lr_logz}")

    # Create evaluation function for accuracy tracking
    def create_eval_fn(trainer, test_real, test_fake, num_samples):
        """Create evaluation function closure for periodic accuracy evaluation."""
        from lgn import LGNEvaluator
        evaluator = LGNEvaluator()

        def eval_fn():
            """Sample LGNs using trainer.sample_greedy_batch and compute accuracy on test data."""
            real_accs = []
            fake_accs = []

            # Use trainer's sample_greedy_batch method (avoids code duplication)
            lgns = trainer.sample_greedy_batch(num_samples)

            # Evaluate all sampled LGNs
            for lgn in lgns:
                # Use evaluate_final_batch (same as reward_fn) for consistency
                real_outputs = evaluator.evaluate_final_batch(lgn, test_real)
                fake_outputs = evaluator.evaluate_final_batch(lgn, test_fake)

                # real_acc: Real samples correctly accepted (output 1)
                real_acc = real_outputs.count(1) / len(test_real)
                real_accs.append(real_acc)

                # fake_acc: Fake samples correctly rejected (output 0)
                fake_acc = fake_outputs.count(0) / len(test_fake)
                fake_accs.append(fake_acc)

            return {
                'real_acc': np.mean(real_accs),
                'fake_acc': np.mean(fake_accs),
            }

        return eval_fn

    eval_fn = create_eval_fn(
        trainer=trainer,
        test_real=test_real,
        test_fake=test_fake,
        num_samples=args.eval_samples,
    )
    print(f"  ✅ Evaluation function created (eval_every={args.eval_every}, samples={args.eval_samples})")

    # Step 5: Train!
    print(f"\n[5/5] Training...")
    print(f"  Running {args.iterations} iterations with batch_size={args.batch_size}")
    if args.profile > 0:
        print(f"  📊 PyTorch Profiler enabled for {args.profile} iterations")
    print()

    # Create visualization wrapper function
    def visualize_fn(lgn, title):
        """Wrapper for visualize_lgn to match trainer's expected signature."""
        return visualize_lgn(lgn, save_path=None, title=title, return_fig_only=True)

    # Run training with optional profiling
    if args.profile > 0:
        # Use PyTorch profiler for detailed GPU/CPU timing
        from torch.profiler import profile, record_function, ProfilerActivity

        profile_dir = Path("experiments/profiles")
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profile_dir / f"profile_{run_name}"

        activities = [ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)

        print(f"  Profiling activities: {[a.name for a in activities]}")
        print(f"  Profile will be saved to: {profile_path}_trace.json")

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=args.profile_detailed,
            on_trace_ready=lambda p: p.export_chrome_trace(str(profile_path) + "_trace.json"),
        ) as prof:
            # Run limited iterations for profiling
            for i in range(args.profile):
                with record_function(f"train_step_{i}"):
                    trainer.train_step(args.batch_size)
                prof.step()

        # Print profiler summary
        print("\n" + "=" * 80)
        print("📊 Profiler Summary (sorted by total CPU time)")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

        if device.type == 'cuda':
            print("\n" + "=" * 80)
            print("📊 Profiler Summary (sorted by total CUDA time)")
            print("=" * 80)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

        # Save detailed text report
        report_path = profile_path.with_suffix('.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CPU Time Summary\n")
            f.write("=" * 80 + "\n")
            f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
            if device.type == 'cuda':
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("CUDA Time Summary\n")
                f.write("=" * 80 + "\n")
                f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
        print(f"\n✅ Detailed report saved to: {report_path}")
        print(f"✅ Chrome trace saved to: {profile_path}_trace.json")
        print("   (Open in chrome://tracing or https://ui.perfetto.dev/)")

        # Continue with full training after profiling
        print(f"\n  Continuing with full training ({args.iterations} iterations)...")

    metrics = trainer.train(
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        log_every=args.print_every,
        verbose=True,
        wandb_log=use_wandb,  # Real-time wandb logging
        eval_fn=eval_fn,  # FN/FP trend tracking
        eval_every=args.eval_every,
        visualize_every=args.visualize_every,
        visualize_fn=visualize_fn,  # Circuit-style LGN visualization
        acc_log_interval=args.log_interval,  # Print accuracy every N steps
    )

    print("\n" + "=" * 80)
    print("Training Completed! ✅")
    print("=" * 80)

    # Print results (TB Loss metrics)
    print("\nFinal Metrics (TB Loss):")
    print(f"  TB Loss: {metrics['loss'][-1]:.4f}")
    print(f"  logZ: {metrics['logZ'][-1]:.4f}")
    print(f"  log P_F: {metrics['log_pf'][-1]:.4f}")
    print(f"  log R: {metrics['log_reward'][-1]:.4f}")

    # Loss trend
    print(f"\nLoss Trend:")
    print(f"  Initial loss: {metrics['loss'][0]:.4f}")
    print(f"  Final loss: {metrics['loss'][-1]:.4f}")
    print(f"  Improvement: {metrics['loss'][0] - metrics['loss'][-1]:.4f}")

    print(f"\n✅ Training complete! Model ready for evaluation.")

    # Log final summary to wandb (real-time logging already done during training)
    if use_wandb:
        wandb.log({
            "final/tb_loss": metrics['loss'][-1],
            "final/logZ": metrics['logZ'][-1],
            "final/log_pf": metrics['log_pf'][-1],
            "final/log_reward": metrics['log_reward'][-1],
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

    # 1. TB Loss Curve
    ax1 = axes[0, 0]
    ax1.plot(iterations_x, metrics['loss'], 'b-', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('TB Loss')
    ax1.set_title('TB Loss Curve (Should Decrease)')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 2. logZ Curve
    ax2 = axes[0, 1]
    ax2.plot(iterations_x, metrics['logZ'], 'r-', linewidth=1, alpha=0.7, label='logZ')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('logZ')
    ax2.set_title('logZ (Partition Function)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. log P_F and log R Curves
    ax3 = axes[1, 0]
    ax3.plot(iterations_x, metrics['log_pf'], 'g-', linewidth=1, alpha=0.7, label='log P_F')
    ax3.plot(iterations_x, metrics['log_reward'], 'purple', linewidth=1, alpha=0.7, label='log R')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Log Value')
    ax3.set_title('log P_F and log R')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Smoothed Loss and logZ
    ax4 = axes[1, 1]
    window = min(100, len(metrics['loss']) // 10) if len(metrics['loss']) > 10 else 1
    if window > 1:
        loss_smooth = np.convolve(metrics['loss'], np.ones(window)/window, mode='valid')
        logz_smooth = np.convolve(metrics['logZ'], np.ones(window)/window, mode='valid')
        smooth_x = list(range(len(loss_smooth)))
        ax4.plot(smooth_x, loss_smooth, 'b-', linewidth=2, label='Loss (smoothed)')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(smooth_x, logz_smooth, 'r-', linewidth=2, label='logZ (smoothed)')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Loss', color='blue')
        ax4_twin.set_ylabel('logZ', color='red')
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
