"""
Benchmark: GPU (Apple MPS) vs CPU performance comparison.

Tests the batched training implementation on both devices.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import numpy as np
from lgn.network import LGNState
from gflownet.policy_network_gnn import LGNGNNPolicy
from gflownet.lgn_mdp import LGNMDP
from gflownet.action_space import LGNActionSpace
from gflownet.training import LGNTrainer


def check_mps_available():
    """Check if Apple MPS (Metal Performance Shaders) is available."""
    if not torch.backends.mps.is_available():
        return False, "MPS not available"
    if not torch.backends.mps.is_built():
        return False, "PyTorch not built with MPS support"
    return True, "MPS available"


def create_trainer(device, num_inputs=10, max_gates=15):
    """Create a trainer on the specified device."""
    policy = LGNGNNPolicy(
        num_inputs=num_inputs,
        max_gates=max_gates,
        node_emb_dim=128,
        num_conv_steps=3
    ).to(device)

    mdp = LGNMDP(num_inputs=num_inputs, max_gates=max_gates)
    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)

    def dummy_reward(lgn: LGNState) -> float:
        return max(0.01, lgn.get_num_gates() * 0.1 + 0.5)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    trainer = LGNTrainer(
        policy=policy,
        mdp=mdp,
        action_space=action_space,
        reward_fn=dummy_reward,
        optimizer=optimizer,
        device=device
    )

    return trainer


def benchmark_train_step(trainer, batch_size, num_iterations, warmup=3):
    """Benchmark train_step performance."""
    # Warmup
    for _ in range(warmup):
        trainer.train_step(batch_size)

    # Benchmark
    times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        loss, metrics = trainer.train_step(batch_size)

        # Synchronize for accurate timing on GPU
        if trainer.device.type == 'mps':
            torch.mps.synchronize()
        elif trainer.device.type == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    return times


def run_benchmark():
    """Run the full benchmark comparison."""
    print("=" * 70)
    print("GPU (MPS) vs CPU Performance Benchmark")
    print("=" * 70)

    # Check MPS availability
    mps_available, mps_msg = check_mps_available()
    print(f"\nMPS Status: {mps_msg}")

    if not mps_available:
        print("Cannot run GPU benchmark - MPS not available")
        print("Running CPU-only benchmark...")

    # Benchmark parameters
    num_inputs = 10
    max_gates = 15
    batch_sizes = [4, 8, 16, 32]
    num_iterations = 10
    warmup = 3

    print(f"\nBenchmark Configuration:")
    print(f"  num_inputs: {num_inputs}")
    print(f"  max_gates: {max_gates}")
    print(f"  batch_sizes: {batch_sizes}")
    print(f"  iterations per batch_size: {num_iterations}")
    print(f"  warmup iterations: {warmup}")

    results = {}

    # CPU Benchmark
    print("\n" + "-" * 70)
    print("CPU Benchmark")
    print("-" * 70)

    cpu_device = torch.device('cpu')
    cpu_trainer = create_trainer(cpu_device, num_inputs, max_gates)

    results['cpu'] = {}
    for batch_size in batch_sizes:
        times = benchmark_train_step(cpu_trainer, batch_size, num_iterations, warmup)
        mean_time = np.mean(times)
        std_time = np.std(times)
        results['cpu'][batch_size] = {'mean': mean_time, 'std': std_time, 'times': times}
        print(f"  batch_size={batch_size:2d}: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms/step")

    # GPU (MPS) Benchmark
    if mps_available:
        print("\n" + "-" * 70)
        print("GPU (MPS) Benchmark")
        print("-" * 70)

        mps_device = torch.device('mps')
        mps_trainer = create_trainer(mps_device, num_inputs, max_gates)

        results['mps'] = {}
        for batch_size in batch_sizes:
            times = benchmark_train_step(mps_trainer, batch_size, num_iterations, warmup)
            mean_time = np.mean(times)
            std_time = np.std(times)
            results['mps'][batch_size] = {'mean': mean_time, 'std': std_time, 'times': times}
            print(f"  batch_size={batch_size:2d}: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms/step")

    # Comparison
    if mps_available:
        print("\n" + "=" * 70)
        print("Performance Comparison (CPU vs MPS)")
        print("=" * 70)
        print(f"\n{'batch_size':>10} | {'CPU (ms)':>12} | {'MPS (ms)':>12} | {'Speedup':>10}")
        print("-" * 50)

        for batch_size in batch_sizes:
            cpu_time = results['cpu'][batch_size]['mean'] * 1000
            mps_time = results['mps'][batch_size]['mean'] * 1000
            speedup = cpu_time / mps_time

            speedup_str = f"{speedup:.2f}x"
            if speedup > 1:
                speedup_str = f"+{speedup_str} (MPS faster)"
            else:
                speedup_str = f"{speedup_str} (CPU faster)"

            print(f"{batch_size:>10} | {cpu_time:>12.2f} | {mps_time:>12.2f} | {speedup_str}")

    # Throughput comparison
    if mps_available:
        print("\n" + "=" * 70)
        print("Throughput Comparison (trajectories/second)")
        print("=" * 70)
        print(f"\n{'batch_size':>10} | {'CPU':>15} | {'MPS':>15} | {'MPS/CPU':>10}")
        print("-" * 55)

        for batch_size in batch_sizes:
            cpu_throughput = batch_size / results['cpu'][batch_size]['mean']
            mps_throughput = batch_size / results['mps'][batch_size]['mean']
            ratio = mps_throughput / cpu_throughput

            print(f"{batch_size:>10} | {cpu_throughput:>12.1f}/s | {mps_throughput:>12.1f}/s | {ratio:>8.2f}x")

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_benchmark()
