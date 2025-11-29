"""
Training Loop for Logic Gate Network GFlowNet.

This module implements Trajectory Balance (TB) Loss for GFlowNet training.

TB Loss Formula:
    Loss = (logZ + log P_F(τ) - log R(x))²

where:
- logZ: Learnable log partition function parameter
- log P_F(τ): Sum of log forward policy probabilities along trajectory
- log R(x): Log reward at terminal state

Key Features:
-------------
- Learnable logZ parameter (nn.Parameter)
- Trajectory-level loss computation
- Gradient clipping for training stability
- Comprehensive logging (loss, logZ, log_pf, log_reward)

Reference:
----------
Based on torchgfn TB implementation and notion.md specifications.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Callable, Union, TypedDict
from dataclasses import dataclass
from collections import Counter
import time
from lgn.network import LGNState
from .policy_network import LGNPolicyNetwork
from .policy_network_gnn import LGNGNNPolicy
from .lgn_mdp import LGNMDP
from .action_space import LGNActionSpace


class TrainStepMetrics(TypedDict):
    """Type definition for train_step return metrics."""
    loss: float
    log_loss: float
    logZ: float
    log_pf: float
    log_reward: float
    num_trajectories: int
    mean_total_gates: float
    mean_connected_gates: float
    mean_connected_inputs: float
    gate_type_counts: Dict[str, int]
    # Distribution metrics for debugging
    termination_by_max_gates: int
    termination_by_all_features: int
    termination_by_stop_action: int
    connectivity_ratio: float  # mean(connected_inputs / total_inputs)
    reward_min: float
    reward_max: float
    reward_std: float
    connected_inputs_min: int
    connected_inputs_max: int
    # Best/Worst trajectory info
    best_traj_reward: float
    best_traj_gates: int
    best_traj_connected_inputs: int
    worst_traj_reward: float
    worst_traj_gates: int
    worst_traj_connected_inputs: int


@dataclass
class TBTrajectory:
    """Trajectory data for TB Loss computation."""
    states: List[LGNState]      # s0, s1, ..., sT (states before each action)
    actions: List[Dict]         # a0, a1, ..., aT-1 (actions including stop)
    log_reward: float           # log R(terminal_state)
    terminal_state: LGNState    # Final state for reward computation
    termination_reason: str     # 'max_gates' or 'all_features' or 'stop_action'


class LGNTrainer:
    """
    Trainer for Logic Gate Network GFlowNet with TB Loss.

    Implements Trajectory Balance (TB) Loss from notion.md Eq. 4:
        Loss = (logZ + log P_F(τ) - log R(x))²

    This class handles:
    - Trajectory sampling using the policy
    - TB loss computation with learnable logZ
    - Optimization (policy + logZ parameters)
    - Logging and metrics

    Parameters:
    -----------
    policy : LGNPolicyNetwork or LGNGNNPolicy
        The policy network that learns Q(s, a) values
    mdp : LGNMDP
        MDP wrapper (used for num_inputs and max_gates)
    action_space : LGNActionSpace
        Action space providing get_valid_actions()
    reward_fn : Callable[[LGNState], float]
        Reward function R(s) for terminal states (returns actual reward, not log)
    optimizer : torch.optim.Optimizer
        Optimizer for training (should include trainer.logZ parameter)
    device : torch.device
        Device to run training on (cpu or cuda)
    init_logZ : float
        Initial value for logZ parameter (default: 0.0, meaning Z=1)
    clip_grad : float
        Gradient clipping value (default: 10.0, 0 to disable)

    Example:
    --------
    >>> policy = LGNGNNPolicy(num_inputs=10, max_gates=15)
    >>> trainer = LGNTrainer(policy=policy, mdp=mdp, ...)
    >>> # Important: Add logZ to optimizer
    >>> optimizer = torch.optim.Adam(
    ...     list(policy.parameters()) + [trainer.logZ],
    ...     lr=1e-3
    ... )
    >>> trainer.optimizer = optimizer
    """

    def __init__(
        self,
        policy: Union[LGNPolicyNetwork, LGNGNNPolicy],
        mdp: LGNMDP,
        action_space: LGNActionSpace,
        reward_fn: Callable[[LGNState], float],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        init_logZ: float = 0.0,
        clip_grad: float = 10.0,
    ):
        self.policy = policy
        self.mdp = mdp
        self.action_space = action_space
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.device = device
        self.clip_grad = clip_grad

        # Move policy to device
        self.policy.to(device)

        # Learnable logZ parameter (TB Loss)
        # logZ = 0.0 means Z = 1.0 (partition function starts at 1)
        self.logZ = nn.Parameter(torch.tensor(init_logZ, dtype=torch.float32, device=device))

        # Training statistics
        self.train_losses = []
        self.step_count = 0

    def sample_trajectory(self) -> TBTrajectory:
        """
        Sample a single trajectory for TB Loss computation.

        Sampling process:
        -----------------
        1. Start from empty LGNState
        2. At each step:
           - Get valid actions from action_space
           - Compute Q-values for all actions (gate additions + stop)
           - Sample action proportional to exp(Q) (Boltzmann sampling)
           - Store (state, action) pair
           - Apply action to get next state
        3. Repeat until stop action is sampled
        4. Compute log_reward from terminal state

        Returns:
        --------
        TBTrajectory
            Contains states, actions, log_reward, and terminal_state
        """
        states = []
        actions = []
        lgn = LGNState(num_inputs=self.mdp.num_inputs, max_gates=self.mdp.max_gates)

        with torch.no_grad():  # Don't track gradients during sampling
            while True:
                # Store current state before action
                states.append(lgn.copy())

                # Get valid actions from current state
                valid_actions = self.action_space.get_valid_actions(lgn)

                # Separate gate actions and stop action
                gate_actions = [a for a in valid_actions if 'gate_type' in a]
                has_stop = any('action' in a and a['action'] == 'stop' for a in valid_actions)

                # Compute Q-values
                action_q, stop_q = self.policy.forward_policy(lgn, gate_actions)

                # Combine Q-values for sampling
                if len(gate_actions) > 0 and has_stop:
                    all_q_values = torch.cat([action_q, stop_q.unsqueeze(0)])
                    all_actions = gate_actions + [{'action': 'stop'}]
                elif len(gate_actions) > 0:
                    all_q_values = action_q
                    all_actions = gate_actions
                else:
                    # No valid gate actions, must stop
                    all_q_values = stop_q.unsqueeze(0)
                    all_actions = [{'action': 'stop'}]

                # Sample action using Boltzmann distribution
                probs = torch.softmax(all_q_values, dim=0)
                cumsum = torch.cumsum(probs, dim=0)
                u = torch.rand(1, device=probs.device)
                action_idx = int(torch.searchsorted(cumsum, u).item())
                action_idx = min(action_idx, len(all_actions) - 1)

                sampled_action = all_actions[action_idx]
                actions.append(sampled_action)

                # Check if stop action
                is_stop = 'action' in sampled_action and sampled_action['action'] == 'stop'

                if is_stop:
                    # Determine termination reason
                    if lgn.get_num_gates() >= lgn.max_gates:
                        termination_reason = 'max_gates'
                    elif len(lgn.get_features_used()) >= lgn.num_inputs:
                        termination_reason = 'all_features'
                    else:
                        termination_reason = 'stop_action'

                    # Compute log reward at terminal state
                    reward = self.reward_fn(lgn)
                    # reward_fn returns actual reward, convert to log
                    log_reward = np.log(reward + 1e-8)
                    break
                else:
                    # Apply action
                    lgn.add_gate(sampled_action['gate_type'], sampled_action['input_indices'])

        return TBTrajectory(
            states=states,
            actions=actions,
            log_reward=log_reward,
            terminal_state=lgn.copy(),
            termination_reason=termination_reason
        )

    def _find_action_index(self, valid_actions: List[Dict], target_action: Dict) -> int:
        """Find index of target action in valid_actions list."""
        for i, action in enumerate(valid_actions):
            if action == target_action:
                return i
            # Handle stop action comparison
            if ('action' in action and action.get('action') == 'stop' and
                'action' in target_action and target_action.get('action') == 'stop'):
                return i
            # Handle gate action comparison
            if ('gate_type' in action and 'gate_type' in target_action and
                action.get('gate_type') == target_action.get('gate_type') and
                action.get('input_indices') == target_action.get('input_indices')):
                return i
        raise ValueError(f"Action {target_action} not found in valid actions")

    def compute_tb_loss(self, trajectory: TBTrajectory) -> Tuple[torch.Tensor, float, float]:
        """
        Compute TB Loss for a single trajectory.

        TB Loss Formula (notion.md Eq. 4, uniform P_B assumption):
            Loss = (logZ + log P_F(τ) - log R(x))²

        where:
            log P_F(τ) = Σ_t log P_F(a_t | s_t)

        Parameters:
        -----------
        trajectory : TBTrajectory
            Sampled trajectory with states, actions, and log_reward

        Returns:
        --------
        Tuple[torch.Tensor, float, float]
            - loss: TB loss tensor (scalar)
            - log_pf: Total log forward probability (for logging)
            - log_reward: Log reward (for logging)
        """
        # Compute log P_F(τ) = Σ_t log P_F(a_t | s_t)
        total_log_pf = torch.tensor(0.0, device=self.device)

        for state, action in zip(trajectory.states, trajectory.actions):
            # Get all valid actions from this state
            valid_actions = self.action_space.get_valid_actions(state)

            # Compute log probabilities for all actions
            log_probs = self.policy.compute_action_logprobs(state, valid_actions, self.device)

            # Find index of the action that was taken
            action_idx = self._find_action_index(valid_actions, action)

            # Add log P_F(a_t | s_t)
            total_log_pf = total_log_pf + log_probs[action_idx]

        # Convert log_reward to tensor
        log_reward = torch.tensor(trajectory.log_reward, device=self.device)

        # TB Loss: (logZ + log P_F - log R)²
        score = total_log_pf - log_reward
        
        #self.logZ = - score.mean(0)

        return score, total_log_pf.item(), trajectory.log_reward

    def sample_batch(self, batch_size: int) -> List[TBTrajectory]:
        """
        Sample a batch of trajectories.

        Parameters:
        -----------
        batch_size : int
            Number of trajectories to sample

        Returns:
        --------
        List[TBTrajectory]
            List of sampled trajectories
        """
        return [self.sample_trajectory() for _ in range(batch_size)]

    def train_step(self, batch_size: int) -> Tuple[float, TrainStepMetrics]:
        """
        Execute one training step with TB Loss.

        TB Loss = mean over trajectories of (logZ + log P_F(τ) - log R(x))²

        Steps:
        ------
        1. Sample batch of trajectories
        2. Compute TB loss for each trajectory
        3. Average losses
        4. Backpropagate
        5. Clip gradients (if enabled)
        6. Update parameters (policy + logZ)

        Parameters:
        -----------
        batch_size : int
            Number of trajectories to sample

        Returns:
        --------
        Tuple[float, TrainStepMetrics]
            - loss: Scalar loss value
            - metrics: Dictionary of training metrics
        """
        # Sample batch of trajectories
        trajectories = self.sample_batch(batch_size)

        # Compute TB loss for each trajectory
        scores = []
        log_pfs = []
        log_rewards = []

        for traj in trajectories:
            score, log_pf, log_reward = self.compute_tb_loss(traj)
            scores.append(score)
            log_pfs.append(log_pf)
            log_rewards.append(log_reward)

        # Average loss over batch
        scores_tensor = torch.stack(scores)
        logZ = -scores_tensor.mean().detach()
        
        losses = [(logZ + score).pow(2) for score in scores]
        
        # scores_tensor = torch.stack(scores)
        # losses = (self.logZ + scores_tensor).pow(2)
        # total_loss = losses.mean()
        
        total_loss = torch.stack(losses).mean()

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping (include logZ in clipping)
        if self.clip_grad > 0:
            all_params = list(self.policy.parameters()) + [self.logZ]
            torch.nn.utils.clip_grad_value_(all_params, self.clip_grad)

        self.optimizer.step()

        # Compute metrics
        mean_log_pf = float(np.mean(log_pfs))
        mean_log_reward = float(np.mean(log_rewards))

        # Collect gate usage statistics from trajectories
        total_gates_list = []
        connected_gates_list = []
        connected_inputs_list = []
        gate_type_totals = Counter()
        termination_counts = {'max_gates': 0, 'all_features': 0, 'stop_action': 0}

        for traj in trajectories:
            stats = traj.terminal_state.get_gate_usage_stats()
            total_gates_list.append(stats['total_gates'])
            connected_gates_list.append(stats['connected_gates'])
            connected_inputs_list.append(stats['connected_inputs'])
            gate_type_totals.update(stats['gate_type_counts'])
            termination_counts[traj.termination_reason] += 1

        # Compute connectivity ratio (connected_inputs / total_inputs) per trajectory
        num_inputs = self.mdp.num_inputs
        connectivity_ratios = [ci / num_inputs for ci in connected_inputs_list]

        # Find best and worst trajectories by reward
        best_idx = int(np.argmax(log_rewards))
        worst_idx = int(np.argmin(log_rewards))
        best_traj = trajectories[best_idx]
        worst_traj = trajectories[worst_idx]
        best_stats = best_traj.terminal_state.get_gate_usage_stats()
        worst_stats = worst_traj.terminal_state.get_gate_usage_stats()

        metrics: TrainStepMetrics = {
            'loss': total_loss.item(),
            'log_loss': float(np.log(total_loss.item() + 1e-8)),
            'logZ': logZ.item(),
            'log_pf': mean_log_pf,
            'log_reward': mean_log_reward,
            'num_trajectories': batch_size,
            # Gate usage statistics
            'mean_total_gates': float(np.mean(total_gates_list)),
            'mean_connected_gates': float(np.mean(connected_gates_list)),
            'mean_connected_inputs': float(np.mean(connected_inputs_list)),
            'gate_type_counts': dict(gate_type_totals),
            # Distribution metrics for debugging
            'termination_by_max_gates': termination_counts['max_gates'],
            'termination_by_all_features': termination_counts['all_features'],
            'termination_by_stop_action': termination_counts['stop_action'],
            'connectivity_ratio': float(np.mean(connectivity_ratios)),
            'reward_min': float(np.min(log_rewards)),
            'reward_max': float(np.max(log_rewards)),
            'reward_std': float(np.std(log_rewards)),
            'connected_inputs_min': int(np.min(connected_inputs_list)),
            'connected_inputs_max': int(np.max(connected_inputs_list)),
            # Best/Worst trajectory info
            'best_traj_reward': log_rewards[best_idx],
            'best_traj_gates': best_stats['total_gates'],
            'best_traj_connected_inputs': best_stats['connected_inputs'],
            'worst_traj_reward': log_rewards[worst_idx],
            'worst_traj_gates': worst_stats['total_gates'],
            'worst_traj_connected_inputs': worst_stats['connected_inputs'],
        }

        # Update statistics
        self.step_count += 1
        self.train_losses.append(metrics['loss'])

        return metrics['loss'], metrics

    def train(
        self,
        num_iterations: int,
        batch_size: int,
        log_every: int = 100,
        verbose: bool = True,
        wandb_log: bool = False,
        eval_fn: Callable[[], Dict[str, float]] = None,
        eval_every: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Run full training loop with TB Loss.

        Parameters:
        -----------
        num_iterations : int
            Number of training steps
        batch_size : int
            Batch size (number of trajectories per step)
        log_every : int
            Print metrics every N iterations (default: 100)
        verbose : bool
            Whether to print training progress (default: True)
        wandb_log : bool
            Whether to log metrics to wandb in real-time (default: False)
        eval_fn : Callable[[], Dict[str, float]]
            Optional evaluation function that returns metrics dict (e.g., FN/FP rates)
        eval_every : int
            Run evaluation every N iterations (default: 100)

        Returns:
        --------
        Dict[str, List[float]]
            Dictionary of training metrics history
        """
        # Import wandb if needed
        if wandb_log:
            try:
                import wandb
            except ImportError:
                print("Warning: wandb not installed, disabling real-time logging")
                wandb_log = False

        all_metrics = {
            'loss': [],
            'log_loss': [],
            'logZ': [],
            'log_pf': [],
            'log_reward': [],
            'mean_total_gates': [],
            'mean_connected_gates': [],
            'mean_connected_inputs': [],
            'gate_type_counts': [],  # List of dicts per step
        }

        start_time = time.time()

        for i in range(num_iterations):
            iter_start = time.time()

            # Show real-time progress on same line
            if verbose:
                progress_pct = (i / num_iterations) * 100
                bar_length = 30
                filled = int(bar_length * i / num_iterations)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r  [{bar}] {i}/{num_iterations} ({progress_pct:.1f}%) - Sampling trajectories...",
                      end='', flush=True)

            loss, metrics = self.train_step(batch_size)

            iter_time = time.time() - iter_start

            # Log metrics
            all_metrics['loss'].append(metrics['loss'])
            all_metrics['log_loss'].append(metrics['log_loss'])
            all_metrics['logZ'].append(metrics['logZ'])
            all_metrics['log_pf'].append(metrics['log_pf'])
            all_metrics['log_reward'].append(metrics['log_reward'])
            all_metrics['mean_total_gates'].append(metrics['mean_total_gates'])
            all_metrics['mean_connected_gates'].append(metrics['mean_connected_gates'])
            all_metrics['mean_connected_inputs'].append(metrics['mean_connected_inputs'])
            all_metrics['gate_type_counts'].append(metrics['gate_type_counts'])

            # Real-time wandb logging
            if wandb_log:
                wandb.log({
                    "train/log_loss": metrics['log_loss'],
                    "train/logZ": metrics['logZ'],
                    "train/log_pf": metrics['log_pf'],
                    "train/log_reward": metrics['log_reward'],
                    "train/iter_time": iter_time,
                    "train/mean_total_gates": metrics['mean_total_gates'],
                    "train/mean_connected_gates": metrics['mean_connected_gates'],
                    "train/mean_connected_inputs": metrics['mean_connected_inputs'],
                    # Distribution metrics
                    "debug/termination_max_gates_ratio": metrics['termination_by_max_gates'] / batch_size,
                    "debug/termination_all_features_ratio": metrics['termination_by_all_features'] / batch_size,
                    "debug/termination_stop_action_ratio": metrics['termination_by_stop_action'] / batch_size,
                    "debug/connectivity_ratio": metrics['connectivity_ratio'],
                    "debug/reward_min": metrics['reward_min'],
                    "debug/reward_max": metrics['reward_max'],
                    "debug/reward_std": metrics['reward_std'],
                    "debug/connected_inputs_min": metrics['connected_inputs_min'],
                    "debug/connected_inputs_max": metrics['connected_inputs_max'],
                }, step=i)

            # Periodic evaluation (FN/FP tracking)
            if eval_fn is not None and (i % eval_every == 0 or i == num_iterations - 1):
                eval_metrics = eval_fn()
                if wandb_log and eval_metrics:
                    wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=i)

            # Print progress
            if verbose and (i % log_every == 0 or i == num_iterations - 1):
                elapsed = time.time() - start_time
                # Clear the progress line and print detailed batch analysis
                print(f"\r{'='*80}")
                print(f"  Step {i+1}/{num_iterations} | Time: {iter_time:.1f}s | Elapsed: {elapsed:.0f}s")
                print(f"  {'─'*76}")

                # Loss and basic metrics
                print(f"  Loss: log_L={metrics['log_loss']:.2f} | "
                      f"logZ={metrics['logZ']:.2f} | "
                      f"log_pf={metrics['log_pf']:.2f} | "
                      f"log_R={metrics['log_reward']:.2f}")

                # Termination reason analysis
                term_max = metrics['termination_by_max_gates']
                term_feat = metrics['termination_by_all_features']
                term_stop = metrics['termination_by_stop_action']
                print(f"  Termination: max_gates={term_max}/{batch_size} ({100*term_max/batch_size:.0f}%) | "
                      f"all_features={term_feat}/{batch_size} ({100*term_feat/batch_size:.0f}%) | "
                      f"stop_action={term_stop}/{batch_size} ({100*term_stop/batch_size:.0f}%)")

                # Connectivity analysis
                num_inputs = self.mdp.num_inputs
                print(f"  Connectivity: ratio={metrics['connectivity_ratio']:.2f} | "
                      f"connected_inputs=[{metrics['connected_inputs_min']}, {metrics['connected_inputs_max']}] "
                      f"(mean={metrics['mean_connected_inputs']:.1f}/{num_inputs})")

                # Reward distribution
                print(f"  Reward: min={metrics['reward_min']:.2f} | "
                      f"max={metrics['reward_max']:.2f} | "
                      f"mean={metrics['log_reward']:.2f} | "
                      f"std={metrics['reward_std']:.2f}")

                # Best/Worst trajectory
                print(f"  Best:  reward={metrics['best_traj_reward']:.2f} | "
                      f"gates={metrics['best_traj_gates']} | "
                      f"connected_inputs={metrics['best_traj_connected_inputs']}/{num_inputs}")
                print(f"  Worst: reward={metrics['worst_traj_reward']:.2f} | "
                      f"gates={metrics['worst_traj_gates']} | "
                      f"connected_inputs={metrics['worst_traj_connected_inputs']}/{num_inputs}")

        # Final newline
        if verbose:
            print()

        return all_metrics
