"""
Training Loop for Logic Gate Network GFlowNet.

This module implements Trajectory Balance (TB) Loss for GFlowNet training.

TB Loss Formula (notion.md Eq. 4, with uniform P_B assumption):
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
from typing import Dict, List, Tuple, Callable, Union
from dataclasses import dataclass
import time
from lgn.network import LGNState
from .policy_network import LGNPolicyNetwork
from .policy_network_gnn import LGNGNNPolicy
from .lgn_mdp import LGNMDP
from .action_space import LGNActionSpace


@dataclass
class TBTrajectory:
    """Trajectory data for TB Loss computation."""
    states: List[LGNState]      # s0, s1, ..., sT (states before each action)
    actions: List[Dict]         # a0, a1, ..., aT-1 (actions including stop)
    log_reward: float           # log R(terminal_state)
    terminal_state: LGNState    # Final state for reward computation


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
            terminal_state=lgn.copy()
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
        loss = (self.logZ + score).pow(2)

        return loss, total_log_pf.item(), trajectory.log_reward

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

    def train_step(self, batch_size: int) -> Tuple[float, Dict[str, float]]:
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
        Tuple[float, Dict[str, float]]
            - loss: Scalar loss value
            - metrics: Dictionary of training metrics
        """
        # Sample batch of trajectories
        trajectories = self.sample_batch(batch_size)

        # Compute TB loss for each trajectory
        losses = []
        log_pfs = []
        log_rewards = []

        for traj in trajectories:
            loss, log_pf, log_reward = self.compute_tb_loss(traj)
            losses.append(loss)
            log_pfs.append(log_pf)
            log_rewards.append(log_reward)

        # Average loss over batch
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
        mean_log_pf = np.mean(log_pfs)
        mean_log_reward = np.mean(log_rewards)
        mean_reward = np.exp(mean_log_reward)  # Convert back to actual reward

        metrics = {
            'loss': total_loss.item(),
            'logZ': self.logZ.item(),
            'log_pf': mean_log_pf,
            'log_reward': mean_log_reward,
            'mean_reward': mean_reward,
            'num_trajectories': batch_size,
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
            'logZ': [],
            'log_pf': [],
            'log_reward': [],
            'mean_reward': [],
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
            all_metrics['logZ'].append(metrics['logZ'])
            all_metrics['log_pf'].append(metrics['log_pf'])
            all_metrics['log_reward'].append(metrics['log_reward'])
            all_metrics['mean_reward'].append(metrics['mean_reward'])

            # Real-time wandb logging (with log-prefixed names)
            if wandb_log:
                wandb.log({
                    "train/tb_loss": metrics['loss'],
                    "train/logZ": metrics['logZ'],
                    "train/log_pf": metrics['log_pf'],
                    "train/log_reward": metrics['log_reward'],
                    "train/mean_reward": metrics['mean_reward'],
                    "train/iter_time": iter_time,
                }, step=i)

            # Periodic evaluation (FN/FP tracking)
            if eval_fn is not None and (i % eval_every == 0 or i == num_iterations - 1):
                eval_metrics = eval_fn()
                if wandb_log and eval_metrics:
                    wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=i)

            # Print progress
            if verbose and (i % log_every == 0 or i == num_iterations - 1):
                elapsed = time.time() - start_time
                iter_loss = metrics['loss']
                iter_logZ = metrics['logZ']
                iter_log_pf = metrics['log_pf']
                iter_log_reward = metrics['log_reward']
                # Clear the progress line and print full metrics
                print(f"\r  Step {i+1}/{num_iterations} | "
                      f"Loss: {iter_loss:.4f} | "
                      f"logZ: {iter_logZ:.4f} | "
                      f"log_pf: {iter_log_pf:.4f} | "
                      f"log_R: {iter_log_reward:.4f} | "
                      f"Iter: {iter_time:.1f}s | "
                      f"Total: {elapsed:.1f}s")

        # Final newline
        if verbose:
            print()

        return all_metrics
