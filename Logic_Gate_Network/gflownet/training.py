"""
Training Loop for Logic Gate Network GFlowNet.

This module implements the Trajectory Balance (TB) training loop for GFlowNet.
It handles:
1. Trajectory sampling (forward passes through the environment)
2. Parent computation (backward transitions using LGNMDP)
3. TB loss computation (inflow/outflow matching)
4. Optimization and logging

The implementation follows the pattern from molecules but simplified for LGN.

Key Features:
-------------
- Trajectory Balance (TB) loss computation
- Balanced loss (terminal vs flow states weighted differently)
- Gradient clipping for training stability
- Comprehensive logging (loss metrics, rewards, etc.)

Reference:
----------
Based on molecules/gflownet.py training loop, adapted for LGN's structure.
"""

import torch
from typing import Dict, Any, List, Tuple, Callable
import time
from ..lgn.network import LGNState
from .policy_network import LGNPolicyNetwork
from .lgn_mdp import LGNMDP
from .action_space import LGNActionSpace


class LGNTrainer:
    """
    Trainer for Logic Gate Network GFlowNet.

    This class handles the complete training loop for GFlowNet, including:
    - Trajectory sampling using the policy
    - TB loss computation
    - Optimization
    - Logging and metrics

    Parameters:
    -----------
    policy : LGNPolicyNetwork
        The policy network that learns Q(s, a) values
    mdp : LGNMDP
        MDP wrapper providing parent_transitions()
    action_space : LGNActionSpace
        Action space providing get_valid_actions()
    reward_fn : Callable[[LGNState], float]
        Reward function R(s) for terminal states
    optimizer : torch.optim.Optimizer
        Optimizer for training (e.g., Adam)
    device : torch.device
        Device to run training on (cpu or cuda)
    balanced_loss : bool
        Whether to use balanced loss (default: True)
    leaf_coef : float
        Coefficient for terminal state loss in balanced mode (default: 10.0)
    log_reg_c : float
        Small constant for numerical stability in logs (default: 1e-6)
    clip_grad : float
        Gradient clipping value (default: 10.0, 0 to disable)

    Example:
    --------
    >>> policy = LGNPolicyNetwork(num_inputs=10, max_gates=15)
    >>> mdp = LGNMDP(num_inputs=10, max_gates=15)
    >>> action_space = LGNActionSpace(num_inputs=10, max_gates=15)
    >>> reward_fn = lambda lgn: compute_reward(lgn)
    >>> optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    >>>
    >>> trainer = LGNTrainer(
    ...     policy=policy,
    ...     mdp=mdp,
    ...     action_space=action_space,
    ...     reward_fn=reward_fn,
    ...     optimizer=optimizer,
    ...     device=torch.device('cpu')
    ... )
    >>>
    >>> # Train for 1000 steps
    >>> for i in range(1000):
    >>>     loss, metrics = trainer.train_step(batch_size=32)
    >>>     if i % 100 == 0:
    >>>         print(f"Step {i}: Loss = {loss:.4f}")
    """

    def __init__(
        self,
        policy: LGNPolicyNetwork,
        mdp: LGNMDP,
        action_space: LGNActionSpace,
        reward_fn: Callable[[LGNState], float],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        balanced_loss: bool = True,
        leaf_coef: float = 10.0,
        log_reg_c: float = 1e-6,
        clip_grad: float = 10.0,
    ):
        self.policy = policy
        self.mdp = mdp
        self.action_space = action_space
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.device = device
        self.balanced_loss = balanced_loss
        self.leaf_coef = leaf_coef
        self.log_reg_c = log_reg_c
        self.clip_grad = clip_grad

        # Move policy to device
        self.policy.to(device)

        # Training statistics
        self.train_losses = []
        self.step_count = 0

    def sample_trajectory(self) -> List[Tuple[Tuple[LGNState, ...], Tuple[Dict[str, Any], ...], float, LGNState, bool]]:
        """
        Sample a single trajectory using the current policy.

        This follows the Grid/Molecules pattern: compute parent_transitions()
        DURING sampling, immediately after each action is applied.

        Sampling process:
        -----------------
        1. Start from empty LGNState
        2. At each step:
           - Get valid actions from action_space
           - Compute Q-values for all actions (gate additions + stop)
           - Sample action proportional to exp(Q) (Boltzmann sampling)
           - Apply action to get next state
           - IMMEDIATELY compute parents of new state (KEY: done during sampling)
           - Store (parents, actions, reward, resulting_state, done)
        3. Repeat until stop action is sampled
        4. Evaluate final state with reward_fn

        Returns:
        --------
        List[Tuple[Tuple[LGNState, ...], Tuple[Dict[str, Any], ...], float, LGNState, bool]]
            List of transition tuples for TB loss computation.
            Each tuple contains:
            - parents: Tuple of all parent states that can transition to resulting_state
            - actions: Tuple of actions taken from each parent (same length as parents)
            - reward: Reward at resulting_state (0 for non-terminal, R(s) for terminal)
            - resulting_state: The state reached after applying action
            - done: Whether resulting_state is terminal
        """
        trajectory = []
        lgn = LGNState(num_inputs=self.mdp.num_inputs, max_gates=self.mdp.max_gates)

        with torch.no_grad():  # Don't track gradients during sampling
            while True:
                # Get valid actions from current state
                actions = self.action_space.get_valid_actions(lgn)

                # Separate gate actions and stop action
                gate_actions = [a for a in actions if 'gate_type' in a]

                # Compute Q-values
                action_q, stop_q = self.policy.forward(lgn, gate_actions)

                # Combine Q-values for sampling
                if len(gate_actions) > 0:
                    all_q_values = torch.cat([action_q, stop_q.unsqueeze(0)])
                    all_actions = gate_actions + [{'action': 'stop'}]
                else:
                    # No valid gate actions, must stop
                    all_q_values = stop_q.unsqueeze(0)
                    all_actions = [{'action': 'stop'}]

                # Sample action using Boltzmann distribution: P(a) proportional to exp(Q(s,a))
                logits = all_q_values
                probs = torch.softmax(logits, dim=0)
                action_idx = int(torch.multinomial(probs, 1).item())
                sampled_action = all_actions[action_idx]

                # Check if stop action
                is_stop = 'action' in sampled_action and sampled_action['action'] == 'stop'

                if is_stop:
                    # Terminal transition: stop action taken from current state
                    reward = self.reward_fn(lgn)

                    # Compute parents of terminal state (with stop action)
                    # The terminal state is lgn itself (stop doesn't change state)
                    parents_list, actions_list = self.mdp.parent_transitions(lgn, used_stop_action=True)

                    # Store: (parents, actions, reward, resulting_state, done)
                    trajectory.append((
                        tuple(parents_list),
                        tuple(actions_list),
                        reward,
                        lgn.copy(),
                        True
                    ))
                    break
                else:
                    # Non-terminal transition: gate addition
                    # Apply action to get next state
                    lgn.add_gate(sampled_action['gate_type'], sampled_action['input_indices'])

                    # KEY: Compute parents of the NEW state (after action applied)
                    # This is the Grid/Molecules pattern - parents computed during sampling
                    parents_list, actions_list = self.mdp.parent_transitions(lgn, used_stop_action=False)

                    # Store: (parents leading to lgn, their actions, reward=0, resulting_state=lgn, done=False)
                    trajectory.append((
                        tuple(parents_list),
                        tuple(actions_list),
                        0.0,
                        lgn.copy(),
                        False
                    ))

        return trajectory

    def sample_batch(self, batch_size: int) -> Tuple[
        List[LGNState],  # parent states
        torch.Tensor,     # parent batch indices
        List[Dict[str, Any]],  # actions
        torch.Tensor,     # rewards
        List[LGNState],  # resulting states
        torch.Tensor,     # done flags
    ]:
        """
        Sample a batch of trajectories and convert to training format.

        This follows the Grid/Molecules pattern: trajectories already contain
        parent information (computed during sampling), so we just flatten them.

        Format matches molecules/gflownet.py and grid/toy_grid_dag.py:
        ---------------------------------------------------------------
        For each transition (parent -> action -> state):
        - p: List of parent states (flattened from all trajectories)
        - pb: Parent batch index (which resulting state this parent leads to)
        - a: Action taken from parent
        - r: Reward at resulting state
        - s: Resulting state
        - d: Done flag (0 for non-terminal, 1 for terminal)

        Parameters:
        -----------
        batch_size : int
            Number of trajectories to sample

        Returns:
        --------
        Tuple of (p, pb, a, r, s, d) as described above
        """
        p_list = []  # parent states
        pb_list = []  # parent batch indices
        a_list = []  # actions
        r_list = []  # rewards
        s_list = []  # resulting states
        d_list = []  # done flags

        for traj_idx in range(batch_size):
            trajectory = self.sample_trajectory()

            # Each trajectory entry is: (parents_tuple, actions_tuple, reward, state, done)
            # where parents and actions were already computed during sampling
            for parents_tuple, actions_tuple, reward, state, done in trajectory:
                # parents_tuple and actions_tuple have the same length
                # (one action per parent leading to the resulting state)

                # For each parent-action pair leading to this state
                for parent, action in zip(parents_tuple, actions_tuple):
                    p_list.append(parent)
                    pb_list.append(len(s_list))  # This parent leads to s_list[len(s_list)]
                    a_list.append(action)

                # Add the resulting state once (parents map to it via pb)
                s_list.append(state)
                r_list.append(reward)
                d_list.append(1.0 if done else 0.0)

        # Convert to tensors
        pb = torch.tensor(pb_list, dtype=torch.long, device=self.device)
        r = torch.tensor(r_list, dtype=torch.float32, device=self.device)
        d = torch.tensor(d_list, dtype=torch.float32, device=self.device)

        return p_list, pb, a_list, r, s_list, d

    def compute_tb_loss(
        self,
        p_list: List[LGNState],
        pb: torch.Tensor,
        a_list: List[Dict[str, Any]],
        r: torch.Tensor,
        s_list: List[LGNState],
        d: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Trajectory Balance (TB) loss.

        TB Loss Formula:
        ----------------
        For each transition (parent -> action -> state):
            inflow = log(sum_{parent, a} exp(Q(parent, a)))
            outflow = log(R(s) + sum_{a'} exp(Q(s, a')))
            loss = (inflow - outflow)^2

        Balanced Loss:
        --------------
        If balanced_loss=True, separately weight terminal vs non-terminal states:
            term_loss = mean((inflow - outflow)^2  for terminal states)
            flow_loss = mean((inflow - outflow)^2  for non-terminal states)
            loss = term_loss * leaf_coef + flow_loss

        This gives terminal states more weight (default 10x), which is critical
        for environments with sparse rewards (like LGN).

        Parameters:
        -----------
        p_list : List[LGNState]
            Parent states
        pb : torch.Tensor
            Parent batch indices (shape: [num_parents])
        a_list : List[Dict[str, Any]]
            Actions taken from parents
        r : torch.Tensor
            Rewards (shape: [num_transitions])
        s_list : List[LGNState]
            Resulting states (shape: [num_transitions])
        d : torch.Tensor
            Done flags (shape: [num_transitions])

        Returns:
        --------
        Tuple[torch.Tensor, Dict[str, float]]
            - loss: Scalar loss tensor
            - metrics: Dictionary with 'term_loss', 'flow_loss', etc.
        """
        ntransitions = len(s_list)

        # ===== INFLOW COMPUTATION =====
        # Compute Q(parent, action) for each parent
        qsa_p = torch.stack([
            self.policy.compute_q_value_for_action(parent, action)
            for parent, action in zip(p_list, a_list)
        ])

        # Sum exp(Q(parent, action)) for all parents leading to same state
        # This uses index_add_ to group parents by their resulting state
        exp_inflow = torch.zeros(ntransitions, dtype=torch.float32, device=self.device)
        exp_inflow = exp_inflow.index_add_(0, pb, torch.exp(qsa_p))
        inflow = torch.log(exp_inflow + self.log_reg_c)

        # ===== OUTFLOW COMPUTATION =====
        # For each state, compute sum of exp(Q(s, a)) over all valid actions
        exp_outflow = torch.stack([
            self.policy.sum_exp_q_values(s, self.action_space.get_valid_actions(s))
            for s in s_list
        ])

        # outflow = log(R + sum exp(Q(s, a)) * (1 - done))
        # Terminal states: log(R), Non-terminal: log(sum exp(Q))
        outflow_plus_r = torch.log(self.log_reg_c + r + exp_outflow * (1 - d))

        # ===== LOSS COMPUTATION =====
        # Squared difference between inflow and outflow
        losses = (inflow - outflow_plus_r).pow(2)

        if self.balanced_loss:
            # Separate terminal and flow losses
            term_loss = (losses * d).sum() / (d.sum() + 1e-20)
            flow_loss = (losses * (1 - d)).sum() / ((1 - d).sum() + 1e-20)
            loss = term_loss * self.leaf_coef + flow_loss
        else:
            # Uniform weighting
            term_loss = (losses * d).sum() / (d.sum() + 1e-20)
            flow_loss = (losses * (1 - d)).sum() / ((1 - d).sum() + 1e-20)
            loss = losses.mean()

        # Metrics for logging
        metrics = {
            'loss': loss.item(),
            'term_loss': term_loss.item(),
            'flow_loss': flow_loss.item(),
            'mean_reward': r[d == 1].mean().item() if (d == 1).any() else 0.0,
            'num_terminals': (d == 1).sum().item(),
            'num_transitions': ntransitions,
        }

        return loss, metrics

    def train_step(self, batch_size: int) -> Tuple[float, Dict[str, float]]:
        """
        Execute one training step.

        Steps:
        ------
        1. Sample batch of trajectories
        2. Compute TB loss
        3. Backpropagate
        4. Clip gradients (if enabled)
        5. Update parameters
        6. Log metrics

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
        # Sample batch
        p_list, pb, a_list, r, s_list, d = self.sample_batch(batch_size)

        # Compute loss
        loss, metrics = self.compute_tb_loss(p_list, pb, a_list, r, s_list, d)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), self.clip_grad)

        self.optimizer.step()

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
    ) -> Dict[str, List[float]]:
        """
        Run full training loop.

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

        Returns:
        --------
        Dict[str, List[float]]
            Dictionary of training metrics history
        """
        all_metrics = {
            'loss': [],
            'term_loss': [],
            'flow_loss': [],
            'mean_reward': [],
        }

        start_time = time.time()

        for i in range(num_iterations):
            loss, metrics = self.train_step(batch_size)

            # Log metrics
            all_metrics['loss'].append(metrics['loss'])
            all_metrics['term_loss'].append(metrics['term_loss'])
            all_metrics['flow_loss'].append(metrics['flow_loss'])
            all_metrics['mean_reward'].append(metrics['mean_reward'])

            # Print progress
            if verbose and (i % log_every == 0 or i == num_iterations - 1):
                elapsed = time.time() - start_time
                iter_loss = metrics['loss']
                iter_term = metrics['term_loss']
                iter_flow = metrics['flow_loss']
                iter_reward = metrics['mean_reward']
                print(f"Step {i}/{num_iterations} | "
                      f"Loss: {iter_loss:.4f} | "
                      f"Term: {iter_term:.4f} | "
                      f"Flow: {iter_flow:.4f} | "
                      f"Reward: {iter_reward:.4f} | "
                      f"Time: {elapsed:.1f}s")

        return all_metrics
