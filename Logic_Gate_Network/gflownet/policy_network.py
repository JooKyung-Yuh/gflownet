"""
Policy Network for Logic Gate Network GFlowNet.

This module implements the neural network that learns Q(state, action) values
for Trajectory Balance (TB) loss computation in GFlowNet training.

Design:
-------
- State Encoder: Flattens LGNState into fixed-size vector representation
- Action Encoder: Encodes gate addition actions (gate_type + input_indices) into fixed-size vectors
- Q-Network: MLP that computes Q(s,a) from concatenated [state_encoding, action_encoding]
- Stop Head: Separate MLP for Q(s, stop) computation

This design handles LGN's variable action space (different number of valid actions per state)
by encoding each (state, action) pair and computing Q-values on demand.

Comparison to Reference Implementations:
----------------------------------------
- Grid: Fixed action space (ndim+1), direct MLP output
- Molecules: Variable action space using graph embeddings + action indexing
- LGN: Variable action space using state/action encoding + Q-value computation

Architecture follows molecules' pattern but uses simpler MLP (like grid) instead of GNN.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
from lgn.network import LGNState
from lgn.gates import GateType, GATE_TYPE_TO_IDX


def make_mlp(layer_sizes: List[int], activation=nn.LeakyReLU(), final_activation=False):
    """
    Create a Multi-Layer Perceptron (MLP).

    Parameters:
    -----------
    layer_sizes : List[int]
        List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
    activation : nn.Module
        Activation function to use between layers (default: LeakyReLU)
    final_activation : bool
        Whether to apply activation after final layer (default: False)

    Returns:
    --------
    nn.Sequential
        MLP module

    Example:
    --------
    >>> mlp = make_mlp([128, 256, 256, 1])
    >>> # Creates: Linear(128, 256) -> LeakyReLU -> Linear(256, 256) -> LeakyReLU -> Linear(256, 1)
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        # Add activation after every layer except the last (unless final_activation=True)
        if i < len(layer_sizes) - 2 or final_activation:
            layers.append(activation)
    return nn.Sequential(*layers)


class LGNPolicyNetwork(nn.Module):
    """
    Policy Network for Logic Gate Network GFlowNet.

    This network learns Q(state, action) values for TB loss computation.
    It consists of:
    1. State encoder: LGNState -> fixed-size vector
    2. Action encoder: Action dict -> fixed-size vector
    3. Q-network: [state_emb, action_emb] -> Q-value
    4. Stop head: state_emb -> Q(stop)

    The network handles variable action spaces by encoding each (state, action)
    pair individually and computing Q-values on demand.

    Parameters:
    -----------
    num_inputs : int
        Number of input features to LGN (e.g., 10 for 10-bit binary vectors)
    max_gates : int
        Maximum number of gates allowed in LGN
    state_emb_dim : int
        Dimension of state embedding (default: 256)
    action_emb_dim : int
        Dimension of action embedding (default: 64)
    hidden_dim : int
        Hidden layer size for MLPs (default: 256)
    num_hidden_layers : int
        Number of hidden layers in Q-network (default: 2)

    Example:
    --------
    >>> policy = LGNPolicyNetwork(num_inputs=10, max_gates=15)
    >>> lgn = LGNState(num_inputs=10, max_gates=15)
    >>> actions = action_space.get_valid_actions(lgn)
    >>>
    >>> # Compute Q-values for all valid actions
    >>> state_emb = policy.encode_state(lgn)
    >>> q_values = policy.compute_action_q_values(state_emb, actions)
    >>> q_stop = policy.compute_stop_q_value(state_emb)
    """

    def __init__(
        self,
        num_inputs: int,
        max_gates: int,
        state_emb_dim: int = 256,
        action_emb_dim: int = 64,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ):
        super().__init__()

        self.num_inputs = num_inputs
        self.max_gates = max_gates
        self.state_emb_dim = state_emb_dim
        self.action_emb_dim = action_emb_dim

        # Calculate state encoding size
        # State encoding: [num_gates, gate_types (one-hot max_gates * 16), connections (max_gates * max_inputs * 2)]
        # Simplified: [metadata + gate_types + connections]
        # - Metadata: [num_gates / max_gates] (1 value)
        # - Gate types: One-hot encoding for each gate slot (max_gates * 16)
        # - Connections: Binary matrix (max_gates * (num_inputs + max_gates))
        max_nodes = num_inputs + max_gates  # inputs + potential gate outputs
        state_dim = 1 + max_gates * 16 + max_gates * max_nodes

        # State encoder: Raw state -> embedding
        self.state_encoder = make_mlp([state_dim, hidden_dim, hidden_dim, state_emb_dim])

        # Action encoder: Raw action -> embedding
        # Action encoding: [gate_type (one-hot 16), input_indices (binary max_nodes)]
        action_dim = 16 + max_nodes
        self.action_encoder = make_mlp([action_dim, hidden_dim, action_emb_dim])

        # Q-network: [state_emb, action_emb] -> Q(s,a)
        q_input_dim = state_emb_dim + action_emb_dim
        q_layers = [q_input_dim] + [hidden_dim] * num_hidden_layers + [1]
        self.q_network = make_mlp(q_layers)

        # Stop head: state_emb -> Q(s, stop)
        self.stop_head = make_mlp([state_emb_dim, hidden_dim, 1])

    def encode_state(self, lgn: LGNState) -> torch.Tensor:
        """
        Encode LGNState into fixed-size vector.

        Encoding format:
        ----------------
        1. Metadata: [num_gates / max_gates]
        2. Gate types: One-hot encoding for each gate slot (max_gates * 16)
           - Empty slots encoded as all zeros
        3. Connections: Binary matrix indicating which nodes feed into each gate
           - Shape: (max_gates, num_inputs + max_gates)
           - connections[i, j] = 1 if node j is input to gate i

        Parameters:
        -----------
        lgn : LGNState
            Logic Gate Network state to encode

        Returns:
        --------
        torch.Tensor
            State encoding tensor of shape (state_dim,)
        """
        device = next(self.parameters()).device

        # Metadata
        num_gates_normalized = len(lgn.gates) / self.max_gates
        metadata = torch.tensor([num_gates_normalized], dtype=torch.float32, device=device)

        # Gate types: One-hot encoding (max_gates * 16)
        gate_types = torch.zeros(self.max_gates, 16, dtype=torch.float32, device=device)
        for i, gate in enumerate(lgn.gates):
            gate_type_idx = GATE_TYPE_TO_IDX[gate.gate_type]
            gate_types[i, gate_type_idx] = 1.0
        gate_types_flat = gate_types.flatten()

        # Connections: Binary matrix (max_gates, num_inputs + max_gates)
        max_nodes = self.num_inputs + self.max_gates
        connections = torch.zeros(self.max_gates, max_nodes, dtype=torch.float32, device=device)
        for i, gate in enumerate(lgn.gates):
            for input_idx in gate.inputs:
                connections[i, input_idx] = 1.0
        connections_flat = connections.flatten()

        # Concatenate all components
        state_encoding = torch.cat([metadata, gate_types_flat, connections_flat])

        # Pass through state encoder
        state_emb = self.state_encoder(state_encoding)
        return state_emb

    def encode_action(self, action: Dict[str, Any]) -> torch.Tensor:
        """
        Encode action dict into fixed-size vector.

        Encoding format:
        ----------------
        1. Gate type: One-hot encoding (16 dimensions)
        2. Input indices: Binary vector indicating which nodes are inputs
           - Shape: (num_inputs + max_gates,)
           - vector[i] = 1 if node i is an input to this gate

        Parameters:
        -----------
        action : Dict[str, Any]
            Action dictionary with keys 'gate_type' and 'input_indices'

        Returns:
        --------
        torch.Tensor
            Action encoding tensor of shape (action_dim,)
        """
        # Type narrowing: action must be a gate action
        assert 'gate_type' in action and 'input_indices' in action

        device = next(self.parameters()).device

        # Gate type: One-hot encoding
        gate_type_idx = GATE_TYPE_TO_IDX[action['gate_type']]
        gate_type_onehot = torch.zeros(16, dtype=torch.float32, device=device)
        gate_type_onehot[gate_type_idx] = 1.0

        # Input indices: Binary vector
        max_nodes = self.num_inputs + self.max_gates
        input_indices_binary = torch.zeros(max_nodes, dtype=torch.float32, device=device)
        for idx in action['input_indices']:
            input_indices_binary[idx] = 1.0

        # Concatenate
        action_encoding = torch.cat([gate_type_onehot, input_indices_binary])

        # Pass through action encoder
        action_emb = self.action_encoder(action_encoding)
        return action_emb

    def compute_action_q_values(
        self,
        state_emb: torch.Tensor,
        actions: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """
        Compute Q(s, a) for each action given state embedding.

        Parameters:
        -----------
        state_emb : torch.Tensor
            State embedding from encode_state(), shape (state_emb_dim,)
        actions : List[Dict[str, Any]]
            List of action dicts (gate additions, NOT stop actions)

        Returns:
        --------
        torch.Tensor
            Q-values for each action, shape (len(actions),)
        """
        if len(actions) == 0:
            return torch.tensor([], dtype=torch.float32, device=state_emb.device)

        # Encode all actions
        action_embs = torch.stack([self.encode_action(a) for a in actions])  # (num_actions, action_emb_dim)

        # Repeat state embedding for each action
        state_embs = state_emb.unsqueeze(0).repeat(len(actions), 1)  # (num_actions, state_emb_dim)

        # Concatenate and compute Q-values
        q_input = torch.cat([state_embs, action_embs], dim=1)  # (num_actions, state_emb_dim + action_emb_dim)
        q_values = self.q_network(q_input).squeeze(-1)  # (num_actions,)

        return q_values

    def compute_stop_q_value(self, state_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute Q(s, stop) given state embedding.

        Parameters:
        -----------
        state_emb : torch.Tensor
            State embedding from encode_state(), shape (state_emb_dim,)

        Returns:
        --------
        torch.Tensor
            Stop Q-value, shape ()
        """
        q_stop = self.stop_head(state_emb).squeeze(-1)
        return q_stop

    def forward(self, lgn: LGNState, actions: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute Q-values for all actions from a state.

        Parameters:
        -----------
        lgn : LGNState
            Current state
        actions : List[Dict[str, Any]]
            List of valid gate addition actions (should NOT include stop action)

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            - action_q_values: Q-values for gate actions, shape (len(actions),)
            - stop_q_value: Q-value for stop action, shape ()
        """
        state_emb = self.encode_state(lgn)
        action_q_values = self.compute_action_q_values(state_emb, actions)
        stop_q_value = self.compute_stop_q_value(state_emb)
        return action_q_values, stop_q_value

    def compute_q_value_for_action(
        self,
        lgn: LGNState,
        action: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute Q(s, a) for a single (state, action) pair.

        This is the key method used in TB loss computation to get Q(parent, action).

        Parameters:
        -----------
        lgn : LGNState
            State
        action : Dict[str, Any]
            Action dict, either:
            - Gate action: {'gate_type': GateType, 'input_indices': tuple}
            - Stop action: {'action': 'stop'}

        Returns:
        --------
        torch.Tensor
            Q(s, a), shape ()
        """
        state_emb = self.encode_state(lgn)

        if 'action' in action and action['action'] == 'stop':
            return self.compute_stop_q_value(state_emb)
        else:
            action_emb = self.encode_action(action)
            q_input = torch.cat([state_emb, action_emb])
            q_value = self.q_network(q_input).squeeze(-1)
            return q_value

    def sum_exp_q_values(self, lgn: LGNState, actions: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute sum of exp(Q(s, a)) over all valid actions.

        This is used in TB loss outflow computation:
            outflow = log(R + sum_a exp(Q(s, a)))

        Parameters:
        -----------
        lgn : LGNState
            Current state
        actions : List[Dict[str, Any]]
            All valid actions (including stop if applicable)

        Returns:
        --------
        torch.Tensor
            sum_a exp(Q(s, a)), shape ()
        """
        # Separate gate actions and stop action
        gate_actions = [a for a in actions if 'gate_type' in a]
        has_stop = any('action' in a and a['action'] == 'stop' for a in actions)

        state_emb = self.encode_state(lgn)

        # Compute exp(Q) for gate actions
        if len(gate_actions) > 0:
            gate_q_values = self.compute_action_q_values(state_emb, gate_actions)
            exp_gate_q = torch.exp(gate_q_values).sum()
        else:
            exp_gate_q = torch.tensor(0.0, device=state_emb.device)

        # Compute exp(Q) for stop action
        if has_stop:
            stop_q = self.compute_stop_q_value(state_emb)
            exp_stop_q = torch.exp(stop_q)
        else:
            exp_stop_q = torch.tensor(0.0, device=state_emb.device)

        return exp_gate_q + exp_stop_q

    def forward_policy(
        self,
        lgn: LGNState,
        gate_actions: List[Dict[str, Any]],
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass compatible with training loop interface.

        This method matches the GNN policy interface for compatibility
        with the training code.

        Parameters:
        -----------
        lgn : LGNState
            Current state
        gate_actions : List[Dict[str, Any]]
            List of gate addition actions (no stop action)
        device : torch.device
            Device to run on (ignored, uses model device)

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            - action_q: Q-values for gate actions [num_gate_actions]
            - stop_q: Q-value for stop action (scalar)
        """
        # Use forward method which already does this
        return self.forward(lgn, gate_actions)
