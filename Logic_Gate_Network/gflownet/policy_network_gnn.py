"""
GNN-based Policy Network for Logic Gate Network GFlowNet.

This module implements a Graph Neural Network that learns Q(state, action) values
following the molecules implementation pattern.

Architecture:
-------------
1. Node embeddings: Gate types (17 types: 0=input, 1-16=gates)
2. Edge embeddings: Connection types
3. GNN layers: NNConv + GRU (multiple message passing steps)
4. Output heads:
   - Node logits: For each node, predict if it should be selected as input
   - Gate type logits: Which gate type to add [16]
   - Stop logit: Whether to stop

This handles variable action space by:
- Predicting which nodes to select as inputs (node-level predictions)
- Predicting which gate type to add (global prediction)

Comparison to Molecules:
------------------------
Molecules: stem_logits (which stem + which block) + mol_logit (stop)
LGN: node_logits (which nodes as inputs) + gate_type_logits (which gate) + stop_logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn
from typing import List, Dict, Any, Tuple, Optional
from lgn.network import LGNState
from lgn.gates import GateType, GATE_TYPE_TO_IDX
from .lgn_to_graph import lgn_to_graph


def make_mlp(layer_sizes: List[int], activation=nn.LeakyReLU(), final_activation=False):
    """Create a Multi-Layer Perceptron."""
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes) - 2 or final_activation:
            layers.append(activation)
    return nn.Sequential(*layers)


class LGNGNNPolicy(nn.Module):
    """
    GNN-based Policy Network for Logic Gate Network GFlowNet.

    This network uses Graph Neural Networks to process the LGN structure
    and output action predictions.

    Parameters:
    -----------
    num_inputs : int
        Number of input features to LGN
    max_gates : int
        Maximum number of gates allowed
    node_emb_dim : int
        Node embedding dimension (default: 128)
    num_conv_steps : int
        Number of GNN message passing steps (default: 3)
    num_gate_types : int
        Number of gate types (default: 16)

    Architecture:
    -------------
    1. Node embeddings (17 types: 0=input, 1-16=gate types)
    2. Edge embeddings (for connection types)
    3. NNConv + GRU layers (num_conv_steps times)
    4. Output heads:
       - node_logits: [num_nodes] (which nodes to select as inputs)
       - gate_type_logits: [16] (which gate type to add)
       - stop_logit: [1] (whether to stop)
    """

    def __init__(
        self,
        num_inputs: int,
        max_gates: int,
        node_emb_dim: int = 128,
        num_conv_steps: int = 3,
        num_gate_types: int = 16,
    ):
        super().__init__()

        self.num_inputs = num_inputs
        self.max_gates = max_gates
        self.node_emb_dim = node_emb_dim
        self.num_conv_steps = num_conv_steps
        self.num_gate_types = num_gate_types

        # Embeddings
        # Node types: 0 = input, 1-16 = gate types
        # Total: 17 types
        self.node_embedding = nn.Embedding(num_gate_types + 1, node_emb_dim)

        # Edge types: source_type, target_type pairs
        # For simplicity, we'll use learned edge features
        self.edge_embedding = nn.Embedding(num_gate_types + 1, node_emb_dim)

        # GNN layer: NNConv (follows molecules pattern)
        # NNConv learns a neural network to compute edge-conditioned weights
        # The edge network must output in_channels * out_channels weights
        edge_network = nn.Sequential(
            nn.Linear(node_emb_dim, node_emb_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(node_emb_dim * 2, node_emb_dim * node_emb_dim),
        )
        self.conv = gnn.NNConv(node_emb_dim, node_emb_dim, edge_network, aggr='mean')

        # Node to embedding (optional preprocessing)
        self.node2emb = nn.Sequential(
            nn.Linear(node_emb_dim, node_emb_dim),
            nn.LeakyReLU(),
            nn.Linear(node_emb_dim, node_emb_dim)
        )

        # GRU for temporal updates (like molecules)
        self.gru = nn.GRU(node_emb_dim, node_emb_dim)

        # Output heads

        # 1. Node selection head: For each node, predict if it should be input
        self.node_selector = make_mlp([node_emb_dim, node_emb_dim, 1])

        # 2. Gate type head: Which gate type to add
        self.gate_type_head = make_mlp([node_emb_dim, node_emb_dim, num_gate_types])

        # 3. Stop head: Whether to stop
        self.stop_head = make_mlp([node_emb_dim, node_emb_dim, 1])

    def forward(self, graph_data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN.

        Parameters:
        -----------
        graph_data : Data
            PyTorch Geometric Data object from lgn_to_graph()
            - x: Node types [num_nodes, 1]
            - edge_index: Edge connectivity [2, num_edges]
            - edge_attr: Edge types [num_edges, 2]

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - node_logits: [num_nodes] (which nodes to select)
            - gate_type_logits: [num_gate_types] (which gate type)
            - stop_logit: [1] (whether to stop)
        """
        # Embed nodes
        # graph_data.x shape: [num_nodes, 1]
        node_types = graph_data.x.squeeze(-1)  # [num_nodes]
        out = self.node_embedding(node_types)  # [num_nodes, node_emb_dim]

        # Embed edges
        # edge_attr shape: [num_edges, 2] (source_type, target_type)
        if graph_data.edge_attr.numel() > 0:
            # Use source type for edge embedding (simplified)
            edge_types = graph_data.edge_attr[:, 0]  # [num_edges]
            edge_emb = self.edge_embedding(edge_types)  # [num_edges, node_emb_dim]

            # For NNConv, we need edge features to be [num_edges, edge_feature_dim]
            # Create a dummy edge network that just passes through
            # In molecules, they use a more complex edge network
            edge_features = edge_emb
        else:
            # No edges yet (empty LGN)
            edge_features = torch.empty((0, self.node_emb_dim), device=out.device)

        # Preprocess node embeddings
        out = self.node2emb(out)

        # GNN message passing with GRU
        h = out.unsqueeze(0)  # [1, num_nodes, node_emb_dim] for GRU

        for i in range(self.num_conv_steps):
            if graph_data.edge_index.numel() > 0:
                # Message passing
                m = F.leaky_relu(self.conv(out, graph_data.edge_index, edge_features))
            else:
                # No edges, skip message passing
                m = out

            # Update with GRU
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)  # [num_nodes, node_emb_dim]

        # Output heads

        # 1. Node selection logits: Which nodes to use as inputs
        node_logits = self.node_selector(out).squeeze(-1)  # [num_nodes]

        # 2. Gate type logits: Which gate type to add
        # Use global pooling (mean over all nodes)
        graph_emb = out.mean(dim=0)  # [node_emb_dim]
        gate_type_logits = self.gate_type_head(graph_emb)  # [num_gate_types]

        # 3. Stop logit
        stop_logit = self.stop_head(graph_emb).squeeze(-1)  # scalar

        return node_logits, gate_type_logits, stop_logit

    def compute_q_values(
        self,
        lgn: LGNState,
        actions: List[Dict[str, Any]],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute Q(s, a) for each action.

        This method handles the variable action space by:
        1. Forward pass to get node_logits, gate_type_logits, stop_logit
        2. For each action, compute Q-value as combination of:
           - Sum of node_logits for selected input nodes
           - gate_type_logit for the chosen gate type

        Parameters:
        -----------
        lgn : LGNState
            Current state
        actions : List[Dict[str, Any]]
            List of actions, each with 'gate_type' and 'input_indices'
            Or special action {'action': 'stop'}
        device : torch.device
            Device to run on

        Returns:
        --------
        torch.Tensor
            Q-values for each action [len(actions)]
        """
        if device is None:
            device = next(self.parameters()).device

        # Convert LGN to graph
        graph_data = lgn_to_graph(lgn, device)

        # Forward pass
        node_logits, gate_type_logits, stop_logit = self.forward(graph_data)

        # Compute Q-value for each action
        q_values = []

        for action in actions:
            if 'action' in action and action['action'] == 'stop':
                # Stop action
                q = stop_logit
            else:
                # Gate addition action
                # Type narrowing: action is a gate action
                assert 'gate_type' in action and 'input_indices' in action
                gate_type = action['gate_type']
                input_indices = action['input_indices']

                # Q-value = sum of node logits + gate type logit
                node_q = node_logits[list(input_indices)].sum()
                gate_type_idx = GATE_TYPE_TO_IDX[gate_type] - 1  # Subtract 1 because logits are 0-indexed
                gate_q = gate_type_logits[gate_type_idx]

                q = node_q + gate_q

            q_values.append(q)

        return torch.stack(q_values)

    def compute_action_logprobs(
        self,
        lgn: LGNState,
        actions: List[Dict[str, Any]],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute log P(a|s) for each action using the policy.

        This converts Q-values to probabilities via softmax:
        P(a|s) = exp(Q(s,a)) / sum_a' exp(Q(s,a'))

        Parameters:
        -----------
        lgn : LGNState
            Current state
        actions : List[Dict[str, Any]]
            List of all valid actions
        device : torch.device
            Device to run on

        Returns:
        --------
        torch.Tensor
            Log probabilities for each action [len(actions)]
        """
        q_values = self.compute_q_values(lgn, actions, device)
        log_probs = F.log_softmax(q_values, dim=0)
        return log_probs

    def sum_exp_q_values(
        self,
        lgn: LGNState,
        actions: List[Dict[str, Any]],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute sum of exp(Q(s, a)) over all valid actions.

        Used in TB loss outflow computation.

        Parameters:
        -----------
        lgn : LGNState
            Current state
        actions : List[Dict[str, Any]]
            All valid actions
        device : torch.device
            Device to run on

        Returns:
        --------
        torch.Tensor
            sum_a exp(Q(s, a))
        """
        q_values = self.compute_q_values(lgn, actions, device)
        return torch.exp(q_values).sum()

    def compute_q_value_for_action(
        self,
        lgn: LGNState,
        action: Dict[str, Any],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute Q(s, a) for a single action.

        This is a convenience method for the training loop.

        Parameters:
        -----------
        lgn : LGNState
            Current state
        action : Dict[str, Any]
            Action to evaluate
        device : torch.device
            Device to run on

        Returns:
        --------
        torch.Tensor
            Q-value for the action (scalar)
        """
        return self.compute_q_values(lgn, [action], device)[0]

    def forward_policy(
        self,
        lgn: LGNState,
        gate_actions: List[Dict[str, Any]],
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass compatible with training loop interface.

        This method separates gate actions and stop action for compatibility
        with the existing training code.

        Parameters:
        -----------
        lgn : LGNState
            Current state
        gate_actions : List[Dict[str, Any]]
            List of gate addition actions (no stop action)
        device : torch.device
            Device to run on

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            - action_q: Q-values for gate actions [num_gate_actions]
            - stop_q: Q-value for stop action (scalar)
        """
        if device is None:
            device = next(self.parameters()).device

        # Compute Q-values for gate actions
        if len(gate_actions) > 0:
            action_q = self.compute_q_values(lgn, gate_actions, device)
        else:
            action_q = torch.empty(0, device=device)

        # Compute Q-value for stop action
        stop_action = [{'action': 'stop'}]
        stop_q = self.compute_q_values(lgn, stop_action, device)[0]

        return action_q, stop_q
