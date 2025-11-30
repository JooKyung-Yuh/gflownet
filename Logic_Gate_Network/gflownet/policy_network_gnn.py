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
from .lgn_to_graph import lgn_to_graph, batch_lgn_to_batched_graph


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

        # Convert LGN to graph ONCE
        graph_data = lgn_to_graph(lgn, device)

        # Forward pass ONCE (instead of calling compute_q_values twice)
        node_logits, gate_type_logits, stop_logit = self.forward(graph_data)

        # Compute Q-values for gate actions
        if len(gate_actions) > 0:
            q_values = []
            for action in gate_actions:
                gate_type = action['gate_type']
                input_indices = action['input_indices']

                # Q-value = sum of node logits + gate type logit
                node_q = node_logits[list(input_indices)].sum()
                gate_type_idx = GATE_TYPE_TO_IDX[gate_type] - 1
                gate_q = gate_type_logits[gate_type_idx]

                q_values.append(node_q + gate_q)

            action_q = torch.stack(q_values)
        else:
            action_q = torch.empty(0, device=device)

        return action_q, stop_logit

    def forward_batched(
        self,
        batch_data: Batch,
        num_nodes_per_graph: List[int],
        has_edges_per_graph: List[bool],
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Batched forward pass through GNN for multiple LGN graphs.

        This method processes multiple graphs in a single forward pass,
        which is much more efficient on GPU than processing one at a time.

        IMPORTANT: Empty graphs (no edges) are handled correctly by tracking
        which graphs have edges and applying message passing selectively.

        Parameters:
        -----------
        batch_data : Batch
            PyTorch Geometric Batch object containing multiple graphs
            Created by batch_lgn_to_batched_graph()
        num_nodes_per_graph : List[int]
            Number of nodes in each graph (for splitting outputs)
        has_edges_per_graph : List[bool]
            Whether each graph has edges (for selective message passing)

        Returns:
        --------
        Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]
            - node_logits_list: List of node logits per graph
              Each element has shape [num_nodes_i] for graph i
            - gate_type_logits: [batch_size, num_gate_types]
            - stop_logits: [batch_size]

        Example:
        --------
        >>> batch, num_nodes_list, _, has_edges = batch_lgn_to_batched_graph(lgns, device)
        >>> node_logits_list, gate_type_logits, stop_logits = policy.forward_batched(batch, num_nodes_list, has_edges)
        >>> # node_logits_list[0] has shape [num_nodes_of_lgn_0]
        >>> # gate_type_logits has shape [batch_size, 16]
        >>> # stop_logits has shape [batch_size]
        """
        batch_size = len(num_nodes_per_graph)
        device = batch_data.x.device

        if batch_size == 0:
            return [], torch.empty(0, self.num_gate_types, device=device), torch.empty(0, device=device)

        # ===== Node Embedding =====
        # batch_data.x shape: [total_nodes, 1]
        node_types = batch_data.x.squeeze(-1)  # [total_nodes]
        out = self.node_embedding(node_types)  # [total_nodes, node_emb_dim]

        # ===== Edge Embedding =====
        if batch_data.edge_attr.numel() > 0:
            edge_types = batch_data.edge_attr[:, 0]  # [total_edges]
            edge_emb = self.edge_embedding(edge_types)  # [total_edges, node_emb_dim]
            edge_features = edge_emb
        else:
            edge_features = torch.empty((0, self.node_emb_dim), device=out.device)

        # ===== Node Preprocessing =====
        out = self.node2emb(out)

        # ===== GNN Message Passing with GRU =====
        # Handle empty graphs (no edges) correctly by processing them separately
        # This ensures identical behavior to forward() for empty graphs

        # Identify node ranges for each graph
        node_ranges = []
        start_idx = 0
        for num_nodes in num_nodes_per_graph:
            end_idx = start_idx + num_nodes
            node_ranges.append((start_idx, end_idx))
            start_idx = end_idx

        # Check if we have any empty graphs (no edges)
        has_any_empty = not all(has_edges_per_graph)

        if has_any_empty:
            # Process graphs separately to handle empty graphs correctly
            # Empty graphs: skip message passing (m = out)
            # Non-empty graphs: apply message passing

            h = out.unsqueeze(0)  # [1, total_nodes, node_emb_dim]

            for conv_step in range(self.num_conv_steps):
                # Initialize m with out (for empty graphs)
                m = out.clone()

                # Apply conv only to nodes belonging to graphs with edges
                if batch_data.edge_index.numel() > 0:
                    # Compute conv for all nodes (but only nodes with edges will be affected)
                    conv_out = F.leaky_relu(self.conv(out, batch_data.edge_index, edge_features))

                    # Replace m values only for graphs that have edges
                    for graph_idx, (start, end) in enumerate(node_ranges):
                        if has_edges_per_graph[graph_idx]:
                            m[start:end] = conv_out[start:end]
                        # else: m[start:end] stays as out[start:end] (no message passing)

                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)
        else:
            # All graphs have edges - can process everything together efficiently
            h = out.unsqueeze(0)

            for conv_step in range(self.num_conv_steps):
                if batch_data.edge_index.numel() > 0:
                    m = F.leaky_relu(self.conv(out, batch_data.edge_index, edge_features))
                else:
                    m = out

                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)

        # ===== Output Heads =====

        # 1. Node selection logits (per-node)
        all_node_logits = self.node_selector(out).squeeze(-1)  # [total_nodes]

        # Split node logits by graph
        node_logits_list = []
        for start, end in node_ranges:
            node_logits_list.append(all_node_logits[start:end])

        # 2. Graph-level embeddings for gate_type and stop heads
        # Use scatter_mean to compute per-graph mean embeddings
        # batch_data.batch: [total_nodes] with values 0, 0, ..., 1, 1, ..., 2, 2, ...
        graph_emb = gnn.global_mean_pool(out, batch_data.batch)  # [batch_size, node_emb_dim]

        # 3. Gate type logits (per-graph)
        gate_type_logits = self.gate_type_head(graph_emb)  # [batch_size, num_gate_types]

        # 4. Stop logits (per-graph)
        stop_logits = self.stop_head(graph_emb).squeeze(-1)  # [batch_size]

        return node_logits_list, gate_type_logits, stop_logits

    def forward_policy_batched(
        self,
        lgns: List[LGNState],
        gate_actions_per_lgn: List[List[Dict[str, Any]]],
        device: Optional[torch.device] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Batched forward policy for multiple LGNs.

        This is the batched version of forward_policy() that processes
        multiple LGNs in a single GNN forward pass.

        Parameters:
        -----------
        lgns : List[LGNState]
            List of LGN states to process
        gate_actions_per_lgn : List[List[Dict[str, Any]]]
            For each LGN, a list of valid gate actions (no stop action)
        device : torch.device
            Device to run on

        Returns:
        --------
        Tuple[List[torch.Tensor], torch.Tensor]
            - action_q_list: List of Q-values for gate actions per LGN
              action_q_list[i] has shape [num_gate_actions_for_lgn_i]
            - stop_q: Q-values for stop action per LGN [batch_size]
        """
        if device is None:
            device = next(self.parameters()).device

        batch_size = len(lgns)

        if batch_size == 0:
            return [], torch.empty(0, device=device)

        # Convert all LGNs to batched graph (single operation)
        batch_data, num_nodes_per_graph, num_inputs_per_graph, has_edges_per_graph = batch_lgn_to_batched_graph(lgns, device)

        # Single batched forward pass
        node_logits_list, gate_type_logits, stop_logits = self.forward_batched(
            batch_data, num_nodes_per_graph, has_edges_per_graph
        )

        # Compute Q-values for each LGN's gate actions
        action_q_list = []

        for i, (node_logits, gate_actions) in enumerate(zip(node_logits_list, gate_actions_per_lgn)):
            if len(gate_actions) > 0:
                q_values = []
                for action in gate_actions:
                    gate_type = action['gate_type']
                    input_indices = action['input_indices']

                    # Q-value = sum of node logits + gate type logit
                    node_q = node_logits[list(input_indices)].sum()
                    gate_type_idx = GATE_TYPE_TO_IDX[gate_type] - 1
                    gate_q = gate_type_logits[i, gate_type_idx]

                    q_values.append(node_q + gate_q)

                action_q = torch.stack(q_values)
            else:
                action_q = torch.empty(0, device=device)

            action_q_list.append(action_q)

        return action_q_list, stop_logits

    def compute_q_values_batched(
        self,
        lgns: List[LGNState],
        actions_per_lgn: List[List[Dict[str, Any]]],
        device: Optional[torch.device] = None
    ) -> List[torch.Tensor]:
        """
        Batched version of compute_q_values() for multiple LGNs.

        Computes Q(s, a) for all actions of multiple LGNs in a single
        batched forward pass.

        Parameters:
        -----------
        lgns : List[LGNState]
            List of LGN states to process
        actions_per_lgn : List[List[Dict[str, Any]]]
            For each LGN, list of all valid actions (including stop action)
        device : torch.device
            Device to run on

        Returns:
        --------
        List[torch.Tensor]
            Q-values for each LGN's actions
            q_values_list[i] has shape [num_actions_for_lgn_i]
        """
        if device is None:
            device = next(self.parameters()).device

        batch_size = len(lgns)

        if batch_size == 0:
            return []

        # Convert all LGNs to batched graph
        batch_data, num_nodes_per_graph, num_inputs_per_graph, has_edges_per_graph = batch_lgn_to_batched_graph(lgns, device)

        # Single batched forward pass
        node_logits_list, gate_type_logits, stop_logits = self.forward_batched(
            batch_data, num_nodes_per_graph, has_edges_per_graph
        )

        # Compute Q-values for each LGN
        q_values_list = []

        for i, (node_logits, actions) in enumerate(zip(node_logits_list, actions_per_lgn)):
            q_values = []

            for action in actions:
                if 'action' in action and action['action'] == 'stop':
                    # Stop action
                    q = stop_logits[i]
                else:
                    # Gate addition action
                    gate_type = action['gate_type']
                    input_indices = action['input_indices']

                    # Q-value = sum of node logits + gate type logit
                    node_q = node_logits[list(input_indices)].sum()
                    gate_type_idx = GATE_TYPE_TO_IDX[gate_type] - 1
                    gate_q = gate_type_logits[i, gate_type_idx]

                    q = node_q + gate_q

                q_values.append(q)

            if len(q_values) > 0:
                q_values_list.append(torch.stack(q_values))
            else:
                q_values_list.append(torch.empty(0, device=device))

        return q_values_list

    def compute_action_logprobs_batched(
        self,
        lgns: List[LGNState],
        actions_per_lgn: List[List[Dict[str, Any]]],
        device: Optional[torch.device] = None
    ) -> List[torch.Tensor]:
        """
        Batched version of compute_action_logprobs() for multiple LGNs.

        Computes log P(a|s) for all actions of multiple LGNs in a single
        batched forward pass.

        This converts Q-values to probabilities via softmax:
        P(a|s) = exp(Q(s,a)) / sum_a' exp(Q(s,a'))

        Parameters:
        -----------
        lgns : List[LGNState]
            List of LGN states to process
        actions_per_lgn : List[List[Dict[str, Any]]]
            For each LGN, list of all valid actions
        device : torch.device
            Device to run on

        Returns:
        --------
        List[torch.Tensor]
            Log probabilities for each LGN's actions
            log_probs_list[i] has shape [num_actions_for_lgn_i]
        """
        q_values_list = self.compute_q_values_batched(lgns, actions_per_lgn, device)

        log_probs_list = []
        for q_values in q_values_list:
            if q_values.numel() > 0:
                log_probs = F.log_softmax(q_values, dim=0)
            else:
                log_probs = q_values  # Empty tensor
            log_probs_list.append(log_probs)

        return log_probs_list
