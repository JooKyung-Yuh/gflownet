"""
Convert LGN to PyTorch Geometric graph representation.

This module converts LogicGateNetwork states into graph data structures
suitable for Graph Neural Network (GNN) processing.

Graph Structure:
----------------
- Nodes: Original inputs + Gates
- Edges: Connections (input→gate, gate→gate)
- Node features: Type (input or gate type)
- Edge features: Connection type

Example:
--------
LGN with 2 inputs, 2 gates:
  input₀ ──→ gate₀ (AND) ──→ gate₁ (OR)
  input₁ ──→ gate₀       ──→ gate₁

Graph:
  Nodes: [input₀, input₁, gate₀, gate₁]
  Edges: [(input₀, gate₀), (input₁, gate₀), (gate₀, gate₁), (gate₀, gate₁)]
"""

import torch
from torch_geometric.data import Data, Batch
from typing import List, Optional, Tuple
from lgn.network import LGNState
from lgn.gates import GateType, GATE_TYPE_TO_IDX

# Node type encoding for GNN:
# - 0: Input node (original input features)
# - 1-16: Gate nodes (GATE_TYPE_TO_IDX + 1)
#
# We add 1 to gate indices to distinguish them from input nodes (type 0)


def lgn_to_graph(lgn: LGNState, device: Optional[torch.device] = None) -> Data:
    """
    Convert LGNState to PyTorch Geometric Data object.

    Node Indexing:
    --------------
    - Nodes 0 to (num_inputs-1): Original inputs
    - Nodes num_inputs to (num_inputs + num_gates - 1): Gates

    Node Features:
    --------------
    - Input nodes: Type = 0 (special input type)
    - Gate nodes: Type = gate_type.value + 1 (1-16)

    Edge Features:
    --------------
    - Source type: Type of source node
    - Target type: Type of target node

    Parameters:
    -----------
    lgn : LGNState
        Logic Gate Network state
    device : torch.device
        Device to place tensors on (default: cpu)

    Returns:
    --------
    Data
        PyTorch Geometric Data object with:
        - x: Node features [num_nodes, 1] (node types)
        - edge_index: Edge connectivity [2, num_edges]
        - edge_attr: Edge features [num_edges, 2] (source_type, target_type)
        - num_inputs: Number of original inputs (stored as attribute)
        - num_gates: Number of gates (stored as attribute)
    """
    if device is None:
        device = torch.device('cpu')

    num_inputs = lgn.num_inputs
    num_gates = len(lgn.gates)
    num_nodes = num_inputs + num_gates

    # --- Node Features ---
    # Input nodes: type = 0
    # Gate nodes: type = GATE_TYPE_TO_IDX[gate_type] + 1 (1-16)
    node_types = torch.zeros(num_nodes, dtype=torch.long, device=device)
    for i, gate in enumerate(lgn.gates):
        node_idx = num_inputs + i
        # Add 1 to distinguish gates from input nodes (type 0)
        node_types[node_idx] = GATE_TYPE_TO_IDX[gate.gate_type] + 1

    # Node features: [num_nodes, 1]
    x = node_types.unsqueeze(1)

    # --- Edges ---
    edges = []
    edge_attrs = []

    for gate_idx, gate in enumerate(lgn.gates):
        target_node = num_inputs + gate_idx  # Gate node index
        # +1 to match node type encoding (gates are 1-16)
        target_type = GATE_TYPE_TO_IDX[gate.gate_type] + 1

        for input_idx in gate.inputs:
            source_node = input_idx  # Could be input or previous gate

            # Source type (match node type encoding)
            if input_idx < num_inputs:
                source_type = 0  # Input node
            else:
                source_gate_idx = input_idx - num_inputs
                # +1 to match node type encoding (gates are 1-16)
                source_type = GATE_TYPE_TO_IDX[lgn.gates[source_gate_idx].gate_type] + 1

            # Add edge: source → target
            edges.append([source_node, target_node])
            edge_attrs.append([source_type, target_type])

    # Convert to tensors
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()  # [2, num_edges]
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long, device=device)  # [num_edges, 2]
    else:
        # Empty graph (no gates yet)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 2), dtype=torch.long, device=device)

    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    # Store metadata
    data.num_inputs = num_inputs
    data.num_gates = num_gates

    return data


def batch_lgn_to_graph(lgns: List[LGNState], device: Optional[torch.device] = None) -> List[Data]:
    """
    Convert multiple LGN states to graph data objects.

    Parameters:
    -----------
    lgns : List[LGNState]
        List of LGN states
    device : torch.device
        Device to place tensors on

    Returns:
    --------
    List[Data]
        List of graph data objects
    """
    return [lgn_to_graph(lgn, device) for lgn in lgns]


def batch_lgn_to_batched_graph(
    lgns: List[LGNState],
    device: Optional[torch.device] = None
) -> Tuple[Batch, List[int], List[int], List[bool]]:
    """
    Convert multiple LGN states to a single batched PyTorch Geometric Batch object.

    This enables efficient batched GNN forward passes by combining multiple
    graphs into one large disconnected graph.

    Parameters:
    -----------
    lgns : List[LGNState]
        List of LGN states to batch together
    device : torch.device
        Device to place tensors on (default: cpu)

    Returns:
    --------
    Tuple[Batch, List[int], List[int], List[bool]]
        - batch: PyTorch Geometric Batch object containing all graphs
                 Access individual graph assignments via batch.batch tensor
        - num_nodes_per_graph: List of node counts for each graph
        - num_inputs_per_graph: List of num_inputs for each graph
        - has_edges_per_graph: List of booleans indicating if each graph has edges

    Example:
    --------
    >>> lgns = [lgn1, lgn2, lgn3]  # 3 LGN states
    >>> batch, num_nodes_list, num_inputs_list, has_edges = batch_lgn_to_batched_graph(lgns, device)
    >>> # batch.x has shape [total_nodes, 1]
    >>> # batch.batch has shape [total_nodes] with values 0, 0, ..., 1, 1, ..., 2, 2, ...
    >>> # Use batch.batch to identify which graph each node belongs to
    >>> # has_edges[i] is True if lgns[i] has at least one gate (edge)

    Notes:
    ------
    - PyG's Batch.from_data_list() automatically handles node index remapping
    - Edge indices are automatically offset to refer to correct nodes in batched graph
    - Use batch.batch tensor to separate outputs per graph after forward pass
    - has_edges_per_graph is needed for correct handling of empty graphs in message passing
    """
    if device is None:
        device = torch.device('cpu')

    if len(lgns) == 0:
        # Return empty batch
        empty_data = Data(
            x=torch.empty((0, 1), dtype=torch.long, device=device),
            edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
            edge_attr=torch.empty((0, 2), dtype=torch.long, device=device),
        )
        return Batch.from_data_list([empty_data]), [], [], []

    # Convert each LGN to graph Data object
    graphs = []
    num_nodes_per_graph = []
    num_inputs_per_graph = []
    has_edges_per_graph = []

    for lgn in lgns:
        graph = lgn_to_graph(lgn, device)
        graphs.append(graph)
        num_nodes_per_graph.append(lgn.num_inputs + len(lgn.gates))
        num_inputs_per_graph.append(lgn.num_inputs)
        has_edges_per_graph.append(len(lgn.gates) > 0)  # Has edges if has gates

    # Create batched graph using PyTorch Geometric's Batch
    # This automatically:
    # - Concatenates node features
    # - Offsets edge indices appropriately
    # - Creates batch.batch tensor for graph membership
    batch = Batch.from_data_list(graphs)

    return batch, num_nodes_per_graph, num_inputs_per_graph, has_edges_per_graph
