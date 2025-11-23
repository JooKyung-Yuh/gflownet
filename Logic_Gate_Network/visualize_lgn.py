"""
LGN Visualization Tool
======================

Visualize Logic Gate Networks sampled from trained GFlowNet models.

Usage:
    python visualize_lgn.py --model experiments/trained_model.pt
"""

import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    print("Error: networkx not installed. Install with: pip install networkx")
    exit(1)

from lgn.network import LGNState
from lgn.gates import GateType
from gflownet.policy_network_gnn import LGNGNNPolicy
from gflownet.action_space import LGNActionSpace
from reward.reward_fn import RewardFunction
from data.generator import RealDataGenerator, FakeDataGenerator
from rules.rule_1 import Rule1_NoConsecutive1s
import numpy as np


def sample_lgn_greedy(policy, action_space, num_inputs, max_gates):
    """Sample LGN using greedy policy."""
    lgn = LGNState(num_inputs=num_inputs, max_gates=max_gates)

    with torch.no_grad():
        for step in range(max_gates + 10):
            valid_actions = action_space.get_valid_actions(lgn)
            if not valid_actions:
                break

            gate_actions = [a for a in valid_actions if 'gate_type' in a]
            if not gate_actions:
                break

            action_q, stop_q = policy.forward_policy(lgn, gate_actions)

            if action_q.max() > stop_q:
                best_idx = action_q.argmax().item()
                best_action = gate_actions[best_idx]
                lgn.add_gate(best_action['gate_type'], best_action['input_indices'])
            else:
                break

    return lgn


def visualize_lgn(lgn, save_path=None, title="Logic Gate Network", sample_data=None):
    """
    Visualize an LGN as a tree-like directed graph.

    Args:
        lgn: LGNState to visualize
        save_path: Optional path to save figure
        title: Figure title
        sample_data: Optional numpy array of input values to show execution
    """
    G = nx.DiGraph()

    num_inputs = lgn.num_inputs
    num_gates = lgn.get_num_gates()

    # Compute node values if sample data provided
    node_values = {}
    if sample_data is not None:
        from lgn.gates import apply_gate

        # Store input values
        for i in range(num_inputs):
            node_values[f"in{i}"] = int(sample_data[i])

        # Compute gate outputs step by step
        for gate_idx, gate in enumerate(lgn.gates):
            # Get input values for this gate
            inputs = []
            for inp_idx in gate.inputs:
                if inp_idx < num_inputs:
                    inputs.append(int(sample_data[inp_idx]))
                else:
                    source_gate_idx = inp_idx - num_inputs
                    inputs.append(node_values[f"g{source_gate_idx}"])

            # Compute gate output using apply_gate function
            gate_output = apply_gate(gate.gate_type, inputs)
            node_values[f"g{gate_idx}"] = int(gate_output)

    # Add input nodes
    for i in range(num_inputs):
        G.add_node(f"in{i}", node_type='input', layer=0)

    # Add gate nodes and compute layers
    gate_layers = []
    for gate_idx, gate in enumerate(lgn.gates):
        gate_id = num_inputs + gate_idx

        # Compute layer based on dependencies
        max_input_layer = 0
        for input_idx in gate.inputs:
            if input_idx >= num_inputs:
                source_gate_idx = input_idx - num_inputs
                if source_gate_idx < len(gate_layers):
                    max_input_layer = max(max_input_layer, gate_layers[source_gate_idx])

        gate_layer = max_input_layer + 1
        gate_layers.append(gate_layer)

        G.add_node(f"g{gate_idx}", node_type='gate', gate_type=gate.gate_type.name,
                   layer=gate_layer, actual_id=gate_id)

        # Add edges from inputs to gate
        for input_idx in gate.inputs:
            if input_idx < num_inputs:
                G.add_edge(f"in{input_idx}", f"g{gate_idx}")
            else:
                # Input from another gate
                source_gate_idx = input_idx - num_inputs
                G.add_edge(f"g{source_gate_idx}", f"g{gate_idx}")

    # Define node colors
    color_map = {
        'input': '#87CEEB',  # Sky blue
        'AND': '#90EE90',    # Light green
        'OR': '#FFB6C1',     # Light pink
        'XOR': '#FFD700',    # Gold
        'NAND': '#98FB98',   # Pale green
        'NOR': '#FFA07A',    # Light salmon
        'XNOR': '#F0E68C',   # Khaki
        'NOT': '#DDA0DD',    # Plum
        'IMPLY': '#87CEFA',  # Light sky blue
        'NIMPLY': '#B0C4DE', # Light steel blue
        'CONVERSE_NIMPLY': '#ADD8E6',  # Light blue
        'default': '#D3D3D3' # Light gray
    }

    node_colors = []
    node_labels = {}

    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data['node_type'] == 'input':
            node_colors.append(color_map['input'])
            label = node.replace('in', 'x')
            if node in node_values:
                label += f"={node_values[node]}"
            node_labels[node] = label
        else:
            gate_type = node_data['gate_type']
            node_colors.append(color_map.get(gate_type, color_map['default']))
            # Shorten gate type names for readability
            short_name = gate_type.replace('CONVERSE_', 'C_').replace('NIMPLY', 'NIM')
            if node in node_values:
                short_name += f"\n={node_values[node]}"
            node_labels[node] = short_name

    # Hierarchical tree layout
    pos = {}

    # Group nodes by layer
    layers = {}
    for node in G.nodes():
        layer = G.nodes[node]['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)

    # Position nodes in tree layout
    max_layer = max(layers.keys())
    layer_height = 1.0 / (max_layer + 1)

    for layer_idx, nodes in layers.items():
        y = 1.0 - layer_idx * layer_height  # Top to bottom
        num_nodes = len(nodes)

        # Sort nodes for consistent layout
        nodes_sorted = sorted(nodes)

        for i, node in enumerate(nodes_sorted):
            x = (i + 1) / (num_nodes + 1)  # Evenly spaced
            pos[node] = (x, y)

    # Draw
    plt.figure(figsize=(12, 8))

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=1500, alpha=0.9)
    nx.draw_networkx_labels(G, pos, node_labels, font_size=10,
                            font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                           arrowsize=20, arrowstyle='->', width=2)

    # Legend
    legend_elements = [
        mpatches.Patch(color=color_map['input'], label='Input'),
        mpatches.Patch(color=color_map['AND'], label='AND'),
        mpatches.Patch(color=color_map['OR'], label='OR'),
        mpatches.Patch(color=color_map['XOR'], label='XOR'),
        mpatches.Patch(color=color_map['NOT'], label='NOT'),
        mpatches.Patch(color=color_map['NAND'], label='NAND'),
        mpatches.Patch(color=color_map['NOR'], label='NOR'),
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize LGN from trained model')
    parser.add_argument('--model', type=str, default='experiments/trained_model.pt',
                        help='Path to trained model')
    parser.add_argument('--num-inputs', type=int, default=6,
                        help='Number of inputs (must match training)')
    parser.add_argument('--max-gates', type=int, default=3,
                        help='Maximum gates (must match training)')
    parser.add_argument('--node-emb-dim', type=int, default=64,
                        help='Node embedding dimension (must match training)')
    parser.add_argument('--num-conv-steps', type=int, default=2,
                        help='Number of GNN convolution steps (must match training)')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of LGNs to visualize')
    parser.add_argument('--save-dir', type=str, default='experiments/visualizations',
                        help='Directory to save visualizations')

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print("\nPlease train a model first:")
        print("  python train_with_real_data.py --num-inputs 6 --max-gates 3 --iterations 30")
        return

    print("="*80)
    print("LGN Visualization")
    print("="*80)

    # Load model
    print(f"\nLoading model from {args.model}...")
    policy = LGNGNNPolicy(num_inputs=args.num_inputs, max_gates=args.max_gates,
                         node_emb_dim=args.node_emb_dim, num_conv_steps=args.num_conv_steps)
    policy.load_state_dict(torch.load(args.model))
    policy.eval()
    print("✅ Model loaded")

    # Create components
    action_space = LGNActionSpace(num_inputs=args.num_inputs, max_gates=args.max_gates)

    # Generate test data for evaluation
    print(f"\nGenerating test data...")
    rule = Rule1_NoConsecutive1s(dimension=args.num_inputs)
    real_gen = RealDataGenerator()
    fake_gen = FakeDataGenerator()
    real_samples = real_gen.generate(rule, count=20)
    fake_samples = fake_gen.generate(rule, real_samples, count=20)
    reward_fn = RewardFunction()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Sample and visualize LGNs
    print(f"\nSampling {args.num_samples} LGNs...")

    for i in range(args.num_samples):
        print(f"\nLGN {i+1}/{args.num_samples}:")

        # Sample LGN
        lgn = sample_lgn_greedy(policy, action_space, args.num_inputs, args.max_gates)

        # Compute metrics
        log_reward = reward_fn.compute_reward(lgn, real_samples, fake_samples)
        reward = np.exp(log_reward)
        real_errors = reward_fn.compute_real_error_count(lgn, real_samples)
        fake_accepts = reward_fn.compute_fake_acceptance_count(lgn, fake_samples)
        real_acc = 1.0 - (real_errors / len(real_samples))
        fake_acc = 1.0 - (fake_accepts / len(fake_samples))

        print(f"  Gates: {lgn.get_num_gates()}")
        print(f"  Real Accuracy: {real_acc:.2%}")
        print(f"  Fake Accuracy: {fake_acc:.2%}")
        print(f"  Log-Reward: {log_reward:.4f}")
        print(f"  Reward: {reward:.2e}")

        # Print gate structure
        print(f"  Structure:")
        for gate_idx, gate in enumerate(lgn.gates):
            input_names = []
            for inp in gate.inputs:
                if inp < args.num_inputs:
                    input_names.append(f"x{inp}")
                else:
                    input_names.append(f"g{inp - args.num_inputs}")
            print(f"    g{gate_idx} = {gate.gate_type.name}({', '.join(input_names)})")

        # Visualize with sample data
        sample = real_samples[0] if len(real_samples) > 0 else None
        title = f"LGN {i+1}: {lgn.get_num_gates()} gates, Real={real_acc:.0%}, Fake={fake_acc:.0%}"
        if sample is not None:
            sample_str = ''.join(str(int(x)) for x in sample)
            title += f" | Input={sample_str}"
        save_path = save_dir / f"lgn_{i+1}.png"
        visualize_lgn(lgn, save_path=save_path, title=title, sample_data=sample)

    print(f"\n✅ Visualization complete!")
    print(f"   Saved {args.num_samples} figures to {save_dir}/")


if __name__ == "__main__":
    main()
