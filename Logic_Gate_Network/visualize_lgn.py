"""
LGN Visualization Tool
======================

Visualize Logic Gate Networks as circuit diagrams.

Usage:
    python visualize_lgn.py --model experiments/trained_model.pt
"""

import torch
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
from pathlib import Path

from lgn.network import LGNState
from lgn.gates import GateType, apply_gate
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


def draw_gate_symbol(ax, x, y, gate_type, size=0.3):
    """
    Draw a logic gate symbol at position (x, y).
    Returns (width, height) of the gate.
    """
    # Colors for all 16 gate types
    gate_colors = {
        # Basic logic gates
        'AND': '#90EE90',    # Light green
        'OR': '#FFB6C1',     # Light pink
        'XOR': '#FFD700',    # Gold
        'NAND': '#98FB98',   # Pale green
        'NOR': '#FFA07A',    # Light salmon
        'XNOR': '#F0E68C',   # Khaki
        # Unary gates
        'NOT': '#DDA0DD',    # Plum
        'BUFFER': '#E6E6FA', # Lavender
        # Implication gates
        'IMPLY': '#87CEFA',           # Light sky blue
        'NIMPLY': '#B0C4DE',          # Light steel blue
        'CONVERSE_IMPLY': '#ADD8E6',  # Light blue
        'CONVERSE_NIMPLY': '#5F9EA0', # Cadet blue
        # Projection gates
        'FIRST': '#F5DEB3',   # Wheat
        'SECOND': '#DEB887',  # Burlywood
        'NFIRST': '#D2B48C',  # Tan
        'NSECOND': '#BC8F8F', # Rosy brown
    }

    color = gate_colors.get(gate_type, '#D3D3D3')

    # Rectangular gate with label
    w, h = size * 1.2, size * 0.8
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # Add NOT bubble for negated gates
    if gate_type in ['NAND', 'NOR', 'XNOR', 'NOT', 'NIMPLY', 'CONVERSE_NIMPLY', 'NFIRST', 'NSECOND']:
        bubble = Circle((x + w/2 + 0.03, y), 0.03, facecolor='white', edgecolor='black', linewidth=1.5)
        ax.add_patch(bubble)

    # Gate label
    short_name = gate_type.replace('CONVERSE_', 'C_')
    ax.text(x, y, short_name, ha='center', va='center', fontsize=8, fontweight='bold')

    return w, h


def visualize_lgn(lgn, save_path=None, title="Logic Gate Network", sample_data=None, return_fig_only=False):
    """
    Visualize an LGN as a circuit diagram with orthogonal wiring.

    Layout: Left-to-right signal flow
    - Inputs on the left
    - Gates arranged in layers (by dependency depth)
    - Output on the right
    - Orthogonal (right-angle) wiring

    Args:
        lgn: LGNState to visualize
        save_path: Optional path to save figure
        title: Figure title
        sample_data: Optional list of input values to show execution
    """
    num_inputs = lgn.num_inputs
    num_gates = lgn.get_num_gates()

    # Compute node values if sample data provided
    node_values = {}
    if sample_data is not None:
        for i in range(num_inputs):
            node_values[i] = int(sample_data[i])
        for gate_idx, gate in enumerate(lgn.gates):
            inputs = []
            for inp_idx in gate.inputs:
                if inp_idx < num_inputs:
                    inputs.append(int(sample_data[inp_idx]))
                else:
                    inputs.append(node_values[inp_idx])
            gate_output = apply_gate(gate.gate_type, inputs)
            node_values[num_inputs + gate_idx] = int(gate_output)

    # Compute gate layers (depth from inputs)
    gate_layers = []
    for gate_idx, gate in enumerate(lgn.gates):
        max_input_layer = 0
        for input_idx in gate.inputs:
            if input_idx >= num_inputs:
                source_gate_idx = input_idx - num_inputs
                if source_gate_idx < len(gate_layers):
                    max_input_layer = max(max_input_layer, gate_layers[source_gate_idx])
        gate_layers.append(max_input_layer + 1)

    max_layer = max(gate_layers) if gate_layers else 0

    # Find root gates (outputs not used by other gates)
    root_gates = lgn.get_root_gates() if hasattr(lgn, 'get_root_gates') else []

    # Find which inputs are actually used by gates
    used_inputs = set()
    for gate in lgn.gates:
        for inp_idx in gate.inputs:
            if inp_idx < num_inputs:
                used_inputs.add(inp_idx)

    # Create figure
    fig_width = max(10, 3 + max_layer * 2.5)
    fig_height = max(6, num_inputs * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Layout parameters
    input_x = 0.5
    gate_start_x = 2.0
    layer_spacing = 2.0
    gate_size = 0.4

    # Position storage: node_id -> (x, y, output_x)
    node_positions = {}

    # Draw input nodes (left side, vertically distributed)
    input_y_start = fig_height - 1.0
    input_spacing = (fig_height - 2.0) / max(num_inputs - 1, 1) if num_inputs > 1 else 0

    for i in range(num_inputs):
        y = input_y_start - i * input_spacing
        is_used = i in used_inputs

        # Input circle (gray if unused)
        color = '#87CEEB' if is_used else '#D3D3D3'
        circle = Circle((input_x, y), 0.15, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)

        # Label
        label = f"x{i}"
        if i in node_values:
            label += f"={node_values[i]}"
        text_color = 'black' if is_used else 'gray'
        ax.text(input_x - 0.35, y, label, ha='right', va='center', fontsize=10, fontweight='bold', color=text_color)

        # Add "unused" label for unused inputs
        if not is_used:
            ax.text(input_x + 0.5, y, "(unused)", ha='left', va='center', fontsize=8, color='gray', style='italic')

        # Horizontal wire from input (only if used)
        wire_end_x = input_x + 0.15
        if is_used:
            ax.plot([input_x + 0.15, wire_end_x + 0.3], [y, y], 'k-', linewidth=1.5)
        else:
            ax.plot([input_x + 0.15, wire_end_x + 0.2], [y, y], color='gray', linewidth=1, linestyle='--')

        node_positions[i] = (input_x, y, wire_end_x + 0.3)

    # Group gates by layer
    layers_dict = {}
    for gate_idx, layer in enumerate(gate_layers):
        if layer not in layers_dict:
            layers_dict[layer] = []
        layers_dict[layer].append(gate_idx)

    # Draw gates layer by layer (left to right)
    for layer_idx in sorted(layers_dict.keys()):
        gates_in_layer = layers_dict[layer_idx]
        layer_x = gate_start_x + (layer_idx - 1) * layer_spacing

        # Position gates based on their input connections (average y of inputs)
        gate_y_positions = []
        for gate_idx in gates_in_layer:
            gate = lgn.gates[gate_idx]
            # Calculate average y position of inputs
            input_ys = []
            for inp_idx in gate.inputs:
                if inp_idx in node_positions:
                    _, inp_y, _ = node_positions[inp_idx]
                    input_ys.append(inp_y)
            if input_ys:
                avg_y = sum(input_ys) / len(input_ys)
            else:
                avg_y = fig_height / 2
            gate_y_positions.append((gate_idx, avg_y))

        # Sort by y position to avoid crossings, then spread if overlapping
        gate_y_positions.sort(key=lambda x: -x[1])  # Sort by y (top to bottom)

        # Spread gates if they're too close
        min_spacing = gate_size * 1.5
        for i in range(1, len(gate_y_positions)):
            prev_y = gate_y_positions[i-1][1]
            curr_y = gate_y_positions[i][1]
            if prev_y - curr_y < min_spacing:
                gate_y_positions[i] = (gate_y_positions[i][0], prev_y - min_spacing)

        for gate_idx, y in gate_y_positions:
            gate = lgn.gates[gate_idx]
            gate_id = num_inputs + gate_idx

            # Draw gate symbol
            w, h = draw_gate_symbol(ax, layer_x, y, gate.gate_type.name, size=gate_size)

            # Gate output value
            if gate_id in node_values:
                ax.text(layer_x + w/2 + 0.15, y + 0.15, f"={node_values[gate_id]}",
                       fontsize=8, color='blue')

            # Mark root gates
            if gate_idx in root_gates:
                ax.text(layer_x, y - h/2 - 0.15, "ROOT", ha='center', fontsize=7,
                       color='red', fontweight='bold')

            # Store position (input side, center y, output side)
            node_positions[gate_id] = (layer_x - w/2, y, layer_x + w/2 + 0.06)

            # Draw input wires with orthogonal routing
            num_gate_inputs = len(gate.inputs)
            input_y_offsets = []
            if num_gate_inputs == 1:
                input_y_offsets = [0]
            elif num_gate_inputs == 2:
                input_y_offsets = [h/4, -h/4]
            else:
                for j in range(num_gate_inputs):
                    offset = (j - (num_gate_inputs - 1) / 2) * (h / (num_gate_inputs + 1))
                    input_y_offsets.append(offset)

            for j, inp_idx in enumerate(gate.inputs):
                src_x, src_y, src_out_x = node_positions[inp_idx]
                dst_x = layer_x - w/2
                dst_y = y + input_y_offsets[j]

                # Wire styling
                wire_color = '#333333'
                wire_width = 1.5

                if abs(src_y - dst_y) < 0.05:
                    # Nearly horizontal - draw direct line
                    ax.plot([src_out_x, dst_x], [src_y, dst_y], color=wire_color, linewidth=wire_width)
                else:
                    # Orthogonal routing: horizontal, then vertical, then horizontal
                    mid_x = (src_out_x + dst_x) / 2
                    ax.plot([src_out_x, mid_x], [src_y, src_y], color=wire_color, linewidth=wire_width)
                    ax.plot([mid_x, mid_x], [src_y, dst_y], color=wire_color, linewidth=wire_width)
                    ax.plot([mid_x, dst_x], [dst_y, dst_y], color=wire_color, linewidth=wire_width)

                # Connection dot at gate input
                ax.plot(dst_x, dst_y, 'ko', markersize=3)

    # Draw output wires from root gates
    if root_gates:
        output_x = gate_start_x + max_layer * layer_spacing + 0.5

        # If multiple root gates, show AND combination (virtual gate - not part of learned network)
        if len(root_gates) > 1:
            # Draw AND gate for combining root outputs (gray dashed border to indicate virtual)
            and_x = output_x
            and_y = fig_height / 2
            w, h = draw_gate_symbol(ax, and_x, and_y, 'AND', size=gate_size)
            # Add gray dashed border overlay to indicate virtual gate
            virtual_rect = FancyBboxPatch(
                (and_x - w/2, and_y - h/2), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor='none', edgecolor='gray', linewidth=2, linestyle='--'
            )
            ax.add_patch(virtual_rect)

            # Connect root gates to AND gate (gray dashed lines for virtual connections)
            for i, root_idx in enumerate(root_gates):
                gate_id = num_inputs + root_idx
                src_x, src_y, src_out_x = node_positions[gate_id]
                dst_x = and_x - w/2
                dst_y = and_y + (i - (len(root_gates) - 1) / 2) * (h / (len(root_gates) + 1))

                mid_x = (src_out_x + dst_x) / 2
                ax.plot([src_out_x, mid_x], [src_y, src_y], color='gray', linestyle='--', linewidth=1.5)
                ax.plot([mid_x, mid_x], [src_y, dst_y], color='gray', linestyle='--', linewidth=1.5)
                ax.plot([mid_x, dst_x], [dst_y, dst_y], color='gray', linestyle='--', linewidth=1.5)

            # Output wire (gray dashed)
            ax.plot([and_x + w/2 + 0.06, and_x + w/2 + 0.5], [and_y, and_y], color='gray', linestyle='--', linewidth=2)
            ax.text(and_x + w/2 + 0.6, and_y, "OUT", ha='left', va='center',
                   fontsize=10, fontweight='bold', color='gray')

            # Show combined output value
            if sample_data is not None:
                root_outputs = [node_values[num_inputs + r] for r in root_gates]
                final_output = int(all(root_outputs))
                ax.text(and_x + w/2 + 0.6, and_y - 0.25, f"={final_output}",
                       fontsize=10, color='blue', fontweight='bold')
        else:
            # Single root gate
            root_idx = root_gates[0]
            gate_id = num_inputs + root_idx
            src_x, src_y, src_out_x = node_positions[gate_id]
            ax.plot([src_out_x, src_out_x + 0.5], [src_y, src_y], 'k-', linewidth=2)
            ax.text(src_out_x + 0.6, src_y, "OUT", ha='left', va='center',
                   fontsize=10, fontweight='bold')

            if gate_id in node_values:
                ax.text(src_out_x + 0.6, src_y - 0.25, f"={node_values[gate_id]}",
                       fontsize=10, color='blue', fontweight='bold')

    # Legend - show only gates that are actually used in this LGN
    used_gate_types = set(gate.gate_type.name for gate in lgn.gates)

    # Always include Input
    legend_elements = [mpatches.Patch(color='#87CEEB', label='Input')]

    # Gate colors for legend (same as draw_gate_symbol)
    legend_gate_colors = {
        'AND': '#90EE90', 'OR': '#FFB6C1', 'XOR': '#FFD700',
        'NAND': '#98FB98', 'NOR': '#FFA07A', 'XNOR': '#F0E68C',
        'NOT': '#DDA0DD', 'BUFFER': '#E6E6FA',
        'IMPLY': '#87CEFA', 'NIMPLY': '#B0C4DE',
        'CONVERSE_IMPLY': '#ADD8E6', 'CONVERSE_NIMPLY': '#5F9EA0',
        'FIRST': '#F5DEB3', 'SECOND': '#DEB887',
        'NFIRST': '#D2B48C', 'NSECOND': '#BC8F8F',
    }

    # Add only used gate types to legend
    for gate_name in ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR', 'NOT', 'BUFFER',
                      'IMPLY', 'NIMPLY', 'CONVERSE_IMPLY', 'CONVERSE_NIMPLY',
                      'FIRST', 'SECOND', 'NFIRST', 'NSECOND']:
        if gate_name in used_gate_types:
            legend_elements.append(mpatches.Patch(color=legend_gate_colors[gate_name], label=gate_name))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Title and formatting
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim(-0.5, fig_width - 0.5)
    ax.set_ylim(-0.5, fig_height + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()

    if return_fig_only:
        # Return figure without saving or showing (for wandb logging)
        return fig

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Circuit diagram saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


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
