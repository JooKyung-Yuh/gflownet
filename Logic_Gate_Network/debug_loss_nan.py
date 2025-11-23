"""
Debug script to identify source of NaN in training loss.

This script runs a single training step with detailed logging
to pinpoint where NaN values first appear.
"""

import torch
import numpy as np
from lgn.network import LGNState
from gflownet.policy_network_gnn import LGNGNNPolicy
from gflownet.training import LGNTrainer
from gflownet.lgn_mdp import LGNMDP
from gflownet.action_space import LGNActionSpace
from reward.reward_fn import RewardFunction
from data.generator import RealDataGenerator, FakeDataGenerator
from rules.rule_1 import Rule1_NoConsecutive1s

def check_nan(tensor, name):
    """Check if tensor contains NaN or Inf."""
    if torch.is_tensor(tensor):
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            print(f"  ❌ {name}: NaN={has_nan}, Inf={has_inf}")
            print(f"     Values: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")
            return True
        else:
            print(f"  ✅ {name}: OK (min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f})")
            return False
    else:
        print(f"  ℹ️  {name}: {tensor} (scalar)")
        return False

def debug_loss_computation(trainer, batch_size=2):
    """
    Run one training step with detailed logging to find NaN source.
    """
    print("\n" + "="*80)
    print("DEBUGGING LOSS NaN - DETAILED TRACE")
    print("="*80)

    # Step 1: Sample batch
    print("\n[1/4] Sampling batch...")
    p_list, pb, a_list, r, s_list, d = trainer.sample_batch(batch_size)

    print(f"  Batch size: {batch_size}")
    print(f"  Num parent states: {len(p_list)}")
    print(f"  Num resulting states: {len(s_list)}")
    print(f"  Num transitions: {len(r)}")

    check_nan(r, "Rewards (r)")
    check_nan(d, "Done flags (d)")

    # Step 2: Compute Q-values for parents
    print("\n[2/4] Computing Q-values for parent states...")
    qsa_p_list = []
    for i, (parent, action) in enumerate(zip(p_list, a_list)):
        qsa = trainer.policy.compute_q_value_for_action(parent, action)
        qsa_p_list.append(qsa)
        if i < 3:  # Show first 3
            print(f"  Q(parent_{i}, action_{i}) = {qsa.item():.4f}")

    qsa_p = torch.stack(qsa_p_list)
    check_nan(qsa_p, "Q(parent, action)")

    # Step 3: Compute inflow
    print("\n[3/4] Computing inflow = log(sum exp(Q(parent, a)))...")

    exp_qsa_p = torch.exp(qsa_p)
    check_nan(exp_qsa_p, "exp(Q(parent, action))")

    ntransitions = len(s_list)
    exp_inflow = torch.zeros(ntransitions, dtype=torch.float32, device=trainer.device)
    exp_inflow = exp_inflow.index_add_(0, pb, exp_qsa_p)
    check_nan(exp_inflow, "sum exp(Q) per state")

    inflow = torch.log(exp_inflow + trainer.log_reg_c)
    check_nan(inflow, "inflow = log(sum exp(Q) + reg)")

    # Step 4: Compute outflow
    print("\n[4/4] Computing outflow = log(R + sum exp(Q(s, a')))...")

    exp_outflow_list = []
    for i, s in enumerate(s_list):
        valid_actions = trainer.action_space.get_valid_actions(s)
        exp_sum = trainer.policy.sum_exp_q_values(s, valid_actions)
        exp_outflow_list.append(exp_sum)
        if i < 3:
            print(f"  State {i}: sum exp(Q(s, a')) = {exp_sum.item():.4f}, num_actions={len(valid_actions)}")

    exp_outflow = torch.stack(exp_outflow_list)
    check_nan(exp_outflow, "sum exp(Q(s, a')) per state")

    outflow_plus_r = torch.log(trainer.log_reg_c + r + exp_outflow * (1 - d))
    check_nan(outflow_plus_r, "outflow = log(reg + R + sum exp(Q) * (1-done))")

    # Step 5: Compute loss
    print("\n[5/5] Computing loss...")

    losses = (inflow - outflow_plus_r).pow(2)
    check_nan(losses, "squared differences (inflow - outflow)^2")

    if trainer.balanced_loss:
        term_loss = (losses * d).sum() / (d.sum() + 1e-20)
        flow_loss = (losses * (1 - d)).sum() / ((1 - d).sum() + 1e-20)
        loss = term_loss * trainer.leaf_coef + flow_loss

        check_nan(term_loss, "terminal loss")
        check_nan(flow_loss, "flow loss")
        check_nan(loss, "total loss")
    else:
        loss = losses.mean()
        check_nan(loss, "mean loss")

    print("\n" + "="*80)
    print("DEBUG TRACE COMPLETE")
    print("="*80)

    return loss


def main():
    print("Setting up GFlowNet components...")

    # Small parameters for faster debugging
    num_inputs = 6
    max_gates = 3

    # Generate minimal data
    rule = Rule1_NoConsecutive1s(dimension=num_inputs)
    real_gen = RealDataGenerator()
    fake_gen = FakeDataGenerator()

    print(f"Generating data (dimension={num_inputs})...")
    real_samples = real_gen.generate(rule, count=20)
    fake_samples = fake_gen.generate(rule, real_samples, count=20)

    # Create GFlowNet components
    policy = LGNGNNPolicy(num_inputs=num_inputs, max_gates=max_gates, node_emb_dim=32, num_conv_steps=1)
    mdp = LGNMDP(num_inputs=num_inputs, max_gates=max_gates)
    action_space = LGNActionSpace(num_inputs=num_inputs, max_gates=max_gates)

    reward_fn = RewardFunction()
    def compute_reward(lgn):
        return reward_fn.compute_reward(lgn, real_samples, fake_samples)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    trainer = LGNTrainer(
        policy=policy,
        mdp=mdp,
        action_space=action_space,
        reward_fn=compute_reward,
        optimizer=optimizer,
        device=torch.device('cpu'),
        balanced_loss=True,
        leaf_coef=10.0
    )

    print(f"\nTrainer created:")
    print(f"  log_reg_c = {trainer.log_reg_c}")
    print(f"  leaf_coef = {trainer.leaf_coef}")
    print(f"  clip_grad = {trainer.clip_grad}")

    # Run debug trace
    loss = debug_loss_computation(trainer, batch_size=2)

    print(f"\n\nFinal Loss: {loss.item():.4f}")

    if torch.isnan(loss):
        print("\n❌ LOSS IS NaN - Check the trace above to see where it first appears")
    else:
        print("\n✅ LOSS IS OK - No NaN detected!")


if __name__ == "__main__":
    main()
