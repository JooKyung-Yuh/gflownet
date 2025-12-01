import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from lgn.gates import GateType, GATE_TYPE_TO_IDX

class JointLGNPolicy(nn.Module):
    def __init__(self, nemb, num_gate_types=16):
        super().__init__()
        # Standard GNN Backbone (TransformerConv is recommended)
        self.convs = nn.ModuleList([
            gnn.TransformerConv(nemb, nemb // 4, heads=4, concat=True)
            for _ in range(3)
        ])
        
        # THE JOINT HEAD (Professor's suggestion)
        # Projects every node to 16 scores (one per gate type)
        # We perform NO global pooling here. We keep the node dimension.
        self.joint_head = nn.Sequential(
            nn.Linear(nemb, nemb), nn.LeakyReLU(),
            nn.Linear(nemb, num_gate_types) 
        )

        # Stop head (Global)
        self.stop_head = nn.Linear(nemb, 1)

    def forward(self, data):
        # 1. GNN Backbone
        x = data.x
        for conv in self.convs:
            x = conv(x, data.edge_index)
        
        # 2. The Dynamic Table (Nodes x 16)
        # x shape: [Num_Nodes_In_Batch, nemb]
        # logits shape: [Num_Nodes_In_Batch, 16]
        # This grows automatically with data.x!
        node_gate_logits = self.joint_head(x)
        
        # 3. Stop Logit (Global)
        global_rep = gnn.global_mean_pool(x, data.batch)
        stop_logit = self.stop_head(global_rep)
        
        return stop_logit, node_gate_logits

    def get_action_dist(self, stop_logit, node_gate_logits):
        """
        Converts the dynamic table into distributions we can sample from.
        """
        # A. Gate Distribution P(GateType)
        # Heuristic: Sum the 'preference' of all nodes for each gate type
        # "How much does the graph want an AND gate vs OR gate?"
        gate_scores = torch.logsumexp(node_gate_logits, dim=0) # Shape [16]
        
        # B. Node Distribution P(Node | GateType)
        # We don't materialize this yet, we just slice the matrix when needed.
        return gate_scores, node_gate_logits

    def sample_action(self, lgn_state, stop_logit, node_gate_logits):
        # 1. Decide Stop
        # (Bernoulli sample on stop_logit...)

        # 2. Sample Gate Type
        gate_scores = torch.logsumexp(node_gate_logits, dim=0)
        gate_type_idx = torch.distributions.Categorical(logits=gate_scores).sample()
        gate_type_enum = list(GATE_TYPE_TO_IDX.keys())[gate_type_idx]

        # 3. Sample Inputs (The "Sample 2" Default)
        # We look at the specific column for the chosen gate type
        # Shape: [Num_Nodes] <-- Variable size!
        col_logits = node_gate_logits[:, gate_type_idx]
        
        # Sample Input A
        probs_a = torch.softmax(col_logits, dim=0)
        idx_a = torch.distributions.Categorical(probs_a).sample()
        
        # Sample Input B (Autoregressive)
        # Mask A to avoid duplicates
        col_logits_b = col_logits.clone()
        col_logits_b[idx_a] = float('-inf')
        probs_b = torch.softmax(col_logits_b, dim=0)
        idx_b = torch.distributions.Categorical(probs_b).sample()

        # 4. The "Discard" Logic
        inputs = [idx_a.item(), idx_b.item()]
        
        if gate_type_enum in [GateType.NOT, GateType.BUFFER]:
            # Discard the second sampled input
            inputs = [inputs[0]]
            
        return {"gate_type": gate_type_enum, "input_indices": tuple(inputs)}