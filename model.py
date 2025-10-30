# model.py
# Defines the GCN model architecture using PyTorch Geometric (PyG).

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

import config

def create_skeleton_graph():
    """
    Creates the static graph connectivity (edge_index) for the 19-joint skeleton.
    Returns edge_index (torch.LongTensor) for PyG, shape [2, num_edges].
    """
    # This 17-edge graph connects 19 joints (0-18).
    # Based on the bone list from the scaffolding.
    bone_list = [
        [0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7],
        [8, 9], [8, 12], [9, 10], [10, 11], [11, 17], [11, 18],
        [12, 13], [13, 14], [14, 15]
    ]
    
    edge_index_list = []
    for bone in bone_list:
        edge_index_list.append([bone[0], bone[1]]) # Edge
        edge_index_list.append([bone[1], bone[0]]) # Reverse Edge
        
    # Add self-loops for every joint (0-18)
    # GCNs perform better when a node's features are included in its
    # own updated representation.
    for i in range(config.NUM_JOINTS): # NUM_JOINTS is 19
         edge_index_list.append([i, i])
        
    # Convert to the shape PyG expects: [2, num_edges]
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    
    return edge_index

class GCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.gcn = GCNConv(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class DualBranchGCN(nn.Module):
    """
    A dual-branch GCN with a shared backbone, a classifier branch for mistake class prediction,
    and a corrector branch for pose correction.
    """
    def __init__(self, in_channels, hidden_channels, out_channels_class, out_channels_corr, num_blocks, dropout):
        super().__init__()
        # --- Shared Backbone (Stack of GCN blocks) ---
        self.backbone = nn.ModuleList()
        self.backbone.append(GCNBlock(in_channels, hidden_channels, dropout))
        for _ in range(num_blocks - 1):
            self.backbone.append(GCNBlock(hidden_channels, hidden_channels, dropout))

        # --- Classifier Branch ---
        self.class_head_gcn = GCNConv(hidden_channels, hidden_channels)
        self.class_pool = global_mean_pool
        self.classifier = nn.Linear(hidden_channels, out_channels_class)

        # --- Corrector Branch ---
        self.corr_head_gcn = GCNConv(hidden_channels, hidden_channels)
        self.feedback_linear = nn.Linear(out_channels_class, hidden_channels)
        self.corr_fuse = nn.Linear(hidden_channels * 2, hidden_channels)
        self.corrector = nn.Linear(hidden_channels, out_channels_corr)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None, labels=None, use_feedback=False):
        # x: [batch_size, num_nodes, num_frames]. We'll reshape for GCN usage.
        B, N, F = x.shape
        x = x.reshape(B * N, F)  # treat frames as features per node
        batch_vec = torch.arange(B, device=x.device).repeat_interleave(N) if batch is None else batch
        # --- Shared Backbone ---
        for blk in self.backbone:
            x = blk(x, edge_index)
        # x: [B*N, hidden_channels]

        # --- Classifier Branch ---
        class_branch = self.class_head_gcn(x, edge_index)
        # Pool to get [B, hidden_channels]
        class_branch_pooled = self.class_pool(class_branch, batch_vec)  # [B, hidden_channels]
        pred_logits = self.classifier(class_branch_pooled)  # [B, out_channels_class]

        # --- Corrector Branch ---
        corr_branch = self.corr_head_gcn(x, edge_index)  # [B*N, hidden_channels]
        # Broadcast feedback features, concatenate
        if use_feedback:
            if labels is None:
                # Softmax prediction feedback
                feedback_vec = torch.softmax(pred_logits.detach(), dim=1)
            else:
                feedback_vec = nn.functional.one_hot(labels, num_classes=pred_logits.shape[1]).float()
            feedback_feat = self.feedback_linear(feedback_vec)  # [B, hidden_channels]
            feedback_feat = feedback_feat.repeat_interleave(N, dim=0)  # [B*N, hidden_channels]
            corr_feat = torch.cat([corr_branch, feedback_feat], dim=1)  # [B*N, 2*hidden_channels]
        else:
            corr_feat = torch.cat([corr_branch, torch.zeros_like(corr_branch)], dim=1)  # [B*N, 2*hidden_channels]
        corr_fused = self.relu(self.corr_fuse(corr_feat))
        corr_out = self.corrector(self.dropout(corr_fused))  # [B*N, out_channels_corr]
        pred_corrected_pose = corr_out.view(B, N, -1)  # [B, N, F_corr]
        return pred_logits, pred_corrected_pose