"""
GCN model implementation.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import BaseGNNLayer, BaseGNNModel, BaseMLP
from registry import MODEL_REGISTRY


class GCNLayer(BaseGNNLayer):
    """Standard graph convolution with symmetric normalization."""

    def __init__(self, in_feature: int, out_feature: int):
        super().__init__(in_feature, out_feature)
        self.linear = nn.Linear(in_feature, out_feature)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_with_self_loop = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        degree = torch.sum(adj_with_self_loop, dim=1)
        degree_inv_sqrt = torch.pow(degree.clamp_min(1e-8), -0.5)
        norm_adj = degree_inv_sqrt.unsqueeze(1) * adj_with_self_loop * degree_inv_sqrt.unsqueeze(0)
        return self.linear(torch.matmul(norm_adj, x))


@MODEL_REGISTRY.register('gcn')
class GCN(BaseGNNModel):
    """Two-stage GCN encoder compatible with the existing FedGCDR interface."""

    def __init__(self, args, in_feature: int, hid_feature: int = 16,
                 out_feature: int = 16, num_layers: int = 2, dropout: float = 0.5):
        super().__init__(args, in_feature, hid_feature, out_feature)
        self.device = args.device
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        self.layers.append(GCNLayer(in_feature, hid_feature).to(self.device))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hid_feature, hid_feature).to(self.device))
        self.layers.append(GCNLayer(hid_feature, out_feature).to(self.device))

    def forward(self, x: torch.Tensor, is_transfer_stage: bool = False,
                domain_attention: torch.Tensor = None,
                transfer_vec: list = None) -> tuple:
        intermediate_embedding = []
        ls, lm = 0, 0

        adj = torch.zeros((len(x), len(x)), device=x.device)
        adj[:, 0] = 1.0
        adj[0, :] = 1.0

        if is_transfer_stage:
            ls = self.compute_ls(x[0], transfer_vec)
            lm = self.compute_lm(x[0], transfer_vec)
            transfer_tensor = torch.stack(transfer_vec)
            x = torch.cat((x, transfer_tensor))
            adj = torch.zeros((len(x), len(x)), device=x.device)
            adj[:, 0] = 1.0
            adj[0, :] = 1.0

        for index, layer in enumerate(self.layers):
            x = layer(x, adj)
            if index < len(self.layers) - 1:
                x = F.relu(x)
            x = self.dropout(x)

            if index == 0:
                intermediate_embedding.append(x[0].data)

        x = F.normalize(x, p=2, dim=1)
        return x, intermediate_embedding, ls, lm


@MODEL_REGISTRY.register('gcn_mlp')
class MLP(BaseMLP):
    def __init__(self, in_feature: int):
        super().__init__(in_feature, hidden_factor=2, dropout=0.5, activation='relu')
