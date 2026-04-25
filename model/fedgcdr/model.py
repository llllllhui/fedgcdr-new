"""
GAT model implementation for FedGCDR.
"""

import os
import sys

import torch
import torch.nn as nn

# Import base classes and registry from model package root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import BaseGNNLayer, BaseGNNModel, BaseMLP
from registry import MODEL_REGISTRY


class GATLayer(BaseGNNLayer):
    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        alpha: float = 0.1,
        use_residual: bool = False,
        use_attention_clamp: bool = True,
        attention_clamp_value: float = 5.0,
    ):
        super().__init__(in_feature, out_feature)
        self.A = nn.Parameter(torch.empty(size=(2 * out_feature, 1)))
        nn.init.xavier_uniform_(self.A.data, nn.init.calculate_gain("relu"))
        self.alpha = alpha
        self.use_residual = use_residual
        self.use_attention_clamp = use_attention_clamp
        self.attention_clamp_value = attention_clamp_value

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = input
        h1 = torch.matmul(h, self.A[self.out_feature :, :])
        h2 = torch.matmul(h, self.A[: self.out_feature, :])
        e = h1 + h2.T

        # Important: clamp logits before masking, otherwise masked entries are revived.
        if self.use_attention_clamp:
            e = torch.clamp(e, -self.attention_clamp_value, self.attention_clamp_value)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=-1)
        ah = torch.matmul(attention, h)
        return ah + h if self.use_residual else ah


@MODEL_REGISTRY.register("gat")
class GAT(BaseGNNModel):
    def __init__(
        self,
        args,
        in_feature: int,
        hid_feature: int = 16,
        out_feature: int = 16,
        alpha: float = 0.1,
        dropout: float = 0,
    ):
        super().__init__(args, in_feature, hid_feature, out_feature)
        self.drop = nn.Dropout(p=dropout)

        self.use_layernorm = getattr(args, "gat_use_layernorm", False)
        self.use_layer_average = getattr(args, "gat_use_layer_average", False)
        self.use_residual = getattr(args, "gat_use_residual", False)
        self.use_attention_clamp = getattr(args, "gat_use_attention_clamp", True)
        self.attention_clamp_value = getattr(args, "gat_attention_clamp_value", 5.0)

        self.in2hidden = GATLayer(
            in_feature,
            hid_feature,
            alpha,
            use_residual=self.use_residual,
            use_attention_clamp=self.use_attention_clamp,
            attention_clamp_value=self.attention_clamp_value,
        ).to(args.device)
        self.hidden2out = GATLayer(
            hid_feature,
            out_feature,
            alpha,
            use_residual=self.use_residual,
            use_attention_clamp=self.use_attention_clamp,
            attention_clamp_value=self.attention_clamp_value,
        ).to(args.device)

        self.ln1 = nn.LayerNorm(hid_feature).to(args.device) if self.use_layernorm else None
        self.ln2 = nn.LayerNorm(out_feature).to(args.device) if self.use_layernorm else None

    def forward(
        self,
        x: torch.Tensor,
        is_transfer_stage: bool = False,
        domain_attention: torch.Tensor = None,
        transfer_vec: list = None,
    ) -> tuple:
        ls, lm = 0, 0
        alpha, beta = 0.01, 0.01
        intermediate_embedding = []

        adj = torch.eye(len(x), device=x.device)
        adj[:, 0] = 1.0
        adj[0, :] = 1.0

        x = self.in2hidden(x, adj)
        if self.ln1 is not None:
            x = self.ln1(x)
        intermediate_embedding.append(x[0].data)
        num_nodes = len(x)

        if is_transfer_stage:
            ls = alpha / 2 * self.compute_ls(x[0], transfer_vec)
            lm = beta / 2 * self.compute_lm(x[0], transfer_vec)
            transfer_vec = torch.stack(transfer_vec)
            x = torch.cat((x, transfer_vec))
            adj = torch.eye(len(x), device=x.device)
            adj[:, 0] = 1.0
            adj[0, :] = 1.0

        x = self.hidden2out(x, adj)
        if self.ln2 is not None:
            x = self.ln2(x)

        if self.use_layer_average:
            x[:num_nodes] = (intermediate_embedding[0] + x[:num_nodes]) / 2

        return x, intermediate_embedding, ls, lm


@MODEL_REGISTRY.register("gat_mlp")
class MLP(BaseMLP):
    def __init__(self, in_feature: int):
        super().__init__(in_feature, hidden_factor=2, dropout=0.0, activation="tanh")
