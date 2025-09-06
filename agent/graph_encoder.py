from typing import Optional, Tuple
import torch
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input: torch.Tensor):
        return input + self.module(input)


class MultiHeadAttention(torch.jit.ScriptModule):
    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        embed_dim: int,
        val_dim: Optional[int] = None,
        key_dim: Optional[int] = None,
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1.0 / math.sqrt(float(key_dim))

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(float(param.size(-1)))
            param.data.uniform_(-stdv, stdv)

    @torch.jit.script_method
    def forward(
        self,
        q: torch.Tensor,                             # [B, n_query, input_dim]
        # [B, graph_size, input_dim]
        h: Optional[torch.Tensor] = None,
        # unused here; kept for API parity
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if h is None:
            h = q  # self-attention

        B = q.size(0)
        n_query = q.size(1)
        graph_size = h.size(1)
        input_dim = h.size(2)

        hflat = h.contiguous().view(-1, input_dim)   # [B*G, D]
        qflat = q.contiguous().view(-1, input_dim)   # [B*Q, D]

        shp = (self.n_heads, B, graph_size, -1)
        shp_q = (self.n_heads, B, n_query,    -1)

        Q = torch.matmul(qflat, self.W_query).view(shp_q)   # [H, B, Q, K]
        K = torch.matmul(hflat, self.W_key).view(shp)       # [H, B, G, K]
        V = torch.matmul(hflat, self.W_val).view(shp)       # [H, B, G, V]

        # compat: [H, B, Q, G]
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        attn = torch.softmax(compatibility, dim=-1)         # [H, B, Q, G]

        heads = torch.matmul(attn, V)                       # [H, B, Q, V]
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous(
            ).view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(B, n_query, self.embed_dim)                  # [B, Q, E]
        return out


class Normalization(nn.Module):
    def __init__(self, embed_dim: int):
        super(Normalization, self).__init__()
        self.normalizer = nn.InstanceNorm1d(embed_dim, affine=True)
        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1.0 / math.sqrt(float(param.size(-1)))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: [B, N, E] -> IN on channels E
        return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)


class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int = 512,
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim),
        )


class GraphAttentionEncoder(torch.jit.ScriptModule):
    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        node_dim: Optional[int] = None,
        feed_forward_hidden: int = 512,
    ):
        super(GraphAttentionEncoder, self).__init__()

        self.init_embed = nn.Linear(
            node_dim, embed_dim) if node_dim is not None else None
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden)
            for _ in range(n_layers)
        ))

    @torch.jit.script_method
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, N, node_dim]
        returns:
          node_emb : [B, N, E]
          graph_emb: [B, E] (mean across N)
        """
        h = x
        if self.init_embed is not None:
            h = self.init_embed(x)
        h_ = self.layers(h)                 # [B, N, E]
        return h_, h_.mean(dim=1)          # per-graph embedding [B, E]
