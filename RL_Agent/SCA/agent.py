from typing import Tuple
import torch as T
import torch.nn as nn

from RL_Agent.SCA.graph_encoder import GraphAttentionEncoder

CPU_DEVICE = T.device("cpu")


class Agent(T.jit.ScriptModule):
    def __init__(
        self,
        n_heads: int = 8,
        n_gae_layers: int = 3,
        input_dim: int = 11,
        embed_dim: int = 128,
        gae_ff_hidden: int = 512,
        tanh_clip: float = 10,
        device=CPU_DEVICE,
    ):
        super(Agent, self).__init__()
        self.n_heads = n_heads
        self.n_gae_layers = n_gae_layers
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.tanh_clip = tanh_clip
        self.device = device
        self.key_size = self.val_size = self.embed_dim // self.n_heads

        self.gae = GraphAttentionEncoder(
            n_heads=n_heads,
            n_layers=n_gae_layers,
            embed_dim=embed_dim,
            node_dim=input_dim,
            feed_forward_hidden=gae_ff_hidden,
        )

        # Per-node 2-class head; applied to node embeddings -> [B, N, 2]
        self.prob_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )
        self.to(device)

    @T.jit.script_method
    def forward(self, features: T.Tensor, mask: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        """
        features: [B, N, input_dim]  (B can be your time horizon T)
        mask    : [B, N] or [B, N, 1] (kept for API compatibility; not forced here)
        returns:
          probs  : [B, N, 2]  (softmax over the class dimension per node)
          entropy: [B]        (mean of per-node entropies per sample)
        """
        B, N, _ = features.shape
        embeddings, _mean_embeddings = self.gae(
            features)  # embeddings: [B, N, E]

        logits = self.prob_head(embeddings)                # [B, N, 2]
        # NOTE: If your mask marks invalid NODES, masking before a per-node
        # 2-class softmax is usually *not* what you want (both classes would be masked).
        # Keep masking for selection outside this module, or adapt here if needed.

        probs = T.softmax(logits, dim=-1)                  # [B, N, 2]

        # Per-node entropy then mean over nodes -> [B]
        # (add epsilon for numerical stability)
        eps = 1e-12
        ent_per_node = -(probs * (probs + eps).log()).sum(dim=-1)  # [B, N]
        entropy = ent_per_node.mean(dim=1)                         # [B]

        return probs, entropy
