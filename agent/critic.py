import torch as T
import torch.nn as nn

from agent.graph_encoder import GraphAttentionEncoder

CPU_DEVICE = T.device("cpu")


class Critic(T.jit.ScriptModule):
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
        super(Critic, self).__init__()
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

        self.value_layers = nn.Sequential(
            nn.Linear(embed_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        self.to(device)

    @T.jit.script_method
    def forward(self, features: T.Tensor) -> T.Tensor:
        """
        features: [B, N, input_dim]  (B can be your time horizon T)
        returns : values per sample [B]
        """
        _node_emb, graph_emb = self.gae(
            features)  # graph_emb: [B, E] (mean over nodes)
        values = self.value_layers(graph_emb).squeeze(-1)  # [B]
        # Preserve your original sign convention (critic returns negative values)
        return -values
