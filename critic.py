import torch
import torch.nn as nn

class HPCCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=3, num_layers=3):
        super(HPCCritic, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(input_dim, 1)  # Single value output

    def forward(self, node_features):
        encoded = self.transformer_encoder(node_features)  # Encode node states
        value = self.output_layer(encoded).mean(dim=1)  # Aggregate node-wise values
        return value.squeeze(-1)  # (batch_size,)

# Initialize and save Critic model
feature_dim = 9  # Match with the actor
critic = HPCCritic(input_dim=feature_dim)

# Save the untrained model
torch.save(critic, "untrained/hpc_critic.pth")
print("Critic model saved successfully.")
