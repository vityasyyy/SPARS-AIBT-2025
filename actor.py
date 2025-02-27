import torch
import torch.nn as nn

class HPCNodeManager(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=3, num_layers=3):
        super(HPCNodeManager, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(input_dim, 1)  # Output 1 value per node (sigmoid activation later)

    def forward(self, node_features):
        encoded = self.transformer_encoder(node_features)  # Encode node states
        logits = self.output_layer(encoded)  # Predict per-node activation score
        predictions = torch.sigmoid(logits)  # Convert logits to probabilities (on/off)
        return predictions.squeeze(-1)  # (batch_size, num_nodes)

# Initialize model
feature_dim = 9  # Set feature dimension
model = HPCNodeManager(input_dim=feature_dim)

# Save the untrained model
torch.save(model, "untrained/hpc_node_manager.pth")
print("Model saved successfully.")
