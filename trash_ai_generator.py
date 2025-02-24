import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class HPCNodeManager(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=2, num_layers=3):
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
        """
        node_features: (batch_size, num_nodes, feature_dim)
        Output: (batch_size, num_nodes, 1) - Binary classification for each node (on/off)
        """
        encoded = self.transformer_encoder(node_features)  # Encode node states
        logits = self.output_layer(encoded)  # Predict per-node activation score
        predictions = torch.sigmoid(logits)  # Convert logits to probabilities (on/off)
        return predictions.squeeze(-1)  # (batch_size, num_nodes)

# Example usage
batch_size, num_nodes, feature_dim = 2, 2, 4
model = HPCNodeManager(input_dim=feature_dim)
torch.save(model, "untrained/hpc_node_manager.pth")