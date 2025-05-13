import torch
import torch.nn as nn

class HPCNodeManager(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(8, 64),  # 4 (global) + 4 (node unique) = 8
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, combined_features):
        # Forward pass on the combined features
        return self.shared_mlp(combined_features).squeeze(-1)

# --- Export Logic ---
if __name__ == "__main__":
    # Initialize untrained model
    model = HPCNodeManager()

    # Save the entire model (architecture + initial weights) to a .pt file
    torch.save(model, "untrained_model.pt")  # <-- EXPLICIT EXPORT
    print("Untrained model exported to 'untrained_model.pt'")
