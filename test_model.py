import torch
import torch.nn as nn
import torch.optim as optim

# Load the saved model
model = torch.load("untrained/hpc_node_manager.pth", map_location=torch.device('cpu'))

# Set model to training mode
model.train()

# Define loss function and optimizer
criterion = nn.MSELoss()  # Example loss function (depends on your use case)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example input and target (assume reward as target)
num_nodes = 3
feature_dim = 4
node_data = torch.randn(1, num_nodes, feature_dim)  # Simulated HPC state
reward = torch.tensor([0.5] * num_nodes).unsqueeze(0)  # Target output (same shape as model output)

# Forward pass
output = model(node_data)  # Get predictions

# Compute loss
loss = criterion(output, reward)

# Backpropagation
optimizer.zero_grad()  # Clear previous gradients
loss.backward()  # Compute gradients
optimizer.step()  # Update weights

print(f"Loss: {loss.item()}")
