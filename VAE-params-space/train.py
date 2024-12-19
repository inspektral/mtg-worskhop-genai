# Example dataset: Replace with your actual 50-parameter snapshots
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from model import Autoencoder
import preprocess

# Dummy Data: Replace with your actual snapshots
snapshots = preprocess.read_json_to_numpy('augmented_dataset.json')
print(snapshots.shape)
snapshots = torch.tensor(snapshots, dtype=torch.float32)

# Create DataLoader
dataset = TensorDataset(snapshots, snapshots)  # Autoencoder input = output
dataloader = DataLoader(dataset, batch_size=450, shuffle=True)

model = Autoencoder(latent_dim=4, input_dim=13)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training Loop
num_epochs = 1000
for epoch in range(num_epochs):
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "autoencoder.pth")
