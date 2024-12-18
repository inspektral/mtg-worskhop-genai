import torch
import torch.nn as nn
import torch.optim as optim

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: Reduce 50 -> 6
        self.encoder = nn.Sequential(
            nn.Linear(50, 1024),   # 50 -> 32
            nn.ReLU(),
            nn.Linear(1024, 16),   # 32 -> 16
            nn.ReLU(),
            nn.Linear(16, 4),    # 16 -> 6
            nn.Tanh()            # Ensure smooth [-1, 1] range for knobs
        )
        # Decoder: Expand 6 -> 50
        self.decoder = nn.Sequential(
            nn.Linear(4, 1024),    # 6 -> 16
            nn.ReLU(),
            nn.Linear(1024, 32),   # 16 -> 32
            nn.ReLU(),
            nn.Linear(32, 50),   # 32 -> 50
            nn.Sigmoid()         # Output range [0, 1] for parameters
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

# Instantiate the model

