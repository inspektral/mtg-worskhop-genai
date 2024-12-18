import torch
import torch.nn as nn
import torch.optim as optim

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=6, input_dim=24):
        super(Autoencoder, self).__init__()
        # Encoder: Reduce 50 -> 6
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),   # 50 -> 32
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),   # 32 -> 16
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, latent_dim),    # 16 -> 6
            nn.Tanh()            # Ensure smooth [-1, 1] range for knobs
        )
        # Decoder: Expand 6 -> 50
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),    # 6 -> 16
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 16),   # 16 -> 32
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, input_dim),   # 32 -> 50
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

# Instantiate the model

