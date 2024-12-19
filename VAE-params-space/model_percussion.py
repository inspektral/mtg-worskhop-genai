import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):

    def __init__(self, latent_dim=4, input_dim=24):

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),  
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8), 
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, latent_dim),    
            nn.Tanh()            
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), 
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, input_dim), 
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def export_onnx_encoder(self, filename):
        dummy_input = torch.randn(1, self.input_dim)
        torch.onnx.export(self.encoder, dummy_input, filename, verbose=True)

    def export_onnx_decoder(self, filename):
        dummy_input = torch.randn(1, self.latent_dim)
        torch.onnx.export(self.decoder, dummy_input, filename, verbose=True)