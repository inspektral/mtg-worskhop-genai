# Example dataset: Replace with your actual 50-parameter snapshots
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from model_percussion import Autoencoder
import preprocess

# load model from file
model = Autoencoder(latent_dim=4, input_dim=16)
model.load_state_dict(torch.load("autoencoder_percussion.pth"))
model.eval()

latent_space_data = torch.tensor([[0.2, -0.2, 0.3, -0.4]], dtype=torch.float32)
prediction = model.decode(latent_space_data)

# denormalization
prediction = prediction.detach().numpy()
prediction = np.clip(prediction, 0, 1)
prediction = preprocess.denormalize(prediction, preprocess.read_json_to_numpy('dataset-percussion.json'))

print(prediction)

def save_to_txt(data, file):
    with open(file, 'w') as f:
        for row in data:
            f.write(' '.join(f'{value:.2f}' for value in row) + '\n')

save_to_txt(prediction, 'decoded_snapshot.txt')

scripted_decoder = torch.jit.script(model.decoder)
scripted_decoder.save("decoder_percussion.ts")