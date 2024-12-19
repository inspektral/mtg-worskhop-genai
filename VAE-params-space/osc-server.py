import socket
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from model import Autoencoder
import preprocess
import math
from pythonosc import udp_client


models = [
    {
        "name": "percussion2",
        "input_dim": 13,
        "latent_dim": 4,
        "model_file": "percussion-13-4.pth",
        "dataset_file": "dataset-percussion2.json"
    },
    {
        "name": "percussion1",
        "input_dim": 16,
        "latent_dim": 6,
        "model_file": "percussion-16-6.pth",
        "dataset_file": "dataset-percussion.json"
    },
    {
        "name": "vctk",
        "input_dim": 24,
        "latent_dim": 4,
        "model_file": "vctk-24-4.pth",
        "dataset_file": "dataset-vctk.json"
    }
]

# calculate denormalization parameters
for model in models:
    dataset = preprocess.read_json_to_numpy(model["dataset_file"])
    model["denormalization_params"] = preprocess.get_denormalization_params(dataset)

# load models from files
autoencoders = []
for model in models:
    autoencoder = Autoencoder(latent_dim=model["latent_dim"], input_dim=model["input_dim"])
    autoencoder.load_state_dict(torch.load(model["model_file"]))
    autoencoder.eval()
    autoencoders.append(autoencoder)

def decode(model, latent_space_data, denormalization_params):
    prediction = model.decode(latent_space_data)
    
    # denormalization
    prediction = prediction.detach().numpy()
    prediction = np.clip(prediction, 0, 1)
    prediction = preprocess.denormalize(prediction, denormalization_params)
    
    return prediction

latent_space_data = []
for model in models:
    latent_space_data.append(torch.tensor([[0 for _ in range(model["latent_dim"])]], dtype=torch.float32))

def midi_rescale(midi):
    return midi*2/127 - 1


midi_maps = [16, 20, 18, 22, 28, 46, 30, 48, 54, 58, 56, 60]



# Define the server address and port
server_address = ('localhost', 5001)

ip = "127.0.0.1"
port = 5000
client = udp_client.SimpleUDPClient(ip, port)

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the address and port
sock.bind(server_address)

print(f"Server listening on {server_address[0]}:{server_address[1]}")

while True:
    # Receive data from the client
    data, client_address = sock.recvfrom(4096)
    if data:
        data = data.decode('utf-8')
        print(f"Received from {client_address}: {data}")
        data = data.split(" ")
        index = midi_maps.index(int(data[0]))
        index_model = math.floor(index/4)

        value = midi_rescale(float(data[1]))

        latent_space_data[index_model][0][index%4] = value
        print(latent_space_data[index_model])

        prediction = decode(autoencoders[index_model], latent_space_data[index_model], models[index_model]["denormalization_params"])
        print(prediction[0].tolist())

        client.send_message(f"/{index_model}", prediction[0].tolist())

