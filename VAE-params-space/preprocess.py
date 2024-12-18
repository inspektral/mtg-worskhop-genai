import numpy as np

def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [list(map(float, line.strip().split())) for line in lines]
    data = np.array(data)

    return data

def interpolate_data(data, num_interpolations=10):
    interpolated_data = []
    for i in range(len(data)-1):
        for j in range(num_interpolations):
            interpolated_data.append(data[i] + j/(num_interpolations+1) * (data[i+1] - data[i]))
    interpolated_data = np.array(interpolated_data)

    return interpolated_data

def augment_data(data, num_augmentations=10):
    augmented_data = []
    for _ in range(num_augmentations):
        augmented_data.append(data + np.random.normal(0, 0.1, data.shape))
    augmented_data = np.array(augmented_data)

    return augmented_data