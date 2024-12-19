import numpy as np
import json

def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        print(lines)
    data = [list(map(float, line.strip().split())) for line in lines]
    data = np.array(data)

    return data

def vertical_normalize(data):
    data = (data) / data.max(axis=0)
    return data

def denormalize(normalized_data, original_data):
    data = normalized_data * original_data.max(axis=0)
    return data

def read_json_to_numpy(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    data = np.array(data)
    
    print(data.shape)
    return data

def interpolate_data(data, num_interpolations=10):
    interpolated_data = []
    for i in range(len(data)-1):
        for j in range(num_interpolations):
            interpolated_data.append(data[i] + j/(num_interpolations+1) * (data[i+1] - data[i]))
    interpolated_data = np.array(interpolated_data)

    print(interpolated_data.shape)
    return interpolated_data

def augment_data(data, num_augmentations=10):
    augmented_data = []
    for _ in range(num_augmentations):
        scale_factors = np.random.uniform(0.9, 1.1, size=data.shape[1])
        augmentation = data * scale_factors
        print(augmentation.shape)
        augmented_data.extend(augmentation)

    augmented_data = np.array(augmented_data)

    print(augmented_data.shape)
    return augmented_data

def write_data(data, file_path):
    with open(file_path, 'w') as file:
        for row in data:
            file.write(' '.join(map(str, row)) + '\n')

def write_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data.tolist(), file)

if __name__ == '__main__':
    data = read_json_to_numpy('dataset-percussion.json')
    print(data.max(axis=0))

    norm_data = vertical_normalize(data)
    print(norm_data)

    interpolated_data = interpolate_data(norm_data)

    augmented_data = augment_data(interpolated_data)
    write_json(augmented_data, 'augmented_dataset_percussion.json')