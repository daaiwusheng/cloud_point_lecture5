import numpy as np


def normalize_point_cloud(data):
    mean = np.mean(data, axis=0)
    data = data - mean
    max_distance = np.max(np.sqrt(np.sum(data ** 2, axis=1)))
    data = data / max_distance
    return data


def rotate_around_z(data, theta):
    ratation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), -np.cos(theta), 0],
                                [0, 0, 1]
                                ])
    data = np.dot(data, ratation_matrix)
    return data


def move_to_device(data, device):
    return data.to(device, non_blocking=True)

