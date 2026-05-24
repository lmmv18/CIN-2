import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

csv_folder = r"C:\Users\Usuario\Desktop\CIN-2\new_csv"
output_folder = r"C:\Users\Usuario\Desktop\CIN-2\new_embeddings"

window_size = 15
step_size = 5

os.makedirs(output_folder, exist_ok=True)


def compute_velocity(coords):
    return np.diff(coords, axis=0)

def compute_speed(velocity):
    return np.linalg.norm(velocity, axis=2)

def compute_distance_matrix(points):
    n = points.shape[1]
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[:, i] - points[:, j], axis=1)
            dists.append(d)
    return np.stack(dists, axis=1)

def extract_features(df):
    columns_xy = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    data = df[columns_xy].values

    if data.shape[1] % 2 != 0:
        data = data[:, :-1]

    num_keypoints = data.shape[1] // 2
    coords = data.reshape(len(df), num_keypoints, 2)

    coords = coords - coords.mean(axis=1, keepdims=True)

    vel = compute_velocity(coords)
    speed = compute_speed(vel)
    dist = compute_distance_matrix(coords)

    min_len = min(len(vel), len(speed), len(dist))

    pos = coords[:min_len]
    vel = vel[:min_len]
    speed = speed[:min_len]
    dist = dist[:min_len]

    pos_flat = pos.reshape(min_len, -1)
    vel_flat = vel.reshape(min_len, -1)

    features = np.concatenate([pos_flat, vel_flat, speed, dist], axis=1)

    return features

def create_windows(features):
    windows = []
    for i in range(0, len(features) - window_size, step_size):
        windows.append(features[i:i+window_size].flatten())
    return np.array(windows)



csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

for file in csv_files:

    print("\nProcesando:", file)

    path = os.path.join(csv_folder, file)

    df = pd.read_csv(path, header=[0,1,2])
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

    df = df.loc[:, ~df.columns.str.contains("likelihood")]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.interpolate().bfill().ffill()

    features = extract_features(df)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    windows = create_windows(features)

    base_name = os.path.splitext(file)[0]
    save_path = os.path.join(output_folder, base_name + ".npy")

    np.save(save_path, windows)

    print(" Guardado:", save_path, windows.shape)
