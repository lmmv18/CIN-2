import os
import re
import joblib
import deeplabcut
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns

CONFIG_PATH = r"C:\Users\Usuario\Desktop\CIN-2\dlc_project\Proyecto1-Laura-2026-04-13\config.yaml"
VIDEO_FOLDER = r"C:\Users\Usuario\Desktop\CIN-2\new_videos"
EMB_FOLDER = r"C:\Users\Usuario\Desktop\CIN-2\new_embeddings"
LABELS_FOLDER = r"C:\Users\Usuario\Desktop\CIN-2\labels"

WINDOW_SIZE = 15
STEP_SIZE = 5
FPS = 30

os.makedirs(EMB_FOLDER, exist_ok=True)

def run_dlc():
    videos_all = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(".mp4")]
    videos_to_process = []

    for video_file in videos_all:
        base_name = os.path.splitext(video_file)[0]

        h5_exists = any(
            f.startswith(base_name) and f.endswith(".h5")
            for f in os.listdir(VIDEO_FOLDER)
        )

        if not h5_exists:
            videos_to_process.append(os.path.join(VIDEO_FOLDER, video_file))

    print(f"Total videos: {len(videos_all)}")
    print(f"Faltantes: {len(videos_to_process)}")

    if len(videos_to_process) > 0:
        deeplabcut.analyze_videos(CONFIG_PATH, videos_to_process, save_as_csv=True)
        deeplabcut.filterpredictions(CONFIG_PATH, videos_to_process)
        print("Procesamiento DLC completado")
    else:
        print("Todos los videos ya fueron procesados")

def compute_velocity(coords):
    return np.diff(coords, axis=0)

def compute_speed(velocity):
    return np.linalg.norm(velocity, axis=2)

def compute_distance_matrix(points):
    n = points.shape[1]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
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

    return np.concatenate([pos_flat, vel_flat, speed, dist], axis=1)

def create_windows(features):
    windows = []
    for i in range(0, len(features) - WINDOW_SIZE, STEP_SIZE):
        windows.append(features[i:i+WINDOW_SIZE].flatten())
    return np.array(windows)

def generate_embeddings():
    csv_files = [
        f for f in os.listdir(VIDEO_FOLDER)
        if f.endswith(".csv") and "DLC" in f
    ]

    for file in csv_files:
        try:
            df = pd.read_csv(os.path.join(VIDEO_FOLDER, file), header=[0,1,2])
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        except:
            continue

        df = df.loc[:, ~df.columns.str.contains("likelihood")]
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.interpolate().bfill().ffill()

        features = extract_features(df)

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        windows = create_windows(features)

        base_name = os.path.splitext(file)[0]
        np.save(os.path.join(EMB_FOLDER, base_name + ".npy"), windows)

        print("Embedding:", file, windows.shape)

def clean_name(name):
    name = name.lower()
    name = re.sub(r"dlc.*", "", name)
    name = name.replace(" ", "")
    return name

def read_csv_safe(path):
    try:
        return pd.read_csv(path, sep=";")
    except:
        return pd.read_csv(path, sep=";", encoding="latin-1")

def expand_time_to_frames(df):
    df = df.dropna(axis=1, how='all')
    times = df.iloc[:, 0].values
    labels = df.iloc[:, 1].values

    frame_labels = []

    for i in range(len(df)-1):
        t1 = float(str(times[i]).replace(",", "."))
        t2 = float(str(times[i+1]).replace(",", "."))
        label = str(labels[i])

        duration = t2 - t1
        n_frames = int(duration * FPS)

        frame_labels.extend([label] * n_frames)

    return np.array(frame_labels)

def build_dataset():
    emb_files = [f for f in os.listdir(EMB_FOLDER) if f.endswith(".npy")]
    label_files = [f for f in os.listdir(LABELS_FOLDER) if f.endswith(".csv")]

    label_map = {clean_name(f): f for f in label_files}

    X_all = []
    y_all = []

    for emb_file in emb_files:
        emb_clean = clean_name(os.path.splitext(emb_file)[0])

        matched_label = None

        for key in label_map:
            if key in emb_clean or emb_clean in key:
                matched_label = label_map[key]
                break

        if matched_label is None:
            continue

        X = np.load(os.path.join(EMB_FOLDER, emb_file))
        df = read_csv_safe(os.path.join(LABELS_FOLDER, matched_label))

        frame_labels = expand_time_to_frames(df)

        window_labels = []

        for i in range(len(X)):
            start = i * STEP_SIZE
            end = start + WINDOW_SIZE

            window = frame_labels[start:end]

            if len(window) == 0:
                window_labels.append("unknown")
                continue

            counts = Counter(window)
            label, freq = counts.most_common(1)[0]

            if freq / len(window) < 0.5:
                window_labels.append("mixed")
            else:
                window_labels.append(label)

        window_labels = np.array(window_labels)

        mask = (
            (window_labels != "mixed") &
            (window_labels != "unknown") &
            (window_labels != "nan")
        )

        X_all.append(X[mask])
        y_all.append(window_labels[mask])

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    np.save("X_final.npy", X_all)
    np.save("y_final.npy", y_all)

    print("X:", X_all.shape)
    print("y:", y_all.shape)

def train_model():
    X = np.load("X_final.npy")
    y = np.load("y_final.npy")

    y = np.array([str(label).strip() for label in y])

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_enc),
        y=y_enc
    )

    class_weights = dict(enumerate(weights))

    timesteps = 15
    features = X.shape[1] // timesteps
    X_seq = X.reshape(len(X), timesteps, features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_cat,
        test_size=0.2,
        stratify=y_enc,
        random_state=42
    )

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
        BatchNormalization(),
        Dropout(0.4),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
        ],
        class_weight=class_weights
    )

    loss, acc = model.evaluate(X_test, y_test)

    print("\\nAccuracy FINAL:", acc)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(
        y_true,
        y_pred_classes,
        target_names=le.classes_
    ))

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title("Matriz de Confusion")
    plt.tight_layout()
    plt.show()

    model.save("lstm_model.keras")
    joblib.dump(le, "label_encoder.pkl")

    print("Modelo guardado")

if __name__ == "__main__":
    run_dlc()
    generate_embeddings()
    build_dataset()
    train_model()
