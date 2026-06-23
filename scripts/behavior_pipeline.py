# ==================================================
# IMPORTS
# ==================================================

import os
import re
import joblib
import deeplabcut
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# CONFIGURACION
# ==================================================
CONFIG_PATH = r"C:\Users\Usuario\Desktop\CIN-2\dlc_project\Proyecto1-Laura-2026-04-13\config.yaml"
VIDEO_FOLDER = r"C:\Users\Usuario\Desktop\CIN-2\videos"
EMB_FOLDER = r"C:\Users\Usuario\Desktop\CIN-2\new_embeddings"
LABELS_FOLDER = r"C:\Users\Usuario\Desktop\CIN-2\labels"
TRAIN_SPLIT = r"C:\Users\Usuario\Desktop\CIN-2\scripts\train_videos.txt"
TEST_SPLIT = r"C:\Users\Usuario\Desktop\CIN-2\scripts\test_videos.txt"

WINDOW_SIZE = 45
STEP_SIZE = 15
FPS = 30

RUN_DLC = False
GENERATE_EMBEDDINGS = False
BUILD_DATASET = False
TRAIN_MODEL = True

os.makedirs(EMB_FOLDER, exist_ok=True)

RESULTS_FOLDER = "results"

os.makedirs(EMB_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
# ==================================================
# DEEPLABCUT
# ==================================================

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

# ==================================================
# FEATURES
# ==================================================
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

    columns_xy = [
        c for c in df.columns
        if c.endswith("_x") or c.endswith("_y")
    ]

    data = df[columns_xy].values

    if data.shape[1] % 2 != 0:
        data = data[:, :-1]

    num_keypoints = data.shape[1] // 2

    coords = data.reshape(
        len(df),
        num_keypoints,
        2
    )

    coords = coords - coords.mean(
        axis=1,
        keepdims=True
    )

    vel = compute_velocity(coords)

    speed = compute_speed(vel)

    dist = compute_distance_matrix(coords)

    acc = np.diff(
        vel,
        axis=0
    )

    min_len = min(
        len(coords),
        len(vel),
        len(acc),
        len(speed),
        len(dist)
    )

    pos = coords[:min_len]
    vel = vel[:min_len]
    acc = acc[:min_len]
    speed = speed[:min_len]
    dist = dist[:min_len]

    pos_flat = pos.reshape(
        min_len,
        -1
    )

    vel_flat = vel.reshape(
        min_len,
        -1
    )

    acc_flat = acc.reshape(
        min_len,
        -1
    )

    return np.concatenate([
        pos_flat,
        vel_flat,
        acc_flat,
        speed,
        dist
    ], axis=1)

def create_windows(features):
    windows = []
    for i in range(0, len(features) - WINDOW_SIZE, STEP_SIZE):
        windows.append(features[i:i+WINDOW_SIZE].flatten())
    return np.array(windows)

# ==================================================
# EMBEDDINGS
# ==================================================
def generate_embeddings():
    
    csv_files = [
        f for f in os.listdir(VIDEO_FOLDER)
        if (
        f.endswith(".csv")
        and "DLC" in f
        and "_filtered" in f
        )
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
# ==================================================
# UTILITIES
# ==================================================
def clean_name(name):

    name = name.lower()

    name = re.sub(r"dlc.*", "", name)

    name = name.replace(" ", "")
    name = name.replace("-", "")
    name = name.replace("_", "")

    return name

def load_split_file(path):

    with open(path, "r") as f:

        return {
            line.strip().lower()
            for line in f
            if line.strip()
        }
    
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

# ==================================================
# DATASET
# ==================================================

def build_dataset():

    train_ids = load_split_file(TRAIN_SPLIT)
    test_ids = load_split_file(TEST_SPLIT)

    emb_files = [f for f in os.listdir(EMB_FOLDER) if f.endswith(".npy")]
    label_files = [f for f in os.listdir(LABELS_FOLDER) if f.endswith(".csv")]

    
    label_map = {clean_name(f): f for f in label_files}

    
    X_train_all = []
    y_train_all = []

    X_test_all = []
    y_test_all = []

    for emb_file in emb_files:

        emb_clean = clean_name(os.path.splitext(emb_file)[0])
    

        matched_label = None
        video_id = None

        for key in label_map:

            if key in emb_clean or emb_clean in key:

                matched_label = label_map[key]

                video_id = clean_name(
                    os.path.splitext(emb_file)[0]
                )

                break

        if matched_label is None:
            print("NO MATCH:", emb_clean)
            continue

        X = np.load(
            os.path.join(EMB_FOLDER, emb_file)
        )

        df = read_csv_safe(
            os.path.join(
                LABELS_FOLDER,
                matched_label
            )
        )

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
            (window_labels != "mixed")
            & (window_labels != "unknown")
            & (window_labels != "nan")
        )

        video_data = X[mask]
        video_labels = window_labels[mask]


        if video_id in train_ids:

            X_train_all.append(video_data)
            y_train_all.append(video_labels)

        elif video_id in test_ids:

            X_test_all.append(video_data)
            y_test_all.append(video_labels)

    print("\nResumen:")
    print("Videos train encontrados:", len(X_train_all))
    print("Videos test encontrados :", len(X_test_all))

    if len(X_train_all) == 0:
        raise ValueError("No se encontraron videos de entrenamiento")

    if len(X_test_all) == 0:
        raise ValueError("No se encontraron videos de prueba")
    
    X_train_all = np.vstack(X_train_all)
    y_train_all = np.concatenate(y_train_all)

    X_test_all = np.vstack(X_test_all)
    y_test_all = np.concatenate(y_test_all)
    
    print("\n===== TRAIN =====")
    print(Counter(y_train_all))

    print("\n===== TEST =====")
    print(Counter(y_test_all))
    
    np.save("X_train.npy", X_train_all)
    np.save("y_train.npy", y_train_all)

    np.save("X_test.npy", X_test_all)
    np.save("y_test.npy", y_test_all)

    print("\n===== DATASET =====")
    print("X_train:", X_train_all.shape)
    print("y_train:", y_train_all.shape)
    print("X_test :", X_test_all.shape)
    print("y_test :", y_test_all.shape)
    
# ==================================================
# LSTM MODEL
# =============================
def train_model():

    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")

    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    

    print("\nTRAIN:")
    print(Counter(y_train))

    print("\nTEST:")
    print(Counter(y_test))

    y_train = np.array([
        str(label).strip()
        for label in y_train
    ])

    y_test = np.array([
        str(label).strip()
        for label in y_test
    ])

    all_labels = np.concatenate([
        y_train,
        y_test
    ])

    le = LabelEncoder()
    le.fit(all_labels)

    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    y_train_cat = to_categorical(y_train_enc)
    y_test_cat = to_categorical(y_test_enc)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_enc),
        y=y_train_enc
    )

    class_weights = dict(
        enumerate(weights)
    )

    timesteps = WINDOW_SIZE

    features = X_train.shape[1] // timesteps

    X_train = X_train.reshape(
        len(X_train),
        timesteps,
        features
    )

    X_test = X_test.reshape(
        len(X_test),
        timesteps,
        features
    )

    model = Sequential([
        LSTM(
            128,
            return_sequences=True,
            input_shape=(timesteps, features)
        ),
        BatchNormalization(),
        Dropout(0.4),

        LSTM(64),
        BatchNormalization(),
        Dropout(0.4),

        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(
            len(le.classes_),
            activation='softmax'
        )
    ])


    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
    X_train,
    y_train_cat,
    epochs=50,
    batch_size=128,
    validation_data=(
        X_test,
        y_test_cat
    ),
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5
        )
    ],
    class_weight=class_weights
    )

    loss, acc = model.evaluate(
        X_test,
        y_test_cat
    )

    print("\\nAccuracy FINAL:", acc)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(
        y_test_cat,
        axis=1
    )

    print(classification_report(
        y_true,
        y_pred_classes,
        target_names=le.classes_
    ))

    cm = confusion_matrix(y_true, y_pred_classes)

    plt.figure(figsize=(10,8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )

    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusion")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "confusion_matrix.png"
        ),
        dpi=300
    )

    print("Matriz guardada en confusion_matrix.png")

    model.save("lstm_model.keras")
    joblib.dump(le, "label_encoder.pkl")

    print("Modelo guardado")


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    if RUN_DLC:
        run_dlc()

    if GENERATE_EMBEDDINGS:
        generate_embeddings()

    if BUILD_DATASET:
        build_dataset()

    if TRAIN_MODEL:
        train_model()

#C:\Users\Usuario\dlc-clean\Scripts\Activate.ps1