import os
import numpy as np
import pandas as pd
from collections import Counter
import re

emb_folder = r"C:\Users\Usuario\Desktop\CIN-2\new_embeddings" # carpeta de los embeddings
labels_folder = r"C:\Users\Usuario\Desktop\CIN-2\labels" # carpetas de las etiquetas manuales

fps = 30
window_size = 15
step_size = 5



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

def expand_time_to_frames(df, fps):

    df = df.dropna(axis=1, how='all')

    times = df.iloc[:, 0].values
    labels = df.iloc[:, 1].values

    frame_labels = []

    for i in range(len(df) - 1):

        t1 = float(str(times[i]).replace(",", "."))
        t2 = float(str(times[i+1]).replace(",", "."))

        label = str(labels[i])

        duration = t2 - t1
        n_frames = int(duration * fps)

        frame_labels.extend([label] * n_frames)

    return np.array(frame_labels)



emb_files = [f for f in os.listdir(emb_folder) if f.endswith(".npy")]
label_files = [f for f in os.listdir(labels_folder) if f.endswith(".csv")]

label_map = {clean_name(f): f for f in label_files}

X_all = []
y_all = []


for emb_file in emb_files:

    emb_base = os.path.splitext(emb_file)[0]
    emb_clean = clean_name(emb_base)

    matched_label = None

    for key in label_map:
        if key in emb_clean or emb_clean in key:
            matched_label = label_map[key]
            break

    if matched_label is None:
        continue

    X = np.load(os.path.join(emb_folder, emb_file))
    df = read_csv_safe(os.path.join(labels_folder, matched_label))

    frame_labels = expand_time_to_frames(df, fps)

    window_labels = []

    for i in range(len(X)):

        start = i * step_size
        end = start + window_size

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

print("\n✅ DATASET FINAL")
print("X:", X_all.shape)
print("y:", y_all.shape)
