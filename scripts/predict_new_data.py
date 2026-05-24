import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from scipy.signal import medfilt

# ================= CONFIG =================

emb_folder = r"C:\Users\Usuario\Desktop\CIN-2\newnew_embeddings"
output_folder = r"C:\Users\Usuario\Desktop\CIN-2\results"

model_path = r"C:\Users\Usuario\Desktop\CIN-2\models\lstm_model.h5"
encoder_path = r"C:\Users\Usuario\Desktop\CIN-2\models\label_encoder.pkl"

timesteps = 15      # igual que entrenamiento
fps = 30
step_size = 5

os.makedirs(output_folder, exist_ok=True)

# ==========================================

# ---------- LOAD MODEL ----------

model = load_model(model_path)
le = joblib.load(encoder_path)

print("✅ Modelo cargado")

# ---------- PROCESAR TODOS LOS ARCHIVOS ----------

files = [f for f in os.listdir(emb_folder) if f.endswith(".npy")]

for file in files:

    print(f"\n🎯 Procesando: {file}")

    # cargar embeddings
    X = np.load(os.path.join(emb_folder, file))

    # reshape para LSTM
    features = X.shape[1] // timesteps
    X_seq = X.reshape(len(X), timesteps, features)

    # ---------- PREDICCIÓN ----------

    y_pred = model.predict(X_seq)

    y_classes = np.argmax(y_pred, axis=1)

    # ✅ SUAVIZADO TEMPORAL (MUY IMPORTANTE)
    y_smooth = medfilt(y_classes, kernel_size=5)

    # convertir a etiquetas
    labels = le.inverse_transform(y_smooth)

    from scipy.signal import medfilt

    y_smooth = medfilt(y_classes, kernel_size=7)
    labels = le.inverse_transform(y_smooth)

    # ---------- TIEMPO ----------

    time_per_window = step_size / fps
    times = np.arange(len(labels)) * time_per_window

    # ---------- GUARDAR RESULTADO ----------

    df = pd.DataFrame({
        "time_sec": times,
        "behavior": labels
    })

    save_path = os.path.join(
        output_folder,
        file.replace(".npy", "_behavior.csv")
    )

    df.to_csv(save_path, index=False)

    print(f"✅ Guardado: {save_path}")

print("\n📊 Conteo de conductas:")
print(df["behavior"].value_counts())

print("\n⏱ Duración (segundos):")
durations = df.groupby("behavior").size() * time_per_window
print(durations)

# transiciones
df["next"] = df["behavior"].shift(-1)
transitions = pd.crosstab(df["behavior"], df["next"])

print("\n🔄 Transiciones:")
print(transitions)

print("\n🚀 TODO COMPLETADO")