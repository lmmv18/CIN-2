import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import joblib


X = np.load("X_final.npy")
y = np.load("y_final.npy")

print("Datos:", X.shape)



y = np.array([str(label).strip() for label in y])


mask = (y != "") & (y != "nan")
X = X[mask]
y = y[mask]

le = LabelEncoder()
y_enc = le.fit_transform(y)

y_cat = to_categorical(y_enc)

print("Clases:", le.classes_)



weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_enc),
    y=y_enc
)

class_weights = dict(enumerate(weights))

print("Pesos:", class_weights)


timesteps = 15
features = X.shape[1] // timesteps

X_seq = X.reshape(len(X), timesteps, features)


X_train, X_test, y_train, y_test = train_test_split(
    X_seq,
    y_cat,
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

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5
)



print("\n Entrenando modelo...")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights
)



loss, acc = model.evaluate(X_test, y_test)

print("\n Accuracy FINAL:", acc)


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)


print("\n Classification Report:\n")
print(classification_report(
    y_true,
    y_pred_classes,
    target_names=le.classes_
))


cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()


print("\n Distribución de clases:")
print(pd.Series(y).value_counts())



model.save("lstm_model.keras")
joblib.dump(le, "label_encoder.pkl")

print("\n Modelo y encoder guardados")

#python train_lstm_model.py  