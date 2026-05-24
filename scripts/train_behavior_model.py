import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier



X = np.load("X_final.npy")
y = np.load("y_final.npy")

print("Datos:", X.shape)



def clean_label(label):
    label = str(label)

    if ";" in label:
        parts = label.split(";")
        if len(parts) > 1:
            label = parts[1]

    return label.strip()

y = np.array([clean_label(l) for l in y])

# quitar basura final
mask = (y != "") & (y != "nan")
X = X[mask]
y = y[mask]



le = LabelEncoder()
y_enc = le.fit_transform(y)

print("Clases:", le.classes_)



X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.2,
    stratify=y_enc,
    random_state=42
)



model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced_subsample", 
    n_jobs=-1
)

print(" Entrenando...")
model.fit(X_train, y_train)



pred = model.predict(X_test)

print("\n RESULTADOS:")
print(classification_report(y_test, pred, target_names=le.classes_))
