## CIN-2

Sistema de clasificación automática de comportamiento animal utilizando DeepLabCut y redes neuronales LSTM.

## Descripción

Este proyecto permite:

1. Extraer coordenadas de puntos corporales mediante DeepLabCut.
2. Generar embeddings espaciales y temporales a partir de las trayectorias.
3. Construir conjuntos de datos etiquetados a partir de anotaciones manuales.
4. Entrenar una red neuronal LSTM para clasificar comportamientos.
5. Realizar inferencia automática sobre videos nuevos.

## Comportamientos clasificados

* Desplazamiento
* Grooming
* Inmovilidad
* Rearing
* Resting

## Arquitectura

```text
Video
  ↓
DeepLabCut
  ↓
Coordenadas (CSV)
  ↓
Extracción de características
  ↓
Embeddings
  ↓
Dataset etiquetado
  ↓
LSTM
  ↓
Clasificación de comportamiento
```


## Requisitos

* Python 3.9
* DeepLabCut 2.3.11
* TensorFlow 2.10.0

Instalación:

```bash
pip install -r requirements.txt
```


## Entrenamiento

Ejecutar:

```bash
python scripts/train/pipeline_completo.py
```

El pipeline realiza:

1. Análisis de videos con DeepLabCut.
2. Generación de embeddings.
3. Construcción del dataset.
4. Entrenamiento de la red LSTM.
5. Evaluación del modelo.

Archivos generados:

* lstm_model.keras
* label_encoder.pkl
* X_final.npy
* y_final.npy


## Resultados obtenidos

Último entrenamiento:

| Métrica  | Valor           |
| -------- | --------------- |
| Accuracy | 93.71%          |
| Macro F1 | 0.93            |
| Dataset  | 40,966 ventanas |

### Desempeño por clase

| Clase          | Precision | Recall | F1   |
| -------------- | --------- | ------ | ---- |
| Desplazamiento | 0.78      | 0.94   | 0.85 |
| Grooming       | 0.88      | 0.99   | 0.93 |
| Inmovilidad    | 0.98      | 0.86   | 0.91 |
| Rearing        | 0.93      | 0.97   | 0.95 |
| Resting        | 0.97      | 1.00   | 0.98 |




