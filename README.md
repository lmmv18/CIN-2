# CIN-2

Sistema de clasificación automática de comportamiento animal utilizando DeepLabCut y aprendizaje profundo.

## Descripción

Este proyecto implementa un pipeline completo para la clasificación automática de comportamiento animal a partir de videos.

El sistema utiliza DeepLabCut para extraer coordenadas corporales, genera características espacio-temporales derivadas del movimiento y entrena una red neuronal LSTM para identificar diferentes comportamientos.

Actualmente el pipeline permite:

* Procesar videos mediante DeepLabCut.
* Generar embeddings a partir de trayectorias corporales.
* Construir datasets etiquetados.
* Entrenar modelos LSTM.
* Evaluar desempeño mediante métricas y matrices de confusión.
* Realizar inferencia sobre nuevos videos.

## Comportamientos clasificados

* Desplazamiento
* Grooming
* Inmovilidad
* Rearing
* Resting

## Pipeline

```text
Video
│
├── DeepLabCut
│
├── Coordenadas corporales
│
├── Extracción de características
│
├── Embeddings
│
├── Dataset etiquetado
│
├── LSTM
│
└── Clasificación de comportamiento
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

El flujo principal del proyecto se encuentra en:

```bash
python scripts/behavior_pipeline.py
```

El pipeline permite:

1. Procesar videos mediante DeepLabCut.
2. Generar embeddings a partir de las trayectorias corporales.
3. Construir conjuntos de datos etiquetados.
4. Entrenar una red neuronal LSTM.
5. Evaluar el desempeño del modelo mediante métricas y matrices de confusión.

Archivos generados:

* `lstm_model.keras`
* `label_encoder.pkl`
* `X_train.npy`
* `y_train.npy`
* `X_test.npy`
* `y_test.npy`

## Características utilizadas

Los embeddings se construyen utilizando información derivada de las trayectorias corporales obtenidas con DeepLabCut:

* Posición relativa de puntos corporales.
* Velocidad.
* Aceleración.
* Magnitud de velocidad.
* Distancias entre puntos corporales.

## Resultados actuales

Los resultados reportados corresponden a una partición por video completo, evitando fuga de información entre los conjuntos de entrenamiento y prueba.

| Métrica                 | Valor  |
| ----------------------- | ------ |
| Accuracy                | 43.2 % |
| Videos de entrenamiento | 51     |
| Videos de prueba        | 16     |
| Clases                  | 5      |

## Observaciones

La mejora más reciente se obtuvo incorporando características de aceleración derivadas de las trayectorias corporales, aumentando el desempeño del modelo respecto a versiones anteriores.

Actualmente el principal reto corresponde al desbalance de clases, especialmente en los comportamientos Grooming y Desplazamiento.

## Trabajo futuro

Las siguientes mejoras se encuentran en evaluación:

* Balanceo de clases para Grooming y Desplazamiento.
* Incorporación de orientación corporal y velocidad angular.
* Optimización de la arquitectura LSTM.
* Validación cruzada por video para evaluar capacidad de generalización.
* Evaluación de nuevas estrategias de extracción de características.

```
```


