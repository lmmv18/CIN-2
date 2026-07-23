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

El mejor desempeño obtenido hasta el momento se logró utilizando un escalado global de las características, ajustando el `StandardScaler` únicamente con el conjunto de entrenamiento y aplicándolo posteriormente tanto al entrenamiento como a la prueba.

El conjunto de datos contiene **8648 ventanas temporales**, distribuidas en **51 videos de entrenamiento** y **16 videos de prueba**.

| Métrica                 | Valor   |
| ----------------------- | ------- |
| Accuracy                | 58.1 %  |
| Macro F1-score          | 0.43    |
| Weighted F1-score       | 0.59    |
| Videos de entrenamiento | 51      |
| Videos de prueba        | 16      |
| Clases                  | 5       |
| Ventanas totales        | 8648    |
## Experimentos realizados

Con el objetivo de mejorar la capacidad de generalización del modelo, se evaluaron distintas estrategias de extracción de características, balanceo de datos y arquitecturas de redes neuronales.

### Configuración base

La configuración con mejor desempeño corresponde a:

* Ventana temporal: **45 frames**
* Arquitectura: **LSTM (128, 64)**
* Características:
  * Posición relativa.
  * Velocidad.
  * Aceleración.
  * Magnitud de velocidad.
  * Distancias entre puntos corporales.
* Embeddings generados **sin escalado previo**.
* Escalado global mediante `StandardScaler`, ajustado únicamente con el conjunto de entrenamiento (`fit` en `X_train` y `transform` en `X_train` y `X_test`).

Desempeño obtenido:

| Métrica | Valor |
|---------|------:|
| Accuracy | **58.1 %** |
| Macro F1 | **0.43** |
| Weighted F1 | **0.59** |

| Experimento | Resultado |
|-------------|-----------|
| Incorporación de aceleración | Mejoró el desempeño respecto a versiones previas. |
| Orientación corporal | No mejoró el desempeño. |
| Velocidad angular | No mejoró el desempeño. |
| Oversampling de clases minoritarias | No produjo mejoras globales. |
| Ventanas temporales de 60 frames | No mejoró el desempeño. |
| Arquitectura GRU | Desempeño similar a LSTM. |
| Eliminación completa del escalado | Incrementó la accuracy, pero redujo considerablemente el desempeño en clases minoritarias. |
| Escalado global (fit en entrenamiento) | Mejor desempeño global obtenido. |
| Entrenamiento sin mezclar ventanas (`shuffle=False`) | Disminuyó significativamente el desempeño del modelo. |
| Arquitectura GRU                    | Desempeño similar al modelo LSTM              |

### Conclusiones

Los experimentos realizados muestran que la calidad del preprocesamiento tiene un impacto mayor que la arquitectura recurrente utilizada.

La mejora más importante se obtuvo modificando la estrategia de normalización, utilizando un escalado global ajustado únicamente con el conjunto de entrenamiento, lo que permitió mejorar significativamente la capacidad de generalización del modelo.

Asimismo, se observó que mantener las ventanas ordenadas durante el entrenamiento (`shuffle=False`) deteriora el aprendizaje, mientras que el entrenamiento con mezcla aleatoria de ventanas proporciona una mejor convergencia del optimizador.

Las principales limitaciones identificadas actualmente son:

* Desbalance entre clases.
* Cantidad limitada de ejemplos para Grooming y Desplazamiento.
* Posible variabilidad en el etiquetado manual.
* Tamaño reducido del conjunto de entrenamiento.

## Observaciones

Actualmente el mejor desempeño del sistema se obtiene utilizando:

* Embeddings sin escalado.
* Escalado global mediante `StandardScaler` ajustado únicamente con el conjunto de entrenamiento.
* Separación completa de videos entre entrenamiento y prueba.
* Entrenamiento con mezcla aleatoria de ventanas (`shuffle=True`).

Esta configuración alcanzó una accuracy del **58.1 %**, constituyendo la línea base actual del proyecto.

## Trabajo futuro
* Incorporación de mecanismos de atención (Attention).
* Evaluación de arquitecturas Transformer para series temporales.
* Evaluación de nuevas estrategias de balanceo de clases.
* Optimización del tamaño de la ventana temporal y del solapamiento.
