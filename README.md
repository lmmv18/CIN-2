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

## Experimentos realizados

Con el objetivo de mejorar la capacidad de generalización del modelo, se evaluaron distintas estrategias de extracción de características, balanceo de datos y arquitecturas de redes neuronales.

### Configuración base

La configuración con mejor desempeño observada corresponde a:

* Ventana temporal: 45 frames.
* Arquitectura: LSTM (128, 64).
* Características:

  * Posición relativa.
  * Velocidad.
  * Aceleración.
  * Magnitud de velocidad.
  * Distancias entre puntos corporales.

Desempeño obtenido:

| Métrica  | Valor  |
| -------- | ------ |
| Accuracy | 43.2 % |

### Experimentos evaluados

| Experimento                         | Resultado                                     |
| ----------------------------------- | --------------------------------------------- |
| Incorporación de aceleración        | Mejora observada respecto a versiones previas |
| Orientación corporal                | No mejoró el desempeño                        |
| Velocidad angular                   | No mejoró el desempeño                        |
| Oversampling de clases minoritarias | No mejoró el desempeño global                 |
| Ventanas temporales de 60 frames    | No mejoró el desempeño                        |
| Arquitectura GRU                    | Desempeño similar al modelo LSTM              |

### Conclusiones

Los resultados obtenidos sugieren que el principal cuello de botella del sistema no corresponde a la arquitectura recurrente utilizada, ya que tanto LSTM como GRU presentaron desempeños similares.

Las principales limitaciones identificadas actualmente son:

* Desbalance entre clases.
* Cantidad limitada de ejemplos para Grooming y Desplazamiento.
* Posible variabilidad en el etiquetado manual.
* Tamaño reducido del conjunto de entrenamiento.

Estos resultados servirán como línea base para futuras evaluaciones de arquitecturas más complejas y nuevas estrategias de extracción de características.

## Observaciones

La mejora más reciente se obtuvo incorporando características de aceleración derivadas de las trayectorias corporales, aumentando el desempeño del modelo respecto a versiones anteriores.

Actualmente el principal reto corresponde al desbalance de clases, especialmente en los comportamientos Grooming y Desplazamiento.

## Trabajo futuro

Las siguientes mejoras se encuentran en evaluación:

* Revisión y validación de etiquetas manuales.
* Incremento del conjunto de entrenamiento para clases minoritarias.
* Validación cruzada por video para evaluar capacidad de generalización.
* Evaluación de arquitecturas basadas en mecanismos de atención.
* Evaluación de nuevas estrategias de extracción de características.
