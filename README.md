# CIN-2

## Clasificación automática del comportamiento animal mediante aprendizaje profundo

Este repositorio contiene el desarrollo de un sistema de clasificación automática de comportamientos de roedores, utilizando DeepLabCut para la estimación de posiciones corporales, características espaciales y cinemáticas del movimiento, y una red neuronal recurrente Long Short-Term Memory (LSTM) para la clasificación temporal.

El proyecto busca automatizar el análisis del comportamiento animal a partir de grabaciones de video, reduciendo la necesidad de clasificación manual y permitiendo analizar patrones de movimiento mediante técnicas de aprendizaje automático.

## 1. Versión actual del proyecto

La versión de referencia es [behavior_pipeline_v06.qmd](scripts/behavior_pipeline_v06.qmd).

También se conserva la [versión v05](scripts/behavior_pipeline_v05.qmd) para realizar comparaciones metodológicas y de desempeño.

Los resultados se documentan en el [reporte comparativo v05/v06 del 12 de octubre de 2026](reports/Reporte_CIN_v06_v05.pdf).

| Archivo | Descripción |
|---|---|
| `scripts/behavior_pipeline_v06.qmd` | Pipeline de referencia del proyecto |
| `scripts/behavior_pipeline_v05.qmd` | Versión anterior para comparación |
| `scripts/behavior_pipeline.py` | Pipeline histórico |
| `reports/Reporte_CIN_v06_v05.pdf` | Reporte comparativo de resultados |
| `docs/README_historico.md` | Documentación histórica |

Los archivos originales de las versiones v05 y v06 se conservan sin modificar su código.


## 2. Objetivo del proyecto

Desarrollar y evaluar un sistema automatizado capaz de identificar y clasificar comportamientos de roedores mediante el procesamiento de datos de movimiento extraídos de grabaciones de video y el entrenamiento de una red neuronal LSTM.

### Objetivos específicos

1. Obtener información de las posiciones corporales de los animales mediante DeepLabCut.
2. Extraer características espaciales, geométricas y cinemáticas a partir de las trayectorias estimadas.
3. Construir secuencias temporales de características para el entrenamiento de modelos de clasificación.
4. Entrenar y evaluar una red neuronal LSTM para identificar comportamientos.
5. Comparar los resultados de las versiones v05 y v06 mediante métricas de clasificación.

## 3. Comportamientos analizados

El sistema contempla cinco categorías de comportamiento:

| Comportamiento | Descripción general |
|---|---|
| Desplazamiento | Movimiento del animal de una posición a otra |
| Grooming | Actividades de acicalamiento corporal |
| Inmovilidad | Ausencia o reducción significativa del desplazamiento |
| Rearing | Elevación del cuerpo, generalmente apoyándose en las extremidades posteriores |
| Resting | Estado de reposo |

Las diferencias operativas entre Inmovilidad y Resting dependen de los criterios utilizados en las anotaciones originales.

## 4. Estructura general del pipeline

El procesamiento de información se organiza en las siguientes etapas:

```text
Grabaciones de video
        |
        v
DeepLabCut
Estimación de posiciones corporales
        |
        v
Extracción de características
Espaciales, geométricas y cinemáticas
        |
        v
Construcción de secuencias temporales
        |
        v
Normalización y preparación de datos
        |
        v
Modelo LSTM
        |
        v
Predicción de comportamientos
        |
        v
Evaluación del modelo
Precisión, Recall y matrices de confusión
```

DeepLabCut permite obtener las coordenadas de puntos corporales identificados en cada fotograma. A partir de estas coordenadas se calculan las características que posteriormente se utilizan para construir las entradas temporales del modelo.

## 5. Base de datos

Los siguientes datos corresponden a los valores documentados en el reporte comparativo del 12 de octubre de 2026.

| Elemento | Cantidad |
|---|---:|
| Archivos de video disponibles | 69 |
| Videos utilizados en v06 | 67 |
| Videos de entrenamiento | 40 |
| Videos de validación | 11 |
| Videos de prueba | 16 |
| Archivos de anotaciones | 71 |
| Registros de comportamiento | 54 066 |
| Fotogramas cubiertos por anotaciones | 161 985 |

### Distribución de los fotogramas por comportamiento

| Comportamiento | Fotogramas cubiertos |
|---|---:|
| Desplazamiento | 13 632 |
| Grooming | 6 237 |
| Inmovilidad | 53 736 |
| Rearing | 32 037 |
| Resting | 56 343 |
| **Total** | **161 985** |


## 6. Características utilizadas en v06

La documentación de la versión v06 reporta:

- **202 características por fotograma.**
- **45 fotogramas por secuencia temporal.**
- **12 estadísticas adicionales calculadas por ventana.**
- **Frecuencia de procesamiento: 30 FPS.**
- **Desplazamiento entre ventanas: 5 fotogramas.**

Las características extraídas describen aspectos geométricos, espaciales y cinemáticos del movimiento animal.

### 6.1. Características espaciales y geométricas

Incluyen:

- Posiciones corporales en coordenadas egocéntricas.
- Distancias entre puntos corporales.
- Posición y movimiento del centroide corporal.
- Orientación de segmentos corporales.
- Representación angular mediante seno y coseno.

### 6.2. Características cinemáticas

Incluyen:

- Velocidades.
- Aceleraciones.
- Rapidez corporal.
- Movimiento relativo de la cabeza.
- Cambios de orientación.
- Variaciones angulares.

### 6.3. Estadísticas adicionales por ventana

La versión v06 contempla las siguientes estadísticas:

| N.º | Estadística |
|---|---|
| 1 | Media de la rapidez corporal |
| 2 | Desviación estándar de la rapidez corporal |
| 3 | Rapidez corporal mínima |
| 4 | Rapidez corporal máxima |
| 5 | Media de la rapidez del centroide |
| 6 | Desviación estándar de la rapidez del centroide |
| 7 | Recorrido |
| 8 | Desplazamiento neto |
| 9 | Rectitud de la trayectoria |
| 10 | Variabilidad de la separación vertical cabeza-cola |
| 11 | Variabilidad del cambio angular corporal |
| 12 | Indicador de periodicidad del movimiento relativo de la cabeza |


## 7. Mejoras incorporadas en v06

La versión v06 incluye modificaciones destinadas a mejorar la consistencia de las etiquetas, la representación del movimiento y la organización de los resultados.

### 7.1. Corrección de alineación temporal

Las etiquetas se asocian con índices absolutos de fotograma mediante:

```python
round(t * FPS)
```

Este procedimiento busca reducir los problemas de desalineación relacionados con el truncamiento de intervalos temporales y permite considerar el desfase inicial de las anotaciones.

### 7.2. Corrección de normalización

Se elimina una segunda normalización aplicada a distancias obtenidas previamente en coordenadas egocéntricas.

Esto busca evitar transformaciones redundantes de la información geométrica.

### 7.3. Representación de variables angulares

Se incorpora una representación de cuatro ángulos de segmentos corporales mediante sus componentes seno y coseno.

Esta representación permite describir la orientación angular evitando la discontinuidad numérica asociada a los límites convencionales de los ángulos.

### 7.4. Continuidad de los cambios angulares

Se utiliza la función `unwrap` para reducir las discontinuidades al calcular cambios de orientación.

### 7.5. Organización de experimentos

Las características de la versión v06 se almacenan separadamente en:

```text
features_v06/
```

También se utilizan directorios de experimentos para organizar modelos y resultados, facilitando la comparación entre distintas ejecuciones.

## 8. Separación de datos

Las versiones v05 y v06 contemplan una separación por video de los conjuntos de entrenamiento, validación y prueba.

Para v06, el conjunto inicial destinado al entrenamiento se subdivide utilizando:

```python
VAL_SIZE = 0.20
```

La separación por video busca reducir el riesgo de que secuencias procedentes de una misma grabación aparezcan simultáneamente en entrenamiento y prueba.

## 9. Resultados de evaluación en Test

Los siguientes resultados fueron transcritos del reporte comparativo v05/v06 y corresponden a la evaluación reportada sobre el conjunto de prueba.

### 9.1. Precisión y Recall

| Comportamiento | Precisión v05 | Recall v05 | Precisión v06 | Recall v06 |
|---|---:|---:|---:|---:|
| Desplazamiento | 11 % | 22 % | 52 % | 83 % |
| Grooming | 0 % | 0 % | 60 % | 59 % |
| Inmovilidad | 48 % | 36 % | 80 % | 55 % |
| Rearing | 54 % | 53 % | 80 % | 93 % |
| Resting | 73 % | 85 % | 82 % | 90 % |
| **Promedio macro** | **37 %** | **39 %** | **71 %** | **76 %** |
| **Promedio ponderado** | **55 %** | **57 %** | **78 %** | **78 %** |

### 9.2. Análisis comparativo

Los resultados reportados muestran una mejora general del desempeño del clasificador en la versión v06 respecto a v05.

La precisión macro aumenta del 37 % al 71 %, mientras que el Recall macro pasa del 39 % al 76 %.

Estas diferencias corresponden a incrementos de 34 y 37 puntos porcentuales, respectivamente.

Se observan mejoras especialmente relevantes en Grooming y Desplazamiento, categorías que presentaban dificultades de clasificación en la versión anterior.

En particular:

- Grooming pasa de una precisión del 0 % al 60 %.
- Desplazamiento aumenta su precisión del 11 % al 52 %.
- Rearing alcanza una precisión del 80 % y un Recall del 93 %.
- Resting alcanza una precisión del 82 % y un Recall del 90 %.

Sin embargo, persisten limitaciones en la identificación de ciertos comportamientos.

La categoría Inmovilidad presenta un Recall del 55 % en v06, lo cual indica que una proporción considerable de los ejemplos reales de esta categoría no es detectada correctamente.

En conjunto, los resultados sugieren que las modificaciones implementadas en v06 están asociadas con un mejor desempeño del modelo. No obstante, no permiten atribuir la mejora a una modificación específica sin experimentos adicionales.

### 9.3. Limitaciones de los resultados

Los resultados de esta sección provienen del reporte y no fueron reproducidos durante la actualización del repositorio.

El reporte no permite identificar de manera inequívoca si la tabla corresponde a predicciones anteriores o posteriores al suavizado temporal.

El valor histórico de **58,1 % de accuracy** documentado anteriormente no corresponde a un resultado verificado de v06 y no debe presentarse como tal.

No se reportan valores independientes de accuracy ni F1 para la versión v06, por lo que no se incluyen estimaciones adicionales de estas métricas.

## 10. Organización del repositorio

```text
CIN-2/
|
|-- docs/
|   |-- README_historico.md
|
|-- reports/
|   |-- Reporte_CIN_v06_v05.pdf
|
|-- scripts/
|   |-- behavior_pipeline.py
|   |-- behavior_pipeline_v05.qmd
|   |-- behavior_pipeline_v06.qmd
|   |-- train_videos.txt
|   |-- test_videos.txt
|
|-- .gitignore
|-- README.md
```

### Archivos no incluidos

Debido al tamaño de la base de datos y de los archivos generados durante el procesamiento, no se incluyen en esta actualización:

- Grabaciones originales de video.
- Archivos de anotaciones originales.
- Proyecto completo de DeepLabCut.
- Conjuntos de características generadas.
- Datasets de entrenamiento.
- Pesos y modelos entrenados.
- Resultados completos de las ejecuciones locales.

Estos recursos se mantienen localmente y son necesarios para reproducir determinadas etapas del procesamiento y el entrenamiento.

## 11. Preparación del entorno

Los archivos `.qmd` corresponden a documentos Quarto que contienen texto y bloques de código Python.

No deben ejecutarse directamente mediante:

```bash
python archivo.qmd
```

El entorno utilizado es:

- Python 3.9.
- DeepLabCut 2.3.11.
- TensorFlow 2.10.0.


### Dependencias principales

Entre las bibliotecas utilizadas se encuentran:

- NumPy.
- pandas.
- SciPy.
- scikit-learn.
- TensorFlow/Keras.
- Matplotlib.
- seaborn.
- joblib.
- DeepLabCut, cuando se ejecuta esa etapa.

Para trabajar con los documentos pueden utilizarse un editor compatible con Python y Jupyter, así como Quarto para su renderización.

## 12. Instrucciones generales de ejecución

### Paso 1. Preparar los datos

Colocar los recursos locales en las rutas esperadas por el código:

```text
videos/
labels/
dlc_project/Proyecto1-Laura-2026-04-13/
```

### Paso 2. Abrir el pipeline

Abrir:

```text
scripts/behavior_pipeline_v06.qmd
```

en un entorno compatible con Quarto y Python.

### Paso 3. Revisar el directorio de trabajo

El código utiliza:

```python
BASE_DIR = os.path.dirname(os.getcwd())
```

Por lo tanto, se debe establecer el directorio de trabajo en:

```text
CIN-2/scripts
```

y comprobar que la ruta base generada corresponda al directorio principal del proyecto.

### Paso 4. Revisar la configuración

Antes de ejecutar, verificar:

- `CONFIG_PATH`.
- Nombres de puntos corporales.
- Rutas de videos.
- Archivos de anotaciones.
- Listas de entrenamiento y prueba.
- Configuración de las etapas del procesamiento.

### Paso 5. Configurar las etapas

El menú de ejecución incluye:

| Parámetro | Valor documentado | Función |
|---|---|---|
| `RUN_DLC` | `False` | Omite la ejecución de DLC |
| `RUN_FEATURES` | `True` | Genera características |
| `RUN_BUILD_DATASET` | `True` | Construye los conjuntos de datos |
| `RUN_TRAIN_MODEL` | `True` | Entrena un modelo |
| `RUN_TEST` | `True` | Realiza la evaluación sobre Test |

Cuando `RUN_DLC = False`, deben existir previamente las salidas requeridas de DeepLabCut.

Durante el desarrollo y ajuste de hiperparámetros se recomienda trabajar con validación y reservar el conjunto Test para la evaluación final.

Si se desea evaluar un modelo previamente entrenado sin ejecutar nuevamente el entrenamiento, deben revisarse los parámetros:

```python
EXISTING_RUN_MODELS
EXISTING_RUN_RESULTS
```

### Paso 6. Ejecutar el pipeline

Ejecutar los bloques de código en orden, comprobando las rutas, las entradas disponibles y las salidas generadas en cada etapa.

El encabezado de los documentos incluye:

```yaml
execute:
  eval: false
```

Por ello, renderizar el documento no ejecuta automáticamente el entrenamiento.

## 13. Salidas generadas

La versión v06 contempla los siguientes directorios de salida:

| Directorio | Contenido |
|---|---|
| `features_v06/` | Características extraídas |
| `datasets/` | Conjuntos de datos y metadatos |
| `models/experiments/` | Modelos entrenados y escaladores |
| `results/experiments/` | Métricas, predicciones, historiales y matrices de confusión |

La disponibilidad de las salidas depende de las etapas ejecutadas y de la configuración utilizada.

## 14. Reproducibilidad y consideraciones

Para reproducir los resultados documentados es necesario disponer de los datos originales, las anotaciones, las salidas de DeepLabCut, la configuración del entorno y la definición de los conjuntos de entrenamiento, validación y prueba.


## 15. Posibles líneas de mejora

Como trabajo futuro se propone:

1. Evaluar con mayor detalle los errores de clasificación entre comportamientos similares.
2. Investigar el bajo Recall de algunas categorías, especialmente Inmovilidad.
3. Analizar el efecto de las características temporales y estadísticas adicionales.
4. Comparar diferentes configuraciones de arquitectura e hiperparámetros.
5. Evaluar sistemáticamente el efecto del suavizado temporal.
6. Mejorar la reproducibilidad mediante archivos de dependencias y configuraciones de ejecución.
7. Realizar estudios de ablación para determinar la contribución individual de las modificaciones incorporadas en v06.

## 16. Estado actual

**Versión de referencia:** v06.

**Versión de comparación:** v05.

**Último reporte incorporado:** 12 de octubre de 2026.

**Estado:** pipeline documentado con resultados comparativos reportados; reproducción independiente y validación completa del entorno pendientes.

--

**Proyecto CIN-2 — Universidad de Costa Rica.**
