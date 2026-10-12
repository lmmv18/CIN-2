# CIN-2

Clasificación automática de comportamiento animal mediante DeepLabCut, características de movimiento y una red LSTM.

## Versión actual

La versión de referencia es [behavior_pipeline_v06.qmd](scripts/behavior_pipeline_v06.qmd). Se incluye [v05](scripts/behavior_pipeline_v05.qmd) para comparación y el [reporte v05/v06 del 12 de octubre de 2026](reports/Reporte_CIN_v06_v05.pdf).

Los archivos originales se conservan sin modificar su código. La v05 recibida mantiene un encabezado interno que dice “v04”; su nombre identifica la versión proporcionada. La v06 corresponde a la última copia enviada, sin los dos bloques auxiliares finales de inventario local.

El archivo `scripts/behavior_pipeline.py` corresponde al pipeline histórico. No es la versión v06. La documentación anterior se conserva en [docs/README_historico.md](docs/README_historico.md).

## Comportamientos

Desplazamiento, Grooming, Inmovilidad, Rearing y Resting.

## Datos reportados

Fuente: reporte adjunto; los conteos no se recalcularon a partir de los videos en esta actualización.

| Elemento | Cantidad |
|---|---:|
| Archivos de video en la base | 69 |
| Videos utilizados en v06 | 67 |
| Videos de entrenamiento | 40 |
| Videos de validación | 11 |
| Videos de prueba | 16 |
| Archivos de anotaciones | 71 |
| Registros de comportamiento | 54 066 |
| Fotogramas cubiertos por anotaciones | 161 985 |

Los fotogramas cubiertos por anotaciones no representan necesariamente fotogramas etiquetados manualmente de manera individual ni el número de ventanas utilizadas para entrenar.

| Comportamiento | Fotogramas cubiertos |
|---|---:|
| Desplazamiento | 13 632 |
| Grooming | 6 237 |
| Inmovilidad | 53 736 |
| Rearing | 32 037 |
| Resting | 56 343 |

## Características y configuración v06

El reporte indica **202 características por fotograma**, secuencias de **45 fotogramas** y **12 estadísticas adicionales por ventana**. El código utiliza 30 FPS y un paso de 5 fotogramas.

Las características incluyen posiciones egocéntricas, velocidades, aceleraciones, rapidez, distancias corporales, movimiento del centroide, movimiento relativo de la cabeza, orientaciones y cambios angulares.

Las estadísticas por ventana son:

1. Media de la rapidez corporal.
2. Desviación estándar de la rapidez corporal.
3. Rapidez corporal mínima.
4. Rapidez corporal máxima.
5. Media de la rapidez del centroide.
6. Desviación estándar de la rapidez del centroide.
7. Recorrido.
8. Desplazamiento neto.
9. Rectitud de la trayectoria.
10. Variabilidad de la separación vertical cabeza-cola.
11. Variabilidad del cambio angular corporal.
12. Indicador de periodicidad del movimiento relativo de la cabeza.

## Cambios incorporados en v06

- Etiquetas ancladas a índices absolutos de fotograma mediante `round(t * FPS)`, evitando la deriva por truncar intervalos y respetando el desfase inicial.
- Eliminación de la segunda normalización de las distancias calculadas en coordenadas egocéntricas.
- Representación de cuatro ángulos de segmentos mediante seno y coseno.
- Uso de `unwrap` para evitar discontinuidades angulares al calcular cambios de orientación.
- Salidas separadas en `features_v06` y directorios de experimentos v06.

Ambas versiones contienen separación por video en entrenamiento, validación y prueba. En v06, el conjunto original de entrenamiento se divide con `VAL_SIZE = 0.20` y semilla 42; el código comprueba que los identificadores de los tres grupos sean disjuntos. Las listas originales del repositorio se conservan; su correspondencia exacta con la ejecución del reporte no se verificó con los datos originales.

## Resultados reportados en Test

Transcripción del reporte adjunto. Estos resultados no se han reproducido en esta actualización. El PDF no identifica si la tabla corresponde a las predicciones antes o después del suavizado.

| Comportamiento | Precisión v05 | Recall v05 | Precisión v06 | Recall v06 |
|---|---:|---:|---:|---:|
| Desplazamiento | 11 % | 22 % | 52 % | 83 % |
| Grooming | 0 % | 0 % | 60 % | 59 % |
| Inmovilidad | 48 % | 36 % | 80 % | 55 % |
| Rearing | 54 % | 53 % | 80 % | 93 % |
| Resting | 73 % | 85 % | 82 % | 90 % |
| Promedio macro | 37 % | 39 % | 71 % | 76 % |
| Promedio ponderado | 55 % | 57 % | 78 % | 78 % |

El **58,1 % de accuracy** del README anterior es un resultado histórico; no se atribuye a v06. El nuevo reporte no incluye una fila independiente de accuracy ni valores de F1, por lo que no se añaden estimaciones de esas métricas.

## Archivos del repositorio

| Ruta | Contenido |
|---|---|
| `scripts/behavior_pipeline_v06.qmd` | Pipeline actual en Quarto con bloques Python |
| `scripts/behavior_pipeline_v05.qmd` | Versión anterior suministrada |
| `scripts/behavior_pipeline.py` | Script histórico |
| `scripts/train_videos.txt`, `scripts/test_videos.txt` | Listas históricas de partición |
| `reports/Reporte_CIN_v06_v05.pdf` | Conteos y métricas reportados |
| `docs/README_historico.md` | Documentación previa completa |

Los videos, anotaciones, proyecto DLC, características, datasets, pesos y resultados de ejecución se mantienen localmente y no están incluidos en esta actualización.

## Preparación y ejecución

Los archivos `.qmd` contienen texto y bloques Python; no se ejecutan con `python archivo.qmd`.

El entorno histórico documentaba Python 3.9, DeepLabCut 2.3.11 y TensorFlow 2.10.0. No se proporcionó un archivo de dependencias fijadas para v05/v06 ni se validó una instalación nueva. Los imports requieren NumPy, pandas, SciPy, scikit-learn, joblib, TensorFlow, Matplotlib y seaborn; DeepLabCut se necesita si se activa esa etapa. Para ejecutar los bloques desde un editor se necesita un entorno Python/Jupyter compatible; para renderizar el documento se necesita Quarto.

1. Colocar los datos locales en `videos/`, `labels/` y `dlc_project/Proyecto1-Laura-2026-04-13/`, según las rutas configuradas.
2. Abrir `scripts/behavior_pipeline_v06.qmd` en un editor compatible con Quarto/Python.
3. Establecer el directorio de trabajo de Python en `CIN-2/scripts`. El código calcula `BASE_DIR = os.path.dirname(os.getcwd())`; comprobar el valor que imprime antes de continuar.
4. Revisar `CONFIG_PATH`, los nombres de puntos corporales, las anotaciones y las listas de videos.
5. Ejecutar los bloques en orden, revisando el menú antes de ejecutar su bloque.

Configuración incluida en el menú:

| Opción | Valor enviado | Acción |
|---|---|---|
| `RUN_DLC` | `False` | Omite análisis DLC; requiere sus salidas existentes |
| `RUN_FEATURES` | `True` | Genera características |
| `RUN_BUILD_DATASET` | `True` | Construye los tres conjuntos |
| `RUN_TRAIN_MODEL` | `True` | Entrena un modelo |
| `RUN_TEST` | `True` | Evalúa en prueba |

Para ajustes durante el desarrollo, cambiar `RUN_TEST = False` y usar validación; reservar Test para la evaluación final. Si se evalúa un modelo existente sin entrenar uno nuevo, completar `EXISTING_RUN_MODELS` y `EXISTING_RUN_RESULTS`.

El encabezado tiene `execute: eval: false`: renderizar el documento no ejecuta el entrenamiento de forma automática.

## Salidas y reproducibilidad

El código guarda características en `features_v06/`, datasets y metadatos en `datasets/`, modelos y escaladores en `models/experiments/`, y métricas, predicciones, historial y matrices de confusión en `results/experiments/`.

Para reproducir los resultados se requieren los datos originales y las salidas de la ejecución correspondiente. En esta actualización se verificó la sintaxis de los bloques Python de ambos archivos y se cotejó la tabla del PDF; no se ejecutó DeepLabCut ni se reentrenaron modelos.
