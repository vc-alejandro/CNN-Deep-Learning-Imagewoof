# Clasificación de razas de perros con una CNN (Imagewoof)
Implementación de un modelo de deep learning (Keras + TensorFlow).

CNN entrenada **desde cero** (sin transfer learning) para clasificar fotografías de perros en 10 razas, usando el dataset [Imagewoof](https://github.com/fastai/imagenette#imagewoof) (versión de 320 px).

## Contenido

| Archivo | Descripción |
|---|---|
| `imagewoof_cnn.ipynb` | Notebook con todo el desarrollo: datos, modelo, ajuste, resultados, conclusiones y aplicación |
| `models/imagewoof_cnn.keras` | Modelo final entrenado (arquitectura + pesos) |

El dataset **no** se incluye: el notebook lo descarga y descomprime automáticamente.

## Resultados

| Modelo | Accuracy (validación) |
|---|---|
| 1. CNN base | 45.9 % |
| 2. CNN + regularización (augmentation, Dropout, label smoothing) | 63.9 % |
| 3. CNN de doble convolución por bloque | **71.9 %** |

El modelo final (3) obtiene **72.7 % de accuracy en el conjunto de prueba** (F1 macro 0.708).

## Cómo ejecutarlo

1. Abre `imagewoof_cnn.ipynb` en Google Colab con una GPU (Entorno de ejecución → Cambiar tipo de entorno).
2. Ejecuta todas las celdas (`Ejecutar todo`). El dataset se descarga solo.
3. Para predecir con el modelo guardado:

```python
import keras

model = keras.models.load_model("models/imagewoof_cnn.keras")
predict_image("ruta/a/tu_imagen.jpg")  # devuelve la raza y su probabilidad
```

`predict_image` está definida en la sección *Aplicación* del notebook.

## Dependencias

Python 3, TensorFlow 2.21 y Keras 3.13 (preinstalados en Colab), además de numpy, matplotlib y scikit-learn.

## Autor

Alejandro Vázquez de la Cruz · Tecnológico de Monterrey
