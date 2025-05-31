#Clasificación de maduración de aguacate

Este repositorio contiene un modelo de aprendizaje automático entrenado para predecir el **nivel de maduración de aguacates** (`ripeness`) a partir de distintas características físicas.

##Contenido

- `avocado_model.pkl`: modelo de clasificación entrenado con scikit-learn.
- `inference.py`: script de inferencia para hacer predicciones con nuevos datos.

##Características de entrada

El modelo fue entrenado con las siguientes 8 características:

1. `firmness`: firmeza del aguacate
2. `hue`: matiz del color
3. `saturation`: saturación del color
4. `brightness`: brillo
5. `color_category`: categoría de color codificada (número)
6. `sound_db`: nivel de sonido (en decibeles)
7. `weight_g`: peso en gramos
8. `size_cm3`: volumen en cm³

## Variable objetivo

- `ripeness`: nivel de maduración (clasificación)

## 🚀 Uso

### 1. Clona el repositorio

#```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo

### 2. Instala kas dependencias necesarias
pip install scikit-learn numpy joblib
### 3. Ejecuta el script inference.py
