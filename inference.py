import joblib
import numpy as np

# Modelo entrenado
model = joblib.load("avocado_model.pkl")

# Ejemplo de datos nuevos con o caracterpisticas
X_nuevo = np.array([[0.55, 120.3, 0.78, 0.65, 2, 58.9, 170.2, 145.6]])

# Predecir nivel de maduración
prediccion = model.predict(X_nuevo)

print("Predicción del nivel de maduración:", prediccion)
