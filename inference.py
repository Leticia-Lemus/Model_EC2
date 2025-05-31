import joblib
import numpy as np

# Modelo entrenado
model = joblib.load("avocado_model.pkl")
scaler = joblib.load("scaler.pkl")

# Ejemplo de datos nuevos con o caracterpisticas
X_nuevo = np.array([[0.55, 120.3, 0.78, 0.65, 2, 58.9, 170.2, 145.6]])
#Aplicar escalamiento
X_nuevo_escalado = scaler.transform(X_nuevo)

# Predecir nivel de maduración
prediccion = model.predict(X_nuevo_escalado)

print("Predicción del nivel de maduración:", prediccion)
