from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cargar el modelo y el escalador
model = joblib.load("avocado_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict")
def predict(
    request: Request,
    firmness: float = Form(...),
    hue: float = Form(...),
    saturation: float = Form(...),
    brightness: float = Form(...),
    color_category: float = Form(...),
    sound_db: float = Form(...),
    weight_g: float = Form(...),
    size_cm3: float = Form(...)
):
    # Convertir a array y escalar
    X = np.array([[firmness, hue, saturation, brightness, color_category, sound_db, weight_g, size_cm3]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)

    return templates.TemplateResponse("form.html", {
        "request": request,
        "result": f"Predicción del nivel de maduración: {prediction[0]}"
    })
