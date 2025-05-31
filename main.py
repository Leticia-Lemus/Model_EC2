from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cargar modelo y scaler
model = joblib.load("avocado_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request,
                  f1: float = Form(...), f2: float = Form(...),
                  f3: float = Form(...), f4: float = Form(...),
                  f5: float = Form(...), f6: float = Form(...),
                  f7: float = Form(...), f8: float = Form(...)):
    
    X = np.array([[f1, f2, f3, f4, f5, f6, f7, f8]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]

    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": prediction
    })
