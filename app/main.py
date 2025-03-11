from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import adadelta
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/adadelta")
def calculo(samples: int, n_features: int, rho: float, eps: float):
    output_file = 'adadelta.png'
    
    # Generar datos de prueba
    np.random.seed(42)
    x = np.linspace(0, 10, samples)
    y = 2 * x + 3 + np.random.normal(0, 2, 100)

    # Inicializar Adadelta
    #rho=0.95
    #eps = 1e-6
    optimizer = adadelta.Adadelta(n_features, rho, eps)
    weights = np.array([0.0, 0.0])
    losses = []

    # Entrenamiento
    for _ in range(100):
        y_pred = weights[0] * x + weights[1]
        error = y_pred - y
        loss = np.mean(error ** 2)
        losses.append(loss)
    
        gradients = np.array([
            np.mean(2 * error * x),  # Gradiente respecto a la pendiente
            np.mean(2 * error)        # Gradiente respecto a la intercepción
        ])
    
        optimizer.update(gradients)
        weights = np.array(optimizer.get_weights())

    # Gráfica de dispersión
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, label="Datos reales")
    plt.plot(x, weights[0] * x + weights[1], color="red", label="Regresión ajustada")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Ajuste de Adadelta")

    # Gráfica de la pérdida
    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Pérdida")
    plt.xlabel("Iteraciones")
    plt.ylabel("MSE")
    plt.title("Evolución de la pérdida")
    plt.legend()
    #plt.show()

    plt.savefig(output_file)
    plt.close()
    
    j1 = {
        "Grafica": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/adadelta-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)
