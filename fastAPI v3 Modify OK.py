# Python 3 server example
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

import uvicorn
from fastapi import FastAPI

#Se importan librerias necesarias
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix


# imports for SQL data part
import pyodbc

import warnings

hostName = "localhost"
serverPort = 8081


#desactivando Warnings
warnings.filterwarnings('ignore')

app = FastAPI()
hits = 0
data = None

matrix = None

U = None
sigma = None
Vt = None

def obtener_datos():
    global data
    print("init call")
    print("conectando...")
    #Lucas --> ODBC Driver 11 for SQL Server
    #Juan --> SQL Server Native Client 11.0
    
    # Carga de data
    
    data = pd.read_csv("/Users/Usuario/Downloads/user_item_data.csv", sep=";", names=["CodCliente", "CodArticu", "Cantidad"])
    data.head(5)
    print("obtenido data")
    print(data)
    return "data obtenida"



#@app.on_event('startup')
#def init_data():
    
    
@app.get("/metric")
async def metric():
    global hits
    hits+=1
    return {"hits": hits}

@app.get("/health")
async def health():
    return "ok"

@app.get("/obtenerData")
async def metric():
    return obtener_datos()

@app.get("/prepararModelo")
async def health():
    global matrix
    global U
    global sigma
    global Vt

    # Se crea matriz usuario-item
    matrix = data.pivot(index='CodCliente', columns='CodArticu', values='Cantidad').fillna(0)

    # Centramos los datos y se verifican valores no numericos y convierten en numericos
    matrix = matrix - np.mean(matrix, axis=0)

    # El SVD requiere una matriz dispersa como input, aca se convierte en eso
    sparse_matrix = csr_matrix(matrix.values)

    # SVD
    U, sigma, Vt = svds(sparse_matrix, k=50) # k is the number of singular values to compute

    # predicciones para todos los usuarios
    mean_item = np.mean(matrix, axis=0)

    
    return "modelo Preparado"



@app.get("/consulta/{customer_id}")
async def consulta(customer_id):
    customer_id_int = int(customer_id)

    user_row = U[customer_id_int-1, :]

    user_predicted_purchase_counts = np.dot(user_row, np.dot(np.diag(sigma), Vt))

    user_recommendations = pd.DataFrame({'CodArticu': matrix.columns, 'predicted_purchase_count': user_predicted_purchase_counts.flatten()})
    user_recommendations = user_recommendations.sort_values('predicted_purchase_count', ascending=False)

    print("Mejores predicciones para el usuario:")
    print(user_recommendations.head(10))

    return "consulta"

#Aca termina Notebook

    
if __name__ == '__main__':
    uvicorn.run(app, host=hostName, port=serverPort)#, log_level="info")


