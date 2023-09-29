from http.server import BaseHTTPRequestHandler, HTTPServer
import time

import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import pyodbc
import warnings

hostName = "localhost"
serverPort = 8081

warnings.filterwarnings('ignore')

app = FastAPI()
hits = 0

data_sales = None
data_users = None
data_items = None

matrix = None
U = None
sigma = None
Vt = None

def obtener_datos():
    global data_sales, data_users, data_items, data
    print("init call")
    print("conectando...")
    cnxn_str = ("Driver={ODBC Driver 11 for SQL Server};"
                "Server=181.169.115.183,1433;"
                "Database=F_SISTEMA;"
                "UID=External;"
                "PWD=external2022_123!;")
    cnxn = pyodbc.connect(cnxn_str, timeout=50000)
    
    data_sales = pd.read_sql("""
        SELECT [CodCliente] as CodCliente,
               RTRIM([CodArticu]) as CodArticu,
               (SUM(CantidadPedida + CantPedidaDerivada) / COUNT(*)) as Cantidad
        FROM [F_CENTRAL].[dbo].[VenFer_PedidoReng]
        WHERE CodCliente <> '' AND CodCliente <> 1176
        GROUP BY CodCliente, CodArticu
        ORDER BY CodCliente
    """, cnxn)
    
    data_users = pd.read_sql("""
        SELECT CodCliente,
               CodigoPostal,
               Vendedor,
               Zona,
               LimiteCredito
        FROM F_central.dbo.Ven_Clientes
    """, cnxn)
    
    data_items = pd.read_sql("""
        SELECT RTRIM(CodArticulo) as CodArticu,
               PrecioCosto,
               ArticuloPatron,
               PrecioUnitario
        FROM F_central.dbo.StkFer_Articulos
    """, cnxn)
    
    data = data_sales.merge(data_users, on="CodCliente").merge(data_items, on="CodArticu")
    return "Data obtenida"

@app.get("/metric")
async def metric():
    global hits
    hits += 1
    return {"hits": hits}

@app.get("/health")
async def health():
    return "ok"   

@app.get("/obtenerData")
async def obtener_data():
    return obtener_datos()

@app.get("/prepararModelo")
async def preparar_modelo():
    global matrix, U, sigma, Vt, data
    
    matrix = data.pivot(index='CodCliente', columns='CodArticu', values='Cantidad').fillna(0)
    matrix.to_csv('matrix.csv', index=True)
    matrix = matrix - np.mean(matrix, axis=0)
    sparse_matrix = csr_matrix(matrix.values)
    sp.save_npz("sparse_matrix.npz", sparse_matrix)
    U, sigma, Vt = svds(sparse_matrix, k=90)
    mean_item = np.mean(matrix, axis=0)
    
    
    return "Modelo preparado"

@app.get("/levantarModelo")
async def levantar_modelo():
    global matrix, U, sigma, Vt
    
    matrix = pd.read_csv('matrix.csv', index_col=0)
    #matrix = matrix - np.mean(matrix, axis=0)
    sparse_matrix = sp.load_npz("sparse_matrix.npz")
    U, sigma, Vt = svds(sparse_matrix, k=90)
    mean_item = np.mean(matrix, axis=0)
    
    
    return "Modelo levantado y preparado"

@app.get("/consulta/{customer_id}")
async def consulta(customer_id):
    if matrix is None:
        return "Primero se debe preparar modelo usando '/prepararModelo'"

    if int(customer_id) not in matrix.index:
        return "El ID de usuario no existe"

    user_index = matrix.index.get_loc(int(customer_id))
    user_row = U[user_index, :]
    
    #user_features_row = data_users[data_users['CodCliente'] == int(customer_id)].iloc[:, 1:]  # Adjust the column index as per your user features
    
    if np.count_nonzero(user_row) == 0:
        # En caso de usuario sin historial
        user_items = data[data['CodCliente'] == int(customer_id)]['CodArticu'].unique()
        all_items = data['CodArticu'].unique()
        cold_start_user_recommendations = list(set(all_items) - set(user_items))
        return {"Usuario Cold Start": cold_start_user_recommendations[:10]}

    user_predicted_purchase_counts = np.dot(user_row, np.dot(np.diag(sigma), Vt))
    user_recommendations = pd.DataFrame({'CodArticu': matrix.columns, 'predicted_purchase_count': user_predicted_purchase_counts.flatten()})
    
    user_recommendations = user_recommendations.merge(data_items, on="CodArticu")
    #user_recommendations = user_recommendations.merge(user_features_row, on="CodCliente")
    
    user_recommendations = user_recommendations.sort_values(['predicted_purchase_count', 'PrecioCosto'], ascending=False)

    user_recommendations = user_recommendations[['CodArticu', 'predicted_purchase_count', 'ArticuloPatron']]

    print("Mejores predicciones para el usuario:")
    print(user_recommendations.head(10))

    return "Consulta"


if __name__ == '__main__':
    uvicorn.run(app, host=hostName, port=serverPort)
