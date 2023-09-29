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

import random
from sklearn.metrics.pairwise import cosine_similarity

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
    
    #cnxn_str = ("Driver={SQL Server Native Client 11.0};"
    cnxn_str = ("Driver={ODBC Driver 11 for SQL Server};"
                "Server=181.169.115.183,1433;"
                "Database=F_SISTEMA;"
                "UID=External;"
                "PWD=external2022_123!;")
    cnxn = pyodbc.connect(cnxn_str, timeout=50000)
    
    data_sales = pd.read_sql("""
         select 
         cli.CodCliente as CodCliente
        ,RTRIM(art.CodArticulo) as CodArticu
        ,(coalesce(SUM(reng.CantidadPedida+reng.CantPedidaDerivada),0)/(COUNT(reng.NroRenglon)+1)) as Cantidad
        from  f_central.dbo.ven_clientes as cli
	  inner join f_central.dbo.StkFer_Articulos as art
              on 1 = 1
	  left join F_CENTRAL.dbo.VenFer_PedidoReng as reng
              on reng.CodCliente = cli.CodCliente
              and reng.CodArticu = art.CodArticulo
        where cli.codCliente in (1176,186,2001,36,35,78,252,154,145,112,201,203)
        group by cli.CodCliente,art.CodArticulo
        order by cli.CodCliente
    """, cnxn)
    #data_sales.to_csv('data_sales.csv', index=True)
    data_users = pd.read_sql("""
        SELECT CodCliente,
               CodigoPostal,
               Vendedor,
               Zona,
               LimiteCredito
        FROM F_central.dbo.Ven_Clientes
    """, cnxn)
    #data_users.to_csv('data_users.csv', index=True)
    data_items = pd.read_sql("""
        SELECT RTRIM(CodArticulo) as CodArticu,
               PrecioCosto,
               ArticuloPatron,
               PrecioUnitario
        FROM F_central.dbo.StkFer_Articulos
    """, cnxn)
    #data_items.to_csv('data_items.csv', index=True)
    
    data = data_sales.merge(data_users, on="CodCliente").merge(data_items, on="CodArticu")
    #data.to_csv('data.csv', index=True)
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
    U, sigma, Vt = svds(sparse_matrix, k=2)
    # sin uso -> mean_item = np.mean(matrix, axis=0)
    
    
    return "Modelo preparado"

@app.get("/levantarModelo")
async def levantar_modelo():
    global matrix, U, sigma, Vt,data_items
    #Levanto la matrix
    matrix = pd.read_csv('matrix.csv', index_col=0)
    #matrix = matrix - np.mean(matrix, axis=0)
    sparse_matrix = sp.load_npz("sparse_matrix.npz")
    print(min(sparse_matrix.shape))
    U, sigma, Vt = svds(sparse_matrix, k=2)
    # sin uso -> mean_item = np.mean(matrix, axis=0)
    
    return "Modelo levantado y preparado"

@app.get("/consulta/{customer_id}")
async def consulta(customer_id):
    if matrix is None:
        return "Primero se debe preparar modelo usando '/prepararModelo'"

    if int(customer_id) not in matrix.index:
        return "El ID de usuario no existe"

    user_index = matrix.index.get_loc(int(customer_id))
    user_row = U[user_index, :]
    
    if np.count_nonzero(user_row) == 0:
        # En caso de usuario sin historial
        user_items = data[data['CodCliente'] == int(customer_id)]['CodArticu'].unique()
        all_items = data['CodArticu'].unique()
        cold_start_user_recommendations = list(set(all_items) - set(user_items))
        return {"Usuario Cold Start": cold_start_user_recommendations[:10]}

    user_predicted_purchase_counts = np.dot(user_row, np.dot(np.diag(sigma), Vt))
    
    user_recommendations = pd.DataFrame({'CodArticu': matrix.columns, 'predicted_purchase_count': user_predicted_purchase_counts.flatten()})

    # Calculate item-item similarity matrix
    item_similarity = cosine_similarity(matrix.T)

    # Calculate diversity scores for each item
    diversity_scores = np.sum(item_similarity, axis=0)

    # Add diversity scores to the user_recommendations DataFrame
    user_recommendations['diversity_score'] = diversity_scores

    # Sort recommendations by predicted purchase count and diversity score
    user_recommendations = user_recommendations.sort_values(['predicted_purchase_count', 'diversity_score'], ascending=[False, False])

    user_recommendations = user_recommendations[['CodArticu', 'predicted_purchase_count']]

    print("Mejores predicciones para el usuario:")
    print(user_recommendations.head(10))

    return "Consulta"


if __name__ == '__main__':
    uvicorn.run(app, host=hostName, port=serverPort)
