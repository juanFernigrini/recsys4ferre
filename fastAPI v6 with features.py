# adding features to v3 modify ok   
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

data_sales = None
data_users = None
data_items = None

matrix = None

U = None
sigma = None
Vt = None

def obtener_datos():
        global data_sales
        global data_users
        global data_items
        global data
        print("init call")
        print("conectando...")
        #Lucas --> ODBC Driver 11 for SQL Server
        #Juan --> SQL Server Native Client 11.0
        cnxn_str = ("Driver={ODBC Driver 11 for SQL Server};"
                    "Server=181.169.115.183,1433;"
                    "Database=F_SISTEMA;"
                    "UID=External;"
                    "PWD=external2022_123!;")
        cnxn = pyodbc.connect(cnxn_str)
        print("conectado a la db ")
        print("comienzo a llenar data sales")
        data_sales = pd.read_sql("SELECT [CodCliente] as CodCliente "+
            " ,RTRIM([CodArticu]) as CodArticu "+
            #" ,count(*) as Cantidad "+ #cantidad de veces que pidio
            #" ,sum(CantidadPedida + CantPedidaDerivada) as Cantidad "+ #cantidad que pidio
            " ,(sum(CantidadPedida + CantPedidaDerivada) / count(*)) as Cantidad "+ #cantidad de veces que pidio / cantidad que pidio
            " FROM [F_CENTRAL].[dbo].[VenFer_PedidoReng] "+
            " WHERE CodCliente <> '' "+
        " GROUP BY CodCliente ,CodArticu"+
        " ORDER BY CodCliente", cnxn)
        data_sales.head(10)
        print(data_sales)

        print("comienzo a llenar data_users")
        data_users = pd.read_sql("select " +
            " CodCliente " +
            " ,CodigoPostal " +
            " ,Vendedor " +
            " ,Zona " +
            " ,LimiteCredito " +
            " from F_central.dbo.Ven_Clientes", cnxn)
        data_users.head(10)
        print(data_users)

        print("comienzo a llenar data_users")
        data_items = pd.read_sql("select " +
            " RTRIM(CodArticulo) as CodArticu" +
            " ,PrecioCosto " +
            " ,ArticuloPatron " +
            " ,PrecioUnitario " +
            " from F_central.dbo.StkFer_Articulos", cnxn)
        data_items.head(10)
        print(data_items)
        

        data = data_sales.merge(data_users, on="CodCliente")
        data = data.merge(data_items, on="CodArticu")
        
        data.head(10)
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

    # Se crea matriz usuario-item con features
    matrix = data.pivot(index='CodCliente', columns='CodArticu', values='Cantidad').fillna(0)

    # Centramos data
    matrix = matrix - np.mean(matrix, axis=0)

    # El SVD requiere una matriz dispersa como input, aca se convierte en eso
    sparse_matrix = csr_matrix(matrix.values)

    # SVD
    U, sigma, Vt = svds(sparse_matrix, k=50) # k is the number of singular values to compute

    # Predicciones para todos los usuarios
    mean_item = np.mean(matrix, axis=0)
    all_user_predicted_purchase_counts = np.dot(np.dot(U, np.diag(sigma)), Vt) + mean_item.values.reshape(1, -1)

        
    return "modelo Preparado"



@app.get("/consulta/{customer_id}")
async def consulta(customer_id):
    customer_id_int = int(customer_id)
    #Aca empieza Notebook

    #Se busca fila de user en la matriz
    user_row = matrix.iloc[customer_id_int, :]
        
    #Se calcula el producto escalar de la fila del usuario con la matriz Vt para cantidad previstos de items recomendados para ese usuario:
    user_predicted_purchase_counts = np.dot(U[customer_id_int,:], np.dot(np.diag(sigma), Vt))

    #Ordenamos obtener los artículos más recomendados para ese usuario e imprimimos:
    user_recommendations = pd.DataFrame({'CodArticu': matrix.columns, 'predicted_purchase_count': user_predicted_purchase_counts})
    user_recommendations = user_recommendations.sort_values('predicted_purchase_count', ascending=False)

    print("Mejores predicciones para el usuario:")
    print(user_recommendations.head(10))

        
    return "consulta"
    #Aca termina Notebook

        
if __name__ == '__main__':
    uvicorn.run(app, host=hostName, port=serverPort)#, log_level="info")


