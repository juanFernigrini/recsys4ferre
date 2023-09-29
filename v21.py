from http.server import BaseHTTPRequestHandler, HTTPServer

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import pyodbc
import warnings

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD

from typing import List

hostName = "0.0.0.0"
#"localhost"
serverPort = 7420

warnings.filterwarnings('ignore')

app = FastAPI()
hits = 0

data_sales = None
data_users = None
data_items = None
item_similarity = None

matrix = None
U = None
sigma = None
Vt = None

# SVD hyperparameters
k = 200  # Number of latent factors
lambda_val = 0.1  # Regularization parameter
alpha = 1.0  # Confidence scaling factor for implicit feedback

def obtener_datos():
    global data_sales, data_users, data_items, data
    print("init call")
    print("conectando...")
    
    #cnxn_str = ("Driver={SQL Server Native Client 11.0};"
    #cnxn_str = ("Driver={ODBC Driver 11 for SQL Server};"
    cnxn_str = ("Driver={ODBC Driver 11 for SQL Server};"
                "Server=181.169.115.183,1433;"
                "Database=F_SISTEMA;"
                "UID=External;"
                "PWD=external2022_123!;")
    cnxn = pyodbc.connect(cnxn_str, timeout=50000)
    
#    data_sales = pd.read_sql("""
#         select 
#         cli.CodCliente as CodCliente
#        ,RTRIM(art.CodArticulo) as CodArticu
#        ,cast((coalesce(SUM((reng.CantidadPedida+reng.CantPedidaDerivada)*reng.PrecioVenta),0)*1+(COUNT(reng.NroRenglon)/100)) as decimal) as Cantidad
#        from  f_central.dbo.ven_clientes as cli
#	  inner join f_central.dbo.StkFer_Articulos as art
#              on 1 = 1
#	  left join F_CENTRAL.dbo.VenFer_PedidoReng as reng
#              on reng.CodCliente = cli.CodCliente
#              and reng.CodArticu = art.CodArticulo
#        where cli.codCliente in (1176,186,2001,36,35,78,252,154,145,112,201,203)
#        group by cli.CodCliente,art.CodArticulo
#        order by cli.CodCliente
#    """, cnxn)
    
    data_types = {
    'CodCliente': int,
    'CodArticu': str,
    'Cantidad': float
    }
    col_names = ['CodCliente', 'CodArticu', 'Cantidad']
    data_sales = pd.read_csv('ConsultaData.csv',
                             header= None,
                             names=col_names,
                             index_col= False,
                             sep=';',
                             dtype= data_types)
    

    
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

exclude_items = []  # Items excluidos


@app.get("/prepararModelo")
async def preparar_modelo(k: int = 200, lambda_val: float = 0.1, alpha: float = 1.0):
    global matrix, U, sigma, Vt, data, item_similarity

    data = data[~data['CodArticu'].isin(exclude_items)]

    matrix = data.pivot(index='CodCliente', columns='CodArticu', values='Cantidad').fillna(0)
    matrix = matrix - np.mean(matrix, axis=0)
    sparse_matrix = csr_matrix(matrix.values)
    sp.save_npz("sparse_matrix.npz", sparse_matrix)

    # matrix item-item similarity 
    item_similarity = cosine_similarity(matrix.T)

    U, sigma, Vt = entrenar_modelo(sparse_matrix, k=k, lambda_val=lambda_val, alpha=alpha) 
    
    return "Modelo preparado"

def entrenar_modelo(sparse_matrix, k, lambda_val, alpha):
    svd = TruncatedSVD(n_components=k, random_state=42)
    svd.fit(sparse_matrix)
    sigma = np.diag(svd.singular_values_)
    U = svd.transform(sparse_matrix)
    Vt = svd.components_
    return U, sigma, Vt

@app.get("/levantarModelo")
async def levantar_modelo():
    global matrix, U, sigma, Vt,data_items
    #Levanto la matrix
    matrix = pd.read_csv('matrix.csv', index_col=0)
    #matrix = matrix - np.mean(matrix, axis=0)
    sparse_matrix = sp.load_npz("sparse_matrix.npz")
    U, sigma, Vt = entrenar_modelo(sparse_matrix) 

    
    return "Modelo levantado y preparado"


@app.get("/consulta/{customer_id}")
async def consulta(customer_id, exclude_items: List[int] = []):
    

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
    user_recommendations = pd.DataFrame(
        {'CodArticu': matrix.columns, 'predicted_purchase_count': user_predicted_purchase_counts.flatten()})

    # Use precomputed item-item similarity matrix
    diversity_scores = np.sum(item_similarity, axis=0)

    user_recommendations['diversity_score'] = diversity_scores
    user_recommendations = user_recommendations.sort_values(
        ['predicted_purchase_count', 'diversity_score'], ascending=[False, False])
    user_recommendations = user_recommendations[['CodArticu', 'predicted_purchase_count']]

    top_recommendations = user_recommendations.head(10)

    novelty_diversity_recommendations = top_recommendations.sample(n=3)

    user_recommendations = user_recommendations.drop(novelty_diversity_recommendations.index)

    model_recommendations = user_recommendations.head(10 - len(novelty_diversity_recommendations))
    final_recommendations = pd.concat([model_recommendations, novelty_diversity_recommendations])
    final_recommendations['recommendation_type'] = ""
    final_recommendations.loc[model_recommendations.index, 'recommendation_type'] = "model"
    final_recommendations.loc[novelty_diversity_recommendations.index, 'recommendation_type'] = "novelty"

    final_recommendations = final_recommendations.sort_values('predicted_purchase_count', axis=0, ascending=False)

    recommendations_dict = final_recommendations.to_dict(orient='records')
    headers = {"Content-Type": "application/json"}
    return JSONResponse(content=recommendations_dict,headers=headers)




if __name__ == '__main__':
    uvicorn.run(app, host=hostName, port=serverPort)
