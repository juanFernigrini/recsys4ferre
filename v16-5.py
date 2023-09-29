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
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD

from typing import List

from sklearn.model_selection import train_test_split, GridSearchCV


from scipy.sparse import save_npz, load_npz
from sklearn.metrics import mean_squared_error


hostName = "localhost"
serverPort = 8081

warnings.filterwarnings('ignore')

app = FastAPI()
hits = 0

data_sales = None
data_users = None
data_items = None
data = None

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
async def preparar_modelo():
    global matrix, U, sigma, Vt, data

    data = data[~data['CodArticu'].isin(exclude_items)]

    matrix = data.pivot(index='CodCliente', columns='CodArticu', values='Cantidad').fillna(0)
    matrix.to_csv('matrix.csv', index=True)
    matrix = matrix - np.mean(matrix, axis=0)
    sparse_matrix = csr_matrix(matrix.values)
    save_npz("sparse_matrix.npz", sparse_matrix)
    
    # Se divide data en train y test
    train_matrix, test_matrix = train_test_split(sparse_matrix, test_size=0.2, random_state=42)
    print(train_matrix.toarray())
    
    # Busequeda mejor valor de k
    param_grid = {'n_components': [1, 10, 20, 30]} # Modify the list of values as needed
    model = GridSearchCV(estimator=TruncatedSVD(), param_grid=param_grid, scoring='neg_mean_absolute_error', n_jobs=-1)
    model.fit(train_matrix)
    best_n_components = model.best_params_['n_components']
    print("Best K:", best_n_components)
    
    # Entrena modelo con el mejor valor de K
    svd = TruncatedSVD(n_components=best_n_components)
    U = svd.fit_transform(train_matrix)
    sigma = svd.explained_variance_
    Vt = svd.components_
    
    # Evaluarlo contra test_data
    #test_predicted_purchase_counts = np.dot(U, np.dot(np.diag(sigma), Vt))
    #test_rmse = np.sqrt(mean_squared_error(test_matrix, test_predicted_purchase_counts))
    #print("Test RMSE:", test_rmse)
    
    return "Modelo preparado"

@app.get("/consulta/{customer_id}")
async def consulta(customer_id, exclude_items: List[int] = []):
    global matrix, U, sigma, Vt, data

    if matrix is None:
        return "Primero se debe preparar el modelo usando '/prepararModelo'"

    if int(customer_id) not in matrix.index:
        return "El ID de usuario no existe"
    
    user_index = matrix.index.get_loc(int(customer_id))
    user_row = U[user_index, :]
    
    if np.count_nonzero(user_row) == 0:
        # In case of a user without a purchase history
        user_items = data[data['CodCliente'] == int(customer_id)]['CodArticu'].unique()
        all_items = data['CodArticu'].unique()
        cold_start_user_recommendations = list(set(all_items) - set(user_items))
        return {"Usuario Cold Start": cold_start_user_recommendations[:10]}

    user_predicted_purchase_counts = np.dot(user_row, np.dot(np.diag(sigma), Vt))
    
    user_recommendations = pd.DataFrame({'CodArticu': matrix.columns, 'predicted_purchase_count': user_predicted_purchase_counts.flatten()})

    # Matrix item-item similarity 
    item_similarity = cosine_similarity(matrix.T)

    # Diversity scores for each item
    diversity_scores = np.sum(item_similarity, axis=0)

    # Add diversity scores to the dataframe
    user_recommendations['diversity_score'] = diversity_scores

    # Sort
    user_recommendations = user_recommendations.sort_values(['predicted_purchase_count', 'diversity_score'], ascending=[False, False])

    user_recommendations = user_recommendations[['CodArticu', 'predicted_purchase_count']]

    # Top 10 recommendations
    top_recommendations = user_recommendations.head(10)

    # Select 3 recommendations randomly based on novelty and diversity
    novelty_diversity_recommendations = top_recommendations.sample(n=3)

    user_recommendations = user_recommendations.drop(novelty_diversity_recommendations.index)

    # Recommendations from the model
    model_recommendations = user_recommendations.head(10 - len(novelty_diversity_recommendations))

    # Combine model recommendations with novelty recommendations
    final_recommendations = pd.concat([model_recommendations, novelty_diversity_recommendations])

    # Add column indicating the recommendation type
    final_recommendations['recommendation_type'] = ""
    final_recommendations.loc[model_recommendations.index, 'recommendation_type'] = "model"
    final_recommendations.loc[novelty_diversity_recommendations.index, 'recommendation_type'] = "novelty"

    # Sort by predicted_purchase_count
    final_recommendations = final_recommendations.sort_values('predicted_purchase_count', axis=0, ascending=False)

    recommendations_dict = final_recommendations.to_dict(orient='records')
    return {"recommendations": recommendations_dict}

if __name__ == '__main__':
    uvicorn.run(app, host=hostName, port=serverPort)