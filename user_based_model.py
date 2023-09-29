from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd

import pyodbc
import warnings

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from typing import List

import pandas as pd

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

from keras.models import load_model

from sklearn.preprocessing import StandardScaler

from fastapi import Depends

import json

import traceback

from sortedcontainers import SortedList


hostName = "localhost"
serverPort = 8081
warnings.filterwarnings('ignore')
app = FastAPI()
hits = 0


trained_model = None
loaded_data = None

def obtener_datos():

    global loaded_data

    #if loaded_data is None:
    #    # Load data if not already loaded
    #    print("Loading data...")
    #    loaded_data = pd.read_csv('purchase_history.csv')
    #    return "Data obtenida"
    
    print("init call")
    print("conectando...")
    
    #cnxn_str = ("Driver={SQL Server Native Client 11.0};"
    #
    cnxn_str = ("Driver={ODBC Driver 11 for SQL Server};"
                "Server=181.169.115.183,1433;"
                "Database=F_SISTEMA;"
                "UID=External;"
                "PWD=external2022_123!;")
    cnxn = pyodbc.connect(cnxn_str, timeout=50000)
    
    #where cli.codCliente in (1176,186,2001,36,35,78,252,154,145,112,201,203)
    #group by cli.CodCliente,art.CodArticulo

    loaded_data = pd.read_sql("""
         select 
         cli.CodCliente as CodCliente
        ,RTRIM(art.CodArticulo) as CodArticu
        ,cast((coalesce(SUM((reng.CantidadPedida+reng.CantPedidaDerivada)*reng.PrecioVenta),0)*1+(COUNT(reng.NroRenglon)/100)) as decimal) as Cantidad
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
    loaded_data.to_csv('loaded_data.csv', index=False)
    return "Data obtenida"
    
@app.get("/obtenerData")
async def obtener_data():
    return obtener_datos()

def preprocess_data():
    global loaded_data
    if loaded_data is None:
        try:
            loaded_data = pd.read_csv('loaded_data.csv')
        except FileNotFoundError:
            print("loaded_data.csv no encontrado, trabajando con purchase_history.csv.")
            loaded_data = pd.read_csv('purchase_history.csv')

    # Creando index para CodCliente usando factorize de pandas
    loaded_data['CodCliente_idx'], _ = pd.factorize(loaded_data['CodCliente'])
    
    # Creando index para CodArticu
    loaded_data['CodArticu_idx'], _ = pd.factorize(loaded_data['CodArticu'])

    loaded_data.to_csv('new_edited_loaded_data.csv', index=False)

    # Asegurando que no haya missing o invalid values en el dataset
    missing_values = loaded_data.isnull().values.any()
    
    # Verificando que todos los CodCliente y CodArticu esten dentro del rango esperado(0 to N-1)
    valid_indices = (
        (loaded_data['CodCliente_idx'] >= 0) &
        (loaded_data['CodCliente_idx'] < loaded_data['CodCliente_idx'].nunique()) &
        (loaded_data['CodArticu_idx'] >= 0) &
        (loaded_data['CodArticu_idx'] < loaded_data['CodArticu_idx'].nunique())
    )
    
    if missing_values:
        print("El Dataset contiene missing or invalid values.")
    else:
        print("El Dataset NO contain missing or invalid values.")
    if valid_indices.all():
        print("Todos los CodCliente y CodArticu estan dentro del rango esperado.")
    else:
        print("Algun CodCliente o CodArticu ESTA FUERA del rango esperado.")
    
    

@app.get("/preprocess")
async def preprocess_data_route():
    preprocess_data()
    return "Preprocess completado."

# Defino flag para indicar si preprocessing ya ha sido completado
preprocessing_completed = False

def preprocess2dict_data():
    global loaded_data
    global count
    global df_test
    global df_train
    global codarticu_idx_to_codarticu
    # load in the data

    if loaded_data is None:
        loaded_data = pd.read_csv('new_edited_loaded_data.csv')


    # Convierto 'CodArticu' en numerico 
    loaded_data['CodArticu'] = pd.to_numeric(loaded_data['CodArticu'], errors='coerce')

    print("Los indices de clientes y articulos van desde:")
    print(loaded_data['CodCliente_idx'].min(), loaded_data['CodCliente_idx'].max())
    print(loaded_data['CodArticu_idx'].min(), loaded_data['CodArticu_idx'].max())

    print("Los id de clientes y articulos van desde:")
    print(loaded_data['CodCliente'].min(), loaded_data['CodCliente'].max())
    print(loaded_data['CodArticu'].min(), loaded_data['CodArticu'].max())

    # StandardScaler para 'Cantidad'
    scaler = StandardScaler()

    # Normalizando 'Cantidad' 
    loaded_data['Cantidad'] = scaler.fit_transform(loaded_data['Cantidad'].values.reshape(-1, 1))

    # Divido en train y test
    loaded_data = shuffle(loaded_data)
    cutoff = int(0.8*len(loaded_data))
    df_train = loaded_data.iloc[:cutoff]
    df_test = loaded_data.iloc[cutoff:]

    # Elimina filas con NaN values en train y test datasets
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # Check for NaN values in train and test datasets
    train_has_nan = df_train.isnull().values.any()
    test_has_nan = df_test.isnull().values.any()

    if train_has_nan:
        print("Train dataset CONTIENE NaN values.")
    else:
        print("Train dataset no contiene NaN values.")

    if test_has_nan:
        print("Test dataset CONTIENE NaN values.")
    else:
        print("Test dataset no contiene NaN values.")


    # Asegurandonos que train y test tengan los mismos CodCliente
    all_users = set(loaded_data.CodCliente_idx.unique())
    users_in_train = set(df_train.CodCliente_idx.unique())
    users_in_test = set(df_test.CodCliente_idx.unique())
    missing_users_in_train = all_users - users_in_train
    missing_users_in_test = all_users - users_in_test

    # Agregando CodClientes faltantes a training set
    missing_users_data = loaded_data[loaded_data.CodCliente_idx.isin(missing_users_in_train)]
    df_train = pd.concat([df_train, missing_users_data])

    # Agregando CodClientes faltantes a test set
    missing_users_data = loaded_data[loaded_data.CodCliente_idx.isin(missing_users_in_test)]
    df_test = pd.concat([df_test, missing_users_data])

    # Ahora df_train and df_test tienen mismos CodCliente
    df_train.to_csv('train_data.csv', index=False)
    df_test.to_csv('test_data.csv', index=False)

    # a dictionary to tell us which users have rated which movies
    CodCliente2CodArticu = {}
    # a dicationary to tell us which movies have been rated by which users
    CodArticu2CodCliente = {}
    # a dictionary to look up ratings
    CodClienteCodArticu2rating = {}
    print("Calling: update_CodCliente2CodArticu_and_CodArticu2CodCliente")
    count = 0
    def update_CodCliente2CodArticu_and_CodArticu2CodCliente(row):
      global count
      count += 1
      if count % 100000 == 0:
        print("processed: %.3f" % (float(count)/cutoff))

      i = int(row.CodCliente)
      j = int(row.CodArticu_idx)
      if i not in CodCliente2CodArticu:
        CodCliente2CodArticu[i] = [j]
      else:
        CodCliente2CodArticu[i].append(j)

      if j not in CodArticu2CodCliente:
        CodArticu2CodCliente[j] = [i]
      else:
        CodArticu2CodCliente[j].append(i)

      CodClienteCodArticu2rating[(i,j)] = row.Cantidad
    df_train.apply(update_CodCliente2CodArticu_and_CodArticu2CodCliente, axis=1)

    # test ratings dictionary
    CodClienteCodArticu2rating_test = {}
    print("Calling: update_CodClienteCodArticu2rating_test")
    count = 0
    def update_CodClienteCodArticu2rating_test(row):
      global count
      count += 1
      if count % 100000 == 0:
        print("processed: %.3f" % (float(count)/len(df_test)))

      i = int(row.CodCliente)
      j = int(row.CodArticu_idx)
      CodClienteCodArticu2rating_test[(i,j)] = row.Cantidad
    df_test.apply(update_CodClienteCodArticu2rating_test, axis=1)

    # note: these are not really JSONs
    with open('CodCliente2CodArticu.json', 'wb') as f:
      pickle.dump(CodCliente2CodArticu, f)

    with open('CodArticu2CodCliente.json', 'wb') as f:
      pickle.dump(CodArticu2CodCliente, f)

    with open('CodClienteCodArticu2rating.json', 'wb') as f:
      pickle.dump(CodClienteCodArticu2rating, f)

    with open('CodClienteCodArticu2rating_test.json', 'wb') as f:
      pickle.dump(CodClienteCodArticu2rating_test, f)

@app.get("/preprocess2dict")
async def preprocess2dict_data_route():
    global preprocessing_completed  # Access a global flag para ver si ya esta completado

    if not preprocessing_completed:
        preprocess2dict_data()
        
        # Pone True a la flag para indicar que ya fue procesado
        preprocessing_completed = True
        
        return "Preprocess2dict esta listo"
    else:
        return "Preprocessing ya fue completado. Use /preprocess2dict para poder volver a correrlo"

# Para reinicar flag
@app.get("/reset_preprocessing_flag")
async def reset_preprocessing_flag():
    global preprocessing_completed
    preprocessing_completed = False
    return "Preprocessing flag reinicada. Puede volver a correr /preprocess2dict."

user_based_executed = False


def user_based_data():
    global loaded_data
    global count
    global trained_model
    global user_based_executed

    with open('CodCliente2CodArticu.json', 'rb') as f:
        CodCliente2CodArticu = pickle.load(f)

    with open('CodArticu2CodCliente.json', 'rb') as f:
        CodArticu2CodCliente = pickle.load(f)

    with open('CodClienteCodArticu2rating.json', 'rb') as f:
        CodClienteCodArticu2rating = pickle.load(f)

    with open('CodClienteCodArticu2rating_test.json', 'rb') as f:
        CodClienteCodArticu2rating_test = pickle.load(f)

    N = np.max(list(CodCliente2CodArticu.keys())) + 1
    m1 = np.max(list(CodArticu2CodCliente.keys()))
    m2 = np.max([m for (u, m), r in CodClienteCodArticu2rating_test.items()])
    M = max(m1, m2) + 1
    print("N:", N, "M:", M)

    if N > 10000:
        print("N =", N, "are you sure you want to continue?")
        print("Comment out these lines if so...")
        exit()

    K = 25
    limit = 5
    neighbors = []
    averages = []
    deviations = []

    def predict(i, m):
        numerator = 0
        denominator = 0
        for neg_w, j in neighbors[i]:
            try:
                numerator += -neg_w * deviations[j][m]
                denominator += abs(neg_w)
            except KeyError:
                pass

        if denominator == 0:
            prediction = averages[i]
        else:
            prediction = numerator / denominator + averages[i]
        prediction = min(5, prediction)
        prediction = max(0.5, prediction)
        return prediction

    for i in range(N):
        movies_i = CodCliente2CodArticu.get(i, [])
        movies_i_set = set(movies_i)

        ratings_i = {movie: CodClienteCodArticu2rating.get((i, movie), 0) for movie in movies_i}
        avg_i = np.mean(list(ratings_i.values()))
        dev_i = {movie: (rating - avg_i) for movie, rating in ratings_i.items()}
        dev_i_values = np.array(list(dev_i.values()))
        sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

        averages.append(avg_i)
        deviations.append(dev_i)

        sl = SortedList()
        for j in range(N):
            if j != i:
                movies_j = CodCliente2CodArticu.get(j, [])
                movies_j_set = set(movies_j)
                common_movies = movies_i_set & movies_j_set
                if len(common_movies) > limit:
                    ratings_j = {movie: CodClienteCodArticu2rating.get((j, movie), 0) for movie in movies_j}
                    avg_j = np.mean(list(ratings_j.values()))
                    dev_j = {movie: (rating - avg_j) for movie, rating in ratings_j.items()}
                    dev_j_values = np.array(list(dev_j.values()))
                    sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                    numerator = sum(dev_i[m] * dev_j[m] for m in common_movies)
                    w_ij = numerator / (sigma_i * sigma_j)

                    sl.add((-w_ij, j))
                    if len(sl) > K:
                        del sl[-1]

        neighbors.append(sl)

        if i % 1 == 0:
            print(i)

    train_predictions = []
    train_targets = []

    for (i, m), target in CodClienteCodArticu2rating.items():
        prediction = predict(i, m)
        train_predictions.append(prediction)
        train_targets.append(target)

    test_predictions = []
    test_targets = []

    for (i, m), target in CodClienteCodArticu2rating_test.items():
        prediction = predict(i, m)
        test_predictions.append(prediction)
        test_targets.append(target)

    def mse(p, t):
        p = np.array(p)
        t = np.array(t)
        return np.mean((p - t)**2)

    print('train mse:', mse(train_predictions, train_targets))
    print('test mse:', mse(test_predictions, test_targets))


@app.get("/user_based")
async def user_based_route():
    global user_based_executed  # Access the global flag
    if not user_based_executed:
        try:
            user_based_data()

            user_based_executed = True  # Set the flag to indicate execution

            return "Modelo entrenado"
        except Exception as e:
            traceback.print_exc()
            return f"Error: {str(e)}"
    else:
        return "mf_keras_deep ya fue completado. Use /reset_mf_keras_flag para correrlo de nuevo."

    


# Para reiniciar flag
@app.get("/reset_mf_keras_flag")
async def reset_mf_keras_flag():
    global mf_keras_deep_executed
    mf_keras_deep_executed = False
    return "mf_keras_deep_executed flag reiniciada. Puede volver a correr /mf_keras_deep."


if __name__ == '__main__':
    uvicorn.run(app, host=hostName, port=serverPort)