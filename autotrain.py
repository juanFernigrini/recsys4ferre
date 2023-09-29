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

hostName = "localhost"
serverPort = 8081
warnings.filterwarnings('ignore')
app = FastAPI()
hits = 0



# Define global variables to store data and model

trained_model = None
loaded_data = None

def obtener_datos():

    global trained_model, loaded_data, M, df_train, mu, movie_idx_to_movie_id

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
        group by cli.CodCliente,art.CodArticulo
        order by cli.CodCliente
    """, cnxn)
    #loaded_data.to_csv('new_edited_loaded_data.csv', index=False)
    

    if loaded_data is None:
        # Load data if not already loaded
        print("Using old data...")
        loaded_data = pd.read_csv('purchase_history.csv')

    ###ACA ARRANCARIA PREPROCCES
    

    # Creando index para CodCliente usando factorize de pandas
    loaded_data['CodCliente_idx'], _ = pd.factorize(loaded_data['CodCliente'])
    
    # Creando index para CodArticu
    loaded_data['CodArticu_idx'], _ = pd.factorize(loaded_data['CodArticu'])

    #loaded_data.to_csv('new_edited_loaded_data.csv', index=False)

    # Ensure there are no missing or invalid values in the dataset
    missing_values = loaded_data.isnull().values.any()
    
    # Verify that all user and movie indices are within the expected range (0 to N-1)
    valid_indices = (
        (loaded_data['CodCliente_idx'] >= 0) &
        (loaded_data['CodCliente_idx'] < loaded_data['CodCliente_idx'].nunique()) &
        (loaded_data['CodArticu_idx'] >= 0) &
        (loaded_data['CodArticu_idx'] < loaded_data['CodArticu_idx'].nunique())
    )
    
    if missing_values:
        print("Dataset contains missing or invalid values.")
        return "Hay datos invalidos, checkear"
    
    if valid_indices.all():
        print("All user and movie indices are within the expected range.")
    else:
        print("Some user or movie indices are out of the expected range.")
        return "Usuarios o Articulos estan fuera de rango"
    
    ##ACA ARRANCARIA PREPROCES2DICT
    
    # Convierto 'CodArticu' en numerico 
    loaded_data['CodArticu'] = pd.to_numeric(loaded_data['CodArticu'], errors='coerce') 

    # Create a StandardScaler instance for 'Cantidad'
    scaler = StandardScaler()

    # Normalize the 'Cantidad' column
    loaded_data['Cantidad'] = scaler.fit_transform(loaded_data['Cantidad'].values.reshape(-1, 1))

    # split into train and test
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
        print("Train dataset contains NaN values.")
        return "Train_Data tiene nan values. checkear"

    if test_has_nan:
        print("Test dataset contains NaN values.")
        return "Test_Data tiene nan values. checkear"


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
    #df_train.to_csv('train_data.csv', index=False)
    #df_test.to_csv('test_data.csv', index=False)
    
    # Creando mapping para CodArticu_idx a CodArticu efficiently
    codarticu_idx_to_codarticu = dict(zip(loaded_data['CodArticu_idx'], loaded_data['CodArticu']))
        
    # Saving the mapping as a JSON file
    with open('codarticu_idx_to_codarticu.json', 'w') as f:
        json.dump(codarticu_idx_to_codarticu, f)


    ###ACA ARRANCA MF_KERAS

    N = loaded_data.CodCliente_idx.max() + 1 # number of users
    M = loaded_data.CodArticu_idx.max() + 1 # number of movies

    # initialize variables
    K = 40 # latent dimensionality
    mu = df_train.Cantidad.mean()
    epochs = 5
    reg = 0.00001 # regularization penalty


    # keras model
    u = Input(shape=(1,))
    m = Input(shape=(1,))
    u_embedding = Embedding(N, K)(u) # (N, 1, K)
    m_embedding = Embedding(M, K)(m) # (N, 1, K)
    u_embedding = Flatten()(u_embedding) # (N, K)
    m_embedding = Flatten()(m_embedding) # (N, K)
    x = Concatenate()([u_embedding, m_embedding]) # (N, 2K)

    # the neural network
    x = Dense(400)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    # x = Dense(100)(x)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = Dense(1)(x)

    model = Model(inputs=[u, m], outputs=x)
    model.compile(
    loss='mse',
    # optimizer='adam',
    # optimizer=Adam(lr=0.01),
    optimizer=SGD(lr=0.0005, momentum=0.3),
    metrics=['mse'],
    )

    r = model.fit(
    x=[df_train.CodCliente_idx.values, df_train.CodArticu_idx.values],
    y=df_train.Cantidad.values - mu,
    epochs=epochs,
    batch_size=128,
    validation_data=(
        [df_test.CodCliente_idx.values, df_test.CodArticu_idx.values],
        df_test.Cantidad.values - mu
    )
    )
        
    trained_model = model

    trained_model.save('your_pretrained_model.h5')

    return trained_model, loaded_data, M, df_train, mu, movie_idx_to_movie_id


    
    
auto_run = obtener_datos()



@app.get("/consulta/{CodCliente}")
async def recommend_top_10_items_for_user(CodCliente: int, top_N: int = 10):
        global trained_model, loaded_data, M, df_train, mu, movie_idx_to_movie_id

        if trained_model is None:
            trained_model = load_model('your_pretrained_model.h5')
            loaded_data = pd.read_csv('edited_loaded_data.csv')
            M = loaded_data.CodArticu_idx.max() + 1 # number of movies
            df_train = pd.read_csv('train_data.csv')
            mu = df_train.Cantidad.mean()
        
        # Se fija si existe el CodCliente ingresado
        if CodCliente not in loaded_data['CodCliente'].values:
            return "Ese CodCliente no existe."  

        # Mapea el CodCliente ingresado con su respectivo indice
        user_idx = loaded_data[loaded_data['CodCliente'] == CodCliente]['CodCliente_idx'].values[0]

        # Busca los indices de todas los articulos
        CodArticu_indices = np.arange(M)

        # Crea array con el CodCliente ingresado y todas los articulos
        user_array = np.array([user_idx] * M)

        # Predice cuan buena es la recomendacion
        predicted_ratings = trained_model.predict([user_array, CodArticu_indices]) + mu

        # Carga diccionario CodArticu_idx - CodArticu
        with open('codarticu_idx_to_codarticu.json', 'rb') as f:
            codarticu_idx_to_codarticu = json.load(f)

        # Crea dataframe con CodArticu_indices, predicted ratings, and CodArticu
        codarticu_ratings = pd.DataFrame({
            'CodArticu_indices': CodArticu_indices,
            'predicted_rating': predicted_ratings.flatten(),
            'CodArticu': [codarticu_idx_to_codarticu[str(i)] for i in CodArticu_indices]
        })


        # Lo ordena en orden descendente
        top_codarticu_ratings = codarticu_ratings.sort_values(by='predicted_rating', ascending=False)

        # Agarra los mejores 10
        top_codarticu_ids = top_codarticu_ratings.head(top_N)['CodArticu'].values

        recommended_codarticu_ids = top_codarticu_ids

        print("Top {} articulos recomendados para cliente (CodCliente) {}:".format(top_N, CodCliente))
        for codarticu_id in recommended_codarticu_ids:
            print("CodArticu:", codarticu_id)
        
        return "listas las recommendaciones"


if __name__ == '__main__':
    uvicorn.run(app, host=hostName, port=serverPort)