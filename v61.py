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
    #global count
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
    
    # Creando mapping para CodArticu_idx a CodArticu efficiently
    codarticu_idx_to_codarticu = dict(zip(loaded_data['CodArticu_idx'], loaded_data['CodArticu']))
        
    # Saving the mapping as a JSON file
    with open('codarticu_idx_to_codarticu.json', 'w') as f:
        json.dump(codarticu_idx_to_codarticu, f)

    return df_test, df_train, codarticu_idx_to_codarticu

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

mf_keras_deep_executed = False

def mf_keras_deep_data():
    global loaded_data
    global count
    global mu
    global trained_model
    global mf_keras_deep_executed
    global M

    if loaded_data is None:
        
        loaded_data = pd.read_csv('new_edited_loaded_data.csv')
        
        # Load data if not already loaded
        #obtener_data()
        #preprocess_data()
        #preprocess2dict_data()
    
    

    df_train = pd.read_csv('train_data.csv')
    df_test = pd.read_csv('test_data.csv')

    N = loaded_data.CodCliente_idx.max() + 1 # Numero de CodClientes
    M = loaded_data.CodArticu_idx.max() + 1 # Numero de CodArticus

    # Inicia variables
    K = 5 # latent dimensionality
    mu = df_train.Cantidad.mean()
    epochs = 1
    reg = 0.001 # regularization penalty


    # keras model
    u = Input(shape=(1,))
    m = Input(shape=(1,))
    u_embedding = Embedding(N, K)(u) # (N, 1, K)
    m_embedding = Embedding(M, K)(m) # (N, 1, K)
    u_embedding = Flatten()(u_embedding) # (N, K)
    m_embedding = Flatten()(m_embedding) # (N, K)
    x = Concatenate()([u_embedding, m_embedding]) # (N, 2K)

    # Hiperparametros de la Red Neuronal
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

    # Muestra grafica de validation loss and MSE
    print("Training Loss:", r.history['loss'])
    print("Validation Loss:", r.history['val_loss'])
    print("Training MSE:", r.history['mse'])
    print("Validation MSE:", r.history['val_mse'])

    # plot losses
    plt.plot(r.history['loss'], label="train loss")
    plt.plot(r.history['val_loss'], label="test loss")
    plt.legend()
    plt.show()

    # plot mse
    plt.plot(r.history['mse'], label="train mse")
    plt.plot(r.history['val_mse'], label="test mse")
    plt.legend()
    plt.show()

    
    return trained_model
    


@app.get("/mf_keras_deep")
async def mf_keras_deep_route():
    global mf_keras_deep_executed  # Access the global flag
    if not mf_keras_deep_executed:
        try:
            mf_keras_deep_data()

            mf_keras_deep_executed = True  # Set the flag to indicate execution

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

@app.get("/consulta/{CodCliente}")
async def recommend_top_10_items_for_user(CodCliente: int, top_N: int = 10):
        global trained_model, loaded_data, M, df_train, mu, codarticu_idx_to_codarticu

        if trained_model is None:
            trained_model = load_model('your_pretrained_model.h5')
            loaded_data = pd.read_csv('new_edited_loaded_data.csv')
            M = loaded_data.CodArticu_idx.max() + 1 # numero de CodArticus
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