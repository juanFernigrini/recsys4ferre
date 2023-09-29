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

    global loaded_data
    if loaded_data is None:
        # Load data if not already loaded
        print("Loading data...")
        loaded_data = pd.read_csv('purchase_history.csv')
    return "Data obtained"
    
    global  df
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
    
    
    df = pd.read_csv('purchase_history.csv')
    return "Data obtenida"
    
@app.get("/obtenerData")
async def obtener_data():
    return obtener_datos()

def preprocess_data():
    global loaded_data
    if loaded_data is None:
        # Load data if not already loaded
        loaded_data = pd.read_csv('purchase_history.csv')

    # Create a mapping for CodCliente using pandas factorize
    loaded_data['CodCliente_idx'], _ = pd.factorize(loaded_data['CodCliente'])
    
    # Create a mapping for CodArticu using pandas factorize
    loaded_data['CodArticu_idx'], _ = pd.factorize(loaded_data['CodArticu'])

    loaded_data.to_csv('edited_loaded_data.csv', index=False)

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
    else:
        print("Dataset does not contain missing or invalid values.")
    if valid_indices.all():
        print("All user and movie indices are within the expected range.")
    else:
        print("Some user or movie indices are out of the expected range.")
    
    

@app.get("/preprocess")
async def preprocess_data_route():
    preprocess_data()
    return "Data preprocessing completed."


# Define a flag to indicate if preprocessing has been completed
preprocessing_completed = False

def preprocess2dict_data():
    global loaded_data
    #global count
    global df_test
    global df_train
    global movie_idx_to_movie_id
    # load in the data

    if loaded_data is None:
        loaded_data = pd.read_csv('edited_loaded_data.csv')


    # Convert 'CodArticu' column to numeric (if it contains numeric values)
    loaded_data['CodArticu'] = pd.to_numeric(loaded_data['CodArticu'], errors='coerce')

    print("Los indices van desde:")
    print(loaded_data['CodCliente_idx'].min(), loaded_data['CodCliente_idx'].max())
    print(loaded_data['CodArticu_idx'].min(), loaded_data['CodArticu_idx'].max())

    print("Los id de clientes y articulos van desde:")
    print(loaded_data['CodCliente'].min(), loaded_data['CodCliente'].max())
    print(loaded_data['CodArticu'].min(), loaded_data['CodArticu'].max())

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
    else:
        print("Train dataset does not contain NaN values.")

    if test_has_nan:
        print("Test dataset contains NaN values.")
    else:
        print("Test dataset does not contain NaN values.")


    # Initialize dictionaries to ensure all users are present in both sets
    all_users = set(loaded_data.CodCliente_idx.unique())
    users_in_train = set(df_train.CodCliente_idx.unique())
    users_in_test = set(df_test.CodCliente_idx.unique())
    missing_users_in_train = all_users - users_in_train
    missing_users_in_test = all_users - users_in_test

    # Add missing users to the training set
    missing_users_data = loaded_data[loaded_data.CodCliente_idx.isin(missing_users_in_train)]
    df_train = pd.concat([df_train, missing_users_data])

    # Add missing users to the test set
    missing_users_data = loaded_data[loaded_data.CodCliente_idx.isin(missing_users_in_test)]
    df_test = pd.concat([df_test, missing_users_data])

    # Now df_train and df_test contain all users
    df_train.to_csv('train_data.csv', index=False)
    df_test.to_csv('test_data.csv', index=False)
    
     # Create a mapping from movie index to movie ID
    movie_idx_to_movie_id = {}
    for index, row in loaded_data.iterrows():
        movie_idx_to_movie_id[row['CodArticu_idx']] = row['CodArticu']

    with open('movie_idx_to_movie_id.json', 'wb') as f:
        pickle.dump(movie_idx_to_movie_id, f)

    return df_test, df_train, movie_idx_to_movie_id

@app.get("/preprocess2dict")
async def preprocess2dict_data_route():
    global preprocessing_completed  # Access the global flag

    if not preprocessing_completed:
        # Run the preprocessing function
        preprocess2dict_data()
        
        # Set the flag to True to indicate that preprocessing has been completed
        preprocessing_completed = True
        
        return "Diccionaries are done and ready"
    else:
        return "Preprocessing has already been completed. Use /preprocess2dict again to re-run."

# Reset the flag when you want to allow preprocessing to be run again
@app.get("/reset_preprocessing_flag")
async def reset_preprocessing_flag():
    global preprocessing_completed
    preprocessing_completed = False
    return "Preprocessing flag reset. You can now run /preprocess2dict again."

mf_keras_deep_executed = False

def mf_keras_deep_data():
    global loaded_data
    global count
    global mu
    global trained_model
    global mf_keras_deep_executed
    global M

    if loaded_data is None:
        
        loaded_data = pd.read_csv('edited_loaded_data.csv')
        
        # Load data if not already loaded
        #obtener_data()
        #preprocess_data()
        #preprocess2dict_data()
    
    

    df_train = pd.read_csv('train_data.csv')
    df_test = pd.read_csv('test_data.csv')

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

    # Print training and validation loss and MSE
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
            # Run the preprocessing function
            mf_keras_deep_data()

            # Set the flag to True to indicate that preprocessing has been completed
            mf_keras_deep_executed = True  # Set the flag to indicate execution

            return "Model ready"
        except Exception as e:
            traceback.print_exc()
            return f"Error: {str(e)}"
    else:
        return "mf_keras_deep has already been completed. Use /reset_mf_keras_flag again to re-run."

    


# Reset the flag when you want to allow preprocessing to be run again
@app.get("/reset_mf_keras_flag")
async def reset_mf_keras_flag():
    global mf_keras_deep_executed
    mf_keras_deep_executed = False
    return "mf_keras_deep_executed flag reset. You can now run /mf_keras_deep again."

@app.get("/consulta/{CodCliente}")
async def recommend_top_10_items_for_user(CodCliente: int, top_N: int = 10):
        global trained_model, loaded_data, M, df_train, mu

        if trained_model is None:
            trained_model = load_model('your_pretrained_model.h5')
            loaded_data = pd.read_csv('new_edited_loaded_data.csv')
            M = loaded_data.CodArticu_idx.max() + 1 # number of movies
            df_train = pd.read_csv('train_data.csv')
            mu = df_train.Cantidad.mean()
        
        # Check if CodCliente exists in loaded_data
        if CodCliente not in loaded_data['CodCliente'].values:
            return "Ese CodCliente no existe."  # Return a message indicating the UserID is not valid

        # Map the user ID to its corresponding index
        user_idx = loaded_data[loaded_data['CodCliente'] == CodCliente]['CodCliente_idx'].values[0]

        # Get the indices of all movies
        CodArticu_indices = np.arange(M)

        # Create an array with the user index repeated for all movies
        user_array = np.array([user_idx] * M)

        # Predict movie ratings for the user
        predicted_ratings = trained_model.predict([user_array, CodArticu_indices]) + mu

        with open('movie_idx_to_movie_id.json', 'rb') as f:
            movie_idx_to_movie_id = pickle.load(f)

        # Create a DataFrame with movie indices, predicted ratings, and movie IDs
        movie_ratings = pd.DataFrame({
            'movie_index': CodArticu_indices,
            'predicted_rating': predicted_ratings.flatten(),
            'movie_id': [movie_idx_to_movie_id[i] for i in CodArticu_indices]
        })

        # Sort the DataFrame by predicted ratings in descending order
        top_movie_ratings = movie_ratings.sort_values(by='predicted_rating', ascending=False)

        # Get the top N recommended movie IDs
        top_movie_ids = top_movie_ratings.head(top_N)['movie_id'].values

        recommended_movie_ids = top_movie_ids

        print("Top {} recommended movies for user (CodCliente) {}:".format(top_N, CodCliente))
        for movie_id in recommended_movie_ids:
            print("Movie ID:", movie_id)
        
        return "salio por fin"


if __name__ == '__main__':
    uvicorn.run(app, host=hostName, port=serverPort)