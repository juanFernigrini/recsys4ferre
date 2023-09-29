# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

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

from sklearn.preprocessing import StandardScaler

# load in the data
df = pd.read_csv('edited_purchase_history.csv')

# Create a StandardScaler instance for 'Cantidad'
scaler = StandardScaler()

# Normalize the 'Cantidad' column
df['Cantidad'] = scaler.fit_transform(df['Cantidad'].values.reshape(-1, 1))

with open('movie_idx_to_movie_id.json', 'rb') as f:
  movie_idx_to_movie_id = pickle.load(f)

N = df.CodCliente_idx.max() + 1 # number of users
M = df.CodArticu_idx.max() + 1 # number of movies

# split into train and test
# ESTO YA ESTA HECHO EN PASO PREVIO, INNECESARIO
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

# initialize variables
K = 10 # latent dimensionality
mu = df_train.Cantidad.mean()
epochs = 5
reg = 0.003 # regularization penalty


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
# x = Dropout(0.5)(x)
# x = Dense(100)(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
x = Dense(1)(x)

model = Model(inputs=[u, m], outputs=x)
model.compile(
  loss='mse',
  # optimizer='adam',
  # optimizer=Adam(lr=0.01),
  optimizer=SGD(lr=0.002, momentum=0.9),
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

def recommend_top_10_items_for_user(CodCliente, model, movie_idx_to_movie_id, top_N=10):
    # Map the user ID to its corresponding index
    user_idx = df[df['CodCliente'] == CodCliente]['CodCliente_idx'].values[0]
    
    # Get the indices of all movies
    CodArticu_indices = np.arange(M)
    
    # Create an array with the user index repeated for all movies
    user_array = np.array([user_idx] * M)
    
    # Predict movie ratings for the user
    predicted_ratings = model.predict([user_array, CodArticu_indices]) + mu
    
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
    
    return top_movie_ids

# Example usage:
CodCliente = 100750  # Replace with the user ID (CodCliente) for whom you want to make recommendations
top_N = 10   # Number of recommendations to generate
recommended_movie_ids = recommend_top_10_items_for_user(CodCliente, model, movie_idx_to_movie_id, top_N)

print("Top {} recommended movies for user (CodCliente) {}:".format(top_N, CodCliente))
for movie_id in recommended_movie_ids:
    print("Movie ID:", movie_id)
    
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
