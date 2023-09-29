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

# load in the data

df = pd.read_csv('edited_purchase_history.csv')

N = df.CodCliente_idx.max() + 1 # number of CodClientes
M = df.CodArticu_idx.max() + 1 # number of CodArticu

# Convert 'CodArticu' column to numeric (if it contains numeric values)
df['CodArticu'] = pd.to_numeric(df['CodArticu'], errors='coerce')

print("Los indices van desde:")
print(df['CodCliente_idx'].min(), df['CodCliente_idx'].max())
print(df['CodArticu_idx'].min(), df['CodArticu_idx'].max())

print(df['CodCliente'].min(), df['CodCliente'].max())
print(df['CodArticu'].min(), df['CodArticu'].max())

# split into train and test
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

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
all_users = set(df.CodCliente_idx.unique())
users_in_train = set(df_train.CodCliente_idx.unique())
users_in_test = set(df_test.CodCliente_idx.unique())
missing_users_in_train = all_users - users_in_train
missing_users_in_test = all_users - users_in_test

# Add missing users to the training set
missing_users_data = df[df.CodCliente_idx.isin(missing_users_in_train)]
df_train = pd.concat([df_train, missing_users_data])

# Add missing users to the test set
missing_users_data = df[df.CodCliente_idx.isin(missing_users_in_test)]
df_test = pd.concat([df_test, missing_users_data])

# Now df_train and df_test contain all users

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

  i = int(row.CodCliente_idx)
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

  i = int(row.CodCliente_idx)
  j = int(row.CodArticu_idx)
  CodClienteCodArticu2rating_test[(i,j)] = row.Cantidad
df_test.apply(update_CodClienteCodArticu2rating_test, axis=1)

# Create a mapping from movie index to movie ID
movie_idx_to_movie_id = {}
for index, row in df.iterrows():
    movie_idx_to_movie_id[row['CodArticu_idx']] = row['CodArticu']

with open('CodCliente2CodArticu.json', 'wb') as f:
  pickle.dump(CodCliente2CodArticu, f)

# note: these are not really JSONs
with open('movie_idx_to_movie_id.json', 'wb') as f:
  pickle.dump(movie_idx_to_movie_id, f)

with open('CodArticu2CodCliente.json', 'wb') as f:
  pickle.dump(CodArticu2CodCliente, f)

with open('CodClienteCodArticu2rating.json', 'wb') as f:
  pickle.dump(CodClienteCodArticu2rating, f)

with open('CodClienteCodArticu2rating_test.json', 'wb') as f:
  pickle.dump(CodClienteCodArticu2rating_test, f)
