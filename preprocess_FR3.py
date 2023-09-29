import pandas as pd

# Load the data
df = pd.read_csv('purchase_history.csv')

# Create a mapping for CodCliente
unique_CodCliente_ids = set(df.CodCliente.values)
CodCliente2idx = {}
count = 0
for CodCliente in unique_CodCliente_ids:
    CodCliente2idx[CodCliente] = count
    count += 1

# Apply the mapping to create the CodCliente_idx column
df['CodCliente_idx'] = df.apply(lambda row: CodCliente2idx[row.CodCliente], axis=1)

# Create a mapping for CodArticu_ids
unique_CodArticu_ids = set(df.CodArticu.values)
CodArticu2idx = {}
count = 0
for CodArticu in unique_CodArticu_ids:
    CodArticu2idx[CodArticu] = count
    count += 1

# Apply the mapping to create the CodArticu_idx column
df['CodArticu_idx'] = df.apply(lambda row: CodArticu2idx[row.CodArticu], axis=1)

# Ensure there are no missing or invalid values in the dataset
missing_values = df.isnull().values.any()

# Verify that all user and movie indices are within the expected range (0 to N-1)
valid_user_indices = (df['CodCliente_idx'] >= 0) & (df['CodCliente_idx'] < len(unique_CodCliente_ids))
valid_movie_indices = (df['CodArticu_idx'] >= 0) & (df['CodArticu_idx'] < len(unique_CodArticu_ids))
valid_indices = valid_user_indices & valid_movie_indices

if missing_values:
    print("Dataset contains missing or invalid values.")
else:
    print("Dataset does not contain missing or invalid values.")

if valid_indices.all():
    print("All user and movie indices are within the expected range.")
else:
    print("Some user or movie indices are out of the expected range.")

df.to_csv('edited_purchase_history.csv', index=False)
