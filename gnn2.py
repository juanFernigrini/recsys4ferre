from http.server import BaseHTTPRequestHandler, HTTPServer
import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import pyodbc
import warnings

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import Dataset, DataLoader

hostName = "localhost"
serverPort = 8081

warnings.filterwarnings('ignore')

app = FastAPI()
hits = 0

data_sales = None
data_users = None
data_items = None
data = None

dataset = None

matrix = None
U = None
sigma = None
Vt = None



def obtener_datos():
    global data_sales, data_users, data_items
    print("conectando...")
    
    #cnxn_str = ("Driver={SQL Server Native Client 11.0};"
    #
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

@app.get("/prepararModelo")
async def preparar_modelo():
    global user_features, GNNRecommender, dataloader, item_features, dataset

    # Step 2: Create Node Features
    user_features = data_users[["CodCliente", "CodigoPostal", "Vendedor", "Zona"]].values
    item_features = data_items[["CodArticu", "PrecioCosto", "ArticuloPatron", "PrecioUnitario"]].values

    # Step 3: Encode Node Features
    label_encoder = LabelEncoder()
    user_features[:, 0] = label_encoder.fit_transform(user_features[:, 0])
    item_features[:, 0] = label_encoder.fit_transform(item_features[:, 0])

    # Step 4: Create Node Index Mapping
    user_mapping = {val: i for i, val in enumerate(user_features[:, 0])}
    item_mapping = {val: i for i, val in enumerate(item_features[:, 0])}

    # Step 5: Create Edge Index and Edge Features
    user_ids = data_sales["CodCliente"].map(user_mapping).values
    item_ids = data_sales["CodArticu"].map(item_mapping).values
    edge_index = np.vstack((user_ids, item_ids)).astype(int)
    edge_attr = data_sales["Cantidad"].values.astype(float)

    # Step 6: Create PyG Data object
    data = Data(x=torch.tensor(user_features, dtype=torch.float),
                y=None,
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float))

    # Step 7: Create Dataset and DataLoader
    dataset = MyDataset(data)

    # Step 8: Create GNN Model
    GNNRecommender = GNNModel()

    # Step 9: Load Pretrained Model
    GNNRecommender.load_state_dict(torch.load('trained_model.pth'))

    # Step 10: Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

    return "Modelo preparado"

def custom_collate(batch):
    batch = [data for data in batch if data is not None]
    return torch_geometric.data.batch.Batch.from_data_list(batch)


class MyDataset(Dataset):
    def __init__(self, graph_data):
        self.graph_data = graph_data

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        data = self.graph_data[idx]
        return data

    @property
    def num_features(self):
        return self.graph_data[0].num_features


class GNNModel(nn.Module):
    def __init__(self, dataset):  # Pass `dataset` as an argument
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = nn.Linear(16, 1)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = global_mean_pool(x, batch=None)
        x = self.lin(x)
        return x

@app.get("/recomendar/{user_id}")
async def recomendar(user_id: int):
    global GNNRecommender, dataloader, user_mapping, item_mapping, item_features

    # Get the user node index from the user_id
    user_node_index = user_mapping.get(user_id)

    if user_node_index is None:
        return f"User with ID {user_id} not found."

    # Get the item node indices
    item_node_indices = np.arange(len(item_features))

    # Create user tensor and item tensor
    user_tensor = torch.tensor([user_node_index], dtype=torch.long)
    item_tensor = torch.tensor(item_node_indices, dtype=torch.long)

    # Repeat user tensor to match the length of item tensor
    user_tensor = user_tensor.repeat(item_tensor.size(0), 1)

    # Create edge tensor using user tensor and item tensor
    edge_tensor = torch.cat((user_tensor, item_tensor.view(-1, 1)), dim=1)

    # Create edge attribute tensor with dummy values (0.0)
    edge_attr_tensor = torch.zeros(item_tensor.size(0), dtype=torch.float)

    # Create PyG Data object for prediction
    data = Data(x=torch.tensor(user_features, dtype=torch.float),
                y=None,
                edge_index=edge_tensor.t().contiguous(),
                edge_attr=edge_attr_tensor)

    # Set the GNN model to evaluation mode
    GNNRecommender.eval()

    # Make predictions
    with torch.no_grad():
        predictions = GNNRecommender(data.x, data.edge_index, data.edge_attr)

    # Sort the predictions in descending order and get the top 5 item indices
    top_indices = torch.argsort(predictions.view(-1), descending=True)[:5]

    # Get the corresponding item codes from the item mapping
    recommended_items = [list(item_mapping.keys())[list(item_mapping.values()).index(idx.item())]
                         for idx in top_indices]

    return {"user_id": user_id, "recommended_items": recommended_items}






if __name__ == '__main__':
    uvicorn.run(app, host=hostName, port=serverPort)