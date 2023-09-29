from http.server import BaseHTTPRequestHandler, HTTPServer
import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import pyodbc
import warnings

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

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
    user_ids = data_sales["CodCliente"].map(user_mapping)
    item_ids = data_sales["CodArticu"].map(item_mapping)

    # Replace NaN values with a unique placeholder value
    user_ids = user_ids.fillna(-1)
    item_ids = item_ids.fillna(-1)

    # Convert to integer
    user_ids = user_ids.astype(int)
    item_ids = item_ids.astype(int)

    edge_index = torch.tensor([user_ids.values, item_ids.values], dtype=torch.long)

    # Step 6: Create Edge Features (e.g., Cantidad)
    edge_features = data_sales["Cantidad"].fillna(0).values

    # Step 7: Define Dataset Class
    class RecommenderDataset(Dataset):
        def __init__(self, edge_index, edge_features, user_features, item_features):
            self.edge_index = edge_index
            self.edge_features = edge_features
            self.user_features = user_features
            self.item_features = item_features

        def __len__(self):
            return len(self.edge_index)

        def __getitem__(self, idx):
            edge_index = self.edge_index[:, idx]
            edge_feature = self.edge_features[idx]
            user_feature = self.user_features[edge_index[0]]
            item_feature = self.item_features[edge_index[1]]
            return edge_index, edge_feature, user_feature, item_feature

    # Step 8: Define GNN Model
    class GNNRecommender(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GNNRecommender, self).__init__()
            self.gcn_layer1 = GraphConvolution(input_dim, hidden_dim)
            self.gcn_layer2 = GraphConvolution(hidden_dim, output_dim)

        def forward(self, edge_index, edge_features, user_features, item_features):
            x = torch.cat((user_features, item_features), dim=0)
            x = self.gcn_layer1(x, edge_index)
            x = F.relu(x)
            x = self.gcn_layer2(x, edge_index)
            user_embeddings, item_embeddings = x.split(len(user_features), dim=0)
            return user_embeddings, item_embeddings

    class GraphConvolution(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(GraphConvolution, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)

        def forward(self, x, edge_index):
            adj_matrix = torch.sparse_coo_tensor(
                edge_index, torch.ones(edge_index.shape[1]), (x.shape[0], x.shape[0])
            ).to_dense()
            adj_matrix = F.normalize(adj_matrix, p=1, dim=1)
            x = torch.spmm(adj_matrix, x)
            x = self.linear(x)
            return x
        
        
    # Step 9: Prepare Data and Dataloader
    
    dataset = RecommenderDataset(edge_index, edge_features, user_features, item_features)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True) 

   
    return "Modelo preparado"

@app.get("/entrenarModelo")
async def entrenarModelo():
    global data_items

    # Step 10: Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = user_features.shape[1] + item_features.shape[1]
    hidden_dim = 64
    output_dim = 32
    model = GNNRecommender(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            edge_index_batch = batch[0].to(device)
            edge_feature_batch = batch[1].to(device)
            user_feature_batch = batch[2].to(device)
            item_feature_batch = batch[3].to(device)
            
            user_embeddings, item_embeddings = model(
                edge_index_batch, edge_feature_batch, user_feature_batch, item_feature_batch
            )
            
            loss = F.mse_loss(user_embeddings, item_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * user_embeddings.size(0)
        
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
 
    
    return "Modelo ENTRENADO"






if __name__ == '__main__':
    uvicorn.run(app, host=hostName, port=serverPort)