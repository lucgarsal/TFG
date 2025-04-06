import os
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import remove_isolated_nodes
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_undirected
from sklearn.decomposition import PCA
from torch_geometric.nn import GCNConv, GATConv, VGAE, SAGEConv
from torch.utils.tensorboard import SummaryWriter
import pykeen
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import numpy as np
import random
import datetime

######################################################################################
## Modelos de entrenamiento                                                         ##
######################################################################################

# GCNLinkPredictor
class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_mixed):
        super(GCNLinkPredictor, self).__init__()
        if use_mixed:
            in_channels = in_channels * 2
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# GATLinkPredictor
class GATLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_mixed):
        super(GATLinkPredictor, self).__init__()
        if use_mixed:
            in_channels = in_channels * 2
        self.conv1 = GATConv(in_channels, out_channels)
        self.conv2 = GATConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# GATSAGELinkPredictor
class GATSAGELinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_mixed):
        super(GATSAGELinkPredictor, self).__init__()
        if use_mixed:
            in_channels = in_channels * 2
        self.conv1 = SAGEConv(in_channels, out_channels)
        self.conv2 = SAGEConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# VGAELinkPredictor
class VGAELinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_mixed):
        super(VGAELinkPredictor, self).__init__()
        if use_mixed:
            in_channels = in_channels * 2
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.vgae = VGAE(self.conv1)

    def forward(self, x, edge_index):
        x = self.vgae.encode(x, edge_index)
        x = F.relu(x)
        x = self.vgae.decode(x, edge_index)
        return x

    
# Definimos el modelo para la predicción de enlaces. En primer lugar definimos un bloque residual para las capas. Esto sirve
# para que no se pierda información en las capas intermedias.
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.adjust = None
        if in_channels != out_channels:
            self.adjust = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.relu(out)

        if self.adjust is not None:
            residual = self.adjust(residual)
        return self.relu(out + residual)

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, layer_sizes, gnn_type, use_mixed):
        super(LinkPredictor, self).__init__()
        if isinstance(layer_sizes, int):
            raise TypeError("layer_sizes debe ser una lista de enteros.")
        
        if gnn_type == 'GCN':
            self.gnn_model = GCNLinkPredictor(in_channels, layer_sizes[0], use_mixed)
        elif gnn_type == 'GAT':
            self.gnn_model = GATLinkPredictor(in_channels, layer_sizes[0], use_mixed)
        elif gnn_type == 'GATSAGE':
            self.gnn_model = GATSAGELinkPredictor(in_channels, layer_sizes[0], use_mixed)
        elif gnn_type == 'VGAE':
            self.gnn_model = VGAELinkPredictor(in_channels, layer_sizes[0], use_mixed)
        else:
            raise ValueError(f"Unknown predictor type: {gnn_type}")
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            # if use_mixed:
            #     layers.append(ResidualBlock(layer_sizes[i]*4, layer_sizes[i + 1]*4))
            # else:
                layers.append(ResidualBlock(layer_sizes[i]*2, layer_sizes[i + 1]*2))
        self.net = torch.nn.Sequential(*layers)
        
        self.final_layer = torch.nn.Linear(layer_sizes[-1]*2, 1)
    
    #Primero hacemos la convolución para obtener una mejor representación del grafo
    #Después extraemos las características de los nodos dados por la lista edge_index y las concatenamos
    #Finalmente pasamos por las capas residuales cuyos tamaños se definen en layer_sizes
    #y por la capa final que nos da la predicción de si existe o no un enlace entre los nodos

    def forward(self, data, edge_index, use_embeddings, use_mixed):
        if use_embeddings:
            x = self.gnn_model(data.embeddings, edge_index)
        elif use_mixed:
            x = torch.cat([data.x, data.embeddings], dim=1)
            x = self.gnn_model(x, edge_index)
        else:
            x = self.gnn_model(data.x, data.edge_index)
        edge_index=edge_index.long()
        src_idx, dest_idx = edge_index
        x_i = x[src_idx]
        x_j = x[dest_idx]
        x = torch.cat([x_i, x_j], dim=-1)
        x = self.net(x)
        x= self.final_layer(x)
        res=torch.sigmoid(x)

        return res

