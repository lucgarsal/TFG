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

USE_EMBEDDINGS = False
GNN_TYPE = 'GATSAGE' # Cambiar a 'GAT', 'GATSAGE' o 'VGAE' según el tipo de predictor que quieras usar

# Cargar el dataset
dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
print(f"Dataset: {dataset.data}")
data = dataset[0]

# Preprocesado básico de los datos

# Quitamos los nodos aislados
# data.x, data.edge_index, data.y, mask = remove_isolated_nodes(data.x, data.edge_index, data.y)

# La normalización se realiza automáticamente con el transform=NormalizeFeatures() al cargar el dataset
# Si las características numéricas tuviesen rangos muy distintos, también podemos normalizarlas usando un escalador
"""
scaler = StandardScaler()
data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float32)
"""

# Si queremos trabajar con un grafo no dirigido, duplicamos las aristas.
# data.edge_index = to_undirected(data.edge_index)

# Si x tiene muchas dimensiones, podemos aplicar PCA o t-SNE.
"""
pca = PCA(n_components=100)  # Reducimos a 100 dimensiones
data.x = torch.tensor(pca.fit_transform(data.x.numpy()), dtype=torch.float32)
"""

# División de Datos
# En este caso, los dataset de Planetoid ya vienen con una división predefinida en entrenamiento, validación y prueba
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

# Asignar las máscaras al objeto data
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

"""
print(f'Número de nodos de entrenamiento: {data.train_mask.sum()}')
print(f'Número de nodos de validación: {data.val_mask.sum()}')
print(f'Número de nodos de prueba: {data.test_mask.sum()}')
"""

# Función de visualización del grafo, que almacena los resultados en la carpeta images
def visualize(h, color, filename, pos_edge_index=None, neg_edge_index=None):
    h = h.detach().cpu().numpy()
    z = TSNE(n_components=2).fit_transform(h)

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color.cpu(), cmap="Set2")

    if pos_edge_index is not None and neg_edge_index is not None:
        for i, j in pos_edge_index.t().cpu().numpy():
            plt.plot([z[i, 0], z[j, 0]], [z[i, 1], z[j, 1]], color='green', alpha=0.5)
        for i, j in neg_edge_index.t().cpu().numpy():
            plt.plot([z[i, 0], z[j, 0]], [z[i, 1], z[j, 1]], color='red', alpha=0.5)

    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig(f'images/{filename}')
    plt.show()

# Mostrar características del conjunto de datos

print(f'Número de nodos: {data.num_nodes}')
print(f'Número de features por nodo: {data.num_node_features}')
print(f'Número de clases: {dataset.num_classes}')
print(f'Número de enlaces: {data.num_edges}')
print(f'Grado medio de los nodos: {data.num_edges / data.num_nodes:.2f}')
print(f'Número de nodos de entrenamiento: {data.train_mask.sum()}')
print(f'Número de nodos de validación: {data.val_mask.sum()}')
print(f'Número de nodos de prueba: {data.test_mask.sum()}')
print(f'Contiene nodos aislados: {data.has_isolated_nodes()}')
print(f'Contiene bucles: {data.has_self_loops()}')
print(f'No es dirigido: {data.is_undirected()}')
print(f"Estructura de los datos: {data}")
print(f"Some features: {data.x[:5]}")


# GCNLinkPredictor
class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLinkPredictor, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# GATLinkPredictor
class GATLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATLinkPredictor, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels)
        self.conv2 = GATConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# GATSAGELinkPredictor
class GATSAGELinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATSAGELinkPredictor, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels)
        self.conv2 = SAGEConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# VGAELinkPredictor
class VGAELinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGAELinkPredictor, self).__init__()
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
    #Este da fallo al multiplicar las matrices por las dimensiones
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
    def __init__(self, in_channels, layer_sizes, gnn_type):
        super(LinkPredictor, self).__init__()
        if isinstance(layer_sizes, int):
            raise TypeError("layer_sizes debe ser una lista de enteros.")
        
        if gnn_type == 'GCN':
            self.gnn_model = GCNLinkPredictor(in_channels, layer_sizes[0])
        elif gnn_type == 'GAT':
            self.gnn_model = GATLinkPredictor(in_channels, layer_sizes[0])
        elif gnn_type == 'GATSAGE':
            self.gnn_model = GATSAGELinkPredictor(in_channels, layer_sizes[0])
        elif gnn_type == 'VGAE':
            self.gnn_model = VGAELinkPredictor(in_channels, layer_sizes[0])
        else:
            raise ValueError(f"Unknown predictor type: {gnn_type}")
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(ResidualBlock(layer_sizes[i], layer_sizes[i + 1]))
        self.net = torch.nn.Sequential(*layers)
        
        self.final_layer = torch.nn.Linear(layer_sizes[-1], 1)
    
    #Primero hacemos la convolución para obtener una mejor representación de los nodos
    #Después extraemos las características de los nodos dados por la lista edge_index y las concatenamos
    #Finalmente pasamos por las capas residuales cuyos tamaños se definen en layer_sizes
    #y por la capa final que nos da la predicción de si existe o no un enlace entre los nodos

    def forward(self, data, edge_index):
        x = self.gnn_model(data.x, data.edge_index)
        #x = self.gnn_model(data, edge_index)
        x=data.x
        edge_index=edge_index.long()
        src_idx, dest_idx = edge_index
        x_i = x[src_idx]
        x_j = x[dest_idx]
        x = torch.cat([x_i, x_j], dim=-1)

        x = self.net(x)

        x= self.final_layer(x)
        res=torch.sigmoid(x)

        return res

# Crear carpeta con timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
path_name = f"uso_embeddings-{USE_EMBEDDINGS}_tipo_gnn-{GNN_TYPE}_{timestamp}"
log_dir = os.path.join("results", path_name)
os.makedirs(log_dir, exist_ok=True)

# Inicializar TensorBoard
writer = SummaryWriter(log_dir=log_dir)
# Entrenamiento y predicción de enlaces
def train_link_predictor(data, model, optimizer, device, use_embeddings, epochs=100):
    model.train()
    data = data.to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
       
        pos_edge_index = sampled_edge_index
        neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1)).to(torch.long)

        pos_edge_index = pos_edge_index.long()
        neg_edge_index = neg_edge_index.long()  
        if use_embeddings:
            pos_out = model(data.embeddings, pos_edge_index)
            neg_out = model(data.embeddings, neg_edge_index)
        else:
            pos_out = model(data, pos_edge_index)
            neg_out = model(data, neg_edge_index)

        #Añadir la opcion de que reciba tanto nodos como los embeddings
        

        pos_loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss.item(), epoch)
        

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

def predict_links(data, model, device, use_embeddings=False, threshold=0.5):
    model.eval()
    data = data.to(device)
    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1)).to(torch.long)
    pos_edge_index = pos_edge_index.long()
    neg_edge_index = neg_edge_index.long()
    if use_embeddings:
            pos_out = model(data.embeddings, pos_edge_index)
            neg_out = model(data.embeddings, neg_edge_index)
    else:
            pos_out = model(data, pos_edge_index)
            neg_out = model(data, neg_edge_index)
    
    pos_pred = (pos_out > threshold).cpu().numpy()
    neg_pred = (neg_out > threshold).cpu().numpy()
    
    return pos_pred, neg_pred, pos_edge_index, neg_edge_index


# Inicializamos el modelo de predicción de enlaces y el optimizador

layer_sizes = [data.num_node_features * 2, 64, 32, 16] if isinstance(data.num_node_features, int) else data.num_node_features
#Habia un problema porque data.num_node_features es un tensor y no un entero
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

use_embeddings = USE_EMBEDDINGS # Cambiar a False para usar características de los nodos
graph = data
edge_list = graph.edge_index
#Lo movemos todos al mismo
graph = graph.to(device)
edge_list = edge_list.to(device).long()

# Seleccionamos aleatoriamente algunas aristas del edge_index
num_edges = edge_list.size(1)
num_sampled_edges = int(0.5 * num_edges)  # Por ejemplo, selecciona el 50% de las aristas
sampled_indices = torch.randperm(num_edges, device=device)[:num_sampled_edges]
sampled_edge_index = edge_list[:, sampled_indices].to(device).long()

print(f"edge_list dtype: {edge_list.dtype}")
print(f"sampled_edge_index dtype: {sampled_edge_index.dtype}")


model = LinkPredictor(in_channels=data.num_node_features, layer_sizes=layer_sizes, gnn_type=GNN_TYPE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Entrenamos el modelo
train_link_predictor(data, model, optimizer, device, use_embeddings, epochs=100)
pos_pred, neg_pred, pos_edge_index, neg_edge_index = predict_links(data, model, device, use_embeddings=False)

def evaluate_model(pos_pred, neg_pred):
    y_true = [1] * len(pos_pred) + [0] * len(neg_pred)
    y_pred = list(pos_pred) + list(neg_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    auc_roc = roc_auc_score(y_true, y_pred)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC-ROC: {auc_roc:.4f}')

writer.close()
evaluate_model(pos_pred, neg_pred)

# Llamar a la función de visualización de embeddings cuando trabajemos con ellos. 
#visualize_embeddings(data, pos_edge_index, neg_edge_index, 'link_predictions.png')