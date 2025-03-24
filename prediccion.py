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
from torch_geometric.nn import GCNConv, GATConv

import pykeen
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import numpy as np

# Cargar el dataset
dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
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

print(f'Número de nodos de entrenamiento: {data.train_mask.sum()}')
print(f'Número de nodos de validación: {data.val_mask.sum()}')
print(f'Número de nodos de prueba: {data.test_mask.sum()}')

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

#Generación del embedding usando pykeen. El problema es que tenemos que cambiar nuestro grafo
#al formato de tripletas con el que trabaja pykeen
"""
sources = data.edge_index[0].tolist()
targets = data.edge_index[1].tolist()
relations = ["linked_to"]*len(sources)

# Convertir a formato de PyKEEN
triples_array = np.array([(str(s), r, str(t)) for s, r, t in zip(sources, relations, targets)])
tf = TriplesFactory.from_labeled_triples(triples_array)

# Entrenar un modelo de embeddings en PyKEEN
result = pipeline(
    training=tf,
    model="TransE",
    training_kwargs={"num_epochs": 100}, 
)

# Obtenemos embeddings de nodos
embedding_model = result.model
node_embeddings = embedding_model.entity_representations[0]  # Representaciones de nodos
data.embeddings = node_embeddings
"""
# GCNLinkPredictor
class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNLinkPredictor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x_i, x_j, edge_index):
        x = torch.cat([x_i, x_j], dim=-1)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

# GATLinkPredictor
class GATLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATLinkPredictor, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, out_channels)

    def forward(self, x_i, x_j, edge_index):
        x = torch.cat([x_i, x_j], dim=-1)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

# GATSAGELinkPredictor
class GATSAGELinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATSAGELinkPredictor, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)
        self.conv3 = GATConv(out_channels, 1)

    def forward(self, x_i, x_j, edge_index):
        x = torch.cat([x_i, x_j], dim=-1)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

# VGAELinkPredictor
class VGAELinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VGAELinkPredictor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.conv3 = GCNConv(out_channels, 1)

    def forward(self, x_i, x_j, edge_index):
        x = torch.cat([x_i, x_j], dim=-1)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

    
# Definimos el modelo para la predicción de enlaces. En primer lugar definimos un bloque residual para las capas. Esto sirve
# para que no se pierda información en las capas intermedias.
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.adjust = None
        if in_channels != out_channels:
            self.adjust = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        residual = x
        out = self.linear(x)
        if self.adjust is not None:
            residual = self.adjust(residual)
        return self.relu(out + residual)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, layer_sizes, predictor_type):
        super(LinkPredictor, self).__init__()
        self.concat_layer = torch.nn.Linear(in_channels * 2, hidden_channels)
        if isinstance(layer_sizes, int):
            raise TypeError("layer_sizes debe ser una lista de enteros.")
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(ResidualBlock(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(torch.nn.Linear(layer_sizes[-1], 1))
        self.net = torch.nn.Sequential(*layers)

        if predictor_type == 'GCN':
            self.link_predictor = GCNLinkPredictor(in_channels, hidden_channels, layer_sizes[-1])
        elif predictor_type == 'GAT':
            self.link_predictor = GATLinkPredictor(in_channels, hidden_channels, layer_sizes[-1])
        elif predictor_type == 'GATSAGE':
            self.link_predictor = GATSAGELinkPredictor(in_channels, hidden_channels, layer_sizes[-1])
        elif predictor_type == 'VGAE':
            self.link_predictor = VGAELinkPredictor(in_channels, hidden_channels, layer_sizes[-1])
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        x = F.relu(self.concat_layer(x))
        return torch.sigmoid(self.net(x))


class GraphConcatenationNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, layer_sizes, edge_index):
        super(GraphConcatenationNetwork, self).__init__()
        self.edge_index = edge_index
        self.concat_layer = torch.nn.Linear(in_channels * 2, hidden_channels)
        if isinstance(layer_sizes, int):
            raise TypeError("layer_sizes debe ser una lista de enteros.")
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(ResidualBlock(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(torch.nn.Linear(layer_sizes[-1], 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, edge_index):
        # Concatenamos las características de los nodos conectados por las aristas
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        x = torch.cat([x_i, x_j], dim=-1)
        
        # Aplicamos la capa de concatenación
        x = F.relu(self.concat_layer(x))

        return torch.sigmoid(self.net(x))
"""
graph = data
edge_list = graph.edge_index
# Seleccionamos aleatoriamente algunas aristas del edge_index
num_edges = edge_list.size(1)
num_sampled_edges = int(0.5 * num_edges)  # Por ejemplo, selecciona el 50% de las aristas
sampled_indices = torch.randperm(num_edges)[:num_sampled_edges]
sampled_edge_index = edge_list[:, sampled_indices]

layer_sizes = [data.num_node_features * 2, 64, 32, 16] if isinstance(data.num_node_features, int) else data.num_node_features
model = GraphConcatenationNetwork(in_channels=graph.num_node_features, hidden_channels=64, layer_sizes=layer_sizes, edge_index=sampled_edge_index)
output = model(graph.x, sampled_edge_index)
"""  

# Llamar a la función de visualización solo cuando tengamos un subgrafo o un dataset pequeño
# visualize(data.x, color=data.y, filename='link_predictions.png', pos_edge_index=pos_edge_index, neg_edge_index=neg_edge_index)

# Entrenamiento y predicción de enlaces
def train_link_predictor(data, model, optimizer, device, use_embeddings, epochs=100):
    """
    Trains a link prediction model.
    Args:
        data (torch_geometric.data.Data): The input data containing node features and edge indices.
        model (torch.nn.Module): The link prediction model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        device (torch.device): The device (CPU or GPU) on which to perform training.
        use_embeddings (bool, optional): If True, use node embeddings for training. If False, use node features. Default is False.
        epochs (int, optional): The number of training epochs. Default is 100.
    Returns:
        None
    """
    model.train()
    data = data.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1))
        
        if use_embeddings:
            pos_out = model(data.embeddings[pos_edge_index[0]], data.embeddings[pos_edge_index[1]])
            neg_out = model(data.embeddings[neg_edge_index[0]], data.embeddings[neg_edge_index[1]])
        else:
            pos_out = model(data.x[pos_edge_index[0]], data.x[pos_edge_index[1]])
            neg_out = model(data.x[neg_edge_index[0]], data.x[neg_edge_index[1]])
        
        pos_loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

def predict_links(data, model, device, use_embeddings=False, threshold=0.5):
    model.eval()
    data = data.to(device)
    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1))
    
    if use_embeddings:
        pos_out = model(data.embeddings[pos_edge_index[0]], data.embeddings[pos_edge_index[1]])
        neg_out = model(data.embeddings[neg_edge_index[0]], data.embeddings[neg_edge_index[1]])
    else:
        pos_out = model(data.x[pos_edge_index[0]], data.x[pos_edge_index[1]])
        neg_out = model(data.x[neg_edge_index[0]], data.x[neg_edge_index[1]])
    
    pos_pred = (pos_out > threshold).cpu().numpy()
    neg_pred = (neg_out > threshold).cpu().numpy()
    
    return pos_pred, neg_pred, pos_edge_index, neg_edge_index


# NOTA: Descomentar el código para ejecutarlo con el modelo que vayamos a entrenar. ¿Se pueden entrenar simultáneamente?

# Inicializamos el modelo de predicción de enlaces y el optimizador
layer_sizes = [data.num_node_features * 2, 64, 32, 16] if isinstance(data.num_node_features, int) else data.num_node_features
#Habia un problema porque data.num_node_features es un tensor y no un entero
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LinkPredictor(in_channels=data.num_node_features, hidden_channels=64, layer_sizes=layer_sizes, predictor_type='GCN').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

use_embeddings = False  # Cambiar a False para usar características de los nodos

train_link_predictor(data, model, optimizer, device, use_embeddings=use_embeddings, epochs=100)
pos_pred, neg_pred, pos_edge_index, neg_edge_index = predict_links(data, model, device, use_embeddings=use_embeddings)

print(f'Positive link predictions: {pos_pred}')
print(f'Negative link predictions: {neg_pred}')

def evaluate_model(pos_pred, neg_pred):
    y_true = [1] * len(pos_pred) + [0] * len(neg_pred)
    y_pred = list(pos_pred) + list(neg_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC-ROC: {auc_roc:.4f}')

evaluate_model(pos_pred, neg_pred)

# Llamar a la función de visualización de embeddings cuando trabajemos con ellos. 
#visualize_embeddings(data, pos_edge_index, neg_edge_index, 'link_predictions.png')