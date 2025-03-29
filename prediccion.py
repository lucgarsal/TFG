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
MODEL_TYPE = 'CompleteGraph' # Cambiar a 'Concatenation' para usar el modelo de concatenación
PREDICTOR_TYPE = 'GCN' # Cambiar a 'GAT', 'GATSAGE' o 'VGAE' según el tipo de predictor que quieras usar

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
"""
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
"""

#Generación del embedding usando pykeen. El problema es que tenemos que cambiar nuestro grafo
#al formato de tripletas con el que trabaja pykeen
# Establecemos la semilla para reproducibilidad. Si no, toma un valor aleatorio muy alto
# y no se puede reproducir el resultado.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def generate_embeddings_with_pykeen(data):
    sources = data.edge_index[0].cpu().numpy()
    targets = data.edge_index[1].cpu().numpy()  
    relations = np.array(["linked_to"] * len(sources), dtype=str) 

    # Convertir a formato de PyKEEN
    triples_array = np.column_stack((sources.astype(str), relations, targets.astype(str)))

    # Crear la TriplesFactory para entrenamiento y prueba (si no no funciona el pipeline)
    tf = TriplesFactory.from_labeled_triples(triples_array)
    tf_train, tf_test = tf.split([0.8, 0.2])

    # Entrenamos un modelo de embeddings en PyKEEN
    result = pipeline(
        training=tf_train,
        testing=tf_test,
        model="TransE",
        random_seed=SEED,
        training_kwargs={
            "num_epochs": 100, 
            "batch_size": 256
        },
        optimizer_kwargs={
            "lr": 0.01
        },
    )
    return result

# Obtenemos embeddings de nodos
if not hasattr(data, "embeddings"):  # Solo entrena si no existen embeddings, para no repetir el entrenamiento
    result = generate_embeddings_with_pykeen(data)
    embedding_model = result.model
    node_embeddings = embedding_model.entity_representations[0]
    data.embeddings = node_embeddings


# GCNLinkPredictor
class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLinkPredictor, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.fc = torch.nn.Linear(out_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return torch.sigmoid(x)

# GATLinkPredictor
class GATLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GATLinkPredictor, self).__init__()
        self.conv = GATConv(in_channels, out_channels, heads=heads, concat=False)
        self.fc = torch.nn.Linear(out_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)  
        x = self.fc(x)
        return torch.sigmoid(x)

# GATSAGELinkPredictor
class GATSAGELinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATSAGELinkPredictor, self).__init__()
        self.conv = SAGEConv(in_channels, out_channels)
        self.fc = torch.nn.Linear(out_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return torch.sigmoid(x)

# VGAELinkPredictor
class VGAELinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VGAELinkPredictor, self).__init__()
        self.encoder = VGAE(
            torch.nn.Sequential(
                GCNConv(in_channels, hidden_channels),
                torch.nn.ReLU(),
                GCNConv(hidden_channels, out_channels)
            )
        )
        self.fc = torch.nn.Linear(out_channels, 1)

    def forward(self, x, edge_index):
        z = self.encoder.encode(x, edge_index)
        x = self.fc(z)
        return torch.sigmoid(x)

    
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
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.adjust = None
        if in_channels != out_channels:
            self.adjust = torch.nn.Linear(in_channels, out_channels)
        
        self.relu =torch.nn.ReLU()
        
        # BatchNorm para estabilizar el entrenamiento
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        out = self.linear(x)
        if self.adjust is not None:
            residual = self.adjust(residual)
        
        out = out + residual
        out = self.bn(out) 
        out = self.relu(out) 

        return out
"""
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, layer_sizes, predictor_type):
        super(LinkPredictor, self).__init__()
        if isinstance(layer_sizes, int):
            raise TypeError("layer_sizes debe ser una lista de enteros.")
        layers = []
        #Concatenamos la capa inicial al tamaño deseado
        self.concat_layer = torch.nn.Linear(in_channels * 2, layer_sizes[0])
        for i in range(len(layer_sizes) - 1):
            layers.append(ResidualBlock(layer_sizes[i], layer_sizes[i + 1]))
        #layers.append(torch.nn.Linear(layer_sizes[-1], 1))
        self.net = torch.nn.Sequential(*layers)
        
        if predictor_type == 'GCN':
            self.link_predictor = GCNLinkPredictor(layer_sizes[-1], layer_sizes[-1])
        elif predictor_type == 'GAT':
            self.link_predictor = GATLinkPredictor(layer_sizes[-1], layer_sizes[-1])
        elif predictor_type == 'GATSAGE':
            self.link_predictor = GATSAGELinkPredictor(layer_sizes[-1], layer_sizes[-1])
        elif predictor_type == 'VGAE':
            self.link_predictor = VGAELinkPredictor(layer_sizes[-1], layer_sizes[-1])
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")

        self.final_layer = torch.nn.Linear(layer_sizes[-1], 1)
        
    def forward(self, x, x_i, x_j, edge_index, model_type):
        if model_type=='CompleteGraph':
            x = torch.cat([x_i, x_j], dim=-1)
        elif model_type=='Concatenation': #Estamos en el de concatenación
            row, col = edge_index
            x_i = x[row]
            x_j = x[col]
            x = torch.cat([x_i, x_j], dim=-1)
            
        # Aplicamos la capa de concatenación
        x = F.relu(self.concat_layer(x))
        x = self.net(x)
        x = self.link_predictor(x, edge_index)
        return torch.sigmoid(x)

# Crear carpeta con timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
path_name = f"tipo_entrada-{MODEL_TYPE}_uso_embeddings-{USE_EMBEDDINGS}_tipo_gnn-{PREDICTOR_TYPE}_{timestamp}"
log_dir = os.path.join("results", path_name)
os.makedirs(log_dir, exist_ok=True)

# Inicializar TensorBoard
writer = SummaryWriter(log_dir=log_dir)
# Entrenamiento y predicción de enlaces
def train_link_predictor(data, model, optimizer, device,model_type, use_embeddings, epochs=100):
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

        if model_type=='Concatenation':
            pos_edge_index = sampled_edge_index
            neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1))
        if use_embeddings:
            pos_out = model(data.embeddings.weight, data.embeddings.weight[pos_edge_index[0]], data.embeddings.weight[pos_edge_index[1]], pos_edge_index, model_type)
            neg_out = model(data.embeddings.weight, data.embeddings.weight[neg_edge_index[0]], data.embeddings.weight[neg_edge_index[1]], neg_edge_index, model_type)
        else:
            pos_out = model(data.x, data.x[pos_edge_index[0]], data.x[pos_edge_index[1]], pos_edge_index, model_type)
            neg_out = model(data.x, data.x[neg_edge_index[0]], data.x[neg_edge_index[1]], neg_edge_index, model_type)

        #Añadir la opcion de que reciba tanto nodos como los embeddings
        

        pos_loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss.item(), epoch)
        

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

def predict_links(data, model, device, model_type, use_embeddings=False, threshold=0.5):
    model.eval()
    data = data.to(device)
    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1))
    
    if model=='Concatenation':
        pos_edge_index = sampled_edge_index
        neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1))

    if use_embeddings:
        pos_out = model(data.embeddings.weight, data.embeddings.weight[pos_edge_index[0]], data.embeddings.weight[pos_edge_index[1]], pos_edge_index, model_type)
        neg_out = model(data.embeddings.weight, data.embeddings.weight[neg_edge_index[0]], data.embeddings.weight[neg_edge_index[1]], neg_edge_index, model_type)
    else:
        pos_out = model(data.x, data.x[pos_edge_index[0]], data.x[pos_edge_index[1]], pos_edge_index, model_type)
        neg_out = model(data.x, data.x[neg_edge_index[0]], data.x[neg_edge_index[1]], neg_edge_index, model_type)
    
    pos_pred = (pos_out > threshold).cpu().numpy()
    neg_pred = (neg_out > threshold).cpu().numpy()
    
    return pos_pred, neg_pred, pos_edge_index, neg_edge_index


# Inicializamos el modelo de predicción de enlaces y el optimizador

layer_sizes = [data.num_node_features * 2, 64, 32, 16] if isinstance(data.num_node_features, int) else data.num_node_features
#Habia un problema porque data.num_node_features es un tensor y no un entero
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

use_embeddings = USE_EMBEDDINGS # Cambiar a False para usar características de los nodos

#Para el modelo de concatenación
graph = data
edge_list = graph.edge_index
#Lo movemos todos al mismo
graph = graph.to(device)
edge_list = edge_list.to(device)

# Seleccionamos aleatoriamente algunas aristas del edge_index
num_edges = edge_list.size(1)
num_sampled_edges = int(0.5 * num_edges)  # Por ejemplo, selecciona el 50% de las aristas
torch.manual_seed(SEED)
sampled_indices = torch.randperm(num_edges, device=device)[:num_sampled_edges]
sampled_edge_index = edge_list[:, sampled_indices].to(device)

model_type = MODEL_TYPE # Cambiar a 'Concatenation' para usar el modelo de concatenación
model = LinkPredictor(in_channels=data.num_node_features, layer_sizes=layer_sizes, predictor_type=PREDICTOR_TYPE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Entrenamos el modelo
train_link_predictor(data, model, optimizer, device, model_type, use_embeddings, epochs=100)
pos_pred, neg_pred, pos_edge_index, neg_edge_index = predict_links(data, model, device, model_type, use_embeddings=False)


#print(f'Positive link predictions: {pos_pred}')
#print(f'Negative link predictions: {neg_pred}')

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