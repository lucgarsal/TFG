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

#dataset = Planetoid(root='/tmp/Cora', name='Cora')
dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
data = dataset[0]

#En primer lugar, realizo un preprocesado básico de los datos. Aunque la mayoría de los datasets de Pytorch Geometric ya vienen preprocesados, 
#es importante tener en cuenta los siguientes aspectos:

# Quitamos los nodos aislados
#data.x, data.edge_index, data.y, mask = remove_isolated_nodes(data.x, data.edge_index, data.y)

# La normalización se realiza automáticamente con el transform=NormalizeFeatures() al cargar el dataset
# Si las características numéricas tuviesen rangos muy distintos, también podemos normalizarlas usando un escalador
"""
scaler = StandardScaler()
data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float32)
"""

# Si queremos trabajar con un grafo no dirigido, duplicamos las aristas.
#data.edge_index = to_undirected(data.edge_index)

#Si x tiene muchas dimensiones, podemos aplicar PCA o t-SNE.
"""
pca = PCA(n_components=100)  # Reducimos a 100 dimensiones
data.x = torch.tensor(pca.fit_transform(data.x.numpy()), dtype=torch.float32)
"""

# División de Datos
# En este caso, los dataset de Planetoid ya vienen con una división predefinida en entrenamiento, validación y prueba
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

# Creación de Máscaras
# Las máscaras ya están creadas en los datasets de Planetoid, pero si estuvieramos trabajando con un dataset diferente, las crearíamos así:
""""
def create_masks(data, num_train, num_val, num_test):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train + num_val]] = True
    test_mask[indices[num_train + num_val:num_train + num_val + num_test]] = True

    return train_mask, val_mask, test_mask

# Ejemplo de uso:
num_train = 140
num_val = 500
num_test = 1000
train_mask, val_mask, test_mask = create_masks(data, num_train, num_val, num_test)
"""

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


# Como podemos observar por pantalla, obtenemos diversas características del conjunto de datos Cora.

print(f'Número de nodos: {data.num_nodes}')
print(f'Número de features por nodo: {data.num_node_features}')
print(f'Número de clases: {dataset.num_classes}')
print(f'Número de enlaces: {data.num_edges}')
print(f'Grado medio de los nodos: {data.num_edges / data.num_nodes:.2f}')
print(f'Número de nodos de entrenamiento: {data.train_mask.sum()}')
print(f'Número de nodos de validación: {data.val_mask.sum()}')
print(f'Número de nodos de tests: {data.test_mask.sum()}')
print(f'Contiene nodos aislados: {data.has_isolated_nodes()}')
print(f'Contiene bucles: {data.has_self_loops()}')
print(f'No es dirigido: {data.is_undirected()}')

# En primer lugar, realizamos una función para predicción de enlaces basándonos únicamente en las características de los nodos.
# Para ello, creamos una red neuronal que recibe como entrada las características de dos nodos y predice si existe un enlace entre ellos.

class LinkPredictor(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(LinkPredictor, self).__init__()
        if isinstance(layer_sizes, int):
            raise TypeError("layer_sizes debe ser una lista de enteros.")
        layers = []
        for i in range(len(layer_sizes) - 1):#corrigeme un error en esta linea de tipo int has no len

            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(layer_sizes[-1], 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        return torch.sigmoid(self.net(x))

def train_link_predictor(data, model, optimizer, device, epochs=100):
    model.train()
    data = data.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1))
        
        pos_out = model(data.x[pos_edge_index[0]], data.x[pos_edge_index[1]])
        neg_out = model(data.x[neg_edge_index[0]], data.x[neg_edge_index[1]])
        
        pos_loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

def predict_links(data, model, device, threshold=0.5):
    model.eval()
    data = data.to(device)
    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1))
    
    pos_out = model(data.x[pos_edge_index[0]], data.x[pos_edge_index[1]])
    neg_out = model(data.x[neg_edge_index[0]], data.x[neg_edge_index[1]])
    
    pos_pred = (pos_out > threshold).cpu().numpy()
    neg_pred = (neg_out > threshold).cpu().numpy()
    
    return pos_pred, neg_pred, pos_edge_index, neg_edge_index


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
layer_sizes = [data.num_node_features * 2, 64, 32, 16] if isinstance(data.num_node_features, int) else data.num_node_features
#Habia un problema porque data.num_node_features es un tensor y no un entero
model = LinkPredictor(layer_sizes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_link_predictor(data, model, optimizer, device, epochs=100)
pos_pred, neg_pred, pos_edge_index, neg_edge_index = predict_links(data, model, device)

print(f'Positive link predictions: {pos_pred}')
print(f'Negative link predictions: {neg_pred}')

def evaluate_model(pos_pred, neg_pred):
    # Crear etiquetas verdaderas
    y_true = [1] * len(pos_pred) + [0] * len(neg_pred)
    # Concatenar predicciones positivas y negativas
    y_pred = list(pos_pred) + list(neg_pred)
    
    # Calcular métricas
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

#Llamar a la función de visualización solo cuando tengamos un subgrafo o un dataset pequeño
#visualize(data.x, color=data.y, filename='link_predictions.png', pos_edge_index=pos_edge_index, neg_edge_index=neg_edge_index)