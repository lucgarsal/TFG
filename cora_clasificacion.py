import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import os
from torch.nn import Linear
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.nn import GCNConv
"""En primer lugar, importamos la clase Planetoid de torch_geometric.datasets. 
A continuación, instanciamos un objeto de la clase Planetoid con los argumentos root='/tmp/Cora' y name='Cora'. 
Accedemos al primer elemento del conjunto de datos y lo almacenamos en la variable data. 
Finalmente, imprimimos la información solicitada."""

from torch_geometric.datasets import Planetoid


dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
print(data)

#Como podemos observar por pantalla, obtenemos diversas características del conjunto de datos Cora.

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


#Mostramos la matriz de enlaces en formato COO.
print(data.edge_index.t())

#Mostramos las etiquetas de los primeros 100 nodos 
print(data.y[0:100])

#Mostramos la máscara que indica que nodos son para entrenamiento viendo que son los primeros 140 
print(data.train_mask[0:150])

#Función de visualización del grafo
def visualize(h, color, filename):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig(f'images/{filename}')
    plt.show()

#Construyo un perceptrón multicapa MLP que opera únicamente sobre las características de los nodos.
#Este será el paso inicial en todos los dataset para ver cómo mejora la clasificación según cambiamos los criterios

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

model_feature = MLP(hidden_channels=16)
out = model_feature(data.x)
visualize(out, color=data.y, filename='CoraUntrained_features.png') #Será similar a todas las imágenes de pre-entrenamiento
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model_feature.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

#Definimos el entrenamiento con la diferencia de que solo usaremos las características de los nodos
def train():
      model_feature.train()
      optimizer.zero_grad()  
      out = model_feature(data.x)  
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()
      optimizer.step()  
      return loss

def test():
      model_feature.eval()
      out = model_feature(data.x)
      pred = out.argmax(dim=1)  
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  
      return test_acc

for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')

model_feature.eval()
out = model_feature(data.x)
visualize(out, color=data.y, filename='CoraTrained_feature.png')


"""
Este código define un modelo de Red Convolucional Gráfica (GCN) utilizando la librería PyTorch Geometric. En primer lugar, se importan los módulos necesarios:
GCNConv de torch_geometric.nn y F de torch.nn.functional. A continuación, se define una clase llamada GCN, que hereda de torch.nn.Module. 
El método constructor __init__ toma un parámetro hidden_channels, que especifica el número de canales ocultos en las capas GCN. 
En el constructor, se establece una semilla aleatoria para la reproducibilidad, y se definen dos capas GCN utilizando GCNConv. 
La primera capa toma como entrada el número de características del conjunto de datos (dataset.num_features) y el número de canales ocultos, 
y la segunda capa toma como entrada el número de canales ocultos y el número de clases del conjunto de datos (dataset.num_classes). 
El método forward toma como entrada las características de los nodos x y los índices de las aristas edge_index, y aplica las dos capas GCN en secuencia. 
Entre las capas, se aplica una función de activación ReLU a la salida de la primera capa y una capa de abandono con una probabilidad de 0,5 para evitar el sobreajuste. 
Por último, se crea una instancia de la clase GCN con hidden_channels=16, y se imprime el modelo para mostrar la estructura de las dos capas GCN."""

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model_gcn = GCN(hidden_channels=16)
print(model_gcn)

model_gcn.eval()
out = model_gcn(data.x, data.edge_index)
visualize(out, color=data.y, filename='CoraUntrained.png')

model_gcn = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model_gcn.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

#Entrenaremos nuestro modelo en 100 epoch utilizando la optimización de Adam y la función de pérdida de entropía cruzada. 
def train():
      model_gcn.train()
      optimizer.zero_grad()
      out = model_gcn(data.x, data.edge_index)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      return loss

def test():
      model_gcn.eval()
      out = model_gcn(data.x, data.edge_index)
      pred = out.argmax(dim=1)
      test_correct = pred[data.test_mask] == data.y[data.test_mask]
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
      return test_acc

#Entrenamos el modelo durante 100 epoch e imprimimos la pérdida en cada época.
for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

#Evaluamos el modelo y obtenemos la precisión en el conjunto de pruebas.
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')

#Visualizamos los datos entrenados
model_gcn.eval()
out = model_gcn(data.x, data.edge_index)
visualize(out, color=data.y, filename='CoraTrained.png')