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
data.edge_index.t()

#Mostramos las etiquetas de los primeros 100 nodos 
data.y[0:100] 

#Mostramos la máscara que indica que nodos son para entrenamiento viendo que son los primeros 140 
data.train_mask[0:150] 

"""Ahora, vamos a definir un modelo para realizar una clasificación de los nodos. Para ello, vamos a usar dos capas GCNConv que implementarán 
la Graph Neural Network. Después de la primera GCN (convierte de la dimensión número de features al número de canales 16) añadimos un ReLU y 
después de la segunda (convierte de 16 al número de clases) un softmax sobre el número de clases. Como se puede ver, las capas se aplican sobre 
los datos con los features de cada nodo y sobre edge_index, que contiene la estructura del grafo."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

model = GCN()
print(model)

"""Ahora, vamos a entrenar el modelo usando 250 epochs (rondas) de los datos. 
Como se puede observar, usamos la máscara de entrenamiento para decir cuáles son los nodos que se tienen que usar para entrenar el modelo."""
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(250):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

#Por último, evaluamos el modelo usando la máscara que indica los nodos de test y vemos que el modelo tiene una buena tasa de acierto.
model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))
