
import networkx as nx
import os
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import numpy as np

# Función de visualización del grafo que guarda la imagen en la carpeta images con el nombre dado como parámetro

def visualize_graph(G, color, filename):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig(f'images/{filename}')
    plt.show()

def visualize_embedding(h, color, filename, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig(f'images/{filename}')
    plt.show() 

from torch_geometric.datasets import KarateClub

dataset = KarateClub()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


edge_index = data.edge_index
print(edge_index.t())
#Visualizamos el grafo
G = to_networkx(data, to_undirected=True)
visualize_graph(G, color=data.y, filename='karateClubGraph.png')

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234) #imputamos 4 categorias de forma manual
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h

model = GCN()
print(model)

#Visualizamos el embedding generado por el modelo, es decir, la representación de los nodos en un espacio de 2 dimensiones.
_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')

visualize_embedding(h, color=data.y, filename='karateClubEmbedding.png')
#Como podemos ver, cada una de las clases es representada por un color distinto en el espacio de 2 dimensiones.


import time
model = GCN()
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

def train(data):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h

for epoch in range(401):
    loss, h = train(data)
    #Visualizamos el embedding cada 50 épocas y guardamos el resultado final
    #TODO guardar los resultados intermedios para añadirlos a la memoria
    if epoch % 50 == 0:
        visualize_embedding(h, color=data.y, filename='karateClubEpoch.png', epoch=epoch, loss=loss)
        time.sleep(0.3)
        ssssss
#Como podemos ver, el modelo aprende a agrupar los nodos de acuerdo a su clase a medida que el entrenamiento avanza.
#En la visualización, podemos ver cómo los nodos de la misma clase son agrupados en regiones distintas del espacio de 2 dimensiones.
#Esto es un indicativo de que el modelo está aprendiendo a representar los nodos de acuerdo a su clase. 
#Además, podemos ver cómo la pérdida disminuye a medida que el entrenamiento avanza.
#En este caso, la representación previa es muy parecida a la que se obtiene al final del entrenamiento. 
#Esto es un indicativo de que el modelo ha convergido a una solución estable.

