from pykeen.triples import TriplesFactory
import pandas as pd
from pykeen.models import TransE
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
import numpy as np
from torch_geometric.datasets import Planetoid
import torch

# Cargar el dataset PubMed
dataset = Planetoid(root='./data', name='PubMed')
data = dataset[0]

# Extraer las aristas (edges) como triples (sujeto, objeto)
edge_index = data.edge_index.numpy()
subjects = edge_index[0]
objects = edge_index[1]

# Crear un DataFrame para los triples
triples_df = pd.DataFrame({
    'subject': subjects.astype(str),  # Convertir a string si es necesario
    'predicate': ['connected_to'] * len(subjects),  # Relación fija
    'object': objects.astype(str)
})

# Convertir a formato de PyKEEN
triples_factory = TriplesFactory.from_labeled_triples(
    triples=triples_df[['subject', 'predicate', 'object']].values
)

# Definir y entrenar el modelo
embedding_dim = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransE(triples_factory=triples_factory, embedding_dim=embedding_dim).to(device)
optimizer = Adam(model.parameters())  # Corregido

training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=triples_factory,
    optimizer=optimizer,
)

training_loop.train(
    triples_factory=triples_factory,
    num_epochs=50,
    batch_size=256,
)

# Obtener las representaciones de los nodos (embeddings)
entity_embeddings = model.entity_representations[0]().detach().cpu().numpy()

# Guardar los embeddings
np.save('./data/embeddings_TransE.npy', entity_embeddings)
