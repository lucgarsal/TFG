from pykeen.triples import TriplesFactory
import pandas as pd
from pykeen.models import TransE
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
import numpy as np
from torch_geometric.datasets import Planetoid

def generate_embeddings(dataset):

    # Extract the edges (triples) from the dataset
    edge_index = dataset[0].edge_index
    subjects = edge_index[0].numpy()
    objects = edge_index[1].numpy()

    # Create a DataFrame for triples
    triples_df = pd.DataFrame({
        'subject': subjects,
        'predicate': ['connected_to'] * len(subjects),
        'object': objects,
        'positive': [1] * len(subjects)
    })

    # Convertir los IDs de los nodos a string
    triples_df['subject'] = triples_df['subject'].astype(str)
    triples_df['object'] = triples_df['object'].astype(str)
    triples_df['predicate'] = triples_df['predicate'].astype(str)

    #triples_df = pd.read_csv('./data/train.txt', sep='\t', header=None, names=['subject', 'predicate', 'object', 'positive']).dropna()
    # Create a dataset from the triples
    triples_factory = TriplesFactory.from_labeled_triples(triples=triples_df.values)

    # Fit the model
    models = []

    #Hago que las dimensiones coincidan para poder concatenarlos después
    embedding_dim = dataset.num_features

    models.append(
        TransE(triples_factory=triples_factory,
                embedding_dim=embedding_dim).to('cuda:0')
    )
    model = models[0]
    optimizer = Adam(params=model.get_grad_params())

    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=triples_factory,
        optimizer=optimizer,
    )

    training_loop.train(
        triples_factory=triples_factory,
        num_epochs=50,
        batch_size=min(256, triples_factory.num_triples),
    )

    entity_embeddings = model.entity_representations[0]().detach().cpu().numpy()

    np.save(f'./data/embeddings_TransE_{dataset.name}PCA.npy', entity_embeddings)