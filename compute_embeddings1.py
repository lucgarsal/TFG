import pandas as pd 
from pykeen.models import TransE
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory
import numpy as np

######################################################################################
## Generación de los embeddings                                                     ##
######################################################################################

def generate_embeddings(data, dataset_name):
    edge_index = data.edge_index
    subjects = edge_index[0].numpy()
    objects = edge_index[1].numpy()

    # Convertimos a string para usar como etiquetas en PyKEEN
    subjects = list(map(str, subjects))
    objects = list(map(str, objects))

    # Todos los nodos esperados como strings
    all_node_ids = set(map(str, range(data.num_nodes)))
    present_nodes = set(subjects + objects)
    missing_nodes = list(all_node_ids - present_nodes)

    # Triples self-loop sintéticos
    self_loop_subjects = np.array(missing_nodes)
    self_loop_objects = np.array(missing_nodes)
    self_loop_predicates = ['connected_to'] * len(missing_nodes)

    # Triples combinados
    all_subjects = np.concatenate([subjects, objects, self_loop_subjects])
    all_objects = np.concatenate([objects, subjects, self_loop_objects])
    all_predicates = ['connected_to'] * (len(subjects) + len(objects)) + self_loop_predicates

    # Creamos el DataFrame de triples
    triples_df = pd.DataFrame({
        'subject': all_subjects,
        'predicate': all_predicates,
        'object': all_objects
    })

    if triples_df.empty:
        raise ValueError("El DataFrame de triples está vacío. Verifica los datos de entrada.")

    # Creamos TriplesFactory (solo las tres columnas necesarias)
    triples_factory = TriplesFactory.from_labeled_triples(
        triples=triples_df[['subject', 'predicate', 'object']].values
    )

    if triples_factory.num_triples == 0:
        raise ValueError("No se han creado triples válidos para entrenamiento.")

    # Definimos el modelo
    embedding_dim = data.num_features or 50  # fallback si no hay features
    model = TransE(
        triples_factory=triples_factory,
        embedding_dim=embedding_dim
    ).to('cuda:0')

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

    # Obtenemos embeddings
    entity_embeddings = model.entity_representations[0]().detach().cpu().numpy()
    np.save(f'./data/embeddings_TransE_{dataset_name}PCA.npy', entity_embeddings)

