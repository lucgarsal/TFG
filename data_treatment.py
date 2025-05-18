
import torch
from torch_geometric.utils import remove_isolated_nodes, remove_self_loops
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_undirected
from sklearn.decomposition import PCA
import numpy as np
from torch_geometric.utils import degree
from compute_embeddings1 import generate_embeddings

######################################################################################
## Preprocesamiento de los datos                                                    ##
######################################################################################
def preprocessing(dataset, use_pca):
    data = dataset[0]
    # Quitamos los nodos aislados si existiesen. NO FUNCIONA REMOVE_ISOLATED_NODES
    if data.has_isolated_nodes():
        #Calculamos el grado de cada nodo
        deg = degree(data.edge_index[0], data.num_nodes)
        non_isolated_mask = deg > 0
        non_isolated_nodes = non_isolated_mask.nonzero(as_tuple=True)[0]

        #Mapeamos índices antiguos a nuevos
        old_to_new = -torch.ones(data.num_nodes, dtype=torch.long)
        old_to_new[non_isolated_nodes] = torch.arange(non_isolated_nodes.size(0))

        #Reindexamos aristas y filtramos las que conectan nodos no aislados
        mask_edge = non_isolated_mask[data.edge_index[0]] & non_isolated_mask[data.edge_index[1]]
        new_edge_index = data.edge_index[:, mask_edge]
        new_edge_index = old_to_new[new_edge_index]

        #Filtramos datos del grafo
        data.x = data.x[non_isolated_nodes]
        data.y = data.y[non_isolated_nodes]
        data.edge_index = new_edge_index
        data.num_nodes = non_isolated_nodes.size(0)

        #Filtramos máscaras si existen
        for mask_name in ['train_mask', 'val_mask', 'test_mask']:
            mask = getattr(data, mask_name, None)
            if mask is not None:
                setattr(data, mask_name, mask[non_isolated_nodes])
    
    # La normalización se realiza automáticamente con el transform=NormalizeFeatures() al cargar el dataset
    # Si las características numéricas tuviesen rangos muy distintos, también podemos normalizarlas usando un escalador
    """
    scaler = StandardScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x.numpy()), dtype=torch.float32)
    """
    # Si el grafo es dirigido, lo convertimos a no dirigido
    if not data.is_undirected():
        data.edge_index = to_undirected(data.edge_index)

    #Si tiene self loops, los eliminamos
    if data.has_self_loops():
        data.edge_index, _ = remove_self_loops(data.edge_index)
    
    # Si la dimensionalidad es muy grande, podemos aplicar PCA para reducirla. Lo recibiremos como parámetro.
    if use_pca:
        pca = PCA(n_components=0.9)  # Reducimos a un 90% de la varianza explicada
        data.x = torch.tensor(pca.fit_transform(data.x.numpy()), dtype=torch.float32)
        dataset.data.x = data.x

        generate_embeddings(data, dataset.name)
        data.embeddings = torch.tensor(np.load(f'data/embeddings_TransE_{dataset.name}PCA.npy'), dtype=torch.float32).to(data.x.device)
    else:
        generate_embeddings(data, dataset.name)
        data.embeddings = torch.tensor(np.load(f'data/embeddings_TransE_{dataset.name}.npy'), dtype=torch.float32).to(data.x.device)

    # Si el dataset ya tiene las máscaras de train, val y test, las asignamos. Si no, generamos unas aleatorias.
    if hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask'):
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
    else:
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)

        train_size = int(num_nodes * 0.8)
        val_size = int(num_nodes * 0.1)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

    # Asignamos las máscaras al objeto data
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    

    return data
    

# Mostramos características del conjunto de datos
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
print(f"Estructura de los datos: {data}")
"""
