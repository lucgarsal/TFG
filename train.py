import os
from torch_geometric.datasets import Planetoid
import torch
import tqdm
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from torch_geometric.transforms import NormalizeFeatures
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import datetime
from torch_geometric.datasets import AttributedGraphDataset

from model import LinkPredictor
from data_treatment import preprocessing

######################################################################################
## Variables de configuración                                                       ##
######################################################################################
valid_combinations = [
    (False, False),  # caracteristicas nodos
    (True, False),   # embeddings
    (False, True)    # embeddings + caracteristicas nodos
]

gnn_types = ['GCN', 'GAT', 'GraphSAGE', 'VGAE']

######################################################################################
## Funciones Principales                                                            ##
######################################################################################
## Entrenamos el modelo de predicción de enlaces.
def train_link_predictor(data, model, optimizer, device, use_embeddings, use_both, writer, epochs=100, epoch_delay=0):
    model.train()
    data = data.to(device)
    
    for epoch in tqdm.tqdm(range(epochs), desc="Epoch Progress", leave=False):
        optimizer.zero_grad()
       
        pos_edge_index = sampled_edge_index
        neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1)).to(torch.long)

        pos_edge_index = pos_edge_index.long()
        neg_edge_index = neg_edge_index.long()  
       
        pos_out = model(data, pos_edge_index, use_embeddings, use_both)
        neg_out = model(data, neg_edge_index, use_embeddings, use_both)

        pos_loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        writer.add_scalar('train/loss', loss.item(), epoch+epoch_delay)
        writer.add_scalar('train/pos_loss', pos_loss.item(), epoch+epoch_delay)
        writer.add_scalar('train/neg_loss', neg_loss.item(), epoch+epoch_delay)

#Evaluamos el modelo de predicción de enlaces.
def predict_links(data, model, device, use_embeddings, use_both, threshold=0.5):
    model.eval()
    data = data.to(device)

    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1)).to(torch.long)
    pos_edge_index = pos_edge_index.long()
    neg_edge_index = neg_edge_index.long()

   
    pos_out = model(data, pos_edge_index, use_embeddings, use_both)
    neg_out = model(data, neg_edge_index, use_embeddings, use_both)
    
    pos_pred = (pos_out > threshold).cpu().numpy()
    neg_pred = (neg_out > threshold).cpu().numpy()
    
    return pos_pred, neg_pred, pos_edge_index, neg_edge_index

def evaluate_model(pos_pred, neg_pred, writer, evaluation_step=0):
    y_true = [1] * len(pos_pred) + [0] * len(neg_pred)
    y_pred = list(pos_pred) + list(neg_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    auc_roc = roc_auc_score(y_true, y_pred)
    
    writer.add_scalar('evaluation/accuracy', accuracy, evaluation_step)
    writer.add_scalar('evaluation/precision', precision, evaluation_step)
    writer.add_scalar('evaluation/recall', recall, evaluation_step)
    writer.add_scalar('evaluation/f1_score', f1, evaluation_step)
    writer.add_scalar('evaluation/auc_roc', auc_roc, evaluation_step)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC-ROC: {auc_roc:.4f}')
    
    return accuracy, precision, recall, f1, auc_roc

#Entrena y evalúa el modelo de predicción de enlaces.
#Cada checkpoint_epochs, guarda el modelo y evalúa su rendimiento.

def train_and_evaluate(data, model, optimizer, device, models_checkpoint_dir, log_dir, use_embeddings, use_both, writer, epochs_total=100, checkpoint_epochs=20):
    number_of_checkpoints = epochs_total // checkpoint_epochs

    for i in tqdm.tqdm(range(number_of_checkpoints), desc="Training Progress"):
        epochs = checkpoint_epochs
        train_link_predictor(data, model, optimizer, device, use_embeddings, use_both, epochs=epochs, epoch_delay=i*checkpoint_epochs, writer=writer)

        # Guardar el modelo en cada checkpoint
        torch.save(model.state_dict(), os.path.join(models_checkpoint_dir, f'model_epoch_{(i+1)*checkpoint_epochs}.pth'))

        # Evaluar el modelo
        pos_pred, neg_pred, pos_edge_index, neg_edge_index = predict_links(data, model, device, use_embeddings, use_both)
        accuracy, precision, recall, f1, auc_roc = evaluate_model(pos_pred, neg_pred, writer, evaluation_step=i)

        # Guardar las métricas en un archivo de texto
        with open(os.path.join(log_dir, 'metrics.txt'), 'a') as f:
            f.write(f'Epoch {(i+1)*checkpoint_epochs}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}\n')

def run_experiment(USE_EMBEDDINGS, USE_BOTH, GNN_TYPE):
    # Creamos carpeta con timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{dataset.name}PCA_embeddings-{USE_EMBEDDINGS}_mixed-{USE_BOTH}_gnn-{GNN_TYPE}_{timestamp}"
    log_dir = os.path.join("logs", model_name)
    models_checkpoint_dir = os.path.join(log_dir, "models")

    os.makedirs(models_checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Inicializamos TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Inicializamos el modelo de predicción de enlaces y el optimizador
    layer_sizes = [data.num_node_features * 2, 64, 32, 16] if isinstance(data.num_node_features, int) else data.num_node_features
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    model = LinkPredictor(in_channels=data.num_node_features, layer_sizes=layer_sizes, gnn_type=GNN_TYPE, use_both=USE_BOTH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Entrenamos el modelo y lo evaluamos
    train_and_evaluate(data, model, optimizer, device,models_checkpoint_dir,log_dir, use_embeddings, use_both, epochs_total=100, checkpoint_epochs=20, writer=writer)

    # Guardamos el modelo final
    torch.save(model.state_dict(), os.path.join(models_checkpoint_dir, 'final_model.pth'))

    writer.close()

######################################################################################
## Preprocesado y entrenamiento                                                     ##
######################################################################################
dataset = AttributedGraphDataset(root='data/Facebook', name='Facebook', transform=NormalizeFeatures())
#dataset = Planetoid(root='data/CiteSeer', name='CiteSeer', transform=NormalizeFeatures())
data= dataset[0]
use_pca=True
data = preprocessing(dataset, use_pca)

device='cpu'
graph = data
edge_list = graph.edge_index

#Lo movemos todos al mismo dispositivo
graph = graph.to(device)
edge_list = edge_list.to(device).long()

num_edges = edge_list.size(1)
num_sampled_edges = int(0.5 * num_edges)  # Seleccionamos el 50% de las aristas
sampled_indices = torch.randperm(num_edges, device=device)[:num_sampled_edges]
sampled_edge_index = edge_list[:, sampled_indices].to(device).long()

for use_embeddings, use_both in valid_combinations:
    for gnn_type in gnn_types:
        print(f"Ejecutando experimento con Embeddings={use_embeddings}, Mixto={use_both}, GNN={gnn_type}")
        run_experiment(use_embeddings, use_both, gnn_type)

 