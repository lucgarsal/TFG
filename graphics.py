import os
import re
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict
import itertools

LOG_DIR = "logs"
OUTPUT_DIR = "graphics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS = {
    "train/loss": "loss",
    "evaluation/accuracy": "accuracy",
    "evaluation/recall": "recall",
    "evaluation/f1_score": "f1_score",
    "evaluation/auc_roc": "auc_roc"
}

# Estructura: results[dataset][metric][(use_emb, use_mix)][gnn_type] = (steps, values)
results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

folder_pattern = re.compile(
    r'(?P<dataset>\w+)PCA_embeddings-(?P<use_emb>True|False)_mixed-(?P<use_mix>True|False)_gnn-(?P<gnn_type>\w+)_\d+'
)

all_gnn_types = set()
all_input_combinations = set()

for root, dirs, files in os.walk(LOG_DIR):
    for file in files:
        if file.startswith("events.out.tfevents"):
            path = os.path.join(root, file)
            match = folder_pattern.search(root)
            if not match:
                continue

            dataset = match.group("dataset")
            use_emb = match.group("use_emb") == "True"
            use_mix = match.group("use_mix") == "True"
            gnn_type = match.group("gnn_type")
            all_gnn_types.add(gnn_type)
            all_input_combinations.add((use_emb, use_mix))

            try:
                ea = EventAccumulator(path)
                ea.Reload()
                tags = ea.Tags().get("scalars", [])

                for tb_tag, metric_name in METRICS.items():
                    if tb_tag not in tags:
                        continue
                    events = ea.Scalars(tb_tag)
                    steps = [e.step for e in events]
                    values = [e.value for e in events]
                    results[dataset][metric_name][(use_emb, use_mix)][gnn_type] = (steps, values)
            except Exception as e:
                print(f"Error procesando {path}: {e}")

# Asignar colores por combinación de entrada
color_map = {
    (False, False): 'red',
    (True, False): 'green',
    (False, True): 'blue'
}

# Estilos únicos por tipo de GNN
gnn_linestyles = {
    'GAT': '-',          # línea continua
    'GCN': '--',         # guiones
    'GraphSAGE': '-.',   # guion-punto
    'VGAE': ':'          # punteada
}

# Expresión regular para extraer el tipo de GNN (sin la fecha)
gnn_type_pattern = re.compile(r'^[A-Za-z]+')

# Graficar por dataset y métrica
for dataset, metrics_dict in results.items():
    for metric, input_combo_dict in metrics_dict.items():
        plt.figure(figsize=(12, 7))

        for (use_emb, use_mix), gnn_data in input_combo_dict.items():
            color = color_map.get((use_emb, use_mix), 'black')
            for gnn_type, (steps, values) in gnn_data.items():
                gnn_type_clean = gnn_type_pattern.match(gnn_type).group(0)
                linestyle = gnn_linestyles.get(gnn_type_clean, '-')

                if metric!= "loss":
                    epochs = [step * 20 for step in steps]
                    plt.plot(epochs, values, color=color, linestyle=linestyle)
                else:
                    plt.plot(steps, values, color=color, linestyle=linestyle)

        plt.title(f"{metric.upper()} - {dataset}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.upper())
        plt.grid(True)

        # Ajuste del eje X según la métrica
        if metric != "loss":
            plt.xticks(range(0, 81, 20))

        # Leyenda de colores (tipo de entrada)
        input_legend = [
            Line2D([0], [0], color='red', lw=2, label='Emb: False, Mix: False'),
            Line2D([0], [0], color='green', lw=2, label='Emb: True, Mix: False'),
            Line2D([0], [0], color='blue', lw=2, label='Emb: False, Mix: True')
        ]

        # Leyenda de estilos de línea (tipo de GNN)
        gnn_legend = [
            Line2D([0], [0], color='black', linestyle='-', label='GAT'),
            Line2D([0], [0], color='black', linestyle='--', label='GCN'),
            Line2D([0], [0], color='black', linestyle='-.', label='GraphSAGE'),
            Line2D([0], [0], color='black', linestyle=':', label='VGAE')
        ]

        if metric == "loss":
            legend1 = plt.legend(handles=input_legend, title='Entrada', loc='upper right')
            legend2 = plt.legend(handles=gnn_legend, title='GNN', loc='upper center')
        else:
            legend1 = plt.legend(handles=input_legend, title='Entrada', loc='lower right')
            legend2 = plt.legend(handles=gnn_legend, title='GNN', loc='lower center')

        plt.gca().add_artist(legend1)

        filename = f"{dataset}_{metric}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()