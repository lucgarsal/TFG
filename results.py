import os
import re
from collections import defaultdict

ruta_modelos = 'logs'  

os.makedirs('resultados', exist_ok=True)

# Diccionario para agrupar resultados por dataset
datos_por_dataset = defaultdict(list)

# Expresión regular para extraer datos del nombre de la carpeta
patron_nombre = re.compile(r'(?P<dataset>.*?)PCA_embeddings-(?P<use_embeddings>True|False)_both-(?P<use_both>True|False)_gnn-(?P<gnn_type>.*?)_')

# Expresión regular para extraer métricas
patron_metrics = re.compile(
    r'Epoch (\d+), Accuracy: ([\d\.]+), Precision: ([\d\.]+), Recall: ([\d\.]+), F1 Score: ([\d\.]+), AUC-ROC: ([\d\.]+)'
)

# Recorremos todas las carpetas de modelos
for carpeta in os.listdir(ruta_modelos):
    carpeta_path = os.path.join(ruta_modelos, carpeta)
    if os.path.isdir(carpeta_path):
        match = patron_nombre.match(carpeta)
        if match:
            dataset = match.group('dataset')
            use_embeddings = match.group('use_embeddings')
            use_both = match.group('use_both')
            gnn_type = match.group('gnn_type')

            metrics_path = os.path.join(carpeta_path, 'metrics.txt')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    for linea in f:
                        match_metrics = patron_metrics.search(linea)
                        if match_metrics:
                            epoch = int(match_metrics.group(1))
                            accuracy = float(match_metrics.group(2))
                            precision = float(match_metrics.group(3))
                            recall = float(match_metrics.group(4))
                            f1_score = float(match_metrics.group(5))
                            auc_roc = float(match_metrics.group(6))

                            datos_por_dataset[dataset].append((
                                use_embeddings, use_both, gnn_type, epoch,
                                accuracy, precision, recall, f1_score, auc_roc
                            ))

# Ahora generamos y guardamos las tablas en LaTeX
for dataset, datos in datos_por_dataset.items():
    datos.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    tabla_latex = "\\begin{table}[H]\n"
    tabla_latex += "\\centering\n"
    tabla_latex += "\\small\n"
    tabla_latex += "\\begin{tabular}{ccccccccc}\n"
    tabla_latex += "\\toprule\n"
    tabla_latex += "Embeddings & Both & GNN Type & Epoch & Accuracy & Precision & Recall & F1 Score & AUC-ROC \\\\\n"
    tabla_latex += "\\midrule\n"

    for fila in datos:
        fila_str = " & ".join(str(x) for x in fila)
        tabla_latex += f"{fila_str} \\\\\n"

    tabla_latex += "\\bottomrule\n"
    tabla_latex += "\\end{tabular}\n"
    tabla_latex += f"\\caption{{Resultados de los experimentos sobre el conjunto de datos {dataset} tras aplicar PCA.}}\n"
    tabla_latex += f"\\label{{tab:{dataset.lower()}-pca-results}}\n"
    tabla_latex += "\\end{table}\n"

    # Guardar en resultados/{dataset}_pca.txt
    output_path = os.path.join('resultados', f'{dataset.lower()}_pca.txt')
    with open(output_path, 'w') as f:
        f.write(tabla_latex)

print("Tablas LaTeX generadas y guardadas en la carpeta 'resultados/'.")
