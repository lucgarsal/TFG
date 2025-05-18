# TFG Lucía García Salido

## Graph Neural Networks

Este Trabajo Fin de Grado explora el uso de Redes Neuronales de Grafos (GNN) y técnicas de embeddings para tareas de predicción de aristas en diferentes conjuntos de datos de grafos.

### Objetivos

1. Predecir si dos nodos están conectados usando únicamente sus características.
2. Predecir si dos nodos están conectados usando embeddings generados.
3. Predecir si dos nodos están conectados combinando embeddings y características.
4. Utilizar capas neuronales para la clasificación de enlaces.

Las arquitecturas GNN empleadas son GCN, GAT, VGAE y GraphSAGE.
---

## Estructura del proyecto
```
.
├── compute_embeddings1.py      # Generación de embeddings para los nodos
├── data_treatment.py          # Preprocesamiento y tratamiento de datos
├── graphics.py                # Generación de gráficas
├── model.py                   # Definición y entrenamiento de modelos GNN
├── results.py                 # Generación de tablas Latex
├── train.py                   # Script principal de entrenamiento
├── data/                      # Datasets y embeddings generados
├── graphics/                  # Gráficas generadas
├── logs/                      # Logs de entrenamiento y experimentos
├── resultados/                # Resultados y tablas en formato LaTeX
└── README.md
```

## Uso

Para lanzar un entrenamiento, en primer lugar debemos definir el dataset sobre el que se aplica. Para ello, modificamos la línea en la que se realiza esta acción, presente en la parte de preprocesado y entrenamiento del script train.py.

Tras definir el conjunto de datos, procedemos a la ejecución del archivo (mediante Run o escribiendo por consola "python train.py"). Observamos cómo se aplica el pipeline de manera secuencial, comenzando por data_treatment.py que a su vez llama a compute_embeddings1.py para generar los embeddings estructurales sobre el conjunto de datos especificado.

A continuación, comenzará el entrenamiento y evaluación de los modelos de manera secuencial. La evolución de las épocas, así como el entrenamiento que está siendo llevado a cabo podrán observarse por consola.



## Resultados
Una vez finalizado el entrenamiento, observamos que se habrán generado 12 directorios en la carpeta logs correspondientes a cada una de las variantes del modelo. Para abrir Tensorboard, escribimos por consola "tensorboard --logdir=logs". Es recomendable abrir la pestaña scalars y marcar la casilla "Ignore outliers in chart scaling".

El scripts graphics.py también debe ejecutarse posteriormente al entrenamiento. Si no se quisiera desplegar tensorboard, graphics.py es la mejor alternativa para observar la evolución de las métricas. No obstante, puesto que extrae la información de los logs de tensorboard, sí es necesario tenerlo instalado. 

Las gráficas de resultados en formato imagen se encuentran en [`graphics/`](graphics/).

Respecto a results.py, fue realizado exclusivamente para automatizar las tablas en formato Latex de la memoria. Sin embargo, es interesante observar cómo se extrae esta información y se transforma al formato elegido.

Las tablas de resultados en formato LaTeX se encuentran en [`resultados/`](resultados/), por ejemplo: [`resultados/wiki_pca.txt`](resultados/wiki_pca.txt).


## Contacto

Lucía García Salido  
Universidad de Sevilla
Correo: [lucgarsal@alum.us.es]




