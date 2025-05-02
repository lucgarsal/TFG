from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_file = "logs/CiteSeerPCA_embeddings-False_mixed-False_gnn-GAT_20250414_165630/events.out.tfevents.1744642590.DESKTOP-JI4A59V.7656.1"  # ajusta con tu ruta local

ea = EventAccumulator(event_file)
ea.Reload()

print("Scalars disponibles:")
print(ea.Tags().get("scalars", []))
