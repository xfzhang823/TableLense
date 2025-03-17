import pickle
from pathlib import Path
from project_config import INFERENCE_EMBEDDINGS_PKL_FILE, TRAINING_EMBEDDINGS_PKL_FILE

# Define paths to your cached embeddings files
training_embeddings_path = TRAINING_EMBEDDINGS_PKL_FILE
inference_embeddings_path = Path(
    r"C:\github\table_lense\pipeline_data\nn_models\temp_backup\inference_embeddings.pkl"
)

# Load training embeddings (expected to be 4-tuple)
with training_embeddings_path.open("rb") as f:
    training_data = pickle.load(f)
if isinstance(training_data, tuple) and len(training_data) == 4:
    training_embeddings, training_labels, training_indices, training_groups = (
        training_data
    )
else:
    raise ValueError("Training embeddings file format is unexpected.")

# Load inference embeddings (might be a 2-tuple or a 4-tuple)
with inference_embeddings_path.open("rb") as f:
    inference_data = pickle.load(f)

if isinstance(inference_data, tuple):
    if len(inference_data) == 4:
        inference_embeddings, inference_labels, inference_indices, inference_groups = (
            inference_data
        )
    elif len(inference_data) == 2:
        inference_embeddings, inference_indices = inference_data
        inference_labels, inference_groups = None, None
    else:
        raise ValueError(
            "Inference embeddings file has an unexpected number of elements."
        )
else:
    raise ValueError("Inference embeddings file format is unexpected.")

print("Training embeddings shape:", training_embeddings.shape)
print("Inference embeddings shape:", inference_embeddings.shape)
