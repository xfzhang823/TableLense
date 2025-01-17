"""
File Name: inference_utilities.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Aug 16

Utility functions for processing data batches and performing inference using a trained model.
"""

import pickle
import logging
import numpy as np
import torch
from tqdm import tqdm
from nn_models.simple_nn import SimpleNN
from nn_models.training_utils import (
    dynamic_batch_processing,
    process_batch_for_embeddings,
)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, input_dim, device):
    """
    Load a neural network model from a saved file, move it to the specified device,
    and prepare it for inference.

    Args:
        - model_path (str): Path to the file containing the saved model state
        (usually a .pth or .pt file).

        - input_dim (int): The number of input features for the model,
        which determines the size of the input layer.

        - device (torch.device): The device on which to load the model
        (e.g., 'cpu' or 'cuda').

    Returns:
        SimpleNN: The loaded and initialized neural network model, ready for inference.
    """
    hidden_dims = [128, 64, 32, 16]  # Example hidden layer dimensions
    model_nn = SimpleNN(input_dim, hidden_dims).to(
        device
    )  # Initialize and move model to device
    model_nn.load_state_dict(
        torch.load(model_path, map_location=device)
    )  # Load the saved model state
    model_nn.eval()  # Set the model to evaluation mode
    logging.info("Model loaded and ready for inference.")  # Log model loading status
    return model_nn  # Return the prepared model


def load_embeddings_from_disk(file_path):
    """Load embeddings, labels, original indices, and groups from disk."""
    logging.info("Loading embeddings from disk...")
    with open(file_path, "rb") as f:
        embeddings, labels, original_indices, groups = pickle.load(f)

    num_batches = len(original_indices)
    if num_batches > 0:
        batch_size = len(embeddings) // num_batches
    else:
        raise ValueError(
            "Original indices data is empty, unable to determine batch size."
        )

    # Reconstruct results from the loaded data
    results = [
        (
            embeddings[i : i + batch_size],
            labels[i : i + batch_size] if labels is not None else None,
            original_indices[i : i + batch_size],
            groups[i : i + batch_size],
        )
        for i in range(0, len(embeddings), batch_size)
    ]

    return results


def save_embeddings_to_disk(file_path, embeddings, labels, original_indices, groups):
    """Save embeddings, labels, original indices, and groups to disk."""
    with open(file_path, "wb") as f:
        pickle.dump((embeddings, labels, original_indices, groups), f)


def generate_embeddings(df, batch_size):
    """Generate embeddings using dynamic batch processing."""
    logging.info("Generating embeddings...")
    results = dynamic_batch_processing(
        df,
        process_batch_for_embeddings,
        batch_size=batch_size,
        is_inference=True,
    )

    embeddings = np.concatenate([r[0] for r in results], axis=0)
    labels = (
        np.concatenate([r[1] for r in results], axis=0)
        if results[0][1] is not None
        else None
    )
    original_indices = np.concatenate([r[2] for r in results], axis=0)
    groups = np.concatenate([r[3] for r in results], axis=0)

    return results, embeddings, labels, original_indices, groups


def classify_data(results, model_nn, device, label_map, df_unlabeled):
    """Classify data using the model and the generated embeddings."""
    processed_results = []
    for batch_embeddings, _, batch_indices, _ in tqdm(
        results, desc="Processing batches"
    ):
        embeddings = torch.tensor(batch_embeddings, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model_nn(embeddings)
            _, predicted = torch.max(outputs, 1)
        predicted_labels = [label_map[pred.item()] for pred in predicted]
        df_batch = df_unlabeled.iloc[batch_indices]
        df_batch["predicted_label"] = predicted_labels
        processed_results.append(df_batch)

    return processed_results
