"""
File Name: inference_utilities.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Aug 16

Utility functions for processing data batches and performing inference using a trained model.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Any, Union, Optional
import pickle
import logging
import pandas as pd
import numpy as np
import torch
from nn_models.simple_nn import SimpleNN
from nn_models.training_utils import (
    dynamic_batch_processing,
    process_batch_for_embeddings,
)

logger = logging.getLogger(__name__)

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


def persist_embeddings_to_disk(file_path, embeddings, labels, original_indices, groups):
    """Save embeddings, labels, original indices, and groups to disk."""
    with open(file_path, "wb") as f:
        pickle.dump((embeddings, labels, original_indices, groups), f)


def update_embeddings_on_disk(
    file_path: Union[str, Path],
    new_embeddings: np.ndarray,
    new_labels: Optional[np.ndarray],
    new_indices: np.ndarray,
    new_groups: np.ndarray,
) -> None:
    """
    Load existing embeddings from disk (if any), check for format consistency and for duplicate records
    (based on original indices), append only the new results, and save back to disk.

    Args:
        file_path (Union[str, Path]): Path to the pickle file.
        new_embeddings (np.ndarray): New embeddings to add.
        new_labels (Optional[np.ndarray]): New labels to add.
        new_indices (np.ndarray): New original indices to add.
        new_groups (np.ndarray): New groups to add.

    Returns:
        None
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_path.exists():
        with open(file_path, "rb") as f:
            old_embeddings, old_labels, old_indices, old_groups = pickle.load(f)

        # Check if the embedding dimensions match
        if old_embeddings.ndim != 2 or new_embeddings.ndim != 2:
            logging.error("Embeddings must be 2-dimensional arrays.")
            return
        if old_embeddings.shape[1] != new_embeddings.shape[1]:
            logging.error(
                f"Embedding dimension mismatch: existing embeddings have shape {old_embeddings.shape}, "
                f"new embeddings have shape {new_embeddings.shape}."
            )
            return

        # Optionally check that labels are 1D arrays if they exist
        if old_labels is not None and new_labels is not None:
            if old_labels.ndim != 1 or new_labels.ndim != 1:
                logging.error("Labels must be 1-dimensional arrays.")
                return

        # Filter out records that already exist (based on original indices)
        # Using numpy's isin function for efficiency
        duplicate_mask = np.isin(new_indices, old_indices)
        if np.any(duplicate_mask):
            num_duplicates = np.sum(duplicate_mask)
            logging.info(f"Found {num_duplicates} duplicate records; skipping them.")
            # Keep only records where duplicate_mask is False (i.e. new records)
            new_embeddings = new_embeddings[~duplicate_mask]
            new_indices = new_indices[~duplicate_mask]
            new_groups = new_groups[~duplicate_mask]
            if new_labels is not None:
                new_labels = new_labels[~duplicate_mask]

        # If after filtering there is no new data, then exit early
        if new_embeddings.size == 0:
            logging.info("No new records to append. The data is already up-to-date.")
            return

        # Concatenate the old and new arrays
        updated_embeddings = np.concatenate([old_embeddings, new_embeddings], axis=0)
        if old_labels is not None and new_labels is not None:
            updated_labels = np.concatenate([old_labels, new_labels], axis=0)
        else:
            updated_labels = new_labels
        updated_indices = np.concatenate([old_indices, new_indices], axis=0)
        updated_groups = np.concatenate([old_groups, new_groups], axis=0)
    else:
        updated_embeddings = new_embeddings
        updated_labels = new_labels
        updated_indices = new_indices
        updated_groups = new_groups

    with open(file_path, "wb") as f:
        pickle.dump(
            (updated_embeddings, updated_labels, updated_indices, updated_groups), f
        )
    logging.info(f"Updated embeddings saved to {file_path}")


def generate_embeddings(df, batch_size):
    """
    Generate embeddings using dynamic batch processing.

    The function orchestrates the generation of embeddings by processing
    the DataFrame in batches. It acts as a high-level wrapper that:
    - Divides the data into manageable chunks.
    - Processes each chunk to extract embeddings and related metadata.
    - Combines the outputs from all batches into consolidated arrays that can be used
    downstream (e.g., for inference or further processing).

    The dynamic_batch_processing function splits the DataFrame into batches
    (while preserving group integrity) and applies the process_batch_for_embeddings
    function to each batch.

    The output, stored in the variable results, is a list where each element is
    a tuple of outputs from processing a batch.

    Returns:
        - results: The raw list of per-batch outputs.
        - embeddings: The combined NumPy array of embeddings.
        - labels: The combined labels array (or None if not applicable).
        - original_indices: The combined array of original indices.
        - groups: The combined array of group identifiers.
    """
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


def classify_data(
    results: List[Tuple[Any, Any, List[int], Any]],
    model_nn: torch.nn.Module,
    label_map: Dict[int, str],
    df_unlabeled: pd.DataFrame,
) -> List[pd.DataFrame]:
    """
    Classify data using the provided model and generated embeddings.

    This function processes a list of batch results where each element is a tuple containing:
      - batch_embeddings: a NumPy array with embeddings for the batch.
      - _ : placeholder for labels (unused in inference mode).
      - batch_indices: a list of original row indices corresponding to the data in the batch.
      - _ : placeholder for group information (unused in inference).

    For each batch, the function:
      1. Converts the batch embeddings to a torch.Tensor and moves it to the global device.
      2. Performs inference using the model (with gradients disabled) to get raw outputs.
      3. Determines the predicted class indices and maps them to label names using label_map.
      4. Selects the corresponding rows from df_unlabeled using the provided batch_indices.
      5. Adds a new column "predicted_label" to that subset DataFrame.
      6. Appends the resulting DataFrame to a list.

    Args:
        - results (List[Tuple[Any, Any, List[int], Any]]): List of batch tuples, each containing
            (batch_embeddings, unused_labels, batch_indices, unused_group_info).
        - model_nn (torch.nn.Module): The trained neural network model for inference.
        - label_map (Dict[int, str]): Dictionary mapping class indices to label names.
        - df_unlabeled (pd.DataFrame): Original DataFrame of unlabeled data.

    Returns:
        List[pd.DataFrame]: A list of DataFrames, one for each batch, with an added
            "predicted_label" column indicating the predicted class label.

    Note:
        This function uses the global variable `device` (e.g., torch.device("cuda") or
        torch.device("cpu"))
        to move the data for inference.
    """
    processed_results: List[pd.DataFrame] = []

    for batch_embeddings, _, batch_indices, _ in results:

        # Convert batch embeddings to a PyTorch tensor and move to the global device.
        embeddings_tensor = torch.tensor(batch_embeddings, dtype=torch.float32).to(
            device
        )

        # Perform model inference without tracking gradients.
        with torch.no_grad():
            outputs = model_nn(embeddings_tensor)
            _, predicted = torch.max(outputs, 1)

        # Map predicted indices to their corresponding label names.
        predicted_labels: List[str] = [label_map[pred.item()] for pred in predicted]

        # Select the corresponding rows from the original DataFrame using batch indices.
        # Using .copy() to ensure we work on a separate DataFrame and avoid SettingWithCopyWarning.
        df_batch = df_unlabeled.iloc[batch_indices].copy()

        # Add the predicted labels as a new column.
        df_batch["predicted_label"] = predicted_labels

        processed_results.append(df_batch)

    return processed_results
