"""
File Name: training_utils.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Aug 1

Utility functions for processing data batches, training a neural network model, 
and applying L2 regularization and early stopping.

Functions:
- process_batch(batch_df): 
Process a batch of data to generate embeddings, extract labels, indices, and group labels.

- dynamic_batch_processing(df, process_batch, batch_size=100): 
Process data in batches while preserving the integrity of groups.

- train_model(model, criterion, optimizer, 
X_train, y_train, X_test, y_test, num_epochs=20, batch_size=32, patience=3, 
model_path="model.pth", test_data_path="test_data.pth", indices_path="train_test_indices.pth"): 
Train the neural network model with L2 Regularization and Early Stopping.
"""

import sys
from pathlib import Path
import logging
import pickle
import os
from typing import Callable, Tuple, Union
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import GroupShuffleSplit

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from utils.read_csv_file import read_csv_file
from utils.read_exce_file import read_excel_file
import logging_config
from project_config import CLASSES

# logger
logger = logging.getLogger(__name__)


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_data(file_path: Union[Path, str]) -> pd.DataFrame:
    """
    Load data from an Excel or CSV file.

    Args:
        file_path (str): Path to the data file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    # Ensure file_path is Path obj
    file_path = Path(file_path)

    # Determine the file extension
    file_extension = file_path.suffix

    # Load the file based on its extension
    if file_extension == ".csv":
        df = read_csv_file(file_path)
    elif file_extension in [".xls", ".xlsx", ".xlsm"]:
        df = read_excel_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    return df


# Function to split data into train, test, etc.
def split_data(
    df: pd.DataFrame,
    combined_embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: NDArray[np.int64],
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Splits the data into training and testing sets while ensuring that samples from
    the same group are not represented in both sets. The function uses GroupShuffleSplit
    to perform the split.

    Args:
        - df (pd.DataFrame): The DataFrame containing the original data.
        Must include a column "original_index".
        - combined_embeddings (np.ndarray): The combined feature embeddings generated
        from the text data.
        - labels (np.ndarray): The labels for the data, corresponding to
        the combined embeddings.
        - groups (np.ndarray): The group labels for the data, ensuring that samples from
        the same group are not split between training and testing sets.
        - test_size (float, optional): The proportion of the data to include in
        the test split
        (train/test split). Default is 0.2.
        - random_state (int, optional): The seed used by the random number generator
        for reproducibility.
        Default is 42.

    Returns:
        tuple: A tuple containing the following elements:
            - X_train (np.ndarray): Training set feature embeddings.
            - X_test (np.ndarray): Testing set feature embeddings.
            - y_train (np.ndarray): Training set labels.
            - y_test (np.ndarray): Testing set labels.
            - train_idx (np.ndarray): Indices of the training set.
            - test_idx (np.ndarray): Indices of the testing set.
            - test_original_indices (np.ndarray):
            Original indices of the test data from the initial DataFrame.

    Example:
        >>> df = pd.read_excel("data.xlsx")
        >>> combined_embeddings = np.array([...])
        >>> labels = np.array([...])
        >>> groups = np.array([...])
        >>> X_train, X_test, y_train, y_test, train_idx, test_idx, test_original_indices = split_data(
        >>>     df, combined_embeddings, labels, groups, test_size=0.2, random_state=42
        >>> )
    """

    # Initializes a GroupShuffleSplit object (gss) to split the data into training and testing,
    # but ensure that each group is not "broken up"
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    # Uses the split method of gss to get indices for the training and testing sets
    # (train_idx and test_idx), use them to split:
    # - combined_embeddings into training and testing sets (X_train and X_test)
    # - the labels into training and testing sets (y_train and y_test)
    # - Extracts the original_index values for the test set (test_original_indices)
    # (to track the original positions of the test data)
    train_idx, test_idx = next(gss.split(combined_embeddings, labels, groups=groups))
    X_train, X_test = combined_embeddings[train_idx], combined_embeddings[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    test_original_indices = df["original_index"].values[test_idx]

    return X_train, X_test, y_train, y_test, train_idx, test_idx, test_original_indices


# Function to process data for a single batch
def process_batch_for_embeddings(
    batch_df: pd.DataFrame, is_inference: bool = False
):  # inference should not have "label" column, but training needs to have it!
    """
    Process a batch of data to generate embeddings, extract labels, indices, and group labels.

    Args:
        batch_df (pd.DataFrame): A DataFrame containing a batch of data to be processed.

    Returns:
        tuple: A tuple containing the following elements:
            - batch_embeddings (np.ndarray): Generated embeddings for the batch.
            - batch_labels (np.ndarray): Extracted labels for the batch.
            - batch_indices (np.ndarray): Original indices of the batch data.
            - batch_groups (np.ndarray): Group labels of the batch data.

    Tokenization and Embedding Mechanism (BERT Specific):
        *The BERT tokenizer processes input text into token IDs using WordPiece tokenization.

        Special tokens added:
        - '[CLS]': Serves as a placeholder for the entire sequence representation.
        - '[SEP]': Marks the end of a sentence or separates two segments.
        - The 'input_ids' are passed to the BERT model, which performs an embedding lookup
        using a learned matrix.
        - The embedding for each token combines token, position, and segment embeddings.
        *- Unlike static embeddings (e.g., Word2Vec), BERT embeddings pass through multiple
        *neural network layers, including self-attention and feed-forward layers,
        *to create **contextualized representations** based on the full sentence.
        - The '[CLS]' embedding (first token) can be used as a summary vector for
        classification tasks.
        - Alternatively, mean pooling can be applied across token embeddings to create
        a sequence representation (#!faster but not as accurate).

    """
    # Check for required columns
    required_columns = [
        "text",
        "row_id",
        "is_title",
        "is_empty",
        "original_index",
        "group",
    ]
    missing_columns = [col for col in required_columns if col not in batch_df.columns]
    if missing_columns:
        # Assuming batch_df has only one unique group, you can do:
        group_name = (
            batch_df["group"].iloc[0] if "group" in batch_df.columns else "Unknown"
        )
        logger.error(f"{missing_columns} missing in group {group_name}.")
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)

    # Tokenize and encode the text
    text_data = batch_df["text"].tolist()
    inputs = tokenizer(
        text_data, return_tensors="pt", padding=True, truncation=True
    ).to(
        device
    )  # tensor here != tensor in NN (only means returned format is a vector - dictionary)

    # Generate embeddings in batches to avoid memory issues
    batch_size = 32  # This batch here is "small" batch; adjust based on GPU memory
    embeddings = []

    # Chunk into batches (resource management)
    # The input tensor to BERT is a dictionary containing input_ids (tokenized text)
    # and attention_mask.
    for i in range(0, len(inputs["input_ids"]), batch_size):
        batch_inputs = {k: v[i : i + batch_size].to(device) for k, v in inputs.items()}
        #
        with torch.no_grad():
            outputs = model(**batch_inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)

    # Concatenate all the embeddings
    batch_embeddings = np.concatenate(embeddings, axis=0)

    # Add row feature (based on row_id column)
    row_feature = batch_df["row_id"].values
    row_feature = np.expand_dims(row_feature, axis=1)

    # Add title feature (based on is_title column)
    # binary encoding: need to convert yes/no to 1/0
    title_feature = batch_df["is_title"].map({"yes": 1, "no": 0}).values
    title_feature = np.expand_dims(title_feature, axis=1)

    # Add "is empty" feature (based on text content)
    # binary encoding: need to convert yes/no to 1/0
    empty_feature = batch_df["is_empty"].map({"yes": 1, "no": 0}).values
    empty_feature = np.expand_dims(empty_feature, axis=1)

    # Concatenate features
    batch_embeddings = np.concatenate(
        [batch_embeddings, row_feature, title_feature, empty_feature], axis=1
    )
    # np.expand_dims method adds an extra dimension to row_feature:
    # if row_feature initially has the shape (N,) where N is the number of samples,
    # np.expand_dims(row_feature, axis=1) will reshape it to (N, 1).
    # Vectorize the first and last column and concatenate
    # batch_df.iloc[:, 0].values & batch_df.iloc[:, -1].values
    # extracts the 1st and last column of batch_df and converts to NumPy arrays

    # If not in inference mode, extract labels
    # (convert label values from text to numeric values for training the model)
    if not is_inference:
        label_to_index = {label: idx for idx, label in enumerate(CLASSES)}
        batch_labels = batch_df["label"].map(label_to_index).values
    else:
        batch_labels = None  # no labels required for inference

    # Extract original indices and group labels
    batch_indices = batch_df["original_index"].values
    batch_groups = batch_df["group"].values

    return batch_embeddings, batch_labels, batch_indices, batch_groups


def dynamic_batch_processing(df, process_batch, batch_size=128, is_inference=False):
    """
    Process data in batches while preserving the integrity of groups
    (each group is a discrete table, which has a max of 80 rows).

    Args:
        - df (pd.DataFrame): The DataFrame containing the data to be processed.
        - process_batch (function): The function to process each batch.
        (When you are passing a reference to the function, which can then be called within
        the other function)
        - batch_size (int, optional): Maximum external batch_size determines how many rows
        (input texts) should be processed together (how many rows (text inputs) are grouped
        together before calling process_batch.) Default to 128.

    Returns:
        list: A list of tuples containing processed batch embeddings, labels, indices,
        and group labels.
    """
    grouped = df.groupby("group")  # Group the DataFrame by the 'group' column
    results = []
    current_batch = []
    current_batch_size = 0

    # Preserves the integrity of each group (table)
    # - ensure that batching does not cut a table in the middle

    # Iterate over each group in the grouped DataFrame
    for group_name, group_data in tqdm(grouped, desc="Processing groups"):
        logger.debug(f"Processing group: {group_name}")
        current_batch.append(group_data)  # Append the group data to the current batch
        current_batch_size += len(group_data)  # Update the current batch size

        # If the current batch size exceeds the specified batch size, process the batch
        # (the data gets "dumped" out to be "processed" & a "new" empty batch starts)
        if current_batch_size >= batch_size:
            batch_df = pd.concat(
                current_batch
            )  # Concatenate the current batch into a single DataFrame
            batch_embeddings, batch_labels, batch_indices, batch_groups = process_batch(
                batch_df, is_inference=is_inference
            )
            # Process the batch
            results.append(
                (batch_embeddings, batch_labels, batch_indices, batch_groups)
            )  # Append the results
            current_batch = []  # Reset the current batch
            current_batch_size = 0  # Reset the current batch size

    # Process any remaining data
    if current_batch:
        batch_df = pd.concat(
            current_batch
        )  # Concatenate the remaining data into a single DataFrame
        batch_embeddings, batch_labels, batch_indices, batch_groups = process_batch(
            batch_df, is_inference=is_inference
        )  # Process the batch
        results.append(
            (batch_embeddings, batch_labels, batch_indices, batch_groups)
        )  # Append the results

    return results  # Return the results


def generate_embeddings(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate embeddings from a DataFrame by processing batches of data.

    This function processes the input DataFrame using dynamic batching and extracts
    embeddings, labels, original indices, and groups associated with each row of the data.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data to process.
            The DataFrame should include columns such as:
            - 'text': Text data for embedding generation.
            - 'row_id': Row identifiers.
            - 'is_title': Flags indicating if a row is a title.
            - 'is_empty': Flags indicating if a row is empty.
            - 'label': Class labels for supervised training.
            - 'group': Group identifiers to preserve group-based batching.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing:
            - embeddings (np.ndarray): Feature embeddings generated from
            the text data.
            - labels (np.ndarray): Labels associated with the embeddings.
            - original_indices (np.ndarray): Original row indices from
            the input DataFrame.
            - groups (np.ndarray): Group identifiers for each data point.

    Raises:
        ValueError: If required columns are missing in the input DataFrame.

    Example:
        >>> df = pd.read_excel("data.xlsx")
        >>> embeddings, labels, original_indices, groups = generate_embeddings(df)
        >>> print(embeddings.shape)  # (num_rows, embedding_dim)
    """
    logger = logging.getLogger(__name__)

    # Log the start of the process
    logger.info("Starting embedding generation...")

    # Ensure required columns exist
    required_columns = ["text", "row_id", "is_title", "is_empty", "label", "group"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Process batches and generate embeddings
    results = dynamic_batch_processing(df, process_batch_for_embeddings)

    # Concatenate batch results into single arrays
    embeddings = np.concatenate([r[0] for r in results], axis=0)
    labels = np.concatenate([r[1] for r in results], axis=0)
    original_indices = np.concatenate([r[2] for r in results], axis=0)
    groups = np.concatenate([r[3] for r in results], axis=0)

    # Log the completion of the process
    logger.info("Embedding generation completed successfully.")

    return embeddings, labels, original_indices, groups


# Function to generate embeddings or load from disk
def load_or_generate_embeddings(
    data_file: Path, embeddings_file: Path, generate_embeddings_func: Callable
):
    """
    Load embeddings from disk if available, otherwise generate embeddings and save them.

    Args:
        training_data_path (str): Path to the training data file.
        embeddings_save_path (str): Path to save or load the embeddings.
        generate_embeddings_func (callable): Function to generate embeddings from data.

    Returns:
        tuple: A tuple containing embeddings, labels, original indices, and groups.
    """
    if embeddings_file.exists():
        logger.info("Loading embeddings from disk...")
        with open(embeddings_file, "rb") as f:
            embeddings, labels, original_indices, groups = pickle.load(f)
    else:
        logger.info("Loading data from Excel file...")
        df = load_data(data_file)
        embeddings, labels, original_indices, groups = generate_embeddings_func(df)
        with open(embeddings_file, "wb") as f:
            pickle.dump((embeddings, labels, original_indices, groups), f)

    return embeddings, labels, original_indices, groups


def train_model(
    model,
    criterion,
    optimizer,
    X_train,
    y_train,
    X_test,
    y_test,
    num_epochs=20,
    batch_size=32,
    patience=3,
    model_path="model.pth",
    test_data_path="test_data.pth",
    indices_path="train_test_indices.pth",
):
    """
    Train the neural network model with L2 Regularization and Early Stopping.

    Args:
        model (nn.Module): The neural network model to be trained.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        X_train (torch.Tensor): Training feature data.
        y_train (torch.Tensor): Training labels.
        X_test (torch.Tensor): Test feature data.
        y_test (torch.Tensor): Test labels.
        num_epochs (int, optional): The number of training epochs. Default is 20.
        batch_size (int, optional): The batch size for training. Default is 32.
        patience (int, optional): The patience for early stopping. Default is 3.
        model_path (str, optional): The path to save the best model. Default is 'model.pth'.
        test_data_path (str, optional): The path to save the test data. Default is 'test_data.pth'.
        indices_path (str, optional): The path to save the train and test indices. Default is 'train_test_indices.pth'.
    """
    best_val_loss = float("inf")
    no_improvement_count = 0

    logger.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        # Using tqdm for progress bar
        for i in tqdm(
            range(0, len(X_train), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            batch_X_train = X_train[i : i + batch_size]
            batch_y_train = y_train[i : i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X_train)
            loss = criterion(outputs, batch_y_train)
            l2_loss = model.get_l2_regularization_loss()
            total_loss = loss + l2_loss
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        # Calculate average training loss
        avg_train_loss = epoch_loss / len(X_train)

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_X_test = X_test[i : i + batch_size]
                batch_y_test = y_test[i : i + batch_size]
                outputs = model(batch_X_test)
                loss = criterion(outputs, batch_y_test)
                l2_loss = model.get_l2_regularization_loss()
                total_loss = loss + l2_loss
                val_loss += total_loss.item()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(X_test)

        # Log the loss every epoch
        logger.info(
            f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_count = 0

            # Save model
            torch.save(model.state_dict(), model_path)  # Save the best model

            # Save test data
            torch.save(
                {
                    "X_test": X_test.cpu().numpy(),
                    "y_test": y_test.cpu().numpy(),
                    "input_dim": X_train.shape[1],
                    "original_indices": np.arange(
                        len(y_test)
                    ),  # Change this if you track indices differently
                },
                test_data_path,
            )

            # Save indices
            torch.save(
                {
                    "train_idx": np.arange(
                        len(y_train)
                    ),  # Change this if you track indices differently
                    "test_idx": np.arange(
                        len(y_test)
                    ),  # Change this if you track indices differently
                    "original_indices": np.arange(
                        len(y_test)
                    ),  # Change this if you track indices differently
                },
                indices_path,
            )
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                logger.info("Early stopping triggered")
                break

    logger.info("Training completed and model saved")
