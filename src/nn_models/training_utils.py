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
from typing import Callable, Optional, Tuple, Union, Set, List
from numpy.typing import NDArray
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import GroupShuffleSplit

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from utils.read_csv_file import read_csv_file
from utils.read_exce_file import read_excel_file
import logging_config
from project_config import CLASSES, INFERENCE_EMBEDDINGS_CACHE_PKL_FILE

# logger
logger = logging.getLogger(__name__)


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device:, {device}")

# Load tokenizer and model once, store globally
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
MODEL = BertModel.from_pretrained("bert-base-uncased").to(device)
MODEL.eval()  # Set model to evaluation mode


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
) -> Tuple[
    np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray
]:  # inference should not have "label" column, but training needs to have it!
    """
    Process a batch of data to generate embeddings, extract labels, indices,
    and group labels.

    This function tokenizes text data with BERT, creates embeddings
    (plus row features like `row_id`, `is_title`, `is_empty`),
    and optionally extracts labels if not in inference mode.

    Args:
        batch_df (pd.DataFrame):
            A DataFrame containing a single batch of data with columns:
            - 'text': The raw text to embed.
            - 'row_id': Positional indicator per table.
            - 'is_title': Whether the row is a title ("yes"/"no").
            - 'is_empty': Whether the row is empty ("yes"/"no").
            - 'original_index': The original row index in the dataset.
            - 'group': Group identifier for preserving table boundaries.
            - 'label': (Required if is_inference=False) The string label
              (e.g., "table_data", "title", etc.).

        is_inference (bool, optional):
            If True, labels are not extracted (batch_labels = None).
            Defaults to False.
            * Training has an extra "label" column.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
            A 4-element tuple containing:
            - batch_embeddings (np.ndarray):
            The combined BERT embeddings plus row, title, and empty indicators.
            Shape: (batch_size, 768 + 3).
            - batch_labels (Optional[np.ndarray]):
            Numeric labels for the batch if training, otherwise None for inference.
            - batch_indices (np.ndarray):
            The 'original_index' values for each row in the batch.
            - batch_groups (np.ndarray):
            The 'group' values for each row, indicating table grouping.

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

    # # Initialize the tokenizer and model
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # model = BertModel.from_pretrained("bert-base-uncased").to(device)

    # Tokenize and encode the text
    text_data = batch_df["text"].tolist()
    inputs = TOKENIZER(
        text_data,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(
        device
    )  # tensor here != tensor in NN (only means returned format is a vector - dictionary)

    # Generate embeddings in batches to avoid memory issues
    inner_batch_size = (
        32  # This batch here is "small" batch; adjust based on GPU memory
    )
    embeddings_chunks = []

    # Chunk into batches (resource management)
    # The input tensor to BERT is a dictionary containing input_ids (tokenized text)
    # and attention_mask.
    for i in range(0, len(inputs["input_ids"]), inner_batch_size):
        batch_inputs = {
            k: v[i : i + inner_batch_size].to(device) for k, v in inputs.items()
        }
        #
        with torch.no_grad():
            outputs = MODEL(**batch_inputs)
            # Mean pooling across tokens
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings_chunks.append(batch_embeddings)

    # Concatenate all the embeddings
    batch_embeddings = np.concatenate(embeddings_chunks, axis=0)

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

    # Extract labels if training mode
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


def dynamic_batch_processing(
    df: pd.DataFrame,
    process_batch: Callable[
        [pd.DataFrame, bool],
        Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray],
    ],
    batch_size: int = 128,
    is_inference: bool = False,
):
    """
    Process data in batches while preserving the integrity of groups
    (each group is a discrete table).

    The function groups rows by the 'group' column, then concatenates
    the data for each group into a batch. Once the batch size exceeds
    `batch_size`, it processes the batch using `process_batch`, clears it,
    and starts a new one. The last partial batch is also processed.

    Args:
        - df (pd.DataFrame):
            The DataFrame containing the data to be processed.
            Must have a 'group' column to batch by.
        *- process_batch (Callable[[pd.DataFrame, bool], Tuple[np.ndarray,
        * Optional[np.ndarray], np.ndarray, np.ndarray]]):
            A function that processes a single DataFrame batch and returns
            a 4-tuple:
              (embeddings, labels (or None), original_indices, groups).
            If `is_inference=True`, labels may be `None`.
        - batch_size (int, optional):
            Maximum batch size in terms of number of rows. Defaults to 128.
        - is_inference (bool, optional):
            Indicates whether this is inference mode. If True, labels may be None.
            Defaults to False.

    Returns:
        List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]]:
            A list where each element corresponds to a processed batch and
            contains:
              - np.ndarray: embeddings array for the batch.
              *- Optional[np.ndarray]: labels array (None if inference).
              - np.ndarray: original_indices for the rows in the batch.
              - np.ndarray: groups for the rows in the batch.

    Example:
        >>> # Example usage:
        >>> results = dynamic_batch_processing(
                        df, process_batch_for_embeddings, 128, is_inference=True
                        )
        >>> for emb, lbl, idx, grp in results:
        ... print(emb.shape, lbl, idx.shape, grp.shape)
    """

    grouped = df.groupby("group")  # Group the DataFrame by the 'group' column
    results = []
    current_batch = []
    current_batch_size = 0

    # Preserves the integrity of each group (table)
    # - ensure that batching does not cut a table in the middle

    # Iterate over each group in the grouped DataFrame
    for group_name, group_data in grouped:

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
            *- 'label': Class labels for supervised training.
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


def dynamic_batch_processing_partial_cache(
    df: pd.DataFrame,
    process_batch: Callable[
        [pd.DataFrame, bool],
        Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray],
    ],
    batch_size: int = 128,
    is_inference: bool = True,
    partial_cache_path: Path = INFERENCE_EMBEDDINGS_CACHE_PKL_FILE,
) -> List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]]:
    """
    Process data in batches while preserving group integrity AND caching partial
    results to disk. Useful for large inference jobs where you might crash or
    time out mid-process.

    - If partial_cache_path exists, loads prior results and skips groups
      that have been completed.
    - If new groups remain, processes them in "mega-batches" of size <= batch_size
      and saves partial results after each batch.

    Groups rows by the 'group' column, accumulates them in a batch, and once
    the batch size is reached (or exceeded), calls 'process_batch'. The result
    is appended to an internal 'results' list, which is also saved to
    'partial_cache_path' so you can resume if the job restarts.

    Args:
        - df (pd.DataFrame):
            The DataFrame containing data to be processed, must have 'group' col.
        - process_batch (Callable[[pd.DataFrame, bool], Tuple[np.ndarray,
        Optional[np.ndarray], np.ndarray, np.ndarray]]):
            A function that processes a single batch, returning
            (embeddings, labels_or_None, indices, groups).
        - batch_size (int, optional):
            The max size of each "mega-batch" before we dump out. Default 128.
        - is_inference (bool, optional):
            If True, labels are None. Default True for inference partial caching.
        - partial_cache_path: partial results are read/written from this file.
            Default to INFERENCE_EMBEDDINGS_CACHE_PKL_FILE.

    Returns:
        List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]]:
            A list of 4-tuples (embeddings, labels_or_None, indices, groups) for
            each processed batch.

    Note:
        1) Groups are never split across two batches. Each group is appended in
           full even if the group size < batch_size.
        2) If partial_cache_path is specified and exists, we skip groups already
           processed. On each processed batch, we save the updated 'results' plus
           a set of 'processed_groups'.
        3) Minimal changes from your original dynamic_batch_processing logic.

    Example:
        >>> results = dynamic_batch_processing_partial_cache(
        ...     df,
        ...     process_batch_for_embeddings,
        ...     batch_size=128,
        ...     is_inference=True,
        ...     partial_cache_path=Path("inference_partial_cache.pkl"),
        ... )
        >>> # If you crash, re-running will skip completed groups.
    """
    # If we have a cache, load partial progress
    if partial_cache_path.exists():
        logger.info(f"Loading partial inference cache from {partial_cache_path}")
        with partial_cache_path.open("rb") as f:
            cache_data = pickle.load(f)
        results = cache_data["results"]
        processed_groups: Set[str] = set(cache_data["processed_groups"])
        logger.info(
            f"Cache has {len(results)} processed batches, skipping {len(processed_groups)} groups."
        )
    else:
        logger.info("No partial cache found, starting fresh inference processing.")
        results = []
        processed_groups: Set[str] = set()

    grouped = df.groupby("group")
    current_batch = []
    current_batch_size = 0

    for group_name, group_data in grouped:
        # If the entire group has already been processed, skip it
        if group_name in processed_groups:
            logger.debug(f"Skipping group {group_name}, found in cache.")
            continue

        # Accumulate this group in the "mega-batch"
        current_batch.append(group_data)
        current_batch_size += len(group_data)

        if current_batch_size >= batch_size:
            # Process the batch
            batch_df = pd.concat(current_batch, ignore_index=True)
            emb, lbl, idx, grp = process_batch(batch_df, is_inference=is_inference)
            results.append((emb, lbl, idx, grp))

            # Mark these groups as processed
            done_groups = batch_df["group"].unique()
            for g in done_groups:
                processed_groups.add(g)

            # Save partial cache
            _save_partial_cache(results, processed_groups, partial_cache_path)

            # Reset accumulators
            current_batch = []
            current_batch_size = 0

    # Process any leftover groups in the final partial batch
    if current_batch:
        batch_df = pd.concat(current_batch, ignore_index=True)
        emb, lbl, idx, grp = process_batch(batch_df, is_inference=is_inference)
        results.append((emb, lbl, idx, grp))

        leftover_groups = batch_df["group"].unique()
        for g in leftover_groups:
            processed_groups.add(g)

        _save_partial_cache(results, processed_groups, partial_cache_path)

    return results


def _save_partial_cache(
    results: List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]],
    processed_groups: Set[str],
    partial_cache_path: Path,
) -> None:
    """Utility to write partial results to disk as a single pickle file."""
    logger.info(f"Saving partial inference results to {partial_cache_path}...")
    with partial_cache_path.open("wb") as f:
        pickle.dump({"results": results, "processed_groups": list(processed_groups)}, f)
        # ? how does it record what's cached already?
        # records the group identifiers so that if the script restarts,
        # it sees those group names in the cache and automatically skips them.


# Function to generate embeddings or load from disk
def load_or_generate_embeddings(
    data_file: Path,
    embeddings_file: Path,
    generate_embeddings_func: Callable[
        [pd.DataFrame], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings from a cache file if it exists; otherwise, generate them using
    a provided function and save the results to disk.

    This function is typically used in a training pipeline to avoid recomputing
    expensive text embeddings. If the embeddings file is found, the data is loaded
    directly from disk; otherwise, the `generate_embeddings_func` is called on
    the loaded dataframe, and the newly computed embeddings are saved for future runs.

    Args:
        - data_file (Path):
            Path to the input file (Excel or CSV) containing raw data.
        - embeddings_file (Path):
            Path to the pickle file where embeddings are cached or will be saved.
        - generate_embeddings_func (Callable[[pd.DataFrame],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
            A function that takes a pandas DataFrame and returns a four-element tuple:
            (embeddings, labels, original_indices, groups).

    * Workflow:
    load_or_generate_embeddings(
                    data_file=...,
                    embeddings_file=...,
                    generate_embeddings_func=generate_embeddings
                )
                    └─> if not cached -> generate_embeddings(...)
                        └─> dynamic_batch_processing(...)
                                └─> process_batch_for_embeddings(...)
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A tuple of:
                - embeddings (np.ndarray): Feature vectors for each row.
                *- labels (np.ndarray): Numeric labels for each row (or None if not training).
                - original_indices (np.ndarray): Original row indices in the dataset.
                - groups (np.ndarray): Group labels corresponding to each row.
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
        for i in range(0, len(X_train), batch_size):
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
