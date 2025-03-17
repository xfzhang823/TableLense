"""
Filename: inference_pipeline_async_batched.py

Asynchronous inference pipeline with batched GPU processing.
Optimized for efficient loading, processing, and saving of inference data.
Includes execution time logging for performance monitoring.
"""

import asyncio
import logging
from pathlib import Path
import pickle
from typing import Union, List, Optional, Tuple
import time
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch import no_grad, Tensor

# User defined
from utils.validate_dataframe_columns import validate_dataframe_columns
from nn_models.training_utils import (
    dynamic_batch_processing_partial_cache,
    process_batch_for_embeddings,
)
from inference.inference_utils import (
    load_model,
    generate_embeddings,
    classify_data,
    update_embeddings_on_disk,
)
from inference.extract_inference_data import extract_and_save_inference_data
from data_processing.post_process_inference_data import (
    process_inference_training_results,
)
from nn_models.simple_nn import SimpleNN
from project_config import (
    TRAINING_INFERENCE_DATA_FILE,
    TRAINING_DATA_FILE,
    MODEL_PTH_FILE,
    TEST_DATA_PTH_FILE,
    INFERENCE_INPUT_DATA_FILE,
    INFERENCE_EMBEDDINGS_PKL_FILE,
    RAW_INFERENCE_OUTPUT_DATA_FILE,
    CLEANED_INFERENCE_OUTPUT_DATA_FILE,
    CLEANED_TRAINING_OUTPUT_DATA_FILE,
    COMBINED_CLEANED_OUTPUT_DATA_FILE,
)

# Setup basic logging configuration
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "text",
    "yearbook_source",
    "section",
    "group",
    "row_id",
    "is_empty",
    "is_title",
    "original_index",
]


async def async_load_csv(path: Path) -> pd.DataFrame:
    """
    Asynchronously loads a CSV file into a Pandas DataFrame.
    Uses asyncio to execute the blocking pandas.read_csv() in a separate thread.

    Args:
        path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    schedules the blocking synchronous task (pd.read_csv) to execute in
    the default thread pool.
    Meanwhile, the main coroutine (async_load_csv) continues asynchronously without blocking.

    * Why: pandas.read_csv(), .to_csv()) that don't have async equivalents built-in.
    """
    loop = asyncio.get_running_loop()
    logger.info(f"Loading data asynchronously from {path}")
    start_time = time.time()
    df = await loop.run_in_executor(None, pd.read_csv, path)
    elapsed_time = time.time() - start_time
    logger.info(f"Loaded {path} in {elapsed_time:.2f} seconds")
    return df


async def async_save_csv(df: pd.DataFrame, path: Path):
    """
    Asynchronously saves a Pandas DataFrame to a CSV file.
    Uses asyncio to execute the blocking pandas.to_csv() in a separate thread.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (Path): Path where the file should be saved.
    """
    loop = asyncio.get_running_loop()
    logger.info(f"Saving data asynchronously to {path}")
    start_time = time.time()

    # Wrap the call in a lambda so that keyword arguments are handled internally.
    await loop.run_in_executor(None, lambda: df.to_csv(path, index=False))

    elapsed_time = time.time() - start_time
    logger.info(f"Saved {path} in {elapsed_time:.2f} seconds")


def classify_inference_results(
    results: List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]],
    model: nn.Module,
    device: torch.device,
) -> pd.DataFrame:
    """
    Concatenate partial batches of embeddings from inference and run classification
    in sub-batches using the provided PyTorch model (the two-step process helps manage
    memory while ensuring the final output covers the entire dataset.)

    This function is designed to handle the output of a partial-caching pipeline
    where each batch in `results` contains:
      (batch_embeddings, None, batch_indices, batch_groups).

    Steps:
    1. Aggregate embeddings and original indices from all partial batches.
    2. Convert them to a single NumPy array each.
    3. Perform inference in sub-batches (to avoid GPU memory overflow) using the model.
    4. Return a DataFrame mapping each 'original_index' to its predicted label ID.

    Args:
        results (List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]]):
            A list of tuples from partial batch processing.
            Each tuple should have:
              - embeddings (np.ndarray): BERT-based embeddings + additional features,
                shape: (batch_size, embedding_dim).
              - labels (Optional[np.ndarray]): In inference mode, typically None.
              - indices (np.ndarray): 'original_index' values per row.
              - groups (np.ndarray): group identifiers (not used here).

        model (nn.Module):
            A trained PyTorch model (e.g., your SimpleNN).
            Must be loaded and in evaluation mode.

        device (torch.device):
            The computation device (CPU or GPU) on which the model will run.

    Returns:
        pd.DataFrame:
            A DataFrame with columns:
              - "original_index": int, row identifier for merging results downstream
              - "predicted_label": int, the predicted class index for each row

    Example:
        >>> partial_results = [
        ...     (emb_array_1, None, idx_array_1, grp_array_1),
        ...     (emb_array_2, None, idx_array_2, grp_array_2),
        ... ]
        >>> df_preds = classify_inference_results(partial_results, model, device)
        >>> print(df_preds.head())
         original_index  predicted_label
        0            100                1
        1            101                0
        2            102                3
        3            103                3
        ...

    Note:
        This function does not map label indices back to string labels.
        If needed, you can do that after merging predictions with a label_map.
    """

    # Ensure the model is in eval mode (no gradient updates).
    model.eval()

    # results: a list of (emb, lbl/None, idx, grp)
    # We only need embeddings & indices to run classification.
    all_embeddings = []
    all_indices = []
    for emb, _, idx, _ in results:
        all_embeddings.append(emb)
        all_indices.append(idx)

    # Concatenate partial arrays into final shape
    embeddings_arr = np.concatenate(all_embeddings, axis=0)
    indices_arr = np.concatenate(all_indices, axis=0)

    logger.info(
        f"Running classification on {embeddings_arr.shape[0]} total rows of embeddings."
    )

    # Perform inference in sub-batches to avoid GPU memory overflow
    predictions = []
    batch_size = 128
    n_samples = embeddings_arr.shape[0]

    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_emb = embeddings_arr[start:end]

        # Move embeddings to PyTorch tensor and device (GPU)
        batch_tensor = torch.tensor(batch_emb, dtype=torch.float32, device=device)

        with no_grad():
            outputs = model(batch_tensor)
            _, predicted = torch.max(outputs, dim=1)
        predictions.extend(predicted.cpu().numpy().tolist())

    # Build a DataFrame with "original_index" and "predicted_label"
    df_preds = pd.DataFrame(
        {"original_index": indices_arr, "predicted_label": predictions}
    )

    return df_preds


async def run_inference_pipeline_async_batched():
    """
    Asynchronous batched inference pipeline.
    Loads data, processes embeddings, classifies data, cleans results,
    and saves the final cleaned dataset asynchronously.

    * Workflow:
    [inference_pipeline_async_batched.py pipeline]
        └─> run_inference_pipeline_async_batched()
            ├─> (Step 1) Setup device & load model
            ├─> (Step 2) Ensure inference data exists via extract_and_save_inference_data()
            ├─> (Step 3) async_load_csv(INFERENCE_INPUT_DATA_FILE)
            ├─> (Step 4) dynamic_batch_processing_partial_cache(
                            df_unlabeled,
                            process_batch_for_embeddings,
                            batch_size=128,
                            is_inference=True,
                            partial_cache_path=INFERENCE_EMBEDDINGS_PKL_FILE
                        )
            │       └─> For each group:
            │               └─> process_batch_for_embeddings(...)
            ├─> (Step 5) classify_inference_results(...)
            │       └─> Concatenate partial embeddings and run model in sub-batches
            ├─> (Step 6) Merge predictions with original data
            ├─> (Step 7) async_save_csv() → RAW_INFERENCE_OUTPUT_DATA_FILE
            ├─> (Step 8) clean_and_relabel_data(...) → CLEANED_INFERENCE_OUTPUT_DATA_FILE
            ├─> (Step 9) clean training data & merge → COMBINED_CLEANED_OUTPUT_DATA_FILE
            └─> (Step 10) Done! (Final output saved)

    Summary:
    - The pipeline loads unlabeled data asynchronously.
    - It processes embeddings in batches using partial caching to skip already processed groups.
    - Predicted labels are computed via batched inference.
    - Predictions are merged back with the original data and cleaned.
    - The final output is saved to disk, ensuring efficient recovery if interrupted.
    """
    try:
        logger.info("Starting inference pipeline...")

        start_time_pipeline = time.time()

        # Step 0: Check for existing output to avoid redundant processing
        if COMBINED_CLEANED_OUTPUT_DATA_FILE.exists():
            logger.info("Inference output already exists. Skipping pipeline.")
            return

        # 1. Setup device & load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        test_data = torch.load(
            TEST_DATA_PTH_FILE, weights_only=False
        )  # Use weights_only=False only if you trust the checkpoint source,
        # as it may execute arbitrary code during unpickling.

        input_dim = test_data["input_dim"]

        model = SimpleNN(input_dim).to(device)
        model.load_state_dict(torch.load(MODEL_PTH_FILE, map_location=device))
        model.eval()
        logger.info("Loaded model & ready for inference.")

        # Step 2: Generate the unlabeled data if not already (filter out training data)
        if not INFERENCE_INPUT_DATA_FILE.exists():
            extract_and_save_inference_data(
                training_inference_data_file=TRAINING_INFERENCE_DATA_FILE,
                training_data_file=TRAINING_DATA_FILE,
                inference_data_file=INFERENCE_INPUT_DATA_FILE,
            )

        # Step 3: Load input inference data
        df_unlabeled = await async_load_csv(INFERENCE_INPUT_DATA_FILE)

        logger.info(f"Unlabeled dataframe to be inferred: {df_unlabeled.head(10)}")

        # Validate df columns
        validate_dataframe_columns(
            df=df_unlabeled, required_cols=REQUIRED_COLUMNS, strict=True
        )
        logger.debug(
            f"Inference input DataFrame validated. Columns: {set(df_unlabeled)}"
        )

        # Step 4: Use partial caching batch function to get the inference results
        #    results -> List[Tuple[embeddings, None, original_indices, groups]]
        partial_results = dynamic_batch_processing_partial_cache(
            df_unlabeled,
            process_batch_for_embeddings,  # from training utils
            batch_size=64,  # desired chunk size
            is_inference=True,  # no label used
            partial_cache_path=INFERENCE_EMBEDDINGS_PKL_FILE,  # use the final
        )

        # Step 5: Classify partial results with the loaded model.
        predictions_df = classify_inference_results(partial_results, model, device)
        logger.info(f"Predictions sample:\n{predictions_df.head(5)}")

        # Step 6: Merge predictions onto df_unlabeled
        final_df = df_unlabeled.merge(predictions_df, on="original_index", how="left")
        logger.info("Merged predictions with unlabeled data.")

        # 7. Save raw inference output (async)
        await async_save_csv(final_df, RAW_INFERENCE_OUTPUT_DATA_FILE)
        logger.info(f"Raw inference results saved to {RAW_INFERENCE_OUTPUT_DATA_FILE}")

        # Step 8: Save raw inference output asynchronously
        await async_save_csv(final_df, RAW_INFERENCE_OUTPUT_DATA_FILE)
        logger.info("Inferred data saved successfully.")

        # Step 9: Clean and relabel inference output
        process_inference_training_results(
            input_file_path=RAW_INFERENCE_OUTPUT_DATA_FILE,
            output_file_path=CLEANED_INFERENCE_OUTPUT_DATA_FILE,
        )
        cleaned_inference_df = await async_load_csv(CLEANED_INFERENCE_OUTPUT_DATA_FILE)
        logger.info(
            f"Inference file cleaned, relabeled, and persisted to {CLEANED_INFERENCE_OUTPUT_DATA_FILE}"
        )
        logger.info(f"Inference output data size: {cleaned_inference_df.shape}")

        # Step 10: Clean and relabel training data
        process_inference_training_results(
            input_file_path=TRAINING_DATA_FILE,
            output_file_path=CLEANED_TRAINING_OUTPUT_DATA_FILE,
        )
        cleaned_training_df = await async_load_csv(CLEANED_TRAINING_OUTPUT_DATA_FILE)
        logger.info(
            f"Training file cleaned, relabeled, and persisted to {CLEANED_INFERENCE_OUTPUT_DATA_FILE}"
        )
        logger.info(f"Training data size: {cleaned_training_df.shape}")

        # Step 11: Combine cleaned inference and training data, and persist to disk
        combined_df = pd.concat(
            [cleaned_inference_df, cleaned_training_df], axis=0, ignore_index=True
        )
        await async_save_csv(
            combined_df, COMBINED_CLEANED_OUTPUT_DATA_FILE
        )  # will insert NaN for empty values
        logger.info("Combined labeled dataset saved.")
        logger.info(f"Combined processed DataFrame size: {combined_df.shape}")

        # Logging time
        elapsed_time_pipeline = time.time() - start_time_pipeline
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_pipeline))

        logger.info("Finished inference pipeline.")
        logger.info(f"Total inference pipeline runtime: {formatted_time}")

    except Exception as e:
        logger.error(f"Error in inference pipeline: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_inference_pipeline_async_batched())

# def run_inference(
#     embeddings: np.ndarray,
#     original_indices: np.ndarray,
#     model: nn.Module,
#     df_unlabed: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Perform inference using precomputed embeddings and return a DataFrame that maps
#     each original index to its predicted label.

#     Args:
#         - embeddings (np.ndarray): Precomputed embeddings.
#         - original_indices (np.ndarray): Array of unique original indices corresponding
#         to each embedding.
#         - model (torch.nn.Module): Trained neural network model.

#     Returns:
#         pd.DataFrame: A DataFrame with columns 'original_index' and 'predicted_label'.
#     """
#     logger.info("Starting inference on precomputed embeddings...")
#     start_time = time.time()

#     # Perform inference using the precomputed embeddings.
#     # Here, classify_data is assumed to return a list of predicted labels.
#     predictions = classify_data(results=embeddings, model_nn=model)

#     # Build a DataFrame with original indices and their corresponding predicted labels.
#     results_df = pd.DataFrame(
#         {"original_index": original_indices, "predicted_label": predictions}
#     )

#     elapsed_time = time.time() - start_time
#     logger.info(f"Inference completed in {elapsed_time:.2f} seconds")

#     return results_df

# def process_embedding_batches_by_group_and_checkpoint(
#     df: pd.DataFrame, target_batch_size: int, pickle_file: Union[str, Path]
# ) -> None:
#     """
#     Process the inference DataFrame in batches based on groups, ensuring that
#     each table (or group) is not split between batches. Accumulate groups until
#     the total row count reaches the target_batch_size, then generate embeddings for
#     that batch and update the checkpoint pickle file incrementally.

#     Args:
#         df (pd.DataFrame): The DataFrame containing inference data. Must include a "group" column.
#         target_batch_size (int): Approximate number of rows per batch.
#         pickle_file (Union[str, Path]): Path to the pickle file where embeddings are stored.

#     Returns:
#         None
#     """
#     if isinstance(pickle_file, str):
#         pickle_file = Path(pickle_file)

#     groups = list(df.groupby("group"))
#     logger.info(f"Found {len(groups)} groups for processing.")

#     batch_groups = []
#     batch_row_count = 0

#     for group_name, group_df in groups:
#         group_size = group_df.shape[0]
#         # If adding this group exceeds the target batch size and we have a current batch,
#         # process the batch first.
#         if batch_row_count + group_size > target_batch_size and batch_groups:
#             # Concatenate all groups in the current batch
#             batch_df = pd.concat(batch_groups)
#             _, batch_embeddings, batch_labels, batch_indices, batch_groups_arr = (
#                 generate_embeddings(batch_df, batch_size=batch_df.shape[0])
#             )
#             update_embeddings_on_disk(
#                 pickle_file,
#                 batch_embeddings,
#                 batch_labels,
#                 batch_indices,
#                 batch_groups_arr,
#             )
#             logger.info(
#                 f"Processed a batch with {batch_row_count} rows and updated embeddings on disk."
#             )
#             # Reset current batch
#             batch_groups = []
#             batch_row_count = 0

#         # Add the current group to the batch
#         batch_groups.append(group_df)
#         batch_row_count += group_size

#     # Process any remaining groups in the batch
#     if batch_groups:
#         batch_df = pd.concat(batch_groups)
#         _, batch_embeddings, batch_labels, batch_indices, batch_groups_arr = (
#             generate_embeddings(batch_df, batch_size=batch_df.shape[0])
#         )
#         update_embeddings_on_disk(
#             pickle_file, batch_embeddings, batch_labels, batch_indices, batch_groups_arr
#         )
#         logger.info(
#             f"Processed final batch with {batch_row_count} rows and updated embeddings on disk."
#         )
