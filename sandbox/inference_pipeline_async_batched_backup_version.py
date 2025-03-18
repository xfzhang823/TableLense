"""
Filename: inference_pipeline_async_batched.py

Asynchronous inference pipeline with batched GPU processing.
Optimized for efficient loading, processing, and saving of inference data.
Includes execution time logging for performance monitoring.
"""

import asyncio
import logging
from pathlib import Path
from typing import Union, List
import time
import pandas as pd
import numpy as np
import torch
import pickle

from inference.inference_utils import (
    load_model,
    generate_embeddings,
    classify_data,
    update_embeddings_on_disk,
)
from inference.extract_inference_data import extract_and_save_inference_data
from data_processing.post_inference_data_cleaning import clean_and_relabel_data
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    await loop.run_in_executor(None, df.to_csv, path, index=False)
    elapsed_time = time.time() - start_time
    logger.info(f"Saved {path} in {elapsed_time:.2f} seconds")


def process_embedding_batches_by_group_and_checkpoint(
    df: pd.DataFrame, target_batch_size: int, pickle_file: Union[str, Path]
) -> None:
    """
    Process the inference DataFrame in batches based on groups, ensuring that
    each table (or group) is not split between batches. Accumulate groups until
    the total row count reaches the target_batch_size, then generate embeddings for
    that batch and update the checkpoint pickle file incrementally.

    Args:
        df (pd.DataFrame): The DataFrame containing inference data. Must include a "group" column.
        target_batch_size (int): Approximate number of rows per batch.
        pickle_file (Union[str, Path]): Path to the pickle file where embeddings are stored.

    Returns:
        None
    """
    if isinstance(pickle_file, str):
        pickle_file = Path(pickle_file)

    groups = list(df.groupby("group"))
    logger.info(f"Found {len(groups)} groups for processing.")

    batch_groups = []
    batch_row_count = 0

    for group_name, group_df in groups:
        group_size = group_df.shape[0]
        # If adding this group exceeds the target batch size and we have a current batch,
        # process the batch first.
        if batch_row_count + group_size > target_batch_size and batch_groups:
            # Concatenate all groups in the current batch
            batch_df = pd.concat(batch_groups)
            _, batch_embeddings, batch_labels, batch_indices, batch_groups_arr = (
                generate_embeddings(batch_df, batch_size=batch_df.shape[0])
            )
            update_embeddings_on_disk(
                pickle_file,
                batch_embeddings,
                batch_labels,
                batch_indices,
                batch_groups_arr,
            )
            logger.info(
                f"Processed a batch with {batch_row_count} rows and updated embeddings on disk."
            )
            # Reset current batch
            batch_groups = []
            batch_row_count = 0

        # Add the current group to the batch
        batch_groups.append(group_df)
        batch_row_count += group_size

    # Process any remaining groups in the batch
    if batch_groups:
        batch_df = pd.concat(batch_groups)
        _, batch_embeddings, batch_labels, batch_indices, batch_groups_arr = (
            generate_embeddings(batch_df, batch_size=batch_df.shape[0])
        )
        update_embeddings_on_disk(
            pickle_file, batch_embeddings, batch_labels, batch_indices, batch_groups_arr
        )
        logger.info(
            f"Processed final batch with {batch_row_count} rows and updated embeddings on disk."
        )


def run_inference(
    embeddings: np.ndarray,
    original_indices: np.ndarray,
    model: torch.nn.Module,
    df_unlabed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform inference using precomputed embeddings and return a DataFrame that maps
    each original index to its predicted label.

    Args:
        - embeddings (np.ndarray): Precomputed embeddings.
        - original_indices (np.ndarray): Array of unique original indices corresponding
        to each embedding.
        - model (torch.nn.Module): Trained neural network model.

    Returns:
        pd.DataFrame: A DataFrame with columns 'original_index' and 'predicted_label'.
    """
    logger.info("Starting inference on precomputed embeddings...")
    start_time = time.time()

    # Perform inference using the precomputed embeddings.
    # Here, classify_data is assumed to return a list of predicted labels.
    predictions = classify_data(results=embeddings, model_nn=model)

    # Build a DataFrame with original indices and their corresponding predicted labels.
    results_df = pd.DataFrame(
        {"original_index": original_indices, "predicted_label": predictions}
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Inference completed in {elapsed_time:.2f} seconds")

    return results_df


async def run_inference_pipeline_async_batched():
    """
    Asynchronous batched inference pipeline.
    Loads data, processes embeddings, classifies data, cleans results,
    and saves the final cleaned dataset asynchronously.
    """
    try:
        start_time_pipeline = time.time()

        # Step 0: Check for existing output to avoid redundant processing
        if COMBINED_CLEANED_OUTPUT_DATA_FILE.exists():
            logger.info("Inference output already exists. Skipping pipeline.")
            return

        # Step 1: Set the device to GPU if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Step 2: Load pre-trained model
        test_data = torch.load(TEST_DATA_PTH_FILE)
        nn_model = load_model(MODEL_PTH_FILE, test_data["input_dim"], device).to(device)
        label_map = {
            0: "table_data",
            1: "title",
            2: "metadata",
            3: "header",
            4: "empty",
        }
        logging.info("Model loaded and ready.")

        # Step 3: Filter out training data from the production dataset and save it
        if not INFERENCE_INPUT_DATA_FILE.exists():
            extract_and_save_inference_data(
                training_inference_data_file=TRAINING_INFERENCE_DATA_FILE,
                training_data_file=TRAINING_DATA_FILE,
                inference_data_file=INFERENCE_INPUT_DATA_FILE,
            )

        # Step 4: Load input inference data
        df_unlabeled = await async_load_csv(INFERENCE_INPUT_DATA_FILE)

        logger.info(f"Unlabeled dataframe to be inferred: {df_unlabeled.head(10)}")

        # Step 5: Incrementally generate or update embeddings.
        if INFERENCE_EMBEDDINGS_PKL_FILE.exists():
            logger.info(
                "Inference embeddings already exist. Skipping embedding generation."
            )
            with open(INFERENCE_EMBEDDINGS_PKL_FILE, "rb") as f:
                saved_embeddings, saved_labels, saved_indices, saved_groups = (
                    pickle.load(f)
                )

            # Determine which records are missing (in unlabelled but not in saved inference data)
            # * (use original indices as unique IDs to filter)
            saved_index_set = set(saved_indices)
            # df_unlabeled["original_index"] = df_unlabeled.index
            missing_df = df_unlabeled[
                ~df_unlabeled["original_index"].isin(saved_index_set)
            ]
            if not missing_df.empty:
                start_time = time.time()
                logger.info(
                    f"Found {missing_df.shape[0]} new rows missing from embeddings. Updating checkpoint..."
                )

                # Process the missing rows in groups (to avoid splitting a group)
                process_embedding_batches_by_group_and_checkpoint(
                    df=missing_df,
                    target_batch_size=128,
                    pickle_file=INFERENCE_EMBEDDINGS_PKL_FILE,
                )

                elapsed_time = time.time() - start_time
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                logger.info(f"Updated embeddings in {formatted_time}")

            else:
                logger.info("No missing records. The checkpoint is complete.")

        else:
            start_time = time.time()
            logger.info(
                "Inference embeddings file does not exist. Generating embeddings from scratch..."
            )

            # Process in batches (for example, 128 rows per batch) and update the pickle file.
            process_embedding_batches_by_group_and_checkpoint(
                df=df_unlabeled,
                target_batch_size=128,
                pickle_file=INFERENCE_EMBEDDINGS_PKL_FILE,
            )

            elapsed_time = time.time() - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            logger.info(f"Generated embeddings in {formatted_time}.")
            # torch.save(embeddings, INFERENCE_EMBEDDINGS_PKL_FILE)

        # Step 6: Load the final embeddings (or use them directly)
        if INFERENCE_EMBEDDINGS_PKL_FILE.exists():
            logger.info("Loading final embeddings from checkpoint file...")
            with open(INFERENCE_EMBEDDINGS_PKL_FILE, "rb") as f:
                embeddings, _, original_indices, _ = pickle.load(f)
        else:
            raise ValueError("Embeddings were not generated successfully.")

        # Step 7: Perform inference using the precomputed embeddings.
        predictions_df = run_inference(
            embeddings=embeddings,
            original_indices=original_indices,
            model=nn_model,
            df_unlabed=df_unlabeled,
        )
        logger.info(f"Predicted labels: \n{predictions_df.head(5)}")

        # Step 8: Merge predictions with the original DataFrame based on the 'original_index'
        df_unlabeled["original_index"] = (
            df_unlabeled.index
        )  # Ensure df has the same identifier
        final_df = df_unlabeled.merge(predictions_df, on="original_index", how="left")
        logger.info(f"Raw inferred data: \n{final_df.head(5)}")

        # Step 8: Save raw inference output asynchronously
        await async_save_csv(final_df, RAW_INFERENCE_OUTPUT_DATA_FILE)
        logger.info("Inferred data saved successfully.")

        # Step 9: Clean and relabel inference output
        clean_and_relabel_data(
            input_path=RAW_INFERENCE_OUTPUT_DATA_FILE,
            output_path=CLEANED_INFERENCE_OUTPUT_DATA_FILE,
        )
        cleaned_inference_df = await async_load_csv(CLEANED_INFERENCE_OUTPUT_DATA_FILE)

        # Step 10: Clean and relabel training data
        clean_and_relabel_data(
            input_path=TRAINING_DATA_FILE,
            output_path=CLEANED_TRAINING_OUTPUT_DATA_FILE,
        )
        cleaned_training_df = await async_load_csv(CLEANED_TRAINING_OUTPUT_DATA_FILE)

        # Step 11: Combine cleaned inference and training data, and persist to disk
        combined_df = pd.concat(
            [cleaned_inference_df, cleaned_training_df], axis=0, ignore_index=True
        )
        await async_save_csv(combined_df, COMBINED_CLEANED_OUTPUT_DATA_FILE)
        logger.info("Combined labeled dataset saved.")

        elapsed_time_pipeline = time.time() - start_time_pipeline
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_pipeline))
        logger.info(f"Total inference pipeline runtime: {formatted_time}")

    except Exception as e:
        logger.error(f"Error in inference pipeline: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_inference_pipeline_async_batched())
