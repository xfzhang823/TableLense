"""
File: inference_pipeline.py
Author: Xiao-Fei Zhang
Date: last updated on 2025 Jan

Description: Pipeline for performing inference with the trained model.
"""

from pathlib import Path
import torch
import os
import logging
import pandas as pd
import tempfile
from src.inference.extract_inference_data import extract_and_save_inference_data
from nn_models.simple_nn import SimpleNN
from data_processing.post_inference_data_cleaning import clean_and_relabel_data
from inference.inference_utils import (
    load_model,
    load_embeddings_from_disk,
    save_embeddings_to_disk,
    generate_embeddings,
    classify_data,
)
from project_config import (
    PREPROCESSED_ALL_DATA_FILE,
    TRAINING_DATA_FILE,
    MODEL_PTH_FILE,
    TEST_DATA_PTH_FILE,
    INFERENCE_INPUT_DATA_FILE,
    INFERENCE_EMBEDDINGS_PKL_FILE,
    RAW_INFERENCE_OUTPUT_DATA_FILE,
    CLEANED_INFERENCE_OUTPUT_DATA_FILE,
    COMBINED_INFERENCE_OUTPUT_DATA_FILE,
)


def build_inference_pipeline(
    model_file: Path | str,
    test_data_file: Path | str,
    all_data_file: Path | str,
    training_data_file: Path | str,
    filtered_data_file: Path | str,
    inference_embedding: Path | str,
    raw_results_file: Path | str,
    cleaned_results_file: Path | str,
    combined_cleaned_results_file: Path | str,
):
    """
    Build and execute the inference pipeline.

    Steps:
    1. Filter out training data from the production dataset.
    2. Load and prepare the filtered data for embedding generation.
    3. Load or generate embeddings for the filtered data.
    4. Use the trained model to classify the filtered data.
    5. Save the classified data.
    6. Clean and relabel the classified data.
    7. Clean and relabel the training data.
    8. Combine the cleaned inference and training data.
    9. Save the combined dataset.
    10. Delete temporary files.
    """
    logging.info("Starting inference pipeline...")
    # Step 1: Filter out training data from the production dataset and save it
    extract_and_save_inference_data(
        training_inference_data_file=all_data_file,
        training_data_file=training_data_file,
        inference_data_file=filtered_data_file,
    )

    # Step 2: Load the filtered data (production excluding training data)
    df_unlabeled = pd.read_csv(filtered_data_file)
    df_unlabeled["original_index"] = df_unlabeled.index

    # Step 3: Load the trained model
    test_data_file = torch.load(test_data_file)
    input_dim = test_data_file["input_dim"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_nn = load_model(model_file, input_dim, device).to(device)
    label_map = {0: "table_data", 1: "title", 2: "metadata", 3: "header", 4: "empty"}
    logging.info("Model loaded and ready.")

    # Step 4: Generate or load embeddings
    if os.path.exists(inference_embedding):
        results = load_embeddings_from_disk(inference_embedding)
    else:
        results, embeddings, labels, original_indices, groups = generate_embeddings(
            df_unlabeled, batch_size=128
        )
        save_embeddings_to_disk(
            inference_embedding, embeddings, labels, original_indices, groups
        )

    # Step 5: Classify data
    processed_results = classify_data(
        results, model_nn, device, label_map, df_unlabeled
    )
    final_df = pd.concat(processed_results)
    final_df.to_csv(raw_results_file, index=False)
    logging.info("Raw classified data saved.")

    # Step 6: Clean, relabel, and save classified data
    clean_and_relabel_data(raw_results_file, cleaned_results_file)
    cleaned_inference_df = pd.read_csv(cleaned_results_file)

    # Step 8: Clean training data
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file_name = temp_file.name
        clean_and_relabel_data(training_data_file, temp_file_name)
    cleaned_training_df = pd.read_csv(temp_file_name)

    # Step 9: Combine cleaned datasets
    combined_df = pd.concat(
        [cleaned_inference_df, cleaned_training_df], axis=0, ignore_index=True
    )
    combined_df.to_csv(combined_cleaned_results_file, index=False)
    os.remove(temp_file_name)
    logging.info("Combined labeled dataset saved.")


def run_inference_pipeline():
    """
    Run the inference pipeline by loading configuration paths and calling the pipeline.
    """
    logging.info("Loading configuration paths for inference...")
    build_inference_pipeline(
        model_file=MODEL_PTH_FILE,
        test_data_file=TEST_DATA_PTH_FILE,
        all_data_file=PREPROCESSED_ALL_DATA_FILE,
        training_data_file=TRAINING_DATA_FILE,
        filtered_data_file=INFERENCE_INPUT_DATA_FILE,
        inference_embedding=INFERENCE_EMBEDDINGS_PKL_FILE,
        raw_results_file=RAW_INFERENCE_OUTPUT_DATA_FILE,
        cleaned_results_file=CLEANED_INFERENCE_OUTPUT_DATA_FILE,
        combined_cleaned_results_file=COMBINED_INFERENCE_OUTPUT_DATA_FILE,
    )
    logging.info("Inference pipeline completed successfully.")
