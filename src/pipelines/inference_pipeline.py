"""
File: inference_pipeline.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Dec

Description: Pipeline for performing inference with the trained model.
"""

# Add the root directory to sys.path
import sys
from pathlib import Path
import os
import logging

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

import torch
from tqdm import tqdm  # Import tqdm for progress bars
import tempfile
import pandas as pd

from utils.read_csv_file import read_csv_file
from preprocessing.post_inference_data_cleaning import clean_and_relabel_data
from inference import run_filter_on_unlabeled_data

from nn_models.simple_nn import SimpleNN
from inference.inference_utils import (
    load_model,
    load_embeddings_from_disk,
    save_embeddings_to_disk,
    generate_embeddings,
    classify_data,
)

from project_config import (
    production_data_path,
    filtered_production_data_path,
    training_data_path,
    model_path,
    test_data_path,
    embeddings_inference_path,
    raw_inference_output_data_path,
    cleaned_inference_output_data_path,
    combined_output_data_path,
)

# from models.training_utils import dynamic_batch_processing, process_batch


# Build the pipeline
def build_inference_pipeline(
    model,
    test_data,
    all_data,
    training_data,
    filtered_data,
    inference_embedding,
    raw_results,
    cleaned_results,
    combined_cleaned_results,
):
    """
    Build and execute the inference pipeline.

    Args:
        model (str): Path to the model file
        test_data (str): Path to the test data file (from model output)
        all_data (str): Path to the production data
        training_data (str): Path to the training data
        filtered_data (str): Path to the filtered production data
        inference_embedding (str): Path to save or load inference embeddings
        raw_results (str): Path to save raw inference results
        cleaned_results (str): Path to save cleaned inference results
        combined_cleaned_results (str): Path to save the combined and cleaned results

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

    # Step 1: Filter out training data from the production dataset
    run_filter_on_unlabeled_data(all_data, training_data, filtered_data)

    # Step 2: Load the filtered data (production excluding training data)
    logging.info(f"Loading filtered data from {filtered_data}")
    df_unlabeled = pd.read_csv(filtered_data)
    logging.info(f"Filtered DataFrame loaded with shape: {df_unlabeled.shape}")

    # Track original indices by adding "original_index" column
    df_unlabeled["original_index"] = df_unlabeled.index

    # Step 3: Load and prepare the trained model
    test_data = torch.load(test_data)
    input_dim = test_data["input_dim"]  # input_dim = number of features
    model_nn = load_model(model_path, input_dim, device).to(
        device
    )  # instantiate neural network model
    label_map = {
        0: "table_data",
        1: "title",
        2: "metadata",
        3: "header",
        4: "empty",
    }  # labels need to be numerics
    logging.info("Model loaded and moved to the device.")

    # Step 4: Generate or load embeddings
    # (use the same batch processing utils functions from models module)

    # Check if embeddings file already exist
    if os.path.exists(inference_embedding):
        results = load_embeddings_from_disk(inference_embedding)
    else:
        results, embeddings, labels, original_indices, groups = generate_embeddings(
            df_unlabeled, batch_size=128
        )
        save_embeddings_to_disk(
            inference_embedding, embeddings, labels, original_indices, groups
        )

    # Step 5: Classify the data using the model
    processed_results = classify_data(
        results, model_nn, device, label_map, df_unlabeled
    )

    # Step 6: Save the classified data
    final_df = pd.concat(processed_results)
    final_df.to_csv(raw_results, index=False)
    logging.info("Classified data saved.")

    # Step 7: Clean, relabel, and save the classified data
    clean_and_relabel_data(raw_results, cleaned_results)
    cleaned_inference_df = pd.read_csv(cleaned_results)

    # Step 8: Clean the training data and store it in a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file_name = temp_file.name
        clean_and_relabel_data(training_data, temp_file_name)
    cleaned_training_df = pd.read_csv(temp_file_name)

    # Step 9: Combine the cleaned inference and training data
    combined_df = pd.concat(
        [cleaned_inference_df, cleaned_training_df], axis=0, ignore_index=True
    )
    combined_df.to_csv(combined_cleaned_results, index=False)

    # Step 10. Delete the temporary file
    os.remove(temp_file_name)
    logging.info("Inference pipeline completed successfully.")


def main():
    """Main script to orchestrate the inference process"""

    logging.info("Starting the inference pipeline...")

    # Set paths with config.py imported paths
    model = model_path
    test_data = test_data_path
    all_data = production_data_path  # all data
    training_data = (
        training_data_path  # training (including both train/test, already labeled)
    )
    filtered_data = (
        filtered_production_data_path  # production minus training data (not labeled)
    )
    embeddings = embeddings_inference_path  # embeddings for inference (production data)
    raw_results = raw_inference_output_data_path  # inference results - not cleaned yet
    cleaned_results = cleaned_inference_output_data_path  # inference results - cleaned
    combined_and_cleaned_results = (
        combined_output_data_path  # cleaned inference + cleaned training
    )

    # Check to see if all the input/output files are there
    if not all(
        os.path.exists(path) for path in [model, test_data, all_data, training_data]
    ):
        raise FileNotFoundError("One or more required files are missing.")

    build_inference_pipeline(
        model,
        test_data,
        all_data,
        training_data,
        filtered_data,
        embeddings,
        raw_results,
        cleaned_results,
        combined_and_cleaned_results,
    )

    logging.info("Inference pipeline completed successfully.")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Check for CUDA to ensure that GPU to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    main()
