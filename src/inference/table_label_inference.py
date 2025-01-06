"""
File Name: table_label_inference.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Aug 1

The script performs inference using a trained model to label table data in a production environment.
It identifies and prints out the mismatched headers.

Steps:
1. Load the processed production data.
2. Instantiate and load the trained model.
3. Perform inference using the model and evaluate with classification metrics.
4. Identify and print misclassified 'header' samples.

The embedding process can take 30 min to a couple of hours depending on your GPU.
"""

# Dependencies
import os
import logging
import logging_config
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from utils.file_encoding_detector import detect_encoding
from read_csv_file import read_csv_file

from nn_models.simple_nn import SimpleNN


# Set up logger
logger = logging.getLogger(__name__)

# Check if CUDA is available (ensure that GPU will be used)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def load_model(model_path, input_dim, device):
    """
    Load the model and prepare it for inference.

    Args:
        model_path (str): Path to the model file.
        input_dim (int): Input dimension of the model.
        device (torch.device): Device to load the model on.

    Returns:
        SimpleNN: The loaded model.
    """
    hidden_dims = [128, 64, 32, 16]
    model_nn = SimpleNN(input_dim, hidden_dims).to(device)
    model_nn.load_state_dict(torch.load(model_path))
    model_nn.eval()
    logger.info("Model loaded and ready for inference.")
    return model_nn


def main():
    # File paths
    model_dir_path = r"C:\github\china stats yearbook RAG\outputs\models"
    model_path = os.path.join(model_dir_path, "simple_nn_model.pth")
    test_data_path = os.path.join(model_dir_path, "test_data.pth")

    # Production data location
    prod_data_path = r"C:\github\china stats yearbook RAG\data\inference\input\yearbook 2012 and 2022 english tables.csv"

    # Where to save data with predicted labels
    predicted_tbl_data_f_path = r"C:\github\china stats yearbook RAG\data\inference\output\yearbook 2012 and 2022 english tables predicted.csv"

    # Load the test_data model to get input_dim
    test_data = torch.load(test_data_path)
    input_dim = test_data["input_dim"]

    # Load the model
    model_nn = load_model(model_path, input_dim, device)

    # Load the production data
    df_prod = read_csv_file(prod_data_path)
    logger.info(f"Production data loaded with {len(df_prod)} rows.")

    # Track original indices
    df_prod["original_index"] = df_prod.index

    # Label map
    label_map = {0: "table_data", 1: "title", 2: "metadata", 3: "header", 4: "empty"}

    # Batch the groups and process
    grouped = df_prod.groupby("group")
    results = []
    batch_size = 500  # Adjust batch size based on memory and performance
    small_batch_size = 100  # Setting this to 90 because the largest table has 87 rows
    current_batch = []
    current_batch_size = 0

    # Iterate through the groups (tables) to append them one by one in each batch
    for group_name, group_data in tqdm(grouped, desc="Processing groups"):
        current_batch.append(group_data)
        current_batch_size += len(group_data)

        if current_batch_size >= batch_size:
            batch_df = pd.concat(current_batch)
            processed_batch_df = process_and_classify(
                batch_df, model_nn, label_map, device
            )
            results.append(processed_batch_df)
            current_batch = []
            current_batch_size = 0

    # Process any remaining data
    if current_batch:
        batch_df = pd.concat(current_batch)
        processed_batch_df = process_and_classify(batch_df, model_nn, label_map, device)
        results.append(processed_batch_df)

    # Concatenate all the results
    final_df = pd.concat(results)

    # Save the classified data
    final_df.to_csv(predicted_tbl_data_f_path, index=False)
    logger.info("Classified data saved.")
    print("Classification and saving completed.")


if __name__ == "__main__":
    main()
