"""
File Name: evaluate_model.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Aug 1

This script evaluates the performance of the trained model and prints out the mismatched headers.

Steps:
1. Load the test data and the original data.
2. Instantiate and load the trained model.
3. Evaluate the model using classification metrics.
4. Identify and print misclassified 'header' samples.
"""

import os
import sys
import logging
import pandas as pd
from pprint import pprint
import torch
from sklearn.metrics import classification_report
import torch.nn as nn
from nn_models.simple_nn import SimpleNN  # Use absolute import

# Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def load_model_and_test_data(model_dir_path, original_data_path):
    """
    Load test data, model, and original data.

    Args:
        - model_dir_path (str): Path to the model directory.
        - original_data_path (str): Path to the original data file (train/text dataset).

    Returns:
        tuple: Contains loaded test data, original data, and text data.
    """
    model_path = os.path.join(model_dir_path, "simple_nn_model.pth")
    test_data_path = os.path.join(model_dir_path, "test_data.pth")

    # Load the original data to get the text field
    original_df = pd.read_excel(original_data_path)
    text_data = original_df["text"].tolist()

    test_data = torch.load(test_data_path)
    X_test = torch.tensor(test_data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(test_data["y_test"], dtype=torch.long)
    input_dim = test_data["input_dim"]  # input_dim is the number of features
    test_original_indices = test_data["original_indices"]

    return model_path, X_test, y_test, input_dim, test_original_indices, text_data


def evaluate_model(model_path, X_test, y_test, input_dim):
    """
    Evaluate the model.

    Args:
        model_path (str): Path to the model file.
        X_test (torch.Tensor): Test data features.
        y_test (torch.Tensor): Test data labels.
        input_dim (int): Input dimension of the model.

    Returns:
        tuple: Contains predicted labels and classification report.
    """
    hidden_dims = [128, 64, 32, 16]
    model_nn = SimpleNN(input_dim, hidden_dims)

    # Load the saved state dictionary into the model
    model_nn.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model_nn.eval()

    # Evaluate the model
    with torch.no_grad():
        outputs = model_nn(X_test)
        _, predicted = torch.max(outputs, 1)
        report = classification_report(
            y_test,
            predicted,
            target_names=["table_data", "title", "metadata", "header", "empty"],
        )

    return predicted, report


def print_misclassified_headers(y_test, predicted, test_original_indices, text_data):
    """
    Print misclassified 'header' samples.

    Args:
        y_test (torch.Tensor): Test data labels.
        predicted (torch.Tensor): Predicted labels.
        test_original_indices (list): Original indices of the test data.
        text_data (list): Original text data.
    """
    header_class_index = 3  # Index of the 'header' class in the target_names list
    misclassified_indices = (y_test == header_class_index) & (y_test != predicted)

    misclassified_samples = test_original_indices[misclassified_indices]
    predicted_labels = predicted[misclassified_indices]

    print("\nMisclassified 'header' samples:")
    for i in range(len(misclassified_samples)):
        sample_index = misclassified_samples[i]
        misclassified_info = {
            "Sample Index": sample_index,
            "Text": text_data[sample_index],
            "True Label": "header",
            "Predicted Label": ["table_data", "title", "metadata", "header", "empty"][
                predicted_labels[i]
            ],
        }
        pprint(misclassified_info)


def main():
    # Training data path
    training_data_path = r"C:\github\china stats yearbook RAG\data\training\training data 2024 Jul 31 wo gap.xlsx"

    # Set needed file paths for eval
    model_dir_path = r"C:\github\china stats yearbook RAG\outputs\models"
    original_data_path = training_data_path
    model_path, X_test, y_test, input_dim, test_original_indices, text_data = (
        load_model_and_test_data(model_dir_path, original_data_path)
    )

    # Evaluate the model
    predicted, report = evaluate_model(model_path, X_test, y_test, input_dim)
    print(report)

    # Print misclassified headers
    print_misclassified_headers(y_test, predicted, test_original_indices, text_data)


if __name__ == "__main__":
    main()


# Optional: Add in the loop to correct and update mismatched results manually
# # Manually verify if the predicted label is actually correct
# is_correct = input("Is the predicted label correct? (yes/no): ").strip().lower()
# if is_correct == "yes":
#     # Update the original DataFrame with the correct label
#     original_df.loc[sample_index, "label"] = "header"
#     print(f"Corrected label for sample {sample_index}")

# Save the corrected DataFrame back to the Excel file
# original_df.to_excel(original_data_path, index=False)
