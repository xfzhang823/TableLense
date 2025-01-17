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

# Standard libraries
from pathlib import Path
import os
import sys
import logging
from typing import List, Optional, Tuple, Union
import pandas as pd
from pprint import pprint
import torch
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn
import matplotlib.pyplot as plt
import altair as alt

# User defined libraries
from nn_models.simple_nn import SimpleNN  # Use absolute import
from utils.read_csv_file import read_csv_file
import logging_config
from project_config import CLASSES


# Logging
logger = logging.getLogger("__main__")


def load_test_data_and_text(
    test_data_file: Path, training_data_file: Path
) -> Tuple[torch.Tensor, torch.Tensor, int, List[int], List[str]]:
    """
    Load test data and original data using paths from project_config.

    Args:
        - test_data_path_file (Path): path to the test data file
        (expect TEST_DATA_PTH_FILE)
        - original_data_file (Path): path to the training data file
        (expect TRAINING_DATA_FILE)

    Returns:
        Tuple: Contains:
            - X_test (torch.Tensor): Test data features.
            - y_test (torch.Tensor): Test data labels.
            - input_dim (int): Number of input features.
            - test_original_indices (List[int]): Original indices of the test data.
            - text_data (List[str]): Original text data.
    """
    # Load the original data to get the text field
    if training_data_file.suffix in [".xls", ".xlsx", ".xlsm"]:
        original_df = read_csv_file(training_data_file, engine="openpyxl")
    elif training_data_file.suffix == ".csv":
        original_df = read_csv_file(training_data_file)
    else:
        raise ValueError(f"Unsupported file format: {training_data_file.suffix}")

    text_data = original_df["text"].tolist()

    # Load test data
    test_data = torch.load(test_data_file)
    X_test = torch.tensor(test_data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(test_data["y_test"], dtype=torch.long)
    input_dim = test_data["input_dim"]
    test_original_indices = test_data["original_indices"]

    return X_test, y_test, input_dim, test_original_indices, text_data


def plot_confusion_matrix_altair(
    cm: List[List[int]],
    classes: List[str],
    file_path: Optional[Union[Path, str]] = None,
) -> Optional[alt.Chart]:
    """
    Plot the confusion matrix using Altair and optionally save it.

    Args:
        cm (List[List[int]]): Confusion matrix as a nested list.
        classes (List[str]): Class names corresponding to labels.
        file_path (Optional[Union[Path, str]]): File path to save the heatmap
        (HTML, PNG, or SVG).
            If None, the chart is only displayed.

    Returns:
        Optional[alt.Chart]: The Altair chart object for further use.

    Raises:
        ValueError: If the file_path format is unsupported.
    """
    try:
        # Convert confusion matrix to DataFrame
        cm_df = pd.DataFrame(cm, columns=classes, index=classes)
        cm_df = cm_df.reset_index().melt(
            id_vars="index", var_name="Predicted", value_name="Count"
        )
        cm_df.rename(columns={"index": "Actual"}, inplace=True)

        # Create Altair heatmap
        chart = (
            alt.Chart(cm_df)
            .mark_rect()
            .encode(
                x=alt.X("Predicted:N", title="Predicted Label"),
                y=alt.Y("Actual:N", title="True Label"),
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["Actual:N", "Predicted:N", "Count:Q"],
            )
            .properties(
                title="Confusion Matrix",
                width=600,  # Adjust the width (default is 400)
                height=400,  # Adjust the height (default is 300)
            )
            .configure_title(fontSize=18)
        )

        # Display the chart
        logger.info("Displaying the confusion matrix heatmap...")
        chart.show()

        # Save the chart if a file path is provided
        if file_path:
            file_path = Path(file_path)  # Ensure file_path is a Path object
            if file_path.suffix in [".png", ".svg", ".html"]:
                chart.save(file_path)
                logger.info(f"Confusion matrix heatmap saved to {file_path}")
            else:
                raise ValueError("Unsupported file format. Use .html, .png, or .svg.")

        # Return the chart object for further use
        return chart

    except Exception as e:
        logger.error(f"Error while plotting confusion matrix: {e}")
        raise


def evaluate_model(
    model_path: Union[Path, str],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    input_dim: int,
) -> Tuple[torch.Tensor, str, List[List[int]], List[str]]:
    """
    Evaluate the model.

    Args:
        model_path (str): Path to the model file.
        X_test (torch.Tensor): Test data features.
        y_test (torch.Tensor): Test data labels.
        input_dim (int): Input dimension of the model.

    Returns:
        Tuple[torch.Tensor, str, List[List[int]], List[str]]:
            - Predicted labels as a tensor.
            - Classification report as a string.
            - Confusion matrix as a nested list.
            - Class labels as a list of strings.
    """
    try:
        # Load model
        hidden_dims = [128, 64, 32, 16]

        # Instantiate model
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
                target_names=CLASSES,
                #! labels preset in project_config file; the order must stay the same!
            )
        logger.info("Generated classification report.")

        # Create confusion_matrix
        cm = confusion_matrix(y_test.cpu().numpy(), predicted.cpu().numpy())
        # need to remove tensors from gput to cpu
        # need to covert to NumPy arrays.
        logger.info("Generated confusion matrix.")

        return predicted, report, cm.tolist(), CLASSES

    except Exception as e:
        logger.error(f"An error occured during model evaluation: {e}")
        raise


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
    for i, sample_index in enumerate(misclassified_samples):
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
    pass


if __name__ == "__main__":
    main()
