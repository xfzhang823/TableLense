"""
File: filter_unlabeled_data.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Aug 10

Description:
Script to filter out rows that have already been labeled during the model training process,
leaving only the unlabeled data for inference.
"""

import logging
from pathlib import Path
import sys
import pandas as pd
from data_processing.data_processing_utils import (
    add_is_empty_column,
    add_is_title_column,
)
from utils.read_csv_file import read_csv_file

logger = logging.getLogger(__name__)


def filter_unlabeled_data(
    production_df: pd.DataFrame, training_df: pd.DataFrame, filter_on: str = "group"
) -> pd.DataFrame:
    """
    Filters out rows in the production data that match rows in the training data.
    This function removes data points from the production dataset that have already
    been seen during the training phase, effectively leaving only the unlabeled data
    for inference.

    Args:
        - production_df (pd.DataFrame): The DataFrame containing the production data.
                                      This is the dataset where inference will be performed.
        - training_df (pd.DataFrame): The DataFrame containing the training data.
                                    These are the labeled examples used during model training.
        - filter_on (str): The column name in both DataFrames that uniquely identifies
        each record (row).

    Returns:
        pd.DataFrame: The filtered production data that excludes any rows found
                      in the training data. The resulting DataFrame contains only the data
                      that has not been used during training.
    """
    # Filter based on the column specified by "filter_on", treating it as a unique identifier
    return production_df[~production_df[filter_on].isin(training_df[filter_on])]


def extract_and_save_inference_data(
    training_inference_data_file: Path | str,
    training_data_file: Path | str,
    inference_data_file: Path | str,
):
    """
    Reads the production and training data from specified file paths, filters
    the production data by removing any rows that exist in the training data,
    and saves the filtered data to an output file.

    The function performs the following steps with error handling:
    1. Converts the input file paths to Path objects.
    2. Validates that the production and training data files exist.
    3. Loads the production (training & inference) data and training data using
    a custom CSV reader.
    4. Filters out rows from the production data that exist in the training data.
    5. Ensures the filtered DataFrame contains the 'is_empty' and 'is_title' columns,
        adding them if necessary.
    6. Saves the filtered DataFrame as a CSV file to the specified output path.

    Args:
        production_data_path (str): The file path to the CSV containing the production data.
                                    This is the dataset where inference will be performed.
        training_data_path (str): The file path to the CSV containing the training data.
                                  These are the labeled examples used during model training.
        output_file (str): The file path where the filtered production data should be saved.
                           The output will be a CSV file containing only the rows
                           from the production data that were not part of the training data.

    Returns:
        None: The function saves the filtered DataFrame to the specified output file
              and does not return anything.

     Raises:
        FileNotFoundError: If any of the input files do not exist.
        Exception: Propagates exceptions encountered during reading, filtering,
        or writing CSV files.
    """
    logger.info(f"Start filtering {training_inference_data_file} for inference data...")

    # Convert inputs to Path objects with error handling
    try:
        training_inference_data_file = Path(training_inference_data_file)
        training_data_file = Path(training_data_file)
        inference_data_file = Path(inference_data_file)
    except Exception as e:
        logger.error("Error converting file paths to Path objects.", exc_info=e)
        raise

    # Verify that the input files exist
    if not training_inference_data_file.exists():
        logger.error(f"Production data file not found: {training_inference_data_file}")
        raise FileNotFoundError(
            f"Production data file not found: {training_inference_data_file}"
        )
    if not training_data_file.exists():
        logger.error(f"Training data file not found: {training_data_file}")
        raise FileNotFoundError(f"Training data file not found: {training_data_file}")

    # Load the production and training data with error handling
    try:
        train_infer_df = read_csv_file(training_inference_data_file)
        training_df = read_csv_file(training_data_file)

        logger.debug(
            f"Training data from {training_data_file}: {training_df.shape}"
        )  # todo: debug
        logger.debug(
            f"Training & inference data from {inference_data_file}: {train_infer_df.shape}"
        )  # todo: debug
        logger.info(
            f"Expected inference data: {len(train_infer_df)-len(training_df)}"
        )  # todo: debug

    except Exception as e:
        logger.error("Error reading input CSV files.", exc_info=e)
        raise

    # Filter out rows in production data that exist in the training data
    try:
        filtered_df = filter_unlabeled_data(train_infer_df, training_df)
    except Exception as e:
        logger.error("Error filtering unlabeled data.", exc_info=e)
        raise

    # Ensure auxiliary columns are present, adding them if necessary
    try:
        filtered_df = add_is_empty_column(filtered_df)
        filtered_df = add_is_title_column(filtered_df)
    except Exception as e:
        logger.error("Error adding auxiliary columns to the DataFrame.", exc_info=e)
        raise

    # Save the filtered data with error handling
    try:
        filtered_df.to_csv(inference_data_file, index=False)

        logger.info(f"Inference data saved to {inference_data_file}")  # Debug

    except Exception as e:
        logger.error("Error saving filtered data to CSV.", exc_info=e)
        raise

    logger.info(f"Filtered data saved to {inference_data_file}")


if __name__ == "__main__":
    production_data_path = sys.argv[1]
    training_data_path = sys.argv[2]
    output_file = sys.argv[3]
    extract_and_save_inference_data(
        production_data_path, training_data_path, output_file
    )
