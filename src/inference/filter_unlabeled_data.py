"""
File: filter_unlabeled_data.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Aug 10

Description:
Script to filter out rows that have already been labeled during the model training process,
leaving only the unlabeled data for inference.
"""

import sys
from preprocessing.preprocessing_utils import add_is_empty_column, add_is_title_column
from read_csv_file import read_csv_file


def filter_unlabeled_data(production_df, training_df, filter_on="group"):
    """
    Filters out rows in the production data that match rows in the training data.
    This function removes data points from the production dataset that have already
    been seen during the training phase, effectively leaving only the unlabeled data for inference.

    Args:
        production_df (pd.DataFrame): The DataFrame containing the production data.
                                      This is the dataset where inference will be performed.
        training_df (pd.DataFrame): The DataFrame containing the training data.
                                    These are the labeled examples used during model training.
        filter_on (str): The column name in both DataFrames that uniquely identifies each record (row).

    Returns:
        pd.DataFrame: The filtered production data that excludes any rows found
                      in the training data. The resulting DataFrame contains only the data
                      that has not been used during training.
    """
    # Filter based on the column specified by "filter_on", treating it as a unique identifier
    return production_df[~production_df[filter_on].isin(training_df[filter_on])]


def main(production_data_path, training_data_path, output_file):
    """
    Reads the production and training data from specified file paths, filters the production data
    by removing any rows that exist in the training data, and saves the filtered data to an output file.

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
    """
    # Load the production and training data
    production_df = read_csv_file(production_data_path)
    training_df = read_csv_file(training_data_path)

    # Filter out training data from production data
    filtered_df = filter_unlabeled_data(production_df, training_df)

    # Check if the dataset has 'is_empty' and 'is_title' columns, and add them if necessary
    filtered_df = add_is_empty_column(filtered_df)
    filtered_df = add_is_title_column(filtered_df)

    # Save the filtered data
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")


if __name__ == "__main__":
    production_data_path = sys.argv[1]
    training_data_path = sys.argv[2]
    output_file = sys.argv[3]
    main(production_data_path, training_data_path, output_file)
