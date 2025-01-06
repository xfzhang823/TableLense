""" Function file to read a csv file into a dataframe w/o encoding issues """

import logging
import pandas as pd
from utils.file_encoding_detector import detect_encoding


def read_csv_file(file_path):
    """
    Read a CSV file into a DataFrame with detected encoding.

    Args:
        input_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    try:
        # Detect encoding of the input file
        encoding, confidence = detect_encoding(file_path)
        logging.info(f"Detected encoding: {encoding} with confidence {confidence}")

        # Read the file into a DataFrame
        df = pd.read_csv(file_path, encoding=encoding, header=0)
        logging.info(f"File read successfully from {file_path}")

        return df
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        raise
