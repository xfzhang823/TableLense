""" Function file to read a csv file into a dataframe w/o encoding issues """

import logging
import pandas as pd
from file_encoding_detector import detect_encoding


def read_excel_file(file_path, sheet_name=0):
    """
    Read an Excel file into a DataFrame with detected encoding.

    Args:
        file_path (str): Path to the input Excel file.
        sheet_name (str or int, optional): The sheet name or index to read (default is the first sheet).

    Returns:
        pd.DataFrame: DataFrame containing the Excel data.
    """
    try:
        # Detect encoding of the input file (Note: This is more relevant to CSVs,
        # but kept for consistency; Excel files typically don't have encoding issues)
        encoding, confidence = detect_encoding(file_path)
        logging.info(f"Detected encoding: {encoding} with confidence {confidence}")

        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        logging.info(f"File read successfully from {file_path}, sheet: {sheet_name}")

        return df

    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        raise
