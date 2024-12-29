"""
File: preprocessing_utils.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Jul 28

Description:
    - Utility class for preprocessing Excel files. 
    - It includes functions to clear sheets, copy content between sheets, 
    process Excel files, convert cell references, etc.
    - This version includes both synchronous and asynchronous methods.

    This version uses asyncio.Semaphore to manage resources; 
    it does not use threading.Lock.

Usage:
    from preprocessing_utils import ExcelPreprocessor

Dependencies: xlwings, pandas, os, logging, sys, asyncio, tempfile, shutil
"""

from pathlib import Path
import os
import logging
import sys
from typing import Union
import asyncio
import numpy as np
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import xlwings as xw
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def add_is_title_column(df):
    """
    Adds an 'is_title' column to the DataFrame, marking the first row of each group as 'yes'
    and the rest as 'no'.
    If the column already exists and is not empty, the function does nothing.

    Args:
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        pd.DataFrame: The modified DataFrame with the 'is_title' column added.
    """
    # Verify if the column exists and is not empty
    col_name = "is_title"

    if col_name in df.columns and not df[col_name].dropna().empty:
        logging.info(f"Column {col_name} exists in the DataFrame and is not empty.")
    else:
        # Initialize the is_title column with "no"
        df[col_name] = "no"

        # Calculate and fill value
        groups = df.group.unique().tolist()
        for group in groups:
            mask = df.group == group
            first_row_idx = df[mask].index[0]
            df.loc[first_row_idx, "is_title"] = "yes"

        logging.info(f"Column {col_name} is added and filled.")

    return df


def is_all_empty(row):
    """
    Check if all items in a row are 'EMPTY', ignoring extra spaces.

    Args:
        row (str): A string representing a row, with items separated by commas.

    Returns:
        bool: True if all items are 'EMPTY' (after stripping spaces), False otherwise.
    """
    items = [item.strip().upper() for item in row.split(",")]
    return all(item == "EMPTY" or item == "" for item in items)


def add_is_empty_column(df):
    """
    Checks if the 'is_empty' column exists in the DataFrame. If it does not exist,
    it creates the column and fills it with 'yes' or 'no' based on whether the 'text'
    column is considered empty.

    Args:
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        pd.DataFrame: The modified DataFrame with the 'is_empty' column added.
    """
    col_name = "is_empty"

    if col_name in df.columns and not df[col_name].dropna().empty:
        logging.info(f"Column '{col_name}' exists in the DataFrame and is not empty.")
    else:
        # Create 'is_empty' column and fill with 'yes' or 'no'
        df[col_name] = df["text"].apply(
            lambda row: "yes" if is_all_empty(row) else "no"
        )
        logging.info(f"Column '{col_name}' has been added and filled.")

    return df


def main():
    preprocessor = ExcelPreprocessor()  # No lock here

    source_path = "source.xlsx"  # Replace with your source workbook path
    target_path = "target.xlsx"  # Replace with your target workbook path
    source_sheet_name = "SourceSheetName"  # Replace with your source sheet name
    target_sheet_name = "TargetSheetName"  # Replace with your target sheet name

    source_start_cell = (
        "A1"  # Replace with the starting cell of the range in the source sheet
    )
    target_start_cell = [
        1,
        1,
    ]  # Replace with the starting cell of the range in the target sheet

    last_row_coord = preprocessor.copy_sheet_to_diff_file(
        source_path,
        target_path,
        source_sheet_name,
        target_sheet_name,
        source_start_cell,
        target_start_cell,
    )
    print(f"Last row with content: {last_row_coord}")


if __name__ == "__main__":
    main()
