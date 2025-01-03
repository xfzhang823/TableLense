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

Dependencies: xlwings, pandas, os, logger, sys, asyncio, tempfile, shutil
"""

from pathlib import Path
import os
import logging
import sys
from typing import Callable, Union
import asyncio
import numpy as np
import pandas as pd
from typing import Callable, List
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import xlwings as xw
import tempfile
import shutil
import logging_config

# Configure logger
logger = logging.getLogger(__name__)


def append_tabular_data_files(
    source_file: Union[Path, str],
    target_file: Union[Path, str],
):
    """
    Append data from the source file to the target file after verifying headers match.

    This function checks if the headers of the two files match and appends the
    content of the `source_file` to the `target_file` if they do.
    If the headers do not match, it raises a ValueError to ensure data consistency.

    Args:
        source_file (Union[Path, str]): Path to the file containing data to append.
        target_file (Union[Path, str]): Path to the file to which data will be appended.

    Raises:
        ValueError: If the headers of the two files do not match.
    """
    source_file, target_file = map(Path, [source_file, target_file])

    # Check if both files exist
    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")
    if not target_file.exists():
        raise FileNotFoundError(f"Target file not found: {target_file}")

    # Validate file types to be csv and excel only
    supported_extensions = {".csv", ".xlsx", ".xls"}
    for file, file_type in [(source_file, "source"), (target_file, "target")]:
        if file.suffix.lower() not in supported_extensions:
            raise ValueError(
                f"Unsupported {file_type} file type: {file.suffix}. Only CSV or Excel files are allowed."
            )

    if have_same_headers(source_file, target_file):
        concatenate_tabular_data_files(
            target_file=target_file,
            source_file=source_file,
            skip_header=True,
        )
        logger.info(f"Appended data from {source_file} to {target_file}")
    else:
        raise ValueError(
            f"{source_file} and {target_file} do not have the same header!"
        )


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
        logger.info(f"Column '{col_name}' exists in the DataFrame and is not empty.")
    else:
        # Create 'is_empty' column and fill with 'yes' or 'no'
        df[col_name] = df["text"].apply(
            lambda row: "yes" if is_all_empty(row) else "no"
        )
        logger.info(f"Column '{col_name}' has been added and filled.")

    return df


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
        logger.info(f"Column {col_name} exists in the DataFrame and is not empty.")
    else:
        # Initialize the is_title column with "no"
        df[col_name] = "no"

        # Calculate and fill value
        groups = df.group.unique().tolist()
        for group in groups:
            mask = df.group == group
            first_row_idx = df[mask].index[0]
            df.loc[first_row_idx, "is_title"] = "yes"

        logger.info(f"Column {col_name} is added and filled.")

    return df


def have_same_headers(file1: Path, file2: Path) -> bool:
    """
    Check if two CSV files have the same headers.

    Args:
        file1 (Path): Path to the first CSV file.
        file2 (Path): Path to the second CSV file.

    Returns:
        bool: True if headers are the same, False otherwise.
    """
    # Validate file type (csv or excel)
    supported_extensions = {".csv", ".xlsx", ".xls"}
    for file, file_type in [(file1, "source"), (file2, "target")]:
        if file.suffix.lower() not in supported_extensions:
            raise ValueError(
                f"Unsupported {file_type} file type: {file.suffix}. Only CSV or Excel files are allowed."
            )

    # Read files to compare headers
    with file1.open("r") as f1, file2.open("r") as f2:
        header1 = f1.readline().strip()  # Read and strip newline/whitespace
        header2 = f2.readline().strip()
        return header1 == header2


def concatenate_tabular_data_files(
    target_file: Path, source_file: Path, skip_header: bool = True
):
    """
    Concatenate the source file to the target file.

    Args:
        target_file (Path): Path to the target file (where to save the combined data).
        source_file (Path): Path to the source file (data to append).
        skip_header (bool): Whether to skip the header of the source file when appending.
    """

    def read_file(file: Path) -> pd.DataFrame:
        if file.suffix == ".csv":
            return pd.read_csv(file)
        elif file.suffix in [".xlsx", ".xls"]:
            return pd.read_excel(file)
        else:
            raise ValueError(f"Unsupported file type: {file.suffix}")

    def write_file(file: Path, df: pd.DataFrame):
        if file.suffix == ".csv":
            df.to_csv(file, index=False)
        elif file.suffix in [".xlsx", ".xls"]:
            df.to_excel(file, index=False)
        else:
            raise ValueError(f"Unsupported file type: {file.suffix}")

    target_data = read_file(target_file)
    source_data = read_file(source_file)

    if skip_header:
        source_data = source_data.iloc[1:]  # Skip the header row

    combined_data = pd.concat([target_data, source_data], ignore_index=True)
    write_file(target_file, combined_data)


def get_filtered_files(
    source_data_dir: Union[Path, str], filter_criterion: Callable[[str], bool]
) -> List[Path]:
    """
    Filter English versions of the excel files based on file names:
    - yearbook_2012_data_dir: excel files end in "e" or "E" are in English
    (i.e., ...\China Year Book 2012\html\O1529e.xls")
    - yearbook_2022_data_dir:  excel files starts in "e" or "E" are in English
    (i.e., ...\China Year Book 2022\zk\html\E24-14.xls")

    Args:
        - source_data_dir (str): Directory path for source data.
        - filter_criterion (Callable[[str], bool]): A function that determines whether
        a file should be included based on whether if the file name starts or ends
        with letter "e"

    Returns:
        List[Path]: Filtered file paths.
    """
    # Ensure source_data_dir is a Path object
    source_data_dir = Path(source_data_dir).resolve()

    # Get all file paths with specified extensions
    file_paths = list(source_data_dir.glob("*.xlsx")) + list(
        source_data_dir.glob("*.xls")
    )
    logger.debug(f"Number of file paths before filtering: {len(file_paths)}")

    # Apply filtering based on the criterion
    filtered_file_paths = [
        file_path for file_path in file_paths if filter_criterion(file_path.stem)
    ]

    logger.debug(
        f"Number of file paths after filtering: {len(filtered_file_paths)}"
    )  # For degugging
    logger.debug(f"First 5 filtered file paths: {filtered_file_paths[:5]}")

    if not filtered_file_paths:
        logger.warning(f"No files matched the filter criterion in {source_data_dir}.")

    return filtered_file_paths


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


def process_file_with_timeout_core(
    process_function: Callable[[str], any], file_path: str, timeout: int
) -> any:
    """
    Core logic for processing a file with a timeout, using a specified function.

    This function provides a standardized way to process files with a timeout
    constraint. It uses a thread pool to execute the provided `process_function`,
    ensuring that the main thread remains responsive. If the processing exceeds the
    specified timeout, it raises a `TimeoutError`.

    Args:
        process_function (Callable): The function that defines the processing logic.
            It should take a single argument (`file_path`) and return the result of
            the processing.
        file_path (str): The path to the file to be processed.
        timeout (int): The maximum time (in seconds) allowed for processing the file.

    Returns:
        Any: The result of the `process_function`, or None if an error occurs
        (e.g., timeout or other exceptions).

    Raises:
        TimeoutError: If the processing does not complete within the specified timeout.
        Exception: If an unexpected error occurs during processing.

    Examples:
        >>> def example_process(file_path):
        >>>     # Simulate processing logic
        >>>     return f"Processed {file_path}"

        >>> result = process_file_with_timeout_core(example_process, "example.txt", timeout=10)
        >>> print(result)
        "Processed example.txt"
    """
    start_time = time.time()
    try:
        with ThreadPoolExecutor() as executor:
            result = executor.submit(process_function, file_path).result(
                timeout=timeout
            )
        end_time = time.time()
        logger.info(f"Processed {file_path} in {end_time - start_time:.2f} seconds.")
        return result
    except TimeoutError:
        logger.error(f"Processing timed out for {file_path}")
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
    return None


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
