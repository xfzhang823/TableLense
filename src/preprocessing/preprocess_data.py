"""
File: preprocess_data.py
Author: Xiao-Fei Zhang
Date: 2024 Jul 22

Script for Processing Excel Files and Aggregating Data (Synchronous Version)

This script:
- processes all Excel files in a specified directory,
- filters the files based on a naming convention (English files only), and
- aggregates their content into a single CSV file,
which will be labeled for train/test to build the classification model (content is broken into
rows - each to be labeled).

The script utilizes multithreading to handle multiple files concurrently and profiles the
performance of the data processing function.

Modules:
    - os: Provides a way of using operating system dependent functionality.
    - sys: Provides access to some variables used or maintained by the interpreter.
    - logger: Provides a way to configure logger in the script.
    - time: Provides various time-related functions.
    - concurrent.futures: Provides a high-level interface for asynchronously executing callables.
    - pandas: Provides data structures and data analysis tools.
    - cProfile: Provides a way to profile Python programs.
    - pstats: Provides statistics object for use with the profiler.
    - StringIO: Provides an in-memory stream for text I/O.
    - preprocessing_tools: Custom module for processing Excel files.
    - get_file_names: Custom module for retrieving file names from a directory.

Functions:
    - profile_function(func, *args, **kwargs): Profiles the given function to track performance.
    - process_file_with_timeout(file_path, timeout=600): Processes an Excel file with a timeout.
    - main(): Main function to process multiple Excel files and save the aggregated data to a CSV file.

Usage:
    Run this script as the main module to process Excel files in the specified directory and save
    the aggregated data to a CSV file.

Example:
    $ python process_excel_files.py

logger:
    The script logs detailed information about the processing steps, including 
    profiling information, file processing status, and any errors encountered 
    during execution. The log output is printed to the standard output stream.

Note:
    - Ensure that the preprocessing_tools and get_file_names modules are available 
    in the Python path.
    - The script requires pandas and xlwings libraries to be installed.
"""

from pathlib import Path
import os
import sys
import logging
import logger_config
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import cProfile
import pstats
from io import StringIO
import threading
from typing import Callable, List, Optional, Union
import pandas as pd
from get_file_names import get_file_names

from preprocessing.preprocessing_utils import ExcelPreprocessor
from project_config import YEARBOOK_2012_DATA_DIR, YEARBOOK_2022_DATA_DIR

training_data_csv_path = r"C:\github\china stats yearbook RAG\data\training data\excel sheet training data yrbk 2012.csv"

# Configure logger
logger = logging.getLogger(__name__)

# Set global lock
excel_lock = threading.Lock()


def profile_function(func: callable, *args, **kwargs) -> any:
    """Profiles the given function to track performance."""
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    s = StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    logger.info(s.getvalue())
    return result


def process_excel_full_range(file_path: Union[Path, str]) -> List[dict]:
    """
    Function to process an Excel file using ExcelPreprocessor.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        list: The processed data from the Excel file.
    """
    preprocessor = ExcelPreprocessor()
    return preprocessor.process_excel_full_range(file_path)


def process_file_with_timeout(file_path, timeout=600):
    """
    Processes an Excel file with a timeout.

    Args:
        file_path (str): The path to the Excel file.
        timeout (int): Timeout in seconds for processing each file.
        Defaults to 600 seconds.

    Returns:
        list: The processed data from the Excel file, or None if an error occurred.
    """
    start_time = time.time()
    try:
        logger.debug(f"Starting processing for {file_path}")
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                profile_function, process_excel_full_range, file_path
            )
            result = future.result(timeout=timeout)
        end_time = time.time()
        logger.info(
            f"Processed {os.path.basename(file_path)} in {end_time - start_time:.2f} seconds."
        )
        return result
    except TimeoutError:
        logger.error(f"Processing timed out for {file_path}")
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
    return None


def get_filtered_files(
    source_data_dir: Union[Path, str], filter_criterion: Callable[[str], bool]
) -> List[str]:
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
        List[str]: Filtered file paths.
    """
    # Ensure source_data_dir is a Path object
    source_data_dir = Path(source_data_dir).resolve()

    # Get all file paths with specified extensions
    file_paths = [str(file) for file in source_data_dir.glob("*.xlsx")] + [
        str(file) for file in source_data_dir.glob("*.xls")
    ]

    # Apply filtering based on directory-specific conventions
    # Apply the filter criterion
    filtered_file_paths = [
        file_path for file_path in file_paths if filter_criterion(Path(file_path).stem)
    ]

    # Log filtered file paths
    logger.info(f"First 10 filtered file paths: {filtered_file_paths[:10]}")
    logger.info(f"Total filtered files: {len(filtered_file_paths)}")

    return filtered_file_paths


def process_files(file_paths: List[str]) -> List[dict]:
    """
    Process multiple files and return aggregated data.

    Args:
        file_paths (List[str]): List of file paths to process.

    Returns:
        List[dict]: Aggregated data from all processed files.
    """
    all_data = []
    for file_path in file_paths:
        file_data = process_file_with_timeout(file_path)
        if file_data:
            all_data.extend(file_data)
            logger.info(f"{os.path.basename(file_path)} processed.")
        else:
            logger.info(
                f"Skipped {os.path.basename(file_path)} due to processing error."
            )
    return all_data


def save_to_csv(data: pd.DataFrame, output_path: Union[Path, str]) -> None:
    """
    Save processed data to a CSV file.

    Args:
        data (List[pd.Dataframe]): Data to save.
        output_path (str): Path to save the CSV file.
    """
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
    else:
        logger.warning("No data to save. Processing might have failed for all files.")


def process_all_excel_files(
    source_data_dir: Union[Path, str], output_csv_path: str
) -> None:
    """
    Encapsulated method to process all excel files in a pipeline-friendly way.

    Args:
        source_data_dir (str): Directory containing source Excel files.
        output_csv_path (str): Path to save the aggregated CSV file.
    """
    logger.info(f"Starting processing Excel files in {source_data_dir}.")

    # Ensure source data dir is Path
    source_data_dir = Path(source_data_dir)

    # Determine the filtering criterion based on the source directory
    if Path(source_data_dir).resolve().samefile(YEARBOOK_2012_DATA_DIR):
        filter_criterion = lambda name: name.endswith(
            "e"
        )  # No .lower() needed for windows (Windows files are case insensitve, unlike Linux)
    elif Path(source_data_dir).resolve().samefile(YEARBOOK_2022_DATA_DIR):
        filter_criterion = lambda name: name.startswith(
            "e"
        )  # No .lower() needed for windows
    else:
        logger.error(
            f"Unknown source directory: {source_data_dir}. Expected one of: "
            f"{YEARBOOK_2012_DATA_DIR}, {YEARBOOK_2022_DATA_DIR}."
        )
        raise ValueError(f"Invalid source directory: {source_data_dir}")

    # Get filtered files
    file_paths = get_filtered_files(
        source_data_dir=source_data_dir, filter_criterion=filter_criterion
    )

    # Check if None
    if not file_paths:
        logger.warning("No files to process. Check the directory and filter criteria.")
        return

    logger.debug(f"Total filtered files: {len(file_paths)}")

    all_data = process_files(file_paths)
    save_to_csv(all_data, output_csv_path)
    logger.info(f"Excel processing in directory {source_data_dir} completed.")


if __name__ == "__main__":
    """Example usage"""
    # # Define directories
    # yearbook_2012_data_dir = r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2012\html"
    yearbook_2022_data_dir = r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2022\zk\html"
    source_data_dir = yearbook_2022_data_dir
    output_csv_path = r"C:\github\china stats yearbook RAG\data\training data\excel sheet training data yrbk 2012.csv"

    # Call the pipeline method
    process_all_excel_files(
        source_data_dir=source_data_dir, output_csv_path=output_csv_path
    )
