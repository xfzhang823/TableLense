"""
File: preprocess_data_sync.py
Author: Xiao-Fei Zhang
Date: 2024 Jul 22

Script for Processing Excel Files and Aggregating Data

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
    - logging: Provides a way to configure logging in the script.
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

Logging:
    The script logs detailed information about the processing steps, including profiling information,
    file processing status, and any errors encountered during execution. The log output is printed to
    the standard output stream.

Note:
    Ensure that the preprocessing_tools and get_file_names modules are available in the Python path.
    The script requires pandas and xlwings libraries to be installed.
"""

import os
import sys
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import cProfile
import pstats
from io import StringIO
import threading
import pandas as pd
from get_file_names import get_file_names
from .preprocessing_utils import ExcelPreprocessor


# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Set global lock
excel_lock = threading.Lock()


def profile_function(func, *args, **kwargs):
    """Profiles the given function to track performance."""
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    s = StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    logging.info(s.getvalue())
    return result


def sync_process_excel_full_range(file_path):
    """
    Synchronous function to process an Excel file using ExcelPreprocessor.

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
        logging.debug(f"Starting processing for {file_path}")
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                profile_function, sync_process_excel_full_range, file_path
            )
            result = future.result(timeout=timeout)
        end_time = time.time()
        logging.info(
            f"Processed {os.path.basename(file_path)} in {end_time - start_time:.2f} seconds."
        )
        return result
    except TimeoutError:
        logging.error(f"Processing timed out for {file_path}")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
    return None


def main():
    """
    Main function to process multiple Excel files and save the aggregated data to a CSV file.
    """
    # Define directories
    yearbook_2012_data_dir = r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2012\html"
    yearbook_2022_data_dir = r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2022\zk\html"
    source_data_dir = yearbook_2012_data_dir

    # Get Excel file names from the directory
    file_paths = get_file_names(
        source_data_dir, full_path=True, file_types=[".xlsx", ".xls"]
    )

    # Filter for English files only
    # - Yearbook 2012: filenames end with “e” are English files, i.e., R0612e, R0613e
    # - Yearbook 2022: filenames start with “E” are English, i.e., E11-02, E11-03

    if source_data_dir == yearbook_2012_data_dir:
        filtered_file_paths = [
            file_path
            for file_path in file_paths
            if os.path.basename(file_path).lower().endswith("e")
        ]
    elif source_data_dir == yearbook_2022_data_dir:
        filtered_file_paths = [
            file_path
            for file_path in file_paths
            if os.path.basename(file_path).lower().startswith("e")
        ]
    else:
        filtered_file_paths = file_paths  # or handle other cases as needed

    # Log the first few filtered file paths
    logging.debug(f"First 10 filtered file paths: {filtered_file_paths[:10]}")
    logging.debug(f"Total filtered files: {len(filtered_file_paths)}")

    start_time = time.time()
    all_data = []

    if not filtered_file_paths:
        logging.warning("No files to process. Check the directory and filter criteria.")
        return

    for file_path in filtered_file_paths:
        file_data = process_file_with_timeout(file_path)
        if file_data is not None:
            all_data.extend(file_data)
            logging.info(f"{os.path.basename(file_path)} processed.")
        else:
            logging.info(
                f"Skipped {os.path.basename(file_path)} due to processing error"
            )

    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds.")

    training_data_csv_path = r"C:\github\china stats yearbook RAG\data\training data\excel sheet training data yrbk 2012.csv"
    yearbook_2012_data_csv_path = r"C:\github\china stats yearbook RAG\data\excel data\yearbook 2012 english tables.csv"

    if all_data:
        output_data_path = training_data_csv_path
        df = pd.DataFrame(all_data)
        df.to_csv(output_data_path, index=False)
        logging.info(f"Data saved to {output_data_path}")
    else:
        logging.warning("No data to save. Processing might have failed for all files.")


if __name__ == "__main__":
    main()
