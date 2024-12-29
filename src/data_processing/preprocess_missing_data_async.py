"""
File: preprocess_missing_data_async.py
Author: Xiao-Fei Zhang
Date: 2024 Aug 6

Script for Processing Excel Files and Aggregating Data Asynchronously

This script:
- processes all Excel files in a specified directory,
- filters the files based on a naming convention (English files only),
- identifies missing files based on a preprocessed dataset, and
- aggregates their content into a single CSV file,
which will be labeled for train/test to build a classification model (content is broken into
rows - each to be labeled).

The script utilizes multithreading to handle multiple files concurrently.

Modules:
    - os: Provides a way of using operating system dependent functionality.
    - logging: Provides a way to configure logging in the script.
    - time: Provides various time-related functions.
    - concurrent.futures: Provides a high-level interface for asynchronously executing callables.
    - pandas: Provides data structures and data analysis tools.
    - asyncio: Provides support for asynchronous I/O, event loops, and coroutines.
    - preprocessing_utils: Custom module for processing Excel files.
    - get_file_names: Custom module for retrieving file names from a directory.
    - file_encoding_detector: Custom module for detecting file encoding.

Functions:
    - sync_process_excel_full_range(file_path): Synchronously processes an Excel file.
    - async_process_file_with_timeout(file_path, timeout=600): Asynchronously processes an Excel file with a timeout.
    - get_missing_files(file_paths, processed_data_path): Identifies the missing files based on the processed dataset and file paths.
    - main(): Main function to process multiple Excel files and save the aggregated data to a CSV file.

Usage:
    Run this script as the main module to process Excel files in the specified directory and save
    the aggregated data to a CSV file.

Example:
    $ python preprocess_missing_data_async.py

Note:
    Ensure that the preprocessing_utils and get_file_names modules are available in the Python path.
    The script requires pandas and xlwings libraries to be installed.
"""

import os
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import pandas as pd
import asyncio
from .preprocessing_utils import ExcelPreprocessor
from get_file_names import get_file_names
from file_encoding_detector import detect_encoding

# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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


async def async_process_file_with_timeout(file_path, timeout=600):
    """
    Asynchronously processes an Excel file with a timeout.

    Args:
        file_path (str): The path to the Excel file.
        timeout (int): Timeout in seconds for processing each file. Defaults to 600 seconds.

    Returns:
        list: The processed data from the Excel file, or None if an error occurred.
    """
    start_time = time.time()
    try:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(
                executor, sync_process_excel_full_range, file_path
            )
            result = await asyncio.wait_for(future, timeout)
        end_time = time.time()
        logging.debug(
            f"Processed {os.path.basename(file_path)} in {end_time - start_time:.2f} seconds."
        )
        return result
    except TimeoutError:
        logging.error(f"Processing timed out for {file_path}")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
    return None


def get_missing_files(file_paths, processed_data_path):
    """
    Gets the missing files based on the processed dataset and file paths.

    Args:
        file_paths (list): List of file paths to check.
        processed_data_path (str): Path to the CSV file containing processed data.

    Returns:
        list: A list of filtered file paths for the missing files.
    """
    # Extract file names from paths
    all_file_names = [
        os.path.splitext(os.path.basename(path))[0] for path in file_paths
    ]

    # Check for file encoding
    encoding, _ = detect_encoding(processed_data_path)

    df_processed = pd.read_csv(processed_data_path, encoding=encoding, header=0)
    processed_file_names = df_processed["group"].unique().tolist()

    missing_file_paths = [
        path
        for path, name in zip(file_paths, all_file_names)
        if name not in processed_file_names
    ]

    return missing_file_paths


async def main():
    """
    Main function to process multiple Excel files and save the aggregated data to a CSV file.
    """
    # Set up source data's directory
    yearbook_2012_data_dir = r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2012\html"
    yearbook_2022_data_dir = r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2022\zk\html"
    source_data_dir = yearbook_2012_data_dir
    csv_path = r"C:\github\china stats yearbook RAG\data\training data\excel sheet training data yrbk 2012.csv"

    # Get Excel file names from the directory
    file_paths = get_file_names(
        source_data_dir, full_path=True, file_types=[".xlsx", ".xls"]
    )

    # Filter for English files only
    if source_data_dir == yearbook_2012_data_dir:
        filtered_file_paths = [
            file_path
            for file_path in file_paths
            if os.path.basename(file_path).lower().endswith("e.xls")
            or os.path.basename(file_path).lower().endswith("e.xlsx")
        ]
    elif source_data_dir == yearbook_2022_data_dir:
        filtered_file_paths = [
            file_path
            for file_path in file_paths
            if os.path.basename(file_path).lower().startswith("e")
        ]
    else:
        filtered_file_paths = file_paths  # or handle other cases as needed

    filtered_file_paths = get_missing_files(filtered_file_paths, csv_path)
    logging.debug(f"Total filtered files: {len(filtered_file_paths)}")

    if not filtered_file_paths:
        logging.warning("No files to process. Check the directory and filter criteria.")
        return

    start_time = time.time()
    all_data = []

    # Limit the number of concurrent tasks
    semaphore = asyncio.Semaphore(15)

    async def process_with_semaphore(file_path):
        async with semaphore:
            result = await async_process_file_with_timeout(file_path)
            return result

    tasks = [process_with_semaphore(file_path) for file_path in filtered_file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for file_path, result in zip(filtered_file_paths, results):
        if isinstance(result, Exception):
            logging.error(
                f"Skipped {os.path.basename(file_path)} due to processing error: {result}"
            )
        elif result is not None:
            all_data.extend(result)
        else:
            logging.error(f"Skipped {os.path.basename(file_path)} due to unknown error")

    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds.")

    output_path = r"C:\github\china stats yearbook RAG\data\training data\excel sheet training data yrbk 2012 missing data.csv"

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_path, index=False)
        logging.info(f"Data saved to {output_path}")
    else:
        logging.warning("No data to save. Processing might have failed for all files.")


if __name__ == "__main__":
    asyncio.run(main())
