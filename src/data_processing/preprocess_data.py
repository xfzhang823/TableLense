"""
#TODO: Need to update the codes (Only the async version is debugged and fixed!)

File: preprocess_data.py
Author: Xiao-Fei Zhang
Last updated: 2024 Dec

Script for Processing Excel Files and Aggregating Data (Synchronous Version)

This script:
- processes all Excel files in a specified directory,
- filters the files based on a naming convention (English files only), and
- aggregates their content into a single CSV file,
which will be labeled for train/test to build the classification model 
(content is broken into rows - each to be labeled).

The script utilizes multithreading to handle multiple files concurrently and 
profiles the performance of the data processing function.

Modules:
    - os: Provides a way of using operating system dependent functionality.
    - sys: Provides access to some variables used or maintained by the interpreter.
    - logger: Provides a way to configure logger in the script.
    - time: Provides various time-related functions.
    - concurrent.futures: Provides a high-level interface for asynchronously 
    executing callables.
    - pandas: Provides data structures and data analysis tools.
    - cProfile: Provides a way to profile Python programs.
    - pstats: Provides statistics object for use with the profiler.
    - StringIO: Provides an in-memory stream for text I/O.
    - preprocessing_tools: Custom module for processing Excel files.
    - get_file_names: Custom module for retrieving file names from a directory.

Functions:
    - profile_function(func, *args, **kwargs): Profiles the given function 
    to track performance.
    - process_file_with_timeout(file_path, timeout=600): 
    Processes an Excel file with a timeout.
    - Process_multiple_excel_files: Main function to process multiple Excel files and save 
    the aggregated data to a CSV file.

Usage:
    Call process_all_excel_files from the pipeline module.

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

Key Features:
- The 'preprocess_data' module supports synchronous file processing, using ThreadPoolExecutor 
for parallelism across multiple files.
- The 'preprocess_data_async' module supports asynchronous workflows by combining asyncio's 
event management with ThreadPoolExecutor for hybrid concurrency and parallelism.

Workflow:
In the synchronous version, ThreadPoolExecutor enables parallel processing, 
where each file is handled in a separate thread.
- Parallelism allows multiple threads to execute tasks simultaneously, leveraging 
multi-core processors if available.
- The processing of individual files is fully synchronous, running in 
a single thread per file.

- In the asynchronous version, asyncio orchestrates concurrency at the pipeline level, 
managing multiple tasks (files) efficiently.
- ThreadPoolExecutor is used to offload blocking file processing tasks to threads.
- Async event management handles scheduling and coordination of tasks but is limited 
to ensuring concurrency across files.
- File processing itself remains synchronous and isolated within its thread.

Key Concepts:
*- Concurrency: Tasks are interleaved logically (e.g., via asyncio) to maximize efficiency 
*without blocking the main event loop.
*- Parallelism: Threads in ThreadPoolExecutor provide parallel execution for tasks, 
*depending on available hardware resources.
*- Hybrid Approach: The asynchronous version combines concurrency (via asyncio) 
*with parallelism (via ThreadPoolExecutor), achieving scalable performance.

The synchronous version is simpler and suitable for I/O-bound tasks with 
a limited number of files.

The asynchronous version is more scalable and ideal for high-concurrency workflows 
or processing large datasets.

!Limitations:
!- The current setup does not enable concurrency within the processing of a single file. 
!Each file's processing logic is synchronous and isolated in its thread.
!- For enabling internal concurrency within file processing, the file processing logic 
!(e.g., `process_excel_full_range`) would need to be rewritten as asynchronous.
"""

from pathlib import Path
import os
import sys
import logging
import logging_config
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import cProfile
import pstats
from io import StringIO
import threading
from typing import Dict, Callable, List, Optional, Union
import pandas as pd

from data_processing.excel_preprocessor import ExcelPreprocessor
from project_config import YEARBOOK_2012_DATA_DIR, YEARBOOK_2022_DATA_DIR
from data_processing.data_processing_utils import (
    get_filtered_files,
    process_file_with_timeout_core,
)

# Configure logger
logger = logging.getLogger(__name__)

# Set global lock
excel_lock = threading.Lock()


def profile_function(func: callable, *args, **kwargs) -> any:
    """
    Profiles the given function to track performance.
    - Measure and log the performance of individual file processing tasks.
    - Identify bottlenecks or inefficiencies in synchronous code execution.

    Implement for sync version only.
    Omitted for async version due to added complexity.
    """
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


def process_excel_file(
    file_path: Union[Path, str], yearbook_source: str
) -> List[Dict[str, Union[str, int]]]:
    """
    Processes a single Excel file by transforming tabular data into a structured,
    text-oriented format. This includes serializing rows, adding metadata, and enriching
    data with logical flags to support downstream training and inference tasks.

    Args:
        file_path (Union[Path, str]): The path to the Excel file.
        yearbook_source (str): Metadata indicating the yearbook source.

    Returns:
        List[Dict[str, Union[str, int]]]: A list of dictionaries, each representing a
        processed row with additional contextual metadata.

    Key Steps:
        1. Flatten Table Rows into Text:
        - Each row is serialized into a single string, with cell values concatenated
            using a delimiter (e.g., commas). Missing cells are replaced with "EMPTY".

        2. Assign Unique Identifiers:
        - Each row is assigned a sequential `row_id` to preserve its original order.

        3. Add Contextual Metadata:
        - Includes fields such as `group`, `yearbook_source`, and other relevant flags.

    Example:
        >>> result = process_excel_file("example.xls", "2012")
        >>> print(result)
        [{'text': 'value1, value2, EMPTY', 'row_id': 1, 'group': 'example',
        'yearbook_source': '2012'}, ...]
    """
    # Ensure file path is a Path obj
    file_path = Path(file_path)

    # Extract group - stem/file name of the file path
    group = file_path.stem

    # Instantiate ExcelPreprocesser class
    preprocessor = ExcelPreprocessor()

    # Return processed data (rows - list of dictionaries)
    return preprocessor.process_excel_full_range(
        file_path=file_path, yearbook_source=yearbook_source, group=group
    )


def process_excel_file_with_timeout(
    file_path: str, yearbook_source: str, timeout: int = 600
) -> list:
    """
    Processes a single Excel file with a timeout.

    This function wraps the `process_excel_file` logic, ensuring the file
    is processed within a specified timeout. If the processing exceeds the
    timeout, a TimeoutError is raised.

    Args:
        - file_path (str): The path to the Excel file.
        - yearbook_source (str): Metadata indicating the yearbook source.
        - timeout (int): The maximum time (in seconds) allowed for processing the file.

    Returns:
        list: A list of dictionaries containing
        - processed data for the file, or
        - None if an error occurs (e.g., timeout or unexpected exception).
    """
    return process_file_with_timeout_core(
        lambda fp: process_excel_file(fp, yearbook_source), file_path, timeout
    )


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


def process_multiple_excel_files(
    source_data_dir: Union[Path, str],
    output_csv_path: str,
    yearbook_source: str,
    timeout: int = 600,
) -> None:
    """
    Encapsulated method to process all Excel files in a pipeline-friendly way.

    Args:
        - source_data_dir (Union[Path, str]): Directory containing source Excel files.
        - output_csv_path (str): Path to save the aggregated CSV file.
        - yearbook_source (str): Metadata to indicate data source; either "2012" or "2022"
        (whether from yearbook 2012 or 2022).
        - timeout (int): Timeout in seconds for processing each file. Defaults to 600 seconds.

    Returns:
        None
    """
    logger.info(f"Starting processing Excel files in {source_data_dir}.")

    # Ensure source_data_dir is Path
    source_data_dir = Path(source_data_dir)

    # Determine the filtering criterion based on the source directory
    if Path(source_data_dir).resolve().samefile(YEARBOOK_2012_DATA_DIR):
        filter_criterion = lambda name: name.endswith(
            ("e", "E")
        )  # English files by suffix
    elif Path(source_data_dir).resolve().samefile(YEARBOOK_2022_DATA_DIR):
        filter_criterion = lambda name: name.startswith(
            ("e", "E")
        )  # English files by prefix
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

    # Check if there are no files to process
    if not file_paths:
        logger.warning("No files to process. Check the directory and filter criteria.")
        return

    logger.info(f"Total filtered files: {len(file_paths)}")

    # Process all files
    all_data = []
    for file_path in file_paths:
        logger.debug(f"Processing file: {file_path}")
        file_data = process_excel_file_with_timeout(
            file_path=file_path, yearbook_source=yearbook_source, timeout=timeout
        )
        if file_data:
            all_data.extend(file_data)
            logger.info(f"Processed file: {os.path.basename(file_path)}")
        else:
            logger.warning(
                f"Skipped file: {os.path.basename(file_path)} due to errors."
            )

        # Convert list of dictionaries to a DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            # Save the DataFrame to CSV
            df.to_csv(output_csv_path, index=False)
            logger.info(f"Data saved to {output_csv_path}")
        else:
            logger.warning(
                "No data to save. Processing might have failed for all files."
            )

    logger.info(f"Excel processing in directory {source_data_dir} completed.")


if __name__ == "__main__":
    """Example usage"""
    # # Define directories
    # yearbook_2012_data_dir = r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2012\html"
    yearbook_2022_data_dir = r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2022\zk\html"
    source_data_dir = yearbook_2022_data_dir
    output_csv_path = r"C:\github\china stats yearbook RAG\data\training data\excel sheet training data yrbk 2012.csv"

    # Call the pipeline method
    process_multiple_excel_files(
        source_data_dir=source_data_dir, output_csv_path=output_csv_path
    )
