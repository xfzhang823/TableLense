"""
File: preprocess_data_async.py
Author: Xiao-Fei Zhang
Last updated: 2024 Dec

Script for Processing Excel Files and Aggregating Data (Asynchronous Version)

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
    - async_process_file_with_timeout(file_path, timeout=1000): 
    Asynchronously processes an Excel file with a timeout.
    - process_all_excel_files_async(source_data_dir, output_csv_path, max_concurrent_tasks=15): 
    Main function to process multiple Excel files asynchronously and save 
    the aggregated data to a CSV file.

Usage:
    Call process_all_excel_files_async from the pipeline module.

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
- The 'preprocess_data' module supports synchronous file processing, using 
ThreadPoolExecutor for parallelism across multiple files.
- The 'preprocess_data_async' module supports asynchronous workflows by combining 
asyncio's event management with ThreadPoolExecutor for hybrid concurrency and parallelism.

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

Limitations:
!- The current setup does not enable concurrency within the processing of a single file. 
!Each file's processing logic is synchronous and isolated in its thread.
!- For enabling internal concurrency within file processing, the file processing logic 
!(e.g., `process_excel_full_range`) would need to be rewritten as asynchronous.
"""

import os
from pathlib import Path
import logging
import logging_config
from typing import Dict, List, Union
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import asyncio
import pandas as pd
import pythoncom

from data_processing.excel_preprocessor import ExcelPreprocessor
from data_processing.preprocessing_utils import (
    get_filtered_files,
    process_file_with_timeout_core,
)
from project_config import YEARBOOK_2012_DATA_DIR, YEARBOOK_2022_DATA_DIR

# Configure logger
logger = logging.getLogger(__name__)


# Function to process a single excel file w/t timeout
async def process_excel_file_with_timeout_async(
    file_path: Union[Path, str], yearbook_source: str, timeout: int = 1000
) -> list:
    """
    Asynchronously processes a single Excel file with a timeout.

    This function wraps the 'process_file_with_timeout_core' logic, running it in a
    non-blocking manner using an executor. It enforces the timeout and ensures proper
    error handling.

    This function integrates a synchronous processing function (`process_excel_full_range`)
    into an asynchronous pipeline by leveraging `loop.run_in_executor`. The '_core' method
    is used to enforce timeouts and ensure consistency between synchronous and asynchronous
    implementations.

    Key Points:
    - 'loop.run_in_executor' ensures that the entire file processing task runs in a
        single thread within the executor's thread pool. This prevents thread-hopping
        and maintains thread-local context for synchronous operations.
    - The async event loop delegates the blocking work (e.g., file I/O, CPU-bound
        computations) to the executor, preventing the event loop from being blocked.
    - By using `_core`, the same logic for enforcing timeouts and managing processing
        can be shared between the synchronous and asynchronous versions of the pipeline.

    Why Use `loop.run_in_executor`:
    - To ensure thread-local variables or resources tied to a specific thread
        (e.g., an Excel application instance) persist throughout the processing
        of a single file.
    - To guarantee that operations for a single file remain in the same thread,
        avoiding potential race conditions or inconsistent state.
    - To offload blocking synchronous work to a thread, ensuring the async
        event loop remains responsive.

    Args:
    - file_path (str): The path to the Excel file.
    - yearbook_source (str): Metadata indicating the yearbook source.
    - timeout (int): Timeout in seconds for processing the file.
    Defaults to 600 seconds.

    Returns:
        list: A list of dictionaries containing
        - processed data for the file,
        - or None if an error occurs (e.g., timeout or unexpected exception).
    """
    # Ensure file path is Path obj
    file_path = Path(file_path)

    # Instantiate ExcelPreprocessor class
    preprocessor = ExcelPreprocessor()

    # Extract 'group' from file path stem
    group = file_path.stem

    # Define the synchronous processing function for `_core`
    # *loop.run_in_executor ensures that a single thread is allocated to process
    # *a single file, and that all operations (including async operations) for
    # *that file will stay within the same thread.
    def processing_function(fp):

        # Initialize COM for current thread
        pythoncom.CoInitialize()  # pylint: disable=no-member

        try:
            # !Must call the sync version process_excel_full_range method
            # !from the ExcelPreprocessor class b/c we are using loop
            return preprocessor.process_excel_full_range(
                file_path=fp,
                yearbook_source=yearbook_source,
                group=group,
            )
        finally:
            # Clean up COM for the thread
            pythoncom.CoUninitialize()  # pylint: disable=no-member

    # Use loop.run_in_executor to offload synchronous `_core` function
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None,
            process_file_with_timeout_core,  # '_core' function
            processing_function,
            file_path,
            timeout,
        )
    except TimeoutError:
        logger.error(f"Processing timed out for {file_path}")
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
    return None


# Function to process multiple excel files
async def process_multiple_excel_files_async(
    source_data_dir: Union[Path, str],
    output_csv_path: str,
    yearbook_source: str,
    max_concurrent_tasks: int = 8,
    timeout: int = 1000,
):
    """
    Asynchronously processes multiple Excel files and saves the aggregated data to a CSV.

    Args:
        - source_data_dir (Union[Path, str]): Directory containing source Excel files.
        - output_csv_path (str): Path to save the aggregated CSV file.
        - yearbook_source (str): Metadata to indicate data source; either "2012" or "2022"
        (whether from yearbook 2012 or 2022).
        - max_concurrent_tasks (int): Maximum number of concurrent file processing tasks.
        - timeout (int): Timeout in seconds for each file. Defaults to 600 seconds.

    Returns:
        None
    """
    logger.info(
        f"Starting asynchronous processing of Excel files in {source_data_dir}."
    )
    source_data_dir = Path(source_data_dir)

    logger.info(f"Resolved source_data_dir: {source_data_dir.resolve()}")
    logger.info(f"YEARBOOK_2012_DATA_DIR: {YEARBOOK_2012_DATA_DIR.resolve()}")
    logger.info(f"YEARBOOK_2022_DATA_DIR: {YEARBOOK_2022_DATA_DIR.resolve()}")

    # Determine filtering criteria based on the directory
    if source_data_dir.resolve().samefile(YEARBOOK_2012_DATA_DIR):
        filter_criterion = lambda name: name.lower().endswith(
            "e"
        )  # English files by suffix
    elif source_data_dir.resolve().samefile(YEARBOOK_2022_DATA_DIR):
        logger.info(
            f"source data dir: {source_data_dir}"
        )  # todo: debugging; delete later
        filter_criterion = lambda name: name.lower().startswith(
            "e"
        )  # English files by prefix
    else:
        logger.error(
            f"Unknown source directory: {source_data_dir}. Expected one of: "
            f"{YEARBOOK_2012_DATA_DIR}, {YEARBOOK_2022_DATA_DIR}."
        )
        raise ValueError(f"Invalid source directory: {source_data_dir}")

    # Get filtered file paths
    file_paths = get_filtered_files(
        source_data_dir=source_data_dir, filter_criterion=filter_criterion
    )
    logger.info(f"file paths: {file_paths}")

    if not file_paths:
        logger.warning("No files to process. Check the directory and filter criteria.")
        return

    logger.info(f"Total files to process: {len(file_paths)}")

    # Limit concurrent tasks with a semaphore
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def process_with_semaphore(file_path):
        async with semaphore:
            return await process_excel_file_with_timeout_async(
                file_path=file_path, yearbook_source=yearbook_source, timeout=timeout
            )

    # Process files concurrently
    tasks = [process_with_semaphore(file_path) for file_path in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Aggregate results
    all_data = []
    for file_path, result in zip(file_paths, results):
        if isinstance(result, Exception):
            logger.error(f"Error processing {file_path}: {result}")
        elif result is not None:
            all_data.extend(result)
        else:
            logger.warning(f"Skipped {file_path} due to unknown error.")

    # Convert aggregated data to a DataFrame and save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Aggregated data saved to {output_csv_path}")
    else:
        logger.warning("No data to save. Processing might have failed for all files.")

    logger.info(
        f"Asynchronous processing of Excel files in {source_data_dir} completed."
    )
