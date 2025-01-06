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

*Key Concepts:
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
import tempfile
from typing import Dict, List, Optional, Union
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import asyncio
import pandas as pd
import pythoncom

# From internal modules
import logging_config
from data_processing.excel_preprocessor import ExcelPreprocessor
from data_processing.data_processing_utils import (
    get_filtered_files,
    process_file_with_timeout_core,
    append_tabular_data_files,
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
    source_data_dir: Optional[Path] = None,
    file_paths: Optional[List[Path]] = None,
    output_csv_file: Path = None,
    yearbook_source: str = "",
    max_concurrent_tasks: int = 10,
    timeout: int = 800,
):
    """
    Asynchronously process multiple Excel files and save results.

    Args:
        - source_data_dir (Optional[Path]): Directory containing Excel files.
        - file_paths (Optional[List[Path]]): List of specific file paths to process.
        - output_csv_file (Path): Path to save the aggregated data.
        - yearbook_source (str): Yearbook identifier (e.g., "2012" or "2022").
        - max_concurrent_tasks (int): Maximum number of concurrent tasks.
        - timeout (int): Timeout in seconds for processing all files.

    Raises:
        - ValueError: If both `source_data_dir` and `file_paths` are provided,
        or if neither is provided.
    """
    # * Validation:
    # * only a directory OR list of file paths is provided, but NOT BOTH
    # * and NOT both None
    if source_data_dir and file_paths:
        raise ValueError(
            "Provide either `source_data_dir` or `file_paths`, but not both."
        )
    if not source_data_dir and not file_paths:
        raise ValueError("Either `source_data_dir` or `file_paths` must be provided.")

    # Determine filtering criteria
    filter_criterion = None  # reset to None
    if source_data_dir:
        if source_data_dir.resolve().samefile(YEARBOOK_2012_DATA_DIR):
            filter_criterion = lambda name: name.lower().endswith(
                "e"
            )  # English files by suffix
        elif source_data_dir.resolve().samefile(YEARBOOK_2022_DATA_DIR):
            filter_criterion = lambda name: name.lower().startswith(
                "e"
            )  # English files by prefix
        else:
            logger.error(
                f"Unknown source directory: {source_data_dir}. Expected one of: "
                f"{YEARBOOK_2012_DATA_DIR}, {YEARBOOK_2022_DATA_DIR}."
            )
            raise ValueError(f"Invalid source directory: {source_data_dir}")

        # Use filtering logic if source_data_dir is provided
        file_paths = get_filtered_files(
            source_data_dir=source_data_dir,
            filter_criterion=filter_criterion,
        )

    # Check if there are files to process
    if not file_paths:
        logger.warning("No files to process.")
        return

    logger.info(
        f"Processing {len(file_paths)} files from {source_data_dir or 'list of paths'}."
    )

    # Semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    # Track success / failure
    successfully_processed = []  # To track successfully processed files
    failed_files = []  # To track failed files

    # * Function to process a SINGLE file (process, save to temp_file, append to aggregate data file)
    async def process_file(file_path: Path):
        async with semaphore:
            try:
                logger.debug(f"Processing file: {file_path}")

                # Process file
                processed_data = await process_excel_file_with_timeout_async(
                    file_path=file_path,
                    yearbook_source=yearbook_source,
                    timeout=timeout,
                )

                # Save results in temp file, then append to aggregate data output file
                if not processed_data:
                    failed_files.append(file_path)
                    logger.warning(f"No data returned for {file_path}. Skipping.")

                # Create a task-specific temporary file, save data to it, and then append
                # temp file to aggregate output file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".csv"
                ) as temp_file:
                    temp_file_path = Path(temp_file.name)
                    pd.DataFrame(processed_data).to_csv(temp_file_path, index=False)
                    logger.info(f"Saved processed data to {temp_file_path}")

                    # Append temp file to the main processed data file
                    try:
                        append_tabular_data_files(
                            source_file=temp_file_path,
                            target_file=output_csv_file,
                        )
                        logger.info(f"Successfully appended data for {file_path}.")
                    except Exception as e:
                        logger.error(f"Error appending data from {file_path}")
                        failed_files.append(file_path)
                    finally:
                        # Delete temp file after appending
                        temp_file_path.unlink()
                        logger.debug(f"Deleted temp file: {temp_file_path}")

            except PermissionError as e:
                logger.warning(
                    f"File lock encountered for {file_path}. Skipping due to permission error."
                )
                failed_files.append(file_path)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout reached for {file_path}. Skipping.")
                failed_files.append(file_path)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                failed_files.append(file_path)

    # Process all files asynchronously
    tasks = [process_file(file_path) for file_path in file_paths]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("Processing timed out.")

    # Log processing statistics
    logger.info(f"Finished processing all files.")
    logger.info(f"Total files: {len(file_paths)}")
    logger.info(f"Successfully processed files: {len(successfully_processed)}")
    logger.info(f"Failed files: {len(failed_files)}")

    logger.info(f"Finished processing {len(successfully_processed)} files.")
