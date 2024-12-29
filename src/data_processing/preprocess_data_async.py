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
    - async_process_file_with_timeout(file_path, timeout=600): 
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
from get_file_names import get_file_names
from data_processing.excel_preprocessor import ExcelPreprocessor
from data_processing.preprocess_data import process_excel_file, get_filtered_files
from project_config import YEARBOOK_2012_DATA_DIR, YEARBOOK_2022_DATA_DIR

# Configure logger
logger = logging.getLogger(__name__)


# Function to process a single excel file w/t timeout
async def process_excel_file_with_timeout_async(
    file_path: str, yearbook_source: str, timeout: int = 600
) -> List[Dict[str, Union[str, int]]]:
    """
    Asynchronously processes a single Excel file with a timeout.

    Args:
        file_path (str): The path to the Excel file.
        yearbook_source (str): Metadata indicating the yearbook source.
        timeout (int): Timeout in seconds for processing the file.
        Defaults to 600 seconds.

    Returns:
        List[Dict[str, Union[str, int]]]: Processed data, or None if processing fails.
    """
    start_time = time.time()
    try:
        logger.debug(f"Starting async processing for {file_path}")

        # Call the asynchronous processing method directly
        preprocessor = ExcelPreprocessor()
        result = await asyncio.wait_for(
            preprocessor.process_excel_full_range_async(file_path, yearbook_source),
            timeout,
        )

        end_time = time.time()
        logger.info(
            f"Async processed {os.path.basename(file_path)} in {end_time - start_time:.2f} seconds."
        )
        return result
    except TimeoutError:
        logger.error(f"Async processing timed out for {file_path}")
    except Exception as e:
        logger.error(f"Error during async processing of {file_path}: {e}")
    return None


# Function to process multiple excel files
async def process_multiple_excel_files_async(
    source_data_dir: Union[Path, str],
    output_csv_path: str,
    yearbook_source: str,
    max_concurrent_tasks: int = 15,
    timeout: int = 600,
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

    # Determine filtering criteria based on the directory
    if source_data_dir.resolve().samefile(YEARBOOK_2012_DATA_DIR):
        filter_criterion = lambda name: name.endswith("e")  # English files by suffix
    elif source_data_dir.resolve().samefile(YEARBOOK_2022_DATA_DIR):
        filter_criterion = lambda name: name.startswith("e")  # English files by prefix
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


# TODO: old code; to be deleted once pipeline debugged
# def sync_process_excel_full_range(file_path):
#     """
#     Synchronous function to process an Excel file using ExcelPreprocessor.

#     Args:
#         file_path (str): The path to the Excel file.

#     Returns:
#         list: The processed data from the Excel file.
#     """
#     preprocessor = ExcelPreprocessor()
#     return preprocessor.process_excel_full_range(file_path)


# async def main():
#     """
#     Main function to process multiple Excel files and save the aggregated data to a CSV file.
#     """
#     # Define directories
#     yearbook_2012_data_dir = r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2012\html"
#     yearbook_2022_data_dir = r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2022\zk\html"
#     source_data_dir = yearbook_2012_data_dir

#     # Get Excel file names from the directory
#     file_paths = get_file_names(
#         source_data_dir, full_path=True, file_types=[".xlsx", ".xls"]
#     )

#     # Filter for English files only
#     if source_data_dir == yearbook_2012_data_dir:
#         filtered_file_paths = [
#             file_path
#             for file_path in file_paths
#             if os.path.basename(file_path).lower().endswith("e.xls")
#             or os.path.basename(file_path).lower().endswith("e.xlsx")
#         ]
#     elif source_data_dir == yearbook_2022_data_dir:
#         filtered_file_paths = [
#             file_path
#             for file_path in file_paths
#             if os.path.basename(file_path).lower().startswith("e")
#         ]
#     else:
#         filtered_file_paths = file_paths  # or handle other cases as needed

#     print(filtered_file_paths)

#     # Log the first few filtered file paths
#     logger.debug(f"First 10 filtered file paths: {filtered_file_paths[:10]}")
#     logger.debug(f"Total filtered files: {len(filtered_file_paths)}")

#     start_time = time.time()
#     all_data = []

#     if not filtered_file_paths:
#         logger.warning("No files to process. Check the directory and filter criteria.")
#         return

#     # Limit the number of concurrent tasks
#     semaphore = asyncio.Semaphore(15)

#     async def process_with_semaphore(file_path):
#         async with semaphore:
#             logger.info(f"Starting processing for {file_path}")
#             result = await async_process_file_with_timeout(file_path)
#             logger.info(f"Finished processing for {file_path}")
#             return result

#     tasks = [process_with_semaphore(file_path) for file_path in filtered_file_paths]
#     results = await asyncio.gather(*tasks, return_exceptions=True)

#     for file_path, result in zip(filtered_file_paths, results):
#         if isinstance(result, Exception):
#             logger.info(
#                 f"Skipped {os.path.basename(file_path)} due to processing error: {result}"
#             )
#         elif result is not None:
#             all_data.extend(result)
#             logger.info(f"{os.path.basename(file_path)} processed.")
#         else:
#             logger.info(f"Skipped {os.path.basename(file_path)} due to unknown error")

#     end_time = time.time()
#     logger.info(f"Total processing time: {end_time - start_time:.2f} seconds.")

#     training_data_csv_path = r"C:\github\china stats yearbook RAG\data\training data\excel sheet training data yrbk 2012.csv"
#     # missing_data_path = r"C:\github\china stats yearbook RAG\data\training data\excel sheet training data yrbk 2012 missing data.csv"

#     if all_data:
#         output_data_path = training_data_csv_path
#         df = pd.DataFrame(all_data)
#         df.to_csv(output_data_path, index=False)
#         logger.info(f"Data saved to {output_data_path}")
#     else:
#         logger.warning("No data to save. Processing might have failed for all files.")


# def process_single_file(file_path):
#     """
#     Synchronously processes an Excel file.

#     Args:
#         file_path (str): The path to the Excel file.

#     Returns:
#         list: The processed data from the Excel file.
#     """
#     try:
#         logger.info(f"Processing file: {file_path}")
#         preprocessor = ExcelPreprocessor()
#         data = preprocessor.process_excel_full_range(file_path)
#         logger.info(f"Finished processing file: {file_path}")
#         return data
#     except Exception as e:
#         logger.error(f"Error processing file {file_path}: {e}")
#         return None


# def main_single_file():
#     """
#     Main function to process a single Excel file and save the aggregated data to a CSV file.
#     """
#     file_path = r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2012\html\B0105e.xls"
#     data = process_single_file(file_path)

#     if data:
#         output_data_path = r"C:\github\china stats yearbook RAG\data\training data\single_file_output.csv"
#         df = pd.DataFrame(data)
#         df.to_csv(output_data_path, index=False)
#         logger.info(f"Data saved to {output_data_path}")
#     else:
#         logger.warning("No data to save. Processing might have failed.")


# if __name__ == "__main__":
#     asyncio.run(main())
