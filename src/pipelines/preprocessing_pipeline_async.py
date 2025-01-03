"""
File: preprocessing_pipeline.py
Author: Xiao-Fei Zhang
Last Updated: December 2024

Description:
    This module orchestrates the asynchronous preprocessing of Excel files for multiple
    yearbook datasets (2012 and 2022). It handles directory-level validation, file filtering,
    data processing, and aggregation of results into structured CSV files.

    The pipeline ensures sequential processing of 2012 data before 2022 data, followed by
    the combination of both datasets into a unified output file. The total execution time
    for the entire pipeline is logged in hours:minutes:seconds format.

Key Features:
    - Asynchronous processing of Excel files for scalability and efficiency.
    - Directory-level validation to ensure input and output paths are valid.
    - Sequential execution of yearbook datasets (2012 followed by 2022).
    - Aggregation and combination of processed data into a single output file.

Modules Used:
    - `preprocess_data_async`: Handles the core asynchronous file processing logic.
    - `pandas`: For data aggregation and saving to CSV.
    - `time`: For tracking pipeline execution time.
    - `logging`: For detailed logging of pipeline progress and errors.

Workflow:
    1. Validate and preprocess the directory containing 2012 yearbook Excel files.
    2. Validate and preprocess the directory containing 2022 yearbook Excel files.
    3. Combine the processed CSV files for 2012 and 2022 into a single file.
    4. Log execution times and completion status.

Functions:
    - `preprocess_files_in_directory_async`: A wrapper for directory-level 
    file processing.
    - `preprocessing_pipeline_async`: The main pipeline to orchestrate preprocessing
      for both datasets and handle file combination.

Configuration:
    Paths and file names for input directories and output files are configured in the
    'project_config' module:
        - `YEARBOOK_2012_DATA_DIR`: Path to the 2012 yearbook directory.
        - `YEARBOOK_2022_DATA_DIR`: Path to the 2022 yearbook directory.
        - `preprocessed_2012_data_file`: Output file for processed 2012 data.
        - `preprocessed_2022_data_file`: Output file for processed 2022 data.
        - `preprocessed_all_data_file`: Output file for the combined dataset.

Usage:
    Call 'preprocessing_pipeline_async()' to execute the entire pipeline.
    Example:
        asyncio.run(preprocessing_pipeline_async())

"""

from pathlib import Path
from typing import Callable, Union
import logging
import pandas as pd
import time
import asyncio
from data_processing.preprocess_data_async import process_multiple_excel_files_async
from data_processing.preprocess_missing_data_async import (
    get_missing_files,
    preprocess_missing_files_async,
    append_tabular_data_files,
)
from data_processing.preprocessing_utils import have_same_headers, get_filtered_files
from utils.file_encoding_detector import detect_encoding
import logging_config
from project_config import (
    YEARBOOK_2012_DATA_DIR,
    YEARBOOK_2022_DATA_DIR,
    PREPROCESSING_DIR,
    PREPROCESSED_2012_DATA_FILE,
    PREPROCESSED_2022_DATA_FILE,
    PREPROCESSED_ALL_DATA_FILE,
    PREPROCESSED_TEMP_MISSING_DATA_FILE,
)

logger = logging.getLogger(__name__)


async def preprocessing_pipeline_async(
    yearbook_source: str,
    source_data_dir: Path,
    processed_data_file: Path,
    filter_function: Callable,
    max_concurrent_tasks: int = 15,
    timeout: int = 800,
):
    """
    Orchestrates preprocessing of a yearbook, including handling missing files.

    Args:
        - yearbook_source (str): Identifier for the yearbook (e.g., "2012" or "2022").
        - source_data_dir (Path): Directory containing source Excel files.
        - processed_data_path (Path): Path to the existing processed data CSV.
        - filter_function (Callable[[str], bool]): Function to filter relevant files.
        max_concurrent_tasks (int): Maximum number of concurrent tasks.
        timeout (int): Timeout in seconds for processing each file.

    Returns:
        None
    """
    start_time = time.time()  # Record start time

    logger.info(f"Starting pipeline for yearbook {yearbook_source}")

    # Preprocess base files: Handle the first-time case

    # Expected headers for the preprocessed data file
    expected_headers = ["text", "row_id", "group", "yearbook_source"]

    if not processed_data_file.exists():
        logger.warning(
            f"{processed_data_file} does not exist. Creating a new file with expected headers."
        )
        # Create an empty CSV with the correct headers
        pd.DataFrame(columns=expected_headers).to_csv(processed_data_file, index=False)

        await preprocess_missing_files_async(
            source_data_dir=source_data_dir,
            processed_data_file=processed_data_file,
            yearbook_source=yearbook_source,
            filter_function=filter_function,
            max_concurrent_tasks=max_concurrent_tasks,
            timeout=timeout,
        )

    # Add a counter to limit the number of looping to just 3 times
    max_attempts = 3  # Limit the number of attempts to process missing files
    attempts = 0  # Track how many times the loop has run

    # Handle missing files iteratively
    while attempts < max_attempts:
        # *Filter out the files supposed to be processed (English excel only!)
        english_xls_files_only = get_filtered_files(
            source_data_dir=source_data_dir, filter_criterion=filter_function
        )
        missing_files = get_missing_files(
            files_to_check_against=english_xls_files_only,
            output_file_to_check=processed_data_file,
        )
        if not missing_files:
            logger.info(f"No more missing files for yearbook {yearbook_source}.")
            break

        # Process and append missing files
        await preprocess_missing_files_async(
            source_data_dir=source_data_dir,
            processed_data_file=processed_data_file,
            yearbook_source=yearbook_source,
            filter_function=filter_function,
            max_concurrent_tasks=max_concurrent_tasks,
            timeout=timeout,
        )

        # Increment attempt counter
        attempts += 1

    end_time = time.time()  # Record end time
    elapsed_time_seconds = end_time - start_time
    elapsed_time_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_seconds))

    logger.info(
        f"Pipeline for yearbook {yearbook_source} completed. \nTotal time: {elapsed_time_hms} (hh:mm:ss)."
    )


# Orchestrate the pipeline
async def run_preprocessing_pipeline_async():
    """
    Orchestrates the preprocessing of multiple yearbook datasets (2012 and 2022).

    This function invokes the `preprocess_yearbook_pipeline` function for each dataset,
    ensuring all files are processed, including iterative handling of missing files.

    Key Terms:
    - processed_data_path: The main file tracking all processed data for the yearbook.
      This acts as a cumulative checkpoint, storing all successfully processed records.
    """
    start_time = time.time()  # Record start time
    logger.info("Start preprocessing pipeline.")

    # Yearbook 2012 dataset
    await preprocessing_pipeline_async(
        yearbook_source="2012",
        source_data_dir=YEARBOOK_2012_DATA_DIR,
        processed_data_file=PREPROCESSED_2012_DATA_FILE,
        filter_function=lambda name: name.lower().endswith("e"),
        max_concurrent_tasks=10,
    )

    # Yearbook 2022 dataset
    await preprocessing_pipeline_async(
        yearbook_source="2022",
        source_data_dir=YEARBOOK_2022_DATA_DIR,
        processed_data_file=PREPROCESSED_2022_DATA_FILE,
        filter_function=lambda name: name.lower().startswith("e"),
    )

    # Combine the two
    try:
        if have_same_headers(PREPROCESSED_2012_DATA_FILE, PREPROCESSED_2022_DATA_FILE):
            encoding, _ = detect_encoding(PREPROCESSED_2012_DATA_FILE)
            df_2012 = pd.read_csv(PREPROCESSED_2012_DATA_FILE, encoding=encoding)

            encoding, _ = detect_encoding(PREPROCESSED_2022_DATA_FILE)
            df_2022 = pd.read_csv(PREPROCESSED_2022_DATA_FILE, encoding=encoding)

            combined_df = pd.concat([df_2012, df_2022], ignore_index=True)
            combined_df.to_csv(PREPROCESSED_ALL_DATA_FILE, index=False)
            logger.info("Combined preprocessed data file created.")
        else:
            raise ValueError(f"Headers do not match.")
    except Exception as e:
        logger.error(f"Error combining datasets: {e}")
        raise

    # Log total pipeline time
    end_time = time.time()  # Record end time
    elapsed_time_seconds = end_time - start_time
    elapsed_time_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_seconds))
    logger.info(
        f"Finished preprocessing pipeline. Total time: {elapsed_time_hms} (hh:mm:ss)."
    )
