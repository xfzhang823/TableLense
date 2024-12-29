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
    `project_config` module:
        - `YEARBOOK_2012_DATA_DIR`: Path to the 2012 yearbook directory.
        - `YEARBOOK_2022_DATA_DIR`: Path to the 2022 yearbook directory.
        - `preprocessed_2012_data_file`: Output file for processed 2012 data.
        - `preprocessed_2022_data_file`: Output file for processed 2022 data.
        - `preprocessed_all_data_file`: Output file for the combined dataset.

Usage:
    Call `preprocessing_pipeline_async()` to execute the entire pipeline.
    Example:
        asyncio.run(preprocessing_pipeline_async())

"""

from pathlib import Path
from typing import Union
import logging
import pandas as pd
import time
import asyncio
from data_processing.preprocess_data_async import process_multiple_excel_files_async
import logging_config
from project_config import (
    YEARBOOK_2012_DATA_DIR,
    YEARBOOK_2022_DATA_DIR,
    preprocessed_2012_data_file,
    preprocessed_2022_data_file,
    preprocessed_all_data_file,
)

logger = logging.getLogger(__name__)


# Preprocess a single directory
async def preprocess_files_in_directory_async(
    source_data_dir: Union[Path, str],
    output_csv_path: Union[Path, str],
    yearbook_source: str,
    max_concurrent_tasks: int = 10,
    timeout: int = 600,
):
    """
    Wrapper to asynchronously process a directory of Excel files and save results to a CSV.
    Logs the total time taken for the process in hours:minutes:seconds format.

    Args:
        source_data_dir (Union[Path, str]): Directory containing Excel files.
        output_csv_path (Union[Path, str]): Path to save the output CSV file.
        yearbook_source (str): Metadata indicating the yearbook source.
        max_concurrent_tasks (int): Maximum concurrent tasks for processing.
        timeout (int): Timeout for each file in seconds.

    Returns:
        None
    """
    start_time = time.time()  # Record start time
    logger.info(f"Starting async pipeline for directory: {source_data_dir}")

    # Ensure path params are Path objects
    source_data_dir = Path(source_data_dir)
    output_csv_path = Path(output_csv_path)

    try:
        # Check input data file
        if not source_data_dir.exists():
            logger.error(f"Source directory does not exist: {source_data_dir}")
            raise FileNotFoundError(f"Source directory not found: {source_data_dir}")
        if not source_data_dir.is_dir():
            logger.error(f"Provided path is not a directory: {source_data_dir}")
            raise NotADirectoryError(f"Expected a directory but got: {source_data_dir}")
        if not any(source_data_dir.glob("*.[xX][lL][sS]*")):  # Check for Excel files
            logger.warning(f"No Excel files found in directory: {source_data_dir}")
            return

        # Check output data file directory exists or not
        if not output_csv_path.parent.exists():
            logger.error(f"Output directory does not exist: {output_csv_path.parent}")
            raise FileNotFoundError(
                f"Output directory not found: {output_csv_path.parent}"
            )

        logger.info(f"Starting async pipeline for directory: {source_data_dir}")

        # Call the processing function from preprocessing_data_async
        await process_multiple_excel_files_async(
            source_data_dir=source_data_dir,
            output_csv_path=output_csv_path,
            yearbook_source=yearbook_source,
            max_concurrent_tasks=max_concurrent_tasks,
            timeout=timeout,
        )

        end_time = time.time()  # record end time
        elapsed_time_seconds = end_time - start_time

        # Convert elapsed time to hrs:mins:seconds
        elapsed_time_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_seconds))

        logger.info(
            f"Pipeline completed for directory: {source_data_dir}. "
            f"Total time taken: {elapsed_time_hms} (hh:mm:ss)."
        )

    except FileNotFoundError as e:
        logger.error(f"File or directory not found: {e}")
    except NotADirectoryError as e:
        logger.error(f"Invalid directory path: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {e}", exc_info=True)


async def preprocessing_pipeline_async():
    """
    Asynchronous preprocessing pipeline to handle multiple yearbook directories.

    Processes 2012 and 2022 yearbook data, combines the results, and
    saves the aggregated data.

    Steps:
        1. Process yearbook 2012 data.
        2. Process yearbook 2022 data.
        3. Combine the two processed files into a single output.

    Returns:
        None
    """
    start_time = time.time()  # Record start time
    logger.info("Start preprocessing pipeline.")

    # Set yearbook source values
    yearbook_source_2012 = "2012"
    yearbook_source_2022 = "2022"

    # Step 1. Process yearbook 2012 data
    logger.info("Processing 2012 yearbook data...")
    await preprocess_files_in_directory_async(
        source_data_dir=YEARBOOK_2012_DATA_DIR,
        output_csv_path=preprocessed_2012_data_file,
        yearbook_source=yearbook_source_2012,
    )

    # Step 2. Process yearbook 2022 data
    logger.info("Processing 2022 yearbook data...")
    await preprocess_files_in_directory_async(
        source_data_dir=YEARBOOK_2022_DATA_DIR,
        output_csv_path=preprocessed_2022_data_file,
        yearbook_source=yearbook_source_2022,
    )

    # Step 3. Combine the two files
    try:
        logger.info("Combining processed files.")
        df_2012 = pd.read_csv(preprocessed_2012_data_file)
        df_2022 = pd.read_csv(preprocessed_2022_data_file)

        # Concatenate DataFrames
        combined_df = pd.concat([df_2012, df_2022], ignore_index=True)

        # Save combined data to CSV
        combined_df.to_csv(preprocessed_all_data_file, index=False)
        logger.info(
            f"Combined data saved to {preprocessed_all_data_file}. Total rows: {len(combined_df)}"
        )
    except Exception as e:
        logger.error(f"Error combining files: {e}", exc_info=True)

    # Log total pipeline time
    end_time = time.time()  # Record end time
    elapsed_time_seconds = end_time - start_time
    elapsed_time_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_seconds))

    logger.info(
        f"Finished preprocessing pipeline. Total time taken: {elapsed_time_hms} (hh:mm:ss)."
    )
