"""
File: preprocess_missing_files.py
Author: Xiao-Fei Zhang
Last Updated: 2024 Dec

Description:
    This module orchestrates the asynchronous preprocessing of Excel files for
    datasets like the 2012 and 2022 yearbook datasets. It handles:
    - Missing file detection and iterative processing,
    - Directory-level validation and filtering,
    - Aggregation of results into structured CSV files.

    The pipeline ensures that all files are processed, even if interrupted, by
    iteratively checking for missing files and appending processed data.

Key Features:
    - Asynchronous processing for scalability and efficiency.
    - Modular and reusable components for handling missing files.
    - Validation to ensure headers match when appending data.
    - Detailed logging to track pipeline progress and issues.

Workflow:
    1. Detect missing files in the directory.
    2. Process all missing files asynchronously in batches.
    3. Append processed data to the preprocessed dataset.
    4. Repeat steps 1-3 until no missing files remain.

Functions:
    - `get_missing_files`: Identifies files that haven't been processed yet.
    - `preprocess_missing_files`: Processes missing files asynchronously and saves the result.
    - `add_preprocessed_missing_data`: Appends newly processed data to the main 
    preprocessed dataset.
    - `preprocess_yearbook_pipeline`: Orchestrates the preprocessing of a single dataset, 
    ensuring all files are processed.

Usage:
    Call `preprocess_yearbook_pipeline` for individual datasets like "2012" and "2022."
    Example:
        asyncio.run(preprocess_yearbook_pipeline(...))

Example:
    $ python preprocess_missing_files.py

Dependencies:
    - pandas: For data aggregation and file I/O.
    - asyncio: For handling asynchronous file processing.
    - logging: For detailed progress and error tracking.
    - Custom modules: `preprocess_data_async`, `preprocessing_utils`, and CSV utilities.

"""

from pathlib import Path
import logging
from typing import Callable, Union
from itertools import islice
import tempfile
import pandas as pd
import asyncio
from data_processing.preprocess_data_async import process_multiple_excel_files_async
from data_processing.data_processing_utils import (
    get_filtered_files,
    have_same_headers,
    concatenate_tabular_data_files,
    append_tabular_data_files,
)
from utils.file_encoding_detector import detect_encoding
import logging_config

# Config logger

# Configure logger
logger = logging.getLogger(__name__)


def get_missing_files(
    files_to_check_against: list[Path], output_file_to_check: Path
) -> list[Path]:
    """
    Identify missing files by comparing a list of files to check against a dataset of
    processed files.

    Args:
        - files_to_check_against (list[Path]): List of file paths to check for missing files.
        - data_output_file_to_check (Path): Path to the CSV file containing the dataset of
        processed files.

    Returns:
        list[Path]: A list of file paths that are missing from the processed dataset.

    Raises:
        FileNotFoundError: If the processed data CSV file does not exist.
        KeyError: If the "group" column is missing in the processed data.
    """
    logger.info("Trying to find missing files...")

    if not output_file_to_check.exists():
        logger.warning(
            f"Processed data CSV file {output_file_to_check} does not exist. Assuming all files are missing."
        )
        return files_to_check_against

    try:
        # Load processed data
        encoding, _ = detect_encoding(output_file_to_check)
        df_processed = pd.read_csv(output_file_to_check, encoding=encoding, header=0)
        processed_file_names = set(
            df_processed["group"].unique()
        )  # Use a set for faster lookups

        logger.info(
            f"processed_file_names (first 10): {list(islice(processed_file_names, 10))}"
        )

        # Identify missing files
        missing_file_paths = [
            path
            for path in files_to_check_against
            if path.stem not in processed_file_names
        ]

        logger.info(f"Identified {len(missing_file_paths)} missing files.")

        # TODO: debugging; delete later
        logger.info(f"Missing files (first 5): {missing_file_paths[:5]}")

        return missing_file_paths

    except KeyError as e:
        logger.error(f"Column 'group' not found in the processed data CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while processing the data: {e}")
        raise


async def preprocess_missing_files_async(
    source_data_dir: Path,
    processed_data_file: Path,
    yearbook_source: str,
    filter_function: Callable,
    max_concurrent_tasks: int = 8,
    timeout: int = 800,
):
    """
    Asynchronously preprocess missing Excel files and append the data to the preprocessed file.

    This function identifies files in the source directory that have not yet been processed
    (tracked in the `processed_data_file`), processes those missing files asynchronously,
    saves their content to a temporary file, and appends the temporary file's content to
    the aggregate preprocessed data file (`processed_data_file`).

    File Inputs and Outputs:
        - 'source_data_dir': The directory containing all Excel files to check for missing files.
        - 'processed_data_file': The main aggregate CSV file that tracks all preprocessed data.
          Newly processed missing files are appended to this file.
        - Temporary File: A temporary CSV file is created during processing to store the
          output of the missing files before appending to `processed_data_file`.
          This file is automatically created and then deleted after appending using temp file lib
          (no need to manually create or input)

    Args:
        - source_data_dir (Path): Directory containing Excel files to check for missing files.
        - processed_data_file (Path): Path to the CSV file that stores the aggregate preprocessed data.
          Missing file content is appended to this file after processing.
        - yearbook_source (str): Identifier for the yearbook (e.g., "2012" or "2022").
        - filter_function (Callable[[str], bool]): A function to filter relevant files based on
        naming criteria.
        - max_concurrent_tasks (int): Maximum number of concurrent tasks for processing files.
        - timeout (int): Timeout in seconds for processing each file.

    Returns:
        None

    Usage:
        Use as part of a pipeline to process and aggregate missing Excel data. The function identifies
        missing files, processes them asynchronously, and appends their content to the preprocessed file.

    Example:
        await preprocess_missing_files_async(
            source_data_dir=Path("/path/to/source/data"),
            processed_data_file=Path("/path/to/processed_data.csv"),
            yearbook_source="2012",
            filter_function=lambda name: name.lower().endswith("e"),
            max_concurrent_tasks=10,
            timeout=800,
        )
    """

    logger.info(f"Starting preprocessing for missing files in {source_data_dir}")

    # From the src file dir, get list of paths of files need to be processed (preprocessed)
    file_paths = get_filtered_files(
        source_data_dir=source_data_dir, filter_criterion=filter_function
    )

    # Get the list of paths of src data files need to process (missing files)
    missing_file_paths = get_missing_files(
        files_to_check_against=file_paths,
        output_file_to_check=processed_data_file,
    )
    logger.debug(f"Total missing files: {len(missing_file_paths)}")

    if not missing_file_paths:
        logger.warning("No missing files to process.")
        return  # If no missing files, then early return

    # Process missing files directly
    await process_multiple_excel_files_async(
        source_data_dir=None,  # We pass None because we are providing specific files
        file_paths=missing_file_paths,
        output_csv_file=processed_data_file,  # Append directly to processed_data_file
        yearbook_source=yearbook_source,
        max_concurrent_tasks=max_concurrent_tasks,
        timeout=timeout,
    )

    logger.info(f"Finished processing missing files for {yearbook_source}.")
