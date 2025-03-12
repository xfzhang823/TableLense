"""
File: data_labeling_pipeline
Author: Xiao-Fei Zhang
Last Updated: 2025 Jan

Description:
    This module defines the data labeling pipeline for preparing training and inference datasets.
    The pipeline reads preprocessed data and a section-group mapping file, merges them, and performs
    several transformation steps to label and structure the data appropriately for downstream tasks.
    The key steps include:
        1. Validating file paths.
        2. Reading the main preprocessed data and mapping files.
        3. Merging the datasets using a VLOOKUP-like operation based on "group" and "yearbook_source".
        4. Adding auxiliary columns such as 'is_empty', 'is_title', and 'original_index'.
        5. Labeling rows based on the presence of title or empty text.
        6. Re-arranging the DataFrame column order.
        7. Saving the final labeled dataset to a CSV file.

Usage:
    To run the entire pipeline, call the function run_data_labeling_pipeline(), which
    checks for the existence of the output file and, if absent, invokes the data_labeling_pipeline()
    function to generate and save the labeled data.
"""

# Dependencies

# From internal / external
from pathlib import Path
from typing import Union
import logging
import pandas as pd

# From project modules
import logging_config
from utils.read_csv_file import read_csv_file
from data_processing.data_processing_utils import (
    add_is_empty_column,
    add_is_title_column,
    clean_text,
)
from project_config import (
    PREPROCESSED_ALL_DATA_FILE,
    TRAINING_INFERENCE_DATA_FILE,
    SECTION_GROUP_MAPPING_FILE,
)

# Setup logger
logger = logging.getLogger()


def data_labeling_pipeline(
    preprocessed_data_file: Path,
    training_inference_data_file: Path,
    section_group_mapping_file: Path,
) -> None:
    """
    Execute the data labeling pipeline.

    This function processes and labels the preprocessed dataset by:
        1. Validating that the input files (preprocessed data and section mapping) exist.
        2. Reading the main preprocessed CSV file using a custom CSV reader to
        handle encoding issues.
        3. Reading the section-group mapping Excel file and merging it with
        the preprocessed data based on shared columns ("group" and "yearbook_source") to
        add section information.
        4. Adding an 'is_empty' column that flags rows with empty or missing text values.
        5. Adding an 'is_title' column that flags the first row of each group as a title.
        6. Creating an 'original_index' column to preserve the original DataFrame index,
        which is useful during training and inference for mapping purposes.
        7. Labeling each row:
            - Default label: "unlabeled"
            - Rows flagged as title are labeled "title"
            - Rows flagged as empty are labeled "empty"
        8. Re-arranging the DataFrame columns into a specified order for consistency.
        9. Saving the final labeled DataFrame as a CSV file to the specified output path.

    Args:
        - preprocessed_data_file (Path): Path to the CSV file containing preprocessed data.
        - training_inference_data_file (Path): Output path for the labeled training/inference
        CSV file.
        - section_group_mapping_file (Path): Path to the Excel file containing section-group
        mapping information.

    Raises:
        FileNotFoundError: If either the preprocessed data file or the section mapping file
        does not exist or is not a file.
    """
    # Step 0. Validate file paths
    for file_path in [preprocessed_data_file, section_group_mapping_file]:
        path = Path(file_path) if isinstance(file_path, str) else file_path
        if not (path.exists() and path.is_file()):
            logger.debug()
            raise FileNotFoundError(f"The file does not exist or is not a file: {path}")

    logger.info("Both data and mapping files exist and are files.")

    # Step 1. Read files

    # Main data file
    df_processed_data = read_csv_file(
        preprocessed_data_file
    )  # Use custom read csv func to cope with encoding issues

    # Step 2. Add section column
    # Read mapping file
    df_mapping = pd.read_excel(SECTION_GROUP_MAPPING_FILE)

    # Add Merge (VLOOKUP equivalent)
    print(df_mapping.columns)  # todo: debugging; delete later
    print(df_processed_data.columns)  # todo: debugging; delete later

    df = pd.merge(
        df_processed_data, df_mapping, on=["group", "yearbook_source"], how="left"
    )
    logger.info(f"df columns after the merge: {df.columns}")

    # Step 3. Add is_empty
    df = add_is_empty_column(df)
    logger.info("Added is_empty column.")

    # Step 4. Add is_title
    df = add_is_title_column(df)
    logger.info("Added is_title column.")

    # Step 5. Add "original_index" (helps with training/inference)
    df["original_index"] = df.index  # Preserve the original DataFrame index as a column

    # Step 6. Label rows based on text content
    df["label"] = "unlabeled"
    df.loc[df["is_title"] == "yes", "label"] = "title"
    df.loc[df["is_empty"] == "yes", "label"] = "empty"
    logger.info("Added label column.")

    # Step 7. Re-arrange column order
    logger.info(f"df columns before re-arrangement: {df.columns}")
    df = df[
        [
            "text",
            "yearbook_source",
            "section",
            "group",
            "row_id",
            "is_empty",
            "is_title",
            "original_index",
            "label",
        ]
    ]
    logger.info("Re-arranged df column order.")

    # Step 5. Save labeled data
    output_file_path = training_inference_data_file
    df.to_csv(output_file_path, index=False)

    # TODO: Not very sure if these rules always apply; keep it out for now
    # df.loc[df["text"].str.contains(r"\(unit\)", case=False, na=False), "label"] = (
    #     "metadata"
    # )
    # df.loc[
    #     df["text"].str.contains(r"Provinces|Regions|District", case=False, na=False),
    #     "label",
    # ] = "header"
    # df.loc[df["text"].str.match(r".*\d+.*", na=False), "label"] = "table_data"


def run_data_labeling_pipeline():
    """
    Run the data labeling pipeline if the labeled training/inference data file does not
    already exist.

    This function serves as an entry point for the data labeling process.
    It performs the following steps:
        1. Logs the start of the labeling pipeline.
        2. Checks if the training/inference data file (output) already exists:
            - If it exists, logs a message indicating that the pipeline is being skipped.
            - If it does not exist, calls data_labeling_pipeline() with the appropriate
            file paths obtained from the project configuration.
        3. Logs the completion of the data labeling pipeline.

    Returns:
        None
    """
    logger.info(f"Start labeling data pipeline.")

    # Check if the output file exists already: if so, skip
    if TRAINING_INFERENCE_DATA_FILE.exists():
        logger.info(
            "Training and interference data file already exists. Skip pipeline!"
        )
        return  # Early return

    data_labeling_pipeline(
        preprocessed_data_file=PREPROCESSED_ALL_DATA_FILE,
        training_inference_data_file=TRAINING_INFERENCE_DATA_FILE,
        section_group_mapping_file=SECTION_GROUP_MAPPING_FILE,
    )
    logger.info("Finished data labeling pipeline!")
