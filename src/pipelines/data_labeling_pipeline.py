"""
File: data_labeling_pipeline
Author: Xiao-Fei Zhang
Last Updated: 2025 Jan
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
    """Run pipeline"""
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
