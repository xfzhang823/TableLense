"""
File: post_inference_data_cleaning.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Aug 8

Description: Further data cleansing and manual data labeling on data after inference.
"""

from pathlib import Path
import logging
from typing import Union
import pandas as pd
from pandas.api.types import is_numeric_dtype

# User defined
from utils.file_encoding_detector import detect_encoding
from data_processing.data_processing_utils import clean_text, is_all_empty
from project_config import NUM_TO_LABEL

# Set up logger
logger = logging.getLogger(__name__)


def add_new_column(df: pd.DataFrame, new_col_name: str) -> pd.DataFrame:
    """
    Add a new column with an empty string if it does not already exist.
    """
    if new_col_name not in df.columns:
        df[new_col_name] = ""
        logger.info(f"New column '{new_col_name}' is created.")
    else:
        logger.warning(
            f"Column '{new_col_name}' already exists; skipping adding new column."
        )
    return df


def convert_numeric_to_text_labels(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Convert numeric labels in the specified column of the DataFrame to textual label values.
    The conversion is based on the NUM_TO_LABEL dictionary defined in project_config.

    This function checks that the column exists, that it is numeric, and that all non-null
    values are integer-like.
    If the column is not numeric or does not contain all integer values, a ValueError is raised.

    Args:
        - df (pd.DataFrame): The DataFrame containing the label column.
        - column_name (str): The name of the column to convert.

    Returns:
        pd.DataFrame: The DataFrame with numeric labels replaced by their corresponding
        text labels.

    Raises:
        ValueError: If the column is not numeric or if not all non-null values are
        integer-like.
    """
    if column_name not in df.columns:
        logger.error(f"Column '{column_name}' not found in DataFrame.")
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[column_name]):
        logger.error(f"Column '{column_name}' is not numeric; conversion aborted.")
        raise ValueError(f"Column '{column_name}' is not numeric; conversion aborted.")

    # Check if all non-null values are integer-like
    non_null = df[column_name].dropna()
    if not ((non_null % 1 == 0).all()):  # check if it's partial integer
        logger.error(
            f"Column '{column_name}' does not contain all integer values; conversion aborted."
        )
        raise ValueError(
            f"Column '{column_name}' does not contain all integer values; conversion aborted."
        )

    try:
        df[column_name] = df[column_name].map(NUM_TO_LABEL)
        logger.info(
            f"Converted numeric labels in column '{column_name}' to text labels."
        )
    except Exception as e:
        logger.error(f"Error converting numeric labels in column '{column_name}': {e}")
        raise

    return df


def change_column_name(
    df: pd.DataFrame, old_col_name: str, new_col_name: str
) -> pd.DataFrame:
    """
    Renames a column from old_col_name to new_col_name if it exists; otherwise logs a warning.
    """
    if old_col_name in df.columns:
        df.rename(columns={old_col_name: new_col_name}, inplace=True)
        logger.info(f"Renamed column '{old_col_name}' to '{new_col_name}'.")
    else:
        logger.warning(f"Column '{old_col_name}' not found; skipping rename.")
    return df


def label_empty_rows(
    df: pd.DataFrame, is_empty_col: str, final_label: str
) -> pd.DataFrame:
    """
    Update the final label column based on text content and the emptiness flag.
    For rows where the text is truly empty (as determined by is_all_empty) or where
    the specified is_empty column equals "yes", set the final label to "empty".
    Otherwise, leave the final label unchanged.
    """
    logger.info("Starting label_empty_rows function.")

    df[final_label] = df.apply(
        lambda row: "empty" if is_all_empty(row["text"]) else row[final_label],
        axis=1,
    )
    logger.info("Applied empty row labeling based on text content in 'text' column.")

    df.loc[df[is_empty_col] == "yes", final_label] = "empty"
    logger.info(
        f"Forced '{final_label}' to 'empty' for rows where '{is_empty_col}' equals 'yes'."
    )

    logger.info("Completed label_empty_rows function.")
    return df


def label_titles(
    df: pd.DataFrame, title_flag_col: str, final_label_col: str
) -> pd.DataFrame:
    """
    Ensure that the final label column is set to "title" for rows where the title flag column equals "yes".
    """
    logger.info("Starting label_title function.")
    df.loc[df[title_flag_col] == "yes", final_label_col] = "title"
    logger.info(
        f"Set '{final_label_col}' to 'title' for rows where '{title_flag_col}' is 'yes'."
    )
    logger.info("Completed label_title function.")
    return df


def merge_labels_based_on_type(
    df: pd.DataFrame,
    actual_label_col: str,
    predicted_label_col: str,
    final_label_col: str,
    label_type_col: str,
) -> pd.DataFrame:
    """
    Create a final label column by choosing between the actual and predicted label
    based on the value in the label type column. If the label_type is "actual", the final
    label is taken from the actual label column; if it is "predicted", it is taken from the
    predicted label column.

    If the predicted label column is missing, the final label is set from the actual label.
    Additionally, if the predicted label column is already in text form (value labels),
    the conversion is skipped.

    Args:
        - df (pd.DataFrame): DataFrame containing the label columns.
        - actual_col (str): Column name for the actual (ground truth) label.
        - predicted_col (str): Column name for the predicted label.
        - label_type_col (str): Column name indicating the label type ("actual" or "predicted").
        - final_label_col (str): Column name for the final merged label.

    Returns:
        pd.DataFrame: The DataFrame with the final label column created.
    """
    logger.info("Starting merge_labels_based_on_type function.")

    if predicted_label_col in df.columns:
        # Check if the predicted column is numeric; if so, convert it.
        if is_numeric_dtype(
            df[predicted_label_col]
        ):  # double check if predicted label -> values already
            df = convert_numeric_to_text_labels(df, predicted_label_col)
            logger.info(
                f"Converted numeric labels in column '{predicted_label_col}' to text labels."
            )
        else:
            logger.info(
                f"Column '{predicted_label_col}' is already in text format; skipping conversion."
            )
    else:
        logger.warning(
            f"Predicted label column '{predicted_label_col}' not found; using actual labels for '{final_label_col}'."
        )
        df[final_label_col] = df[actual_label_col]
        logger.info("Completed merge_labels_based_on_type function.")
        return df

    def choose_label(row):
        if str(row[label_type_col]).lower() == "actual":
            return row[actual_label_col]
        else:
            return row[predicted_label_col]

    df[final_label_col] = df.apply(choose_label, axis=1)
    logger.info(
        f"Final label column '{final_label_col}' created based on '{label_type_col}'."
    )
    logger.info("Completed merge_labels_based_on_type function.")
    return df


def read_csv_file(input_path: str) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame using detected encoding.
    """
    try:
        encoding, confidence = detect_encoding(input_path)
        logger.info(f"Detected encoding: {encoding} with confidence {confidence}")
        df = pd.read_csv(input_path, encoding=encoding, header=0)
        logger.info(f"File read successfully from {input_path}")
        return df
    except Exception as e:
        logger.error(f"Error reading {input_path}: {e}")
        raise


def reclassify_header(df: pd.DataFrame, final_label_col: str) -> pd.DataFrame:
    """
    Reclassify rows labeled as "header" if they are sandwiched between "table_data" rows within the same group.
    """
    for i in range(1, len(df) - 1):
        if df.at[i, final_label_col] == "header":
            if (
                df.at[i - 1, final_label_col] == "table_data"
                and df.at[i + 1, final_label_col] == "table_data"
                and df.at[i - 1, "group"] == df.at[i, "group"]
                and df.at[i + 1, "group"] == df.at[i, "group"]
            ):
                df.at[i, final_label_col] = "table_data"
    return df


def reclassify_metadata(df: pd.DataFrame, final_label_col: str) -> pd.DataFrame:
    """
    Reclassify rows labeled as "metadata" based on their neighbors in the final label column.
    """
    for i in range(1, len(df) - 1):
        if df.at[i, final_label_col] == "metadata":
            if (
                df.at[i - 1, final_label_col] == "header"
                and df.at[i + 1, final_label_col] == "header"
            ):
                df.at[i, final_label_col] = "header"
            elif (
                df.at[i - 1, final_label_col] == "table_data"
                and df.at[i + 1, final_label_col] == "table_data"
            ):
                df.at[i, final_label_col] = "table_data"
    logger.info("Metadata checked & reclassified.")
    return df


def reclassify_empty(df: pd.DataFrame, final_label_col: str) -> pd.DataFrame:
    """
    Reclassify rows labeled as "empty" but not truly empty by adopting the label of the row above,
    or defaulting to "metadata" if no valid neighbor is found.
    """
    for i in range(1, len(df) - 1):
        if df.at[i, final_label_col] == "empty" and not is_all_empty(df.at[i, "text"]):
            label_above = df.at[i - 1, final_label_col]
            label_below = df.at[i + 1, final_label_col]
            default_row = "metadata"
            if label_above in ["header", "metadata", "table_data"]:
                df.at[i, final_label_col] = label_above
            elif label_below in ["header", "metadata", "table_data"]:
                df.at[i, final_label_col] = label_below
            else:
                df.at[i, final_label_col] = default_row
    return df


def set_label_type(
    df: pd.DataFrame, actual_col: str, label_type_col: str
) -> pd.DataFrame:
    """
    Set the label type based on the value in the actual label column.
    If the actual label is exactly "unlabeled" (case-insensitive),
    the label type is set to "predicted"; otherwise, it is set to "actual".

    Args:
        - df (pd.DataFrame): The DataFrame containing the actual label column.
        - actual_col (str): The name of the column with the actual label.
        - label_type_col (str): The name of the column to store the label type.

    Returns:
        pd.DataFrame: The updated DataFrame with the label type column.
    """
    logger.info("Starting set_label_type function.")
    try:
        df[label_type_col] = df[actual_col].apply(
            lambda x: "predicted" if str(x).lower() == "unlabeled" else "actual"
        )
        logger.info(
            f"Set '{label_type_col}' to 'predicted' for unlabeled rows, else 'actual'."
        )
    except Exception as e:
        logger.error(f"Error in set_label_type: {e}")
        raise
    logger.info("Completed set_label_type function.")
    return df


def process_inference_training_results(
    input_file_path: Union[Path, str], output_file_path: Union[Path, str]
) -> None:
    """
    Process the data by merging label columns, updating label types, reclassifying rows,
    cleaning text, sorting, and finally saving the cleaned DataFrame.

    This function:
      1. Reads the CSV file with detected encoding.
      2. Converts numeric predicted labels to text (if the column exists).
      3. Merges label columns: renames 'label' to 'actual_label', creates
      a new 'label' column,
         and sets the final label based on the label type.
      4. Sorts the DataFrame.
      5. Cleans text in all cells.
      6. Reclassifies rows based on emptiness, title flags, metadata, and
        header criteria.
      7. Saves the final cleaned DataFrame to disk.

    Args:
        input_path (Union[Path, str]): Path to the input CSV file.
        output_path (Union[Path, str]): Path to save the cleaned CSV file.

    Raises:
        Exception: Propagates any exceptions encountered during processing.
    """
    logger.info(f"Start processing file ({input_file_path})...")

    # Ensure Path objects
    input_file_path, output_file_path = Path(input_file_path), Path(output_file_path)

    try:
        # Step 0. Read data
        df = read_csv_file(input_file_path)
        logger.info(f"DataFrame loaded:\n{df.head(5)}")

        # Step 1. Convert numeric predicted labels to text if available
        if "predicted_label" in df.columns:
            df = convert_numeric_to_text_labels(df, "predicted_label")
            logger.info("Converted predicted labels from numeric to text.")
        else:
            logger.info("Predicted label column not found; skipping conversion.")

        # Step 2. Merge labels:
        # * 2a. Rename 'label' to 'actual_label' and create a new 'label' column
        df = change_column_name(df, "label", "actual_label")
        df = add_new_column(df, "label")
        df["label"] = df["actual_label"]
        logger.info(
            "Merged actual labels; final 'label' initialized from 'actual_label'."
        )

        # * 2b. Set and merge label type (actual vs. predicted)
        df = add_new_column(df, "label_type")
        df = set_label_type(df, "actual_label", "label_type")
        df = merge_labels_based_on_type(
            df,
            actual_label_col="actual_label",
            predicted_label_col="predicted_label",
            final_label_col="label",
            label_type_col="label_type",
        )

        # Step 3. Sort DataFrame
        df = df.sort_values(by=["yearbook_source", "group", "row_id"], ascending=True)
        logger.info("DataFrame sorted.")

        # Step 4. Clean text in all cells
        df = df.applymap(clean_text)
        logger.info("Text cleaned.")

        # Step 5. Reclassify rows:
        df = label_empty_rows(df, is_empty_col="is_empty", final_label="label")
        logger.info("Empty rows updated.")
        df = label_titles(df, title_flag_col="is_title", final_label_col="label")
        logger.info("Title rows updated.")
        df = reclassify_empty(df, final_label_col="label")
        df = reclassify_metadata(df, final_label_col="label")
        df = reclassify_header(df, final_label_col="label")
        logger.info("Reclassification of empty, metadata, and header rows completed.")

        # Step 6. Save cleaned DataFrame
        df.to_csv(output_file_path, encoding="utf-8", index=False)
        logger.info(f"Cleaned data saved to {output_file_path}")

    except Exception as e:
        logger.error(f"Error in clean_and_relabel_data: {e}")
        raise


def main():
    """
    Main function to clean and relabel data.
    """
    input_f_path = "your file path"
    output_f_path = "your file path"
    process_inference_training_results(input_f_path, output_f_path)


if __name__ == "__main__":
    main()
