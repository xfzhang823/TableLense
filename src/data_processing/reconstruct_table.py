""" "
Filename: reconstruct_table.py
Author: Xiao-Fei Zhang

Functions to deserialize final inference + training data to normal table format.

#* These functions themselves are pure CPU-bound functions that work on in-memory data
#* (strings, lists, arrays) and don't involve I/O or blocking calls. Therefore no need
#* to declare as async.

"""

from pathlib import Path
import logging
from typing import List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def deserialize_text(text: str, delimiter: str = ",") -> List[str]:
    """
    Convert a flattened text cell into a list of cell values.

    Args:
        text (str): The flattened string to be deserialized.
        delimiter (str): Delimiter used to split the text. Default is a comma.

    Returns:
        List[str]: A list of cell values with whitespace stripped.
    """
    if isinstance(text, str):
        # Split the text using the given delimiter and remove extra whitespace from each part.
        return [item.strip() for item in text.split(delimiter)]
    return []


def split_and_pad_rows(rows: List[str], delimiter: str = ",") -> np.ndarray:
    """
    Splits a list of comma-delimited strings into a 2D numpy array.
    Each row is split on the delimiter, extra spaces are stripped, and rows are padded to
    the same length.

    Args:
        rows (List[str]): List of flattened row strings.
        delimiter (str): Delimiter used to split each row. Default is a comma.

    Returns:
        np.ndarray: A 2D numpy array of type object where each row has been padded to
        the maximum row length.

    Raises:
        ValueError: If the input list 'rows' is empty.
    """
    if len(rows) == 0:
        raise ValueError("No rows to split and pad")

    # Split each row into a list of elements and strip whitespace from each element.
    rows_split: List[List[str]] = [row.split(delimiter) for row in rows]
    rows_split = [[element.strip() for element in row] for row in rows_split]

    # Determine the maximum number of elements in any row.
    max_length: int = max(len(row) for row in rows_split)

    # Pad each row with empty strings so that all rows have the same length.
    padded_rows: List[List[str]] = [
        row + [""] * (max_length - len(row)) for row in rows_split
    ]

    return np.array(padded_rows, dtype=object)


def merge_rows(array: np.ndarray) -> np.ndarray:
    """
    Merge multiple header rows (given as a 2D numpy array) into a single header row.
    For each column, unique non-empty values (excluding any value that is "empty"
    regardless of case) are joined with a space.

    Args:
        array (np.ndarray): A 2D numpy array where each row represents a header row.

    Returns:
        np.ndarray: A 1D numpy array representing the merged header row.
    """
    merged_row: List[str] = []

    # Iterate through each column in the array.
    for col in range(array.shape[1]):
        seen: set = set()
        unique_values: List[str] = []
        # Loop through each row in the current column.
        for value in array[:, col]:
            # Exclude values that are empty, whitespace only, or the string "empty" (case-insensitive).
            if (
                value
                and value.strip()
                and value.lower() != "empty"
                and value not in seen
            ):
                unique_values.append(value)
                seen.add(value)
        # Join all unique values for this column with a space.
        merged_row.append(" ".join(unique_values))

    return np.array(merged_row)


def reconstruct_table_from_group(
    group_df: pd.DataFrame, delimiter: str = ","
) -> pd.DataFrame:
    """
    Reconstructs a single table from a grouped DataFrame.
    The input DataFrame must include at least the columns "row_id", "label", and "text".

    Steps:
      1. Sorts rows by 'row_id'.
      2. Deserializes the "text" column into lists of cells.
      3. Separates header rows and table data rows.
      4. Merges header rows into one header using merge_rows.
      5. Ensures that the header and data rows have the same number of columns.
      6. Returns a DataFrame with the reconstructed table.

    Args:
        group_df (pd.DataFrame): DataFrame containing rows for a single table group.
            Must include columns "row_id", "label", and "text".
        delimiter (str): Delimiter used in the flattened text. Default is a comma.

    Returns:
        pd.DataFrame: A DataFrame representing the reconstructed table.
    """
    # Sort the group by 'row_id' to preserve the original order.
    group_df = group_df.sort_values(by="row_id").reset_index(drop=True)

    # Apply deserialization on the 'text' column. Replace "EMPTY" with an empty string.
    group_df["cells"] = group_df["text"].apply(
        lambda x: [
            cell if cell != "EMPTY" else "" for cell in deserialize_text(x, delimiter)
        ]
    )

    # Extract header rows and table data rows.
    header_lists: List[List[str]] = group_df[group_df["label"] == "header"][
        "cells"
    ].tolist()
    data_lists: List[List[str]] = group_df[group_df["label"] == "table_data"][
        "cells"
    ].tolist()

    # Process header rows.
    if header_lists:
        # Convert each header row to a flattened string.
        header_strings: List[str] = [", ".join(row) for row in header_lists]
        # Split and pad the header rows to form a 2D array.
        header_array: np.ndarray = split_and_pad_rows(header_strings, delimiter)
        # Merge multiple header rows into a single header row.
        header_row: np.ndarray = merge_rows(header_array)
    else:
        # If no header exists, create default column names based on the first data row.
        header_row = (
            ["Column_" + str(i + 1) for i in range(len(data_lists[0]))]
            if data_lists
            else []
        )

    # Process data rows.
    if data_lists:
        data_strings: List[str] = [", ".join(row) for row in data_lists]
        data_array: np.ndarray = split_and_pad_rows(data_strings, delimiter)
    else:
        data_array = np.array([])

    # Ensure that header and data arrays have the same number of columns.
    if data_array.size != 0:
        n_data_cols: int = data_array.shape[1]
        n_header_cols: int = len(header_row)
        if n_header_cols < n_data_cols:
            header_row = list(header_row) + [""] * (n_data_cols - n_header_cols)
        elif n_data_cols < n_header_cols:
            data_array = np.pad(
                data_array,
                ((0, 0), (0, n_header_cols - n_data_cols)),
                constant_values="",
            )

    # Create a DataFrame using the merged header row as columns.
    table_df: pd.DataFrame = pd.DataFrame(data_array, columns=header_row)
    return table_df
