""" "
Filename: reconstruct_table.py
Author: Xiao-Fei Zhang

Functions to deserialize final inference + training data to normal table format.

#* These functions themselves are pure CPU-bound functions that work on in-memory data
#* (strings, lists, arrays) and don't involve I/O or blocking calls. Therefore no need
#* to declare as async.

"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

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
    logger.debug(f"Deserializing text: {text}")
    if isinstance(text, str):
        result = [item.strip() for item in text.split(delimiter)]
        logger.debug(f"Deserialized result: {result}")
        return result
    logger.warning("Provided text is not a string; returning empty list.")
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
    logger.info("Splitting and padding rows.")
    if len(rows) == 0:
        logger.error("No rows to split and pad.")
        raise ValueError("No rows to split and pad")

    # Split each row into list of elements and strip whitespace.
    rows_split: List[List[str]] = [row.split(delimiter) for row in rows]
    rows_split = [[element.strip() for element in row] for row in rows_split]
    logger.debug(f"Rows after split and strip: {rows_split}")

    # Determine the maximum number of elements in any row.
    max_length: int = max(len(row) for row in rows_split)
    logger.info(f"Maximum row length determined: {max_length}")

    # Pad each row with empty strings to reach the maximum length.
    padded_rows: List[List[str]] = [
        row + [""] * (max_length - len(row)) for row in rows_split
    ]
    logger.debug(f"Padded rows: {padded_rows}")

    return np.array(padded_rows, dtype=object)


def merge_rows(array: np.ndarray) -> np.ndarray:
    """
    Merge multiple header rows (given as a 2D numpy array) into a single header row.
    For each column, unique non-empty values (excluding any value that is "empty"
    regardless of case)
    are joined with a space.

    Args:
        array (np.ndarray): A 2D numpy array where each row represents
        a header row.

    Returns:
        np.ndarray: A 1D numpy array representing the merged header row.
    """
    logger.info("Merging header rows.")
    merged_row: List[str] = []

    for col in range(array.shape[1]):
        seen: set = set()
        unique_values: List[str] = []
        for value in array[:, col]:
            # Exclude empty strings and any variation of "empty"
            if (
                value
                and value.strip()
                and value.lower() != "empty"
                and value not in seen
            ):
                unique_values.append(value)
                seen.add(value)
        col_merged = " ".join(unique_values)
        logger.debug(f"Column {col}: merged value = '{col_merged}'")
        merged_row.append(col_merged)

    logger.info(f"Merged header row: {merged_row}")
    return np.array(merged_row)


def reconstruct_objects_from_group(
    group_df: pd.DataFrame, delimiter: str = ","
) -> Dict[str, Any]:
    """
    Reconstructs a single table group into three separate objects: title, metadata, and table data.
    The input DataFrame must include at least the columns "row_id", "label", and "text".

    The function performs these steps:
      1. Sort rows by 'row_id'.
      2. Deserialize the "text" column into lists of cells.
      3. Separates rows into title, metadata, header, and table data.
      4. Merges header rows into one header using merge_rows.
      5. Processes table data rows, ensuring they match header length.
      6. Returns a dictionary with keys "title", "metadata", and "table".
         If no table data is available, logs an error and includes an "error" key.

    Args:
        group_df (pd.DataFrame): DataFrame containing rows for a single table group.
            Must include columns "row_id", "label", and "text".
        delimiter (str): Delimiter used in the flattened text. Default is a comma.

    Returns:
        Dict[str, Any]: A dictionary with:
            - "title": Merged title string.
            - "metadata": Merged metadata string.
            - "table": DataFrame representing the main table data, or
            None if table data is missing.
            - "error": (Optional) Error message if reconstruction failed.
    """
    logger.info("Reconstructing objects from group.")
    group_df = group_df.sort_values(by="row_id").reset_index(drop=True)
    logger.debug(f"Sorted group dataframe: {group_df[['row_id', 'label']].head()}")

    group_df["cells"] = group_df["text"].apply(
        lambda x: [
            cell if cell != "EMPTY" else "" for cell in deserialize_text(x, delimiter)
        ]
    )
    logger.debug("Deserialized text into cells.")

    title_rows: List[List[str]] = group_df[group_df["label"] == "title"][
        "cells"
    ].tolist()
    metadata_rows: List[List[str]] = group_df[group_df["label"] == "metadata"][
        "cells"
    ].tolist()
    header_rows: List[List[str]] = group_df[group_df["label"] == "header"][
        "cells"
    ].tolist()
    data_rows: List[List[str]] = group_df[group_df["label"] == "table_data"][
        "cells"
    ].tolist()

    logger.info(
        f"Found {len(title_rows)} title rows, {len(metadata_rows)} metadata rows, "
        f"{len(header_rows)} header rows, and {len(data_rows)} table data rows."
    )

    title_str: str = " ".join(" ".join(row) for row in title_rows) if title_rows else ""
    metadata_str: str = (
        " ".join(" ".join(row) for row in metadata_rows) if metadata_rows else ""
    )

    if header_rows:
        header_strings: List[str] = [", ".join(row) for row in header_rows]
        logger.debug(f"Header strings: {header_strings}")
        header_array: np.ndarray = split_and_pad_rows(header_strings, delimiter)
        header_row: List[str] = list(merge_rows(header_array))
    else:
        header_row = (
            ["Column_" + str(i + 1) for i in range(len(data_rows[0]))]
            if data_rows
            else []
        )
        logger.info("No header rows found; using default column names.")

    if data_rows:
        data_strings: List[str] = [", ".join(row) for row in data_rows]
        data_array: np.ndarray = split_and_pad_rows(data_strings, delimiter)
    else:
        data_array = np.array([])
        logger.warning("No table data rows found.")

    # If no table data, log error and return a record indicating failure.
    if data_array.size == 0:
        error_msg = "No table data available for this group."
        logger.error(error_msg)
        return {
            "title": title_str,
            "metadata": metadata_str,
            "table": None,
            "error": error_msg,
        }

    # Ensure header and data arrays have the same number of columns.
    n_data_cols: int = data_array.shape[1]
    n_header_cols: int = len(header_row)
    logger.info(f"Header columns: {n_header_cols}, Data columns: {n_data_cols}")
    if n_header_cols < n_data_cols:
        header_row = header_row + [""] * (n_data_cols - n_header_cols)
        logger.debug("Padded header row to match data columns.")
    elif n_data_cols < n_header_cols:
        data_array = np.pad(
            data_array, ((0, 0), (0, n_header_cols - n_data_cols)), constant_values=""
        )
        logger.debug("Padded data array to match header columns.")

    table_df: pd.DataFrame = pd.DataFrame(data_array, columns=header_row)

    logger.info("Group reconstruction complete.")
    return {"title": title_str, "metadata": metadata_str, "table": table_df}


# def reconstruct_table_from_group(
#     group_df: pd.DataFrame, delimiter: str = ","
# ) -> pd.DataFrame:
#     """
#     Reconstructs a single table from a grouped DataFrame.
#     The input DataFrame must include at least the columns "row_id", "label", and "text".

#     It performs these steps:
#       1. Sorts rows by 'row_id'.
#       2. Deserializes the "text" column into lists of cells.
#       3. Separates header rows and table data rows.
#       4. Merges header rows into one header using merge_rows.
#       5. Ensures that the header and data rows have the same number of columns.
#       6. Returns a DataFrame with the reconstructed table.

#     Args:
#         group_df (pd.DataFrame): DataFrame containing rows for a single table group.
#             Must include columns "row_id", "label", and "text".
#         delimiter (str): Delimiter used in the flattened text. Default is a comma.

#     Returns:
#         pd.DataFrame: A DataFrame representing the reconstructed table.
#     """
#     logger.info("Reconstructing table from group.")

#     # Sort by 'row_id' to preserve original order.
#     group_df = group_df.sort_values(by="row_id").reset_index(drop=True)
#     logger.debug(f"Sorted group dataframe: {group_df[['row_id', 'label']].head()}")

#     # Deserialize text column, replacing "EMPTY" with an empty string.
#     group_df["cells"] = group_df["text"].apply(
#         lambda x: [
#             cell if cell != "EMPTY" else "" for cell in deserialize_text(x, delimiter)
#         ]
#     )
#     logger.debug("Deserialized text into cells.")

#     # Extract header rows and table data rows.
#     header_lists: List[List[str]] = group_df[group_df["label"] == "header"][
#         "cells"
#     ].tolist()
#     data_lists: List[List[str]] = group_df[group_df["label"] == "table_data"][
#         "cells"
#     ].tolist()
#     logger.info(
#         f"Found {len(header_lists)} header rows and {len(data_lists)} data rows."
#     )

#     # Process header rows.
#     if header_lists:
#         header_strings: List[str] = [", ".join(row) for row in header_lists]
#         logger.debug(f"Header strings: {header_strings}")
#         header_array: np.ndarray = split_and_pad_rows(header_strings, delimiter)
#         header_row: np.ndarray = merge_rows(header_array)
#     else:
#         header_row = (
#             ["Column_" + str(i + 1) for i in range(len(data_lists[0]))]
#             if data_lists
#             else []
#         )
#         logger.info("No header rows found; using default column names.")

#     # Process data rows.
#     if data_lists:
#         data_strings: List[str] = [", ".join(row) for row in data_lists]
#         data_array: np.ndarray = split_and_pad_rows(data_strings, delimiter)
#     else:
#         data_array = np.array([])
#         logger.warning("No table data rows found.")

#     # Ensure header and data arrays have the same number of columns.
#     if data_array.size != 0:
#         n_data_cols: int = data_array.shape[1]
#         n_header_cols: int = len(header_row)
#         logger.info(f"Header columns: {n_header_cols}, Data columns: {n_data_cols}")
#         if n_header_cols < n_data_cols:
#             header_row = list(header_row) + [""] * (n_data_cols - n_header_cols)
#             logger.debug("Padded header row to match data columns.")
#         elif n_data_cols < n_header_cols:
#             data_array = np.pad(
#                 data_array,
#                 ((0, 0), (0, n_header_cols - n_data_cols)),
#                 constant_values="",
#             )
#             logger.debug("Padded data array to match header columns.")

#     table_df: pd.DataFrame = pd.DataFrame(data_array, columns=header_row)
#     logger.info("Table reconstruction complete.")
#     return table_df
