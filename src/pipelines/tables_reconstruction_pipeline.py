"""
todo: still debugging... almost complete

"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def deserialize_text(text, delimiter=","):
    """
    Convert a flattened text cell into a list of cell values.
    """
    if isinstance(text, str):
        return [item.strip() for item in text.split(delimiter)]
    return []


def split_and_pad_rows(rows):
    """
    Splits a list of comma-delimited strings into a 2D numpy array.
    Each row is split on commas, extra spaces are stripped, and rows are padded to
    the same length.
    """
    if len(rows) == 0:
        raise ValueError("No rows to split and pad")
    rows_split = [row.split(",") for row in rows]
    rows_split = [[element.strip() for element in row] for row in rows_split]
    max_length = max(len(row) for row in rows_split)
    padded_rows = [row + [""] * (max_length - len(row)) for row in rows_split]
    return np.array(padded_rows, dtype=object)


def merge_rows(array):
    """
    Merge multiple header rows (given as a 2D numpy array) into a single header row.
    For each column, unique non-empty values (excluding any value that is "empty" regardless of case)
    are joined with a space.
    """
    merged_row = []
    for col in range(array.shape[1]):
        seen = set()
        unique_values = []
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
        merged_row.append(" ".join(unique_values))
    return np.array(merged_row)


def reconstruct_table_from_group(group_df, delimiter=","):
    """
    Reconstructs a single table from a grouped DataFrame.
    The input DataFrame must have at least the columns "row_id", "label", and "text".

    It performs these steps:
      1. Sorts rows by row_id.
      2. Deserializes the "text" column into lists of cells.
      3. Separates header rows and table data rows.
      4. Merges header rows into one header using merge_rows.
      5. Ensures that the header and data rows have the same number of columns.
      6. Returns a DataFrame with the reconstructed table.
    """
    group_df = group_df.sort_values(by="row_id").reset_index(drop=True)
    group_df["cells"] = group_df["text"].apply(lambda x: deserialize_text(x, delimiter))
    header_lists = group_df[group_df["label"] == "header"]["cells"].tolist()
    data_lists = group_df[group_df["label"] == "table_data"]["cells"].tolist()

    if header_lists:
        header_strings = [", ".join(row) for row in header_lists]
        header_array = split_and_pad_rows(header_strings)
        header_row = merge_rows(header_array)
    else:
        header_row = (
            ["Column_" + str(i + 1) for i in range(len(data_lists[0]))]
            if data_lists
            else []
        )

    if data_lists:
        data_strings = [", ".join(row) for row in data_lists]
        data_array = split_and_pad_rows(data_strings)
    else:
        data_array = np.array([])

    if data_array.size != 0:
        n_data_cols = data_array.shape[1]
        n_header_cols = len(header_row)
        if n_header_cols < n_data_cols:
            header_row = list(header_row) + [""] * (n_data_cols - n_header_cols)
        elif n_data_cols < n_header_cols:
            data_array = np.pad(
                data_array,
                ((0, 0), (0, n_header_cols - n_data_cols)),
                constant_values="",
            )

    table_df = pd.DataFrame(data_array, columns=header_row)
    return table_df


def reconstruct_tables(inference_csv, output_dir, delimiter=",", save_as_excel=False):
    """
    Reconstructs tables from an inference CSV file.

    This function:
      - Loads the inference output CSV.
      - Groups rows by the combination of "group" and "yearbook_source".
      - For each group, calls reconstruct_table_from_group to get a structured table.
      - Saves each reconstructed table as a CSV file.
      - Optionally, saves all tables into a single Excel workbook.

    Parameters:
        inference_csv (str or Path): Path to the inference output CSV file.
        output_dir (str or Path): Directory where reconstructed tables will be saved.
        delimiter (str): Delimiter used in the flattened text (default is comma).
        save_as_excel (bool): If True, saves all tables in a single Excel workbook.

    Returns:
        dict: Mapping of table names to their reconstructed DataFrames.
    """
    df = pd.read_csv(inference_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby(["group", "yearbook_source"])
    tables = {}

    for (group, yearbook_source), group_df in grouped:
        try:
            table_df = reconstruct_table_from_group(group_df, delimiter)
            table_name = f"{group}_{yearbook_source}"
            tables[table_name] = table_df
            csv_path = output_dir / f"{table_name}.csv"
            table_df.to_csv(csv_path, index=False)
            logger.info(f"Saved table {table_name} to {csv_path}")
        except Exception as e:
            logger.error(
                f"Error reconstructing table for group {group} ({yearbook_source}): {e}"
            )

    if save_as_excel:
        excel_path = output_dir / "reconstructed_tables.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            for name, table in tables.items():
                sheet_name = name[:31]
                table.to_excel(writer, sheet_name=sheet_name, index=False)
        logger.info(f"All tables saved in single Excel workbook: {excel_path}")
    else:
        logger.info(f"Tables saved as individual CSV files in: {output_dir}")

    return tables


def run_tables_reconstruction_pipeline(
    inference_csv="inference_output.csv",
    output_dir="reconstructed_tables",
    save_as_excel=True,
):
    """
    Entry point for the table reconstruction pipeline.

    Calls reconstruct_tables with the given parameters.
    """
    return reconstruct_tables(inference_csv, output_dir, save_as_excel=save_as_excel)


if __name__ == "__main__":
    run_reconstruct_tables_pipeline()
