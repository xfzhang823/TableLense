"""
File: preprocessing_utils.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Jul 28

Description:
    Utility class for preprocessing Excel files. 
    It includes functions to clear sheets, copy content between sheets, 
    process Excel files, convert cell references, etc.
    This version includes both synchronous and asynchronous methods.

    This version uses asyncio.Semaphore to manage resources; 
    it does not use threading.Lock.

Usage:
    from preprocessing_utils import ExcelPreprocessor

Dependencies: xlwings, pandas, os, logging, sys, asyncio, tempfile, shutil
"""

import os
import logging
import sys
import asyncio
import numpy as np
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import xlwings as xw
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class ExcelPreprocessor:
    """
    A class to encapsulate Excel preprocessing operations using xlwings.

    Methods:
    -------
    a1_to_rc(a1)
        Convert Excel A1 notation to 1-indexed row and column.
    rc_to_a1(row, col)
        Convert 1-indexed row and column to Excel A1 notation.
    convert_to_indices(cell)
        Convert cell reference to 1-indexed row and column, whether in A1 or [row, col] format.
    clear_sheet(file_path, sheet_name)
        Clear the target sheet before copying.
    copy_sheet_to_diff_file(src_path, tgt_path, src_sheet_name, tgt_sheet_name, src_start_cell, tgt_start_cell)
        Copy content from one sheet to another in different workbooks.
    find_data_area(sheet)
        Find the row/col of the area with data.
    find_table_area(sheet)
        Find the row/col of the table area.
    is_stand_alone(sheet, cell_address)
        Check if a cell is 'stand-alone' (doesn't border other non-empty cells).
    process_excel_full_range(file_path)
        Processes the content of an Excel file, handling merged cells by filling each cell
        in the formerly merged area with the original merged cell's value.
    _read_and_fill_merged_cells(file_path)
        Reads data from the specified Excel sheet, handling merged cells by filling each cell in
        the formerly merged area with the original merged cell's value.
    async_process_excel_full_range(file_path)
        Asynchronously processes the content of an Excel file, handling merged cells by filling each cell
        in the formerly merged area with the original merged cell's value.
    async_read_and_fill_merged_cells(file_path)
        Asynchronously reads data from the specified Excel sheet, handling merged cells by
        filling each cell in the formerly merged area with the original merged cell's value.
    _read_excel(file_path)
        Private helper function to read the content of an Excel file synchronously.
    """

    def __init__(self):
        self.app = xw.App(visible=False)

    def __del__(self):
        """Ensures the Excel application is properly closed."""
        self.app.quit()

    def a1_to_rc(self, a1):
        """Convert Excel A1 notation to 1-indexed row and column."""
        col = 0
        row = 0
        for i, c in enumerate(a1):
            if c.isdigit():
                row = int(a1[i:])  # Excel rows are 1-indexed
                break
            col = col * 26 + (ord(c.upper()) - ord("A") + 1)
        return row, col

    def rc_to_a1(self, row, col):
        """Convert 1-indexed row and column to Excel A1 notation."""
        col_str = ""
        while col > 0:
            col, remainder = divmod(col - 1, 26)
            col_str = chr(65 + remainder) + col_str
        return f"{col_str}{row}"

    def convert_to_indices(self, cell):
        """Convert cell reference to 1-indexed row and column, whether in A1 or [row, col] format."""
        if isinstance(cell, str):
            return self.a1_to_rc(cell)
        elif isinstance(cell, (list, tuple)) and len(cell) == 2:
            return cell[0], cell[1]
        else:
            raise ValueError(
                "Invalid cell reference format. Use A1 notation or [row, col] format."
            )

    def clear_sheet(self, file_path, sheet_name):
        """
        Clear the target sheet before copying.

        Parameters:
        file_path (str): Path to the Excel file.
        sheet_name (str): Name of the sheet to clear.
        """
        logging.info(f"Clearing sheet: {sheet_name} in file: {file_path}")
        wb = self.app.books.open(file_path)
        sht = wb.sheets[sheet_name]
        sht.clear()
        wb.save()
        wb.close()

    def copy_sheet_to_diff_file(
        self,
        src_path,
        tgt_path,
        src_sheet_name,
        tgt_sheet_name,
        src_start_cell,
        tgt_start_cell,
    ):
        """
        Copy content from one sheet to another in different workbooks.

        Parameters:
        src_path (str): Path to the source Excel file.
        tgt_path (str): Path to the target Excel file.
        src_sheet_name (str): Name of the source sheet.
        tgt_sheet_name (str): Name of the target sheet.
        src_start_cell (str or list): Starting cell in the source sheet (A1 or [row, col]).
        tgt_start_cell (str or list): Starting cell in the target sheet (A1 or [row, col]).

        Returns:
        str: The last row with content in A1 notation.
        """
        logging.info(
            f"Copying sheet: {src_sheet_name} from file: {src_path} to file: {tgt_path}"
        )
        source_wb = self.app.books.open(src_path)
        target_wb = self.app.books.open(tgt_path)

        source_sheet = source_wb.sheets[src_sheet_name]
        target_sheet = target_wb.sheets[tgt_sheet_name]

        src_start_row, src_start_col = self.convert_to_indices(src_start_cell)
        tgt_start_row, tgt_start_col = self.convert_to_indices(tgt_start_cell)

        source_range = source_sheet.range((src_start_row, src_start_col)).expand()
        source_values = source_range.value

        target_sheet.range((tgt_start_row, tgt_start_col)).value = source_values
        tgt_last_row = (
            target_sheet.range((tgt_start_row, tgt_start_col)).expand().last_cell.row
        )

        last_row_coord = self.rc_to_a1(tgt_last_row, tgt_start_col)

        target_wb.save()
        source_wb.close()
        target_wb.close()

        return last_row_coord

    def find_data_area(self, sheet):
        """Find the row/col of the area with data."""
        logging.info("Finding data area")
        max_row, max_col = 0, 0

        for row in range(1, sheet.used_range.last_cell.row + 1):
            for col in range(1, sheet.used_range.last_cell.column + 1):
                cell_value = sheet.range((row, col)).value
                if cell_value not in [None, "", " "]:
                    max_row, max_col = max(max_row, row), max(max_col, col)

        min_row = next(
            row
            for row in range(1, max_row + 1)
            if any(
                sheet.range((row, col)).value not in [None, "", " "]
                for col in range(1, max_col + 1)
            )
        )

        min_col = next(
            col
            for col in range(1, max_col + 1)
            if any(
                sheet.range((row, col)).value not in [None, "", " "]
                for row in range(1, max_row + 1)
            )
        )

        logging.info(
            f"Data area determined: min_row={min_row}, max_row={max_row}, min_col={min_col}, max_col={max_col}"
        )
        return min_row, max_row, min_col, max_col

    def find_table_area(self, sheet):
        """Find the row/col of the table area."""
        logging.info("Finding table area")

        min_row, max_row, min_col, max_col = self.find_data_area(sheet)

        table_max_row = (
            next(
                (
                    row
                    for row in range(min_row, max_row + 1)
                    if all(
                        sheet.range((row, col)).value in [None, "", " "]
                        for col in range(min_col, max_col + 1)
                    )
                ),
                max_row + 1,
            )
            - 1
        )

        table_max_col = (
            next(
                (
                    col
                    for col in range(min_col, max_col + 1)
                    if all(
                        sheet.range((row, col)).value in [None, "", " "]
                        for row in range(min_row, max_row + 1)
                    )
                ),
                max_col + 1,
            )
            - 1
        )

        logging.info(
            f"Table area determined: min_row={min_row}, max_row={table_max_row}, min_col={min_col}, max_col={table_max_col}"
        )
        return min_row, table_max_row, min_col, table_max_col

    def is_stand_alone(self, sheet, cell_address):
        """Check if a cell is 'stand-alone' (doesn't border other non-empty cells)."""
        cell = sheet.range(cell_address)
        row, col = cell.row, cell.column

        neighbors = [
            sheet.range((row - 1, col)).value,  # Above
            sheet.range((row + 1, col)).value,  # Below
            sheet.range((row, col - 1)).value,  # Left
            sheet.range((row, col + 1)).value,  # Right
        ]

        return all(neighbor in [None, "", " "] for neighbor in neighbors)

    def process_excel_full_range(self, file_path):
        """
        Processes the content of an Excel file, handling merged cells by filling each cell
        in the formerly merged area with the original merged cell's value.
        Also:
        - Adds a group field to track the table source (which table)
        - Adds a row_id to track the order of each row within each group.

        Args:
            file_path (str): The path to the Excel file.

        Returns:
            list: List of dictionaries containing the processed data, group label, and row_id.
        """
        logging.info(f"Processing file: {file_path}")
        data, merged_ranges = self._read_and_fill_merged_cells(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        processed_data = []
        row_id = 1  # Initialize row_id counter
        label = ""

        for row in data:
            row = ["EMPTY" if cell is None else cell for cell in row]
            row = [str(cell).replace(",", " ") for cell in row]

            processed_data.append(
                {
                    "text": ", ".join([str(cell) for cell in row]),
                    "group": file_name,
                    "row_id": row_id,
                    "label": label,
                }
            )
            row_id += 1

        return processed_data

    def _read_and_fill_merged_cells(self, file_path):
        """
        Reads data from the specified Excel sheet, handling merged cells by filling each cell in
        the formerly merged area with the original merged cell's value.

        Parameters:
        - file_path (str): Path to the Excel workbook.

        Returns:
        - list: Nested list representing the content of the Excel file with filled merged cells.
        - list: List of merged cell ranges.
        """
        logging.info(f"Reading and filling merged cells in file: {file_path}")
        try:
            temp_dir = tempfile.mkdtemp()
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".xls", dir=temp_dir
            ) as temp_file:
                temp_file_path = temp_file.name
                shutil.copy(file_path, temp_file_path)
                logging.info(f"Temporary file created at: {temp_file_path}")

            wb = self.app.books.open(temp_file_path)
            sheet = wb.sheets[0]

            merged_ranges = [cell for cell in sheet.used_range if cell.merge_cells]
            for merge_range in merged_ranges:
                merge_range.unmerge()

            min_row, max_row, min_col, max_col = self.find_data_area(sheet)

            data_with_merge_info = []
            processed_cells = set()

            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    cell = sheet.range((row, col))
                    cell_address = cell.address

                    if cell_address in processed_cells:
                        continue

                    cell_value = cell.value
                    merge_area = cell.merge_area

                    if merge_area.count > 1:
                        top_left_cell = merge_area[0, 0]
                        top_left_value = top_left_cell.value

                        for area_cell in merge_area:
                            processed_cells.add(area_cell.address)
                            data_with_merge_info.append(
                                (area_cell.row, area_cell.column, top_left_value)
                            )
                        if merge_area.address not in [
                            rng.address for rng in merged_ranges
                        ]:
                            merged_ranges.append(merge_area)
                    else:
                        data_with_merge_info.append((cell.row, cell.column, cell_value))

            wb.close()

            max_row = max(row for row, _, _ in data_with_merge_info)
            max_col = max(col for _, col, _ in data_with_merge_info)

            data_matrix = [[None for _ in range(max_col)] for _ in range(max_row)]

            for row, col, value in data_with_merge_info:
                data_matrix[row - 1][col - 1] = value

            logging.info(f"Finished processing file: {file_path}")
            return data_matrix, merged_ranges

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            return [], []

        finally:
            shutil.rmtree(temp_dir)

    async def async_process_excel_full_range(self, file_path):
        """
        Asynchronously processes the content of an Excel file, handling merged cells by filling each cell
        in the formerly merged area with the original merged cell's value.
        Also adds a row_id field to track the order of each row within each group.

        Args:
            file_path (str): The path to the Excel file.

        Returns:
            list: List of dictionaries containing the processed data, group label, and row_id.
        """

        data = await self.async_read_and_fill_merged_cells(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        processed_data = []
        row_id = 1

        for row in data:
            row = ["EMPTY" if cell is None else cell for cell in row]
            row = [str(cell).replace(",", " ") for cell in row]

            processed_data.append(
                {
                    "text": ", ".join([str(cell) for cell in row]),
                    "group": file_name,
                    "row_id": row_id,
                }
            )
            row_id += 1

        return processed_data

    async def async_read_and_fill_merged_cells(self, file_path):
        """
        Asynchronously reads data from the specified Excel sheet, handling merged cells by
        filling each cell in the formerly merged area with the original merged cell's value.

        Parameters:
        - file_path (str): Path to the Excel workbook.

        Returns:
        - list: Nested list representing the content of the Excel file with filled merged cells.
        """
        loop = asyncio.get_running_loop()

        with ThreadPoolExecutor() as pool:
            data = await loop.run_in_executor(
                pool, self._read_and_fill_merged_cells, file_path
            )

        return data

    def _read_excel(self, file_path):
        """
        Private helper function to read the content of an Excel file synchronously.

        Args:
            file_path (str): The path to the Excel file.

        Returns:
            list: Nested list representing the content of the Excel file.
        """
        wb = self.app.books.open(file_path)
        sheet = wb.sheets[0]
        data = sheet.used_range.value
        wb.close()
        return data


def add_is_title_column(df):
    """
    Adds an 'is_title' column to the DataFrame, marking the first row of each group as 'yes'
    and the rest as 'no'.
    If the column already exists and is not empty, the function does nothing.

    Args:
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        pd.DataFrame: The modified DataFrame with the 'is_title' column added.
    """
    # Verify if the column exists and is not empty
    col_name = "is_title"

    if col_name in df.columns and not df[col_name].dropna().empty:
        logging.info(f"Column {col_name} exists in the DataFrame and is not empty.")
    else:
        # Initialize the is_title column with "no"
        df[col_name] = "no"

        # Calculate and fill value
        groups = df.group.unique().tolist()
        for group in groups:
            mask = df.group == group
            first_row_idx = df[mask].index[0]
            df.loc[first_row_idx, "is_title"] = "yes"

        logging.info(f"Column {col_name} is added and filled.")

    return df


def is_all_empty(row):
    """
    Check if all items in a row are 'EMPTY', ignoring extra spaces.

    Args:
        row (str): A string representing a row, with items separated by commas.

    Returns:
        bool: True if all items are 'EMPTY' (after stripping spaces), False otherwise.
    """
    items = [item.strip().upper() for item in row.split(",")]
    return all(item == "EMPTY" or item == "" for item in items)


def add_is_empty_column(df):
    """
    Checks if the 'is_empty' column exists in the DataFrame. If it does not exist,
    it creates the column and fills it with 'yes' or 'no' based on whether the 'text'
    column is considered empty.

    Args:
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        pd.DataFrame: The modified DataFrame with the 'is_empty' column added.
    """
    col_name = "is_empty"

    if col_name in df.columns and not df[col_name].dropna().empty:
        logging.info(f"Column '{col_name}' exists in the DataFrame and is not empty.")
    else:
        # Create 'is_empty' column and fill with 'yes' or 'no'
        df[col_name] = df["text"].apply(
            lambda row: "yes" if is_all_empty(row) else "no"
        )
        logging.info(f"Column '{col_name}' has been added and filled.")

    return df


def main():
    preprocessor = ExcelPreprocessor()  # No lock here

    source_path = "source.xlsx"  # Replace with your source workbook path
    target_path = "target.xlsx"  # Replace with your target workbook path
    source_sheet_name = "SourceSheetName"  # Replace with your source sheet name
    target_sheet_name = "TargetSheetName"  # Replace with your target sheet name

    source_start_cell = (
        "A1"  # Replace with the starting cell of the range in the source sheet
    )
    target_start_cell = [
        1,
        1,
    ]  # Replace with the starting cell of the range in the target sheet

    last_row_coord = preprocessor.copy_sheet_to_diff_file(
        source_path,
        target_path,
        source_sheet_name,
        target_sheet_name,
        source_start_cell,
        target_start_cell,
    )
    print(f"Last row with content: {last_row_coord}")


if __name__ == "__main__":
    main()
