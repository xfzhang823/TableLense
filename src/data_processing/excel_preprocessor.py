"""
File: preprocessing_utils.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Dec

Description:
    - Utility class for preprocessing Excel files. 
    - It includes functions to clear sheets, copy content between sheets, 
    process Excel files, convert cell references, etc.
    - This version includes both synchronous and asynchronous methods.

    This version uses asyncio.Semaphore to manage resources; 
    it does not use threading.Lock.

Usage:
    from preprocessing_utils import ExcelPreprocessor

Dependencies: xlwings, pandas, os, logger, sys, asyncio, tempfile, shutil, etc.
"""

from pathlib import Path
import os
import logging
import sys
import shutil
import tempfile
from typing import Any, Dict, List, Tuple, Union
import asyncio
import numpy as np
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import xlwings as xw
import logging_config


# Setup logger
logger = logging.getLogger(__name__)


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
    copy_sheet_to_diff_file(src_path, tgt_path, src_sheet_name, tgt_sheet_name, src_start_cell,
    tgt_start_cell)
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
        Asynchronously processes the content of an Excel file, handling merged cells by
        filling each cell in the formerly merged area with the original merged cell's value.
    async_read_and_fill_merged_cells(file_path)
        Asynchronously reads data from the specified Excel sheet, handling merged cells by
        filling each cell in the formerly merged area with the original merged cell's value.
    """

    def __init__(self):
        """
        Use lazy initiation instead of eager initiation to avoid issues when running
        multi-threading and async.

        COM requires the Excel application instance to remain within the thread where
        it was created.
        """
        self.app = None

    def _intialize_app(self):
        """Initialize only when called upon"""
        if self.app is None:
            self.app = xw.App(visible=False)

    def _close_app(self):
        """Close the app to clean up resources."""
        if self.app is not None:
            self.app.quit()
            self.app = None

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
        """
        Convert cell reference to 1-indexed row and column, whether in A1 or [row, col]
        format.
        """
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
        logger.info(f"Clearing sheet: {sheet_name} in file: {file_path}")
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
        logger.info(
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

    def find_data_area(self, sheet) -> Tuple[int, int, int, int]:
        """
        Find the row/col boundaries of the area containing data in the Excel sheet.

        Args:
            sheet: xlwings sheet object.

        Returns:
            Tuple[int, int, int, int]: (min_row, max_row, min_col, max_col)
            - min_row: First row containing non-empty data.
            - max_row: Last row containing non-empty data.
            - min_col: First column containing non-empty data.
            - max_col: Last column containing non-empty data.

        *The method identifies the boundaries of meaningful data in an Excel sheet
        *Method Steps:
        1. Bulk Read Data: Fetch the entire data range from the Excel sheet into a 2D list
        using 'sheet.range(sheet.used_range.address).value'.
        2. Determine Dimensions: Calculate the number of rows (`rows`) and
        columns ('cols') in the 2D list.
        3. Find Non-Empty Rows:
            - Loop through each row and check if any cell contains meaningful data
            ('not in [None, "", " "]').
            - Identify the first (`min_row`) and last (`max_row`) rows with non-empty data.
        4. Find Non-Empty Columns:
            - Loop through each column and check if any cell in that column contains
            meaningful data.
            - Identify the first (`min_col`) and last (`max_col`) columns with
            non-empty data.
        5. Log and Return Results:
            - Log the calculated 'min_row', 'max_row', 'min_col', and 'max_col'
            for debugging.
            - Return the tuple '(min_row, max_row, min_col, max_col)' indicating
            the bounding rectangle of the data.

        Example output: (1, 3, 2, 3)

        This structured approach efficiently  while minimizing redundant operations.
        """
        logger.info("Finding data area")

        # Read all the cell values into a 2D list
        data = sheet.range(sheet.used_range.address).value  # 2D list

        # Initialize variables
        rows = len(data)
        cols = len(data[0]) if rows > 0 else 0

        # Identify the minimum and maximum rows with non-empty cells
        min_row = min(
            r + 1
            for r in range(rows)
            if any(cell not in [None, "", " "] for cell in data[r])
        )
        max_row = max(
            r + 1
            for r in range(rows)
            if any(cell not in [None, "", " "] for cell in data[r])
        )

        # Identify the minimum and maximum columns with non-empty cells
        min_col = min(
            c + 1
            for c in range(cols)
            if any(data[r][c] not in [None, "", " "] for r in range(rows))
        )
        max_col = max(
            c + 1
            for c in range(cols)
            if any(data[r][c] not in [None, "", " "] for r in range(rows))
        )

        logger.info(
            f"Data area determined: min_row={min_row}, max_row={max_row}, min_col={min_col}, max_col={max_col}"
        )
        return min_row, max_row, min_col, max_col

    # Todo: older version - to be deleted after debugging
    # def find_data_area_old_version(self, sheet):
    #     """Find the row/col of the area with data."""
    #     logger.info("Finding data area")
    #     max_row, max_col = 0, 0

    #     for row in range(1, sheet.used_range.last_cell.row + 1):
    #         for col in range(1, sheet.used_range.last_cell.column + 1):
    #             cell_value = sheet.range((row, col)).value
    #             if cell_value not in [None, "", " "]:
    #                 max_row, max_col = max(max_row, row), max(max_col, col)

    #     min_row = next(
    #         row
    #         for row in range(1, max_row + 1)
    #         if any(
    #             sheet.range((row, col)).value not in [None, "", " "]
    #             for col in range(1, max_col + 1)
    #         )
    #     )

    #     min_col = next(
    #         col
    #         for col in range(1, max_col + 1)
    #         if any(
    #             sheet.range((row, col)).value not in [None, "", " "]
    #             for row in range(1, max_row + 1)
    #         )
    #     )

    #     logger.info(
    #         f"Data area determined: min_row={min_row}, max_row={max_row}, min_col={min_col}, max_col={max_col}"
    #     )
    #     return min_row, max_row, min_col, max_col

    def find_table_area(self, sheet):
        """Find the row/col of the table area."""
        logger.info("Finding table area")

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

        logger.info(
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

    def process_excel_full_range(
        self, file_path: Union[Path, str], yearbook_source: str, group: str
    ) -> List[Dict[str, Any]]:
        """
        Processes the content of an Excel file, handling merged cells by filling each cell
        in the formerly merged area with the original merged cell's value.

        Also adds metadata related fields.

        Args:
            - file_path (str): The path to the Excel file.
            - yearbook_source (str): data source (e.g., "2012").
            - group (str): The group or table identifier (e.g., derived from the file name).

        Returns:
            Tuple[List[List[Any]], List[Any]]:
                - Nested list representing the content of the Excel file with filled
                merged cells.
                - List of merged cell ranges.
        """

        # Initialize xlwings
        self._intialize_app()

        try:
            logger.info(f"Processing file: {file_path}")

            # Ensure file_path is Path obj
            file_path = Path(file_path)

            # Core processing
            raw_data, _ = self._read_and_fill_merged_cells(file_path)

            # Generate a new list by adding "row_id" and "EMPTY"
            processed_data = [
                {
                    "text": ", ".join(
                        ["EMPTY" if cell is None else str(cell) for cell in row]
                    ),
                    "row_id": idx
                    + 1,  # Sequential row ID for a table ("order" feature for training)
                    "group": group,  # table number (data file name)
                    "yearbook_source": yearbook_source,  # 2012 or 2022 yearbook
                }
                for idx, row in enumerate(raw_data)
            ]
            logger.info(f"Finished processing {file_path}.")
            return processed_data

        finally:
            # Close the instance to free up memory slot
            self._close_app()

    def _read_and_fill_merged_cells(self, file_path):
        """
        Reads data from the specified Excel sheet, handling merged cells by filling
        each cell in the formerly merged area with the original merged cell's value.

        Parameters:
            - file_path (str): Path to the Excel workbook.

        Returns:
            Tuple[List[List[Any]], List[Any]]:
                - Nested list representing the content of the Excel file with filled
                merged cells.
                - List of merged cell ranges.

        Raises:
            Exception: If there are issues opening or processing the file.
        """
        logger.info(f"Reading and filling merged cells in file: {file_path}")
        try:
            temp_dir = tempfile.mkdtemp()
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".xls", dir=temp_dir
            ) as temp_file:
                temp_file_path = temp_file.name
                shutil.copy(file_path, temp_file_path)
                logger.info(f"Temporary file created at: {temp_file_path}")

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

            logger.info(f"Finished processing file: {file_path}")
            return data_matrix, merged_ranges

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return [], []

        finally:
            shutil.rmtree(temp_dir)

    async def process_excel_full_range_async(
        self, file_path: str, group: str, yearbook_source: str
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously processes the content of an Excel file, handling merged cells
        by filling each cell in the formerly merged area with the original merged cell's value.
        Also adds a row_id field and incorporates the group and yearbook_source into the output.

        Args:
            - file_path (str): The path to the Excel file.
            - group (str): The group or table identifier (e.g., derived from the file name).
            yearbook_source (str): The yearbook source (e.g., "2012").

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing processed rows
            with keys `text`, `row_id`, `group`, and `yearbook_source`.
        """
        # Initialize app - xlwings
        self._intialize_app()

        try:
            logger.info(f"Processing file asynchronously: {file_path}")

            # Ensure file path is Paht obj
            file_path = Path(file_path)

            # Read and process the file asynchronously
            data = await self.read_and_fill_merged_cells_async(file_path)

            # Process each row
            processed_data = [
                {
                    "text": ", ".join(
                        ["EMPTY" if cell is None else str(cell) for cell in row]
                    ),
                    "row_id": idx + 1,
                    "group": group,
                    "yearbook_source": yearbook_source,
                }
                for idx, row in enumerate(data)
            ]

            return processed_data

        finally:
            self._close_app()

    async def read_and_fill_merged_cells_async(self, file_path: str) -> List[List[Any]]:
        """
        Asynchronously reads data from the specified Excel sheet, handling merged cells by
        filling each cell in the formerly merged area with the original merged cell's value.

        Args:
            file_path (str): Path to the Excel workbook.

        Returns:
            List[List[Any]]: Nested list representing the content of the Excel file with
            filled merged cells.
        """
        logger.info(
            f"Reading and filling merged cells asynchronously for file: {file_path}"
        )
        temp_dir = None
        try:
            # Create a temporary directory and copy the file
            temp_dir = tempfile.mkdtemp()
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".xls", dir=temp_dir
            ) as temp_file:
                temp_file_path = temp_file.name
                shutil.copy(file_path, temp_file_path)
                logger.info(f"Temporary file created at: {temp_file_path}")

            loop = asyncio.get_running_loop()

            # Run synchronous reading logic in a thread pool
            result = await loop.run_in_executor(
                None, self._read_and_fill_merged_cells, temp_file_path
            )
            return result[0]  # Return only the data matrix

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []

        finally:
            # Ensure temporary directory is cleaned up
            if temp_dir:
                shutil.rmtree(temp_dir)
