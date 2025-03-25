"""
todo: still debugging... almost complete

"""

from pathlib import Path
import asyncio
import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from data_processing.reconstruct_table import reconstruct_objects_from_group
from project_config import RECONSTRUCTED_TABLES_DIR, COMBINED_CLEANED_OUTPUT_DATA_FILE


logger = logging.getLogger(__name__)


async def load_csv_async(path: Path | str) -> pd.DataFrame:
    """Load CSV file asynchronously."""
    path = Path(path) if isinstance(path, str) else path
    loop = asyncio.get_running_loop()
    logger.info(f"Loading CSV from {path}")
    return await loop.run_in_executor(None, pd.read_csv, path)


async def save_csv_async(df: pd.DataFrame, path: Path | str):
    """Save a DataFrame to CSV asynchronously."""
    path = Path(path) if isinstance(path, str) else path
    loop = asyncio.get_running_loop()
    logger.info(f"Saving CSV to {path}")
    await loop.run_in_executor(None, lambda: df.to_csv(path, index=False))


def process_group(group_tuple: tuple, delimiter: str) -> Dict[str, Any]:
    """
    Process a single group tuple by reconstructing its title, metadata, and table.

    Args:
        group_tuple (tuple): A tuple (group, group_df) for a single table.
        delimiter (str): Delimiter used in the flattened text.
        yearbook_source (str): Yearbook source identifier (e.g., "2012").

    Returns:
        Dict[str, Any]: A dictionary containing:
            - "group": The group identifier.
            - "yearbook_source": The yearbook source.
            - "title": Title string.
            - "metadata": Metadata string.
            - "table": DataFrame of table data (or None if processing failed).
            - "error": (Optional) Error message if reconstruction failed.
    """
    group, group_df = group_tuple

    logger.info(f"Processing group: {group}")

    # Extract yearbook source from the dataframe
    yearbook_source_value = (
        group_df["yearbook_source"].iloc[0]
        if "yearbook_source" in group_df.columns
        else ""
    )
    objects = reconstruct_objects_from_group(group_df, delimiter)
    objects["group"] = group
    objects["yearbook_source"] = yearbook_source_value
    return objects


async def process_all_groups_async(
    input_csv: str,
    delimiter: str,
) -> List[Dict[str, Any]]:
    """
    Processes all groups from the input CSV concurrently.

    Args:
        input_csv (str): Path to the flattened CSV file.
        delimiter (str): Delimiter used in the flattened text.
        yearbook_source (str): Yearbook source identifier to be added to each object.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, one per group, each containing
            keys: "group", "yearbook_source", "title", "metadata", "table", and possibly "error".
    """
    df = pd.read_csv(input_csv)
    groups = list(df.groupby("group"))
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(executor, process_group, group_tuple, delimiter)
            for group_tuple in groups
        ]
        results = await asyncio.gather(*tasks)
    return results


async def tables_reconstruction_pipeline_async(
    input_csv: str,
    output_dir: str,
    delimiter: str = ",",
    save_as_excel: bool = False,
) -> List[Dict[str, Any]]:
    """
    Reverse preprocesses the entire dataset by reconstructing each table group into
    three separate objects: title, metadata, and table data.

    For each group, this function saves three CSV files using the naming convention:
        {yearbook_source}_{group}_title.csv
        {yearbook_source}_{group}_metadata.csv
        {yearbook_source}_{group}_table.csv

    Args:
        input_csv (str): Path to the flattened CSV file.
        output_dir (str): Directory where reconstructed CSV files will be saved.
        delimiter (str): Delimiter used in the flattened text. Default is a comma.
        save_as_excel (bool): If True, also saves all objects into
        a single Excel workbook.
        yearbook_source (str): Yearbook source identifier (e.g., "2012").

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, one per group, with keys:
            "group", "yearbook_source", "title", "metadata", and "table".
            If a group failed to reconstruct (e.g., missing table data),
            an "error" key is included.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all groups concurrently.
    groups_objects = await process_all_groups_async(
        input_csv,
        delimiter,
    )

    loop = asyncio.get_running_loop()
    save_tasks = []
    failed_tables = []  # Record tables that failed reconstruction

    for obj in groups_objects:
        if obj.get("error"):
            failed_tables.append(
                {
                    "group": obj["group"],
                    "yearbook_source": obj["yearbook_source"],
                    "error": obj["error"],
                }
            )
        else:
            group = obj["group"]
            y_source = obj["yearbook_source"]

            # Save title CSV.
            title_csv = output_dir / f"{y_source}_{group}_title.csv"
            title_df = pd.DataFrame({"title": [obj["title"]]})
            logger.info(f"Scheduling save for title of group {group} to {title_csv}")
            save_tasks.append(save_csv_async(title_df, title_csv))

            # Save metadata CSV.
            metadata_csv = output_dir / f"{y_source}_{group}_metadata.csv"
            metadata_df = pd.DataFrame({"metadata": [obj["metadata"]]})
            logger.info(
                f"Scheduling save for metadata of group {group} to {metadata_csv}"
            )
            save_tasks.append(save_csv_async(metadata_df, metadata_csv))

            # Save table CSV.
            table_csv = output_dir / f"{y_source}_{group}_table.csv"
            logger.info(
                f"Scheduling save for table data of group {group} to {table_csv}"
            )
            save_tasks.append(save_csv_async(obj["table"], table_csv))

    # Await all save tasks concurrently.
    await asyncio.gather(*save_tasks)

    # Optionally, save all tables into a single Excel workbook.
    if save_as_excel:
        from pandas import ExcelWriter

        excel_path = output_dir / "reconstructed_tables.xlsx"

        def write_excel():
            with ExcelWriter(excel_path) as writer:
                for obj in groups_objects:
                    group = obj["group"]
                    y_source = obj["yearbook_source"]
                    # Save title sheet.
                    sheet_name_title = f"{y_source}_{group}_tit"[:31]
                    pd.DataFrame({"title": [obj["title"]]}).to_excel(
                        writer, sheet_name=sheet_name_title, index=False
                    )
                    # Save metadata sheet.
                    sheet_name_meta = f"{y_source}_{group}_meta"[:31]
                    pd.DataFrame({"metadata": [obj["metadata"]]}).to_excel(
                        writer, sheet_name=sheet_name_meta, index=False
                    )
                    # Save table sheet if reconstruction was successful.
                    if obj.get("table") is not None:
                        sheet_name_table = f"{y_source}_{group}_table"[:31]
                        obj["table"].to_excel(
                            writer, sheet_name=sheet_name_table, index=False
                        )

        await loop.run_in_executor(None, write_excel)
        logger.info(f"Saved combined Excel workbook to {excel_path}")

    # Save the list of failed tables to a CSV for further analysis.
    if failed_tables:
        error_df = pd.DataFrame(failed_tables)
        error_file = output_dir / "failed_tables.csv"
        error_df.to_csv(error_file, index=False)
        logger.info(f"Saved failed tables report to {error_file}")

    logger.info("Reverse preprocessing pipeline completed.")
    return groups_objects


async def run_tables_reconstruction_pipeline_async(
    inference_training_csv_file: str = str(COMBINED_CLEANED_OUTPUT_DATA_FILE),
    output_dir: str = str(RECONSTRUCTED_TABLES_DIR),
    save_as_excel: bool = False,
):
    """
    Entry point for the table reconstruction pipeline.

    Calls tables_reconstruction_pipeline with the given parameters.
    """
    logger.info("Start running tables reconstruction pipeline...")
    await tables_reconstruction_pipeline_async(
        input_csv=inference_training_csv_file,
        output_dir=output_dir,
        save_as_excel=save_as_excel,
    )
    logger.info("Finished running tables reconstruction pipeline...")


if __name__ == "__main__":
    asyncio.run(run_tables_reconstruction_pipeline_async())
