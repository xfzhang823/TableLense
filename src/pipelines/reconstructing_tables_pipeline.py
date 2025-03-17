import pandas as pd
import os
from pathlib import Path


def reconstruct_tables(inference_csv, output_dir, save_as_excel=False):
    """
    Reconstructs tables from an inference CSV file.

    Args:
        inference_csv (str or Path): Path to the inference output CSV file.
        output_dir (str or Path): Directory where reconstructed tables will be saved.
        save_as_excel (bool): If True, saves all tables in a single Excel workbook.
    """
    # Load inference output
    df = pd.read_csv(inference_csv)

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by table (group, yearbook_source)
    grouped = df.groupby(["group", "yearbook_source"])

    tables = {}

    for (group, yearbook_source), table_df in grouped:
        # Sort rows by their original row position
        table_df = table_df.sort_values(by=["row_id"]).reset_index(drop=True)

        # Split text column into individual cells
        table_df["cells"] = table_df["text"].apply(
            lambda x: x.split(", ") if isinstance(x, str) else []
        )

        # Extract headers and table data
        headers = table_df[table_df["label"] == "header"]["cells"].tolist()
        data = table_df[table_df["label"] == "table_data"]["cells"].tolist()

        # Flatten headers into a single row
        if headers:
            header_row = headers[0]  # Use the first header row (simplification)
        else:
            header_row = (
                ["Column_" + str(i + 1) for i in range(len(data[0]))] if data else []
            )

        # Create DataFrame
        table_df = pd.DataFrame(data, columns=header_row)

        # Save table to dictionary
        table_name = f"{group}_{yearbook_source}"
        tables[table_name] = table_df

        # Save as CSV
        table_df.to_csv(output_dir / f"{table_name}.csv", index=False)

    # Save all tables into a single Excel file if requested
    if save_as_excel:
        excel_path = output_dir / "reconstructed_tables.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            for name, table in tables.items():
                table.to_excel(
                    writer, sheet_name=name[:31], index=False
                )  # Excel limits sheet names to 31 chars
        print(f"All tables saved in: {excel_path}")
    else:
        print(f"Tables saved as individual CSV files in: {output_dir}")


# Example usage
reconstruct_tables("inference_output.csv", "reconstructed_tables", save_as_excel=True)
