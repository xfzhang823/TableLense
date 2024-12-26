""" Function file to remove the gap between header and table data in table data """

import pandas as pd


def remove_header_table_gap(input_path, output_path):
    """
    Remove the empty row between rows labeled 'header' and 'table_data' in the specified input Excel file.
    Save the cleaned data to the specified output path.

    Args:
        input_path (str): Path to the input Excel file.
        output_path (str): Path to save the cleaned Excel file.
    """
    # Read into dataframe
    df = pd.read_excel(input_path, header=0)

    # Initialize a list to collect the indices of rows to delete
    rows_to_delete = []

    # Group the DataFrame by 'group'
    grouped = df.groupby("group")

    for group_name, group_data in grouped:
        # Find the index of the first occurrence of "header" and "table_data"
        header_indices = group_data.index[group_data.label == "header"]
        table_data_indices = group_data.index[group_data.label == "table_data"]

        # Proceed only if both "header" and "table_data" are found in the group
        if not header_indices.empty and not table_data_indices.empty:
            header_index = header_indices[0]
            table_data_index = table_data_indices[0]

            if header_index < table_data_index:
                # Filter rows between these indices
                between_df = group_data.loc[header_index:table_data_index]

                # Find rows where label is "empty"
                empty_rows = between_df[between_df.label == "empty"]

                # Collect indices of rows to delete
                rows_to_delete.extend(empty_rows.index)

    # Drop all collected rows at once
    df_cleaned = df.drop(rows_to_delete)

    # Save the cleaned DataFrame to a new file
    df_cleaned.to_excel(output_path, index=False)
    print("File saved!")


def main():
    training_data_path = r"C:\github\china stats yearbook RAG\data\training data\training data 2024 Jul 31.xlsx"
    data_out_path = r"C:\github\china stats yearbook RAG\data\training data\training data 2024 Jul 31 wo gap.xlsx"
    remove_header_table_gap(training_data_path, data_out_path)


if __name__ == "__main__":
    main()
