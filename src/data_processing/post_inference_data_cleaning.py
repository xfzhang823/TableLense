"""
File: post_inference_data_cleaning.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Aug 8

Description: Further data cleansing and manual data labeling on data after inference.
"""

import re
import logging
import pandas as pd
from file_encoding_detector import detect_encoding

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def change_label_columns(df):
    """
    Change predicted_label to label & add label_type column:
    - predicted (for inferenced data)
    - actual (for training data)
    """
    # Ensure 'label_type' column exists
    if "label_type" not in df.columns:
        df["label_type"] = ""

    # Handle renaming and setting 'label_type'
    if "predicted_label" in df.columns:
        df.rename(columns={"predicted_label": "label"}, inplace=True)
        df["label_type"] = "predicted"
    elif "label" in df.columns:
        df["label_type"] = "actual"
    else:
        raise KeyError("'label' or 'predicted_label' are missing from the DataFrame.")

    return df


def clean_text(value):
    """
    Clean a single value by removing non-printable characters and excessive spaces,
    and strip extra spaces.

    Args:
        value (str): The value to clean.

    Returns:
        str: The cleaned value.
    """
    if isinstance(value, str):
        # Remove non-printable characters
        cleaned_value = re.sub(r"[^\x20-\x7E]", "", value)
        # Remove any excessive spaces within the value
        cleaned_value = " ".join(cleaned_value.split()).strip()
        return cleaned_value
    return value  # Return the value as is if it's not a string


def is_all_empty(row):
    """
    Check if all items in a row are "EMPTY".

    Args:
        row (pd.Series or str): The row to check.

    Returns:
        bool: True if all items are "EMPTY" or "", False otherwise.
    """
    if isinstance(row, str):
        items = [item.strip() for item in row.split(",")]
    else:
        items = [str(item).strip() for item in row]

    return all(item == "EMPTY" or item == "" for item in items)


def label_empty_rows(df):
    """
    Label empty rows as "empty" if they are not already labeled as "empty".

    Args:
        df (pd.DataFrame): The DataFrame to label.

    Returns:
        pd.DataFrame: The re-labeled DataFrame.
    """
    # Set label to "empty" if text cell is "truly empty"
    df["label"] = df.apply(
        lambda row: (
            "empty"
            if is_all_empty(row["text"]) and row["label"] != "empty"
            else row["label"]
        ),
        axis=1,
    )

    # Make sure that label is set to "empty" if is_empty is "yes"
    df.loc[df["is_empty"] == "yes", "label"] = "empty"

    return df


def label_title(df):
    """
    Double check to make sure that label is "title" if is_title is "yes".

    Args:
        df (pd.DataFrame): The DataFrame to label.

    Returns:
        pd.DataFrame: The re-labeled DataFrame.
    """
    # Make sure that label is set to "empty" if is_empty is "yes"
    df.loc[df["is_title"] == "yes", "label"] = "title"

    return df


def read_csv_file(input_path):
    """
    Read a CSV file into a DataFrame with detected encoding.

    Args:
        input_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    try:
        # Detect encoding of the input file
        encoding, confidence = detect_encoding(input_path)
        logging.info(f"Detected encoding: {encoding} with confidence {confidence}")

        # Read the file into a DataFrame
        df = pd.read_csv(input_path, encoding=encoding, header=0)
        logging.info(f"File read successfully from {input_path}")

        return df
    except Exception as e:
        logging.error(f"Error reading {input_path}: {e}")
        raise


def reclassify_header(df):
    """
    Reclassify rows labeled as header if they are sandwiched immediately
    between table_data (within the same group/table).
    - these rows are likely to be table data, i.e., a table can have an extra row
    as the category name of items in rows below but has values)
    - rare scenario but exists

    Args:
        df (pd.DataFrame): The DataFrame to reclassify.

    Returns:
        pd.DataFrame: The reclassified DataFrame.
    """
    for i in range(1, len(df) - 1):
        if df.at[i, "label"] == "header":
            if (
                df.at[i - 1, "label"] == "table_data"
                and df.at[i + 1, "label"] == "table_data"
                and df.at[i - 1, "group"] == df.at[i, "group"]
                and df.at[i + 1, "group"] == df.at[i, "group"]
            ):
                df.at[i, "label"] = "table_data"
    return df


def reclassify_metadata(df):
    """
    Reclassify rows labeled as metadata if they are positioned between headers or table_data.
    (metadata should not be immediately in between headers or table_data
    - these rows are likely header or table data, i.e., a table can have an extra row
    as the category name of items in rows below but has values)

    Args:
        df (pd.DataFrame): The DataFrame to reclassify.

    Returns:
        pd.DataFrame: The reclassified DataFrame.
    """
    for i in range(1, len(df) - 1):
        if df.at[i, "label"] == "metadata":
            if df.at[i - 1, "label"] == "header" and df.at[i + 1, "label"] == "header":
                df.at[i, "label"] = "header"
            elif (
                df.at[i - 1, "label"] == "table_data"
                and df.at[i + 1, "label"] == "table_data"
            ):
                df.at[i, "label"] = "table_data"
    return df


def reclassify_empty(df):
    """
    Reclassify rows labeled as "empty" but are not really empty.
    Set their label value to that of the row immediately above.
    (usually these are in between table data - an index only row,
    with the rest of the columns empty)

    Args:
        df (pd.DataFrame): The DataFrame to reclassify.

    Returns:
        pd.DataFrame: The reclassified DataFrame.
    """
    for i in range(1, len(df) - 1):  # Adjust to avoid index out of range error
        if df.at[i, "label"] == "empty" and not is_all_empty(df.at[i, "text"]):
            label_above = df.at[i - 1, "label"]
            label_below = df.at[i + 1, "label"]
            default_row = "metadata"

            if label_above in ["header", "metadata", "table_data"]:
                df.at[i, "label"] = label_above
            elif label_below in ["header", "metadata", "table_data"]:
                df.at[i, "label"] = label_below
            else:
                df.at[i, "label"] = default_row
    return df


def clean_and_relabel_data(input_path, output_path):
    """
    Clean the data and label empty rows as "empty" (if needed),
    and save to another file.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the cleaned CSV file.
    """
    try:
        # Step 1. Read data from csv file
        df = read_csv_file(input_path)

        # Step 2. Add label_type column and rename predicted_label to label
        df = change_label_columns(df)
        logging.info("label_type added.")

        # Step 3. Sort dataframe (ascending)
        df = df.sort_values(by=["yearbook_source", "group", "row_id"], ascending=True)
        logging.info("Data sorted.")

        # Step 4. Clean up scrambled and other problematic characters in the row
        df = df.applymap(clean_text)
        logging.info("Text cleaned.")

        # Step 5. Check for actual empty rows but not labeled as "empty"
        # and relabel them as "empty"
        df = label_empty_rows(df)
        logging.info("Empty rows relabeled.")

        # Step 6. Check to make sure that label is "title" if is_title is "yes"
        df = label_title(df)
        logging.info("is_title rows' labels double checked.")

        # Step 7. Check for non-empty rows but are classified as "empty"
        # and relabel them with the label of the row above, or default to "metadata"
        df = reclassify_empty(df)
        logging.info("Non-empty rows reclassified.")

        # Step 8. Check for metadata labeled rows "sandwiched" between
        # headers or between table data, and reclassify them as headers or table_data
        df = reclassify_metadata(df)
        logging.info("Metadata reclassified.")

        # Step 9. Check for header labeled rows "sandwiched" between
        # headers or between table data, and reclassify them as table_data
        df = reclassify_header(df)
        logging.info("Header reclassified.")

        # Step 10. Save the cleaned DataFrame back to a CSV file
        df.to_csv(output_path, encoding="utf-8", index=False)
        logging.info(f"Cleaned data saved to {output_path}")

    except Exception as e:
        logging.error(f"Error in clean_and_relabel_data: {e}")
        raise


def main():
    """
    Main function to clean and relabel data.
    """
    # Paths to input and output files
    input_f_path = r"C:\github\china stats yearbook RAG\data\inference\output\yearbook 2012 and 2022 english tables predicted.csv"
    output_f_path = r"C:\github\china stats yearbook RAG\data\inference\output\yearbook 2012 and 2022 english tables predicted cleaned version.csv"

    # Clean and label the data
    clean_and_relabel_data(input_f_path, output_f_path)


if __name__ == "__main__":
    main()
