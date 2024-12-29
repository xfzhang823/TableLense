"""
File: normalizing_data_for_inference.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Aug 5

Description: 
Data normalization and cleansing to get data ready for inference (label prediction).
"""

import pandas as pd
from file_encoding_detector import detect_encoding


def read_and_merge_data(source_data_path, mapping_f_path):
    """
    Read and merge data with mapping file to perform VLOOKUP equivalent.

    Args:
        source_data_path (str): Path to the source data file.
        mapping_f_path (str): Path to the mapping file.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    encoding, confidence = detect_encoding(source_data_path)
    df = pd.read_csv(source_data_path, encoding=encoding, header=0)
    mapping = pd.read_csv(mapping_f_path, header=0)
    df_merged = df.merge(mapping, on="group", how="inner")

    # Add is_title field
    df_merged["is_title"] = ""
    groups = df_merged.group.unique().tolist()
    for group in groups:
        mask = df_merged.group == group
        first_row_idx = df_merged[mask].index[0]
        df_merged.loc[first_row_idx, "is_title"] = "yes"

    return df_merged


def read_2022_data(source_data_path):
    """
    Read 2022 yearbook data and add necessary fields.

    Args:
        source_data_path (str): Path to the 2022 data file.

    Returns:
        pd.DataFrame: DataFrame with added fields.
    """
    df = pd.read_excel(source_data_path, header=0)
    df["label"] = ""
    return df


def normalize_and_combine_data(
    source_data_path_2012, mapping_f_path_2012, source_data_path_2022, output_path
):
    """
    Normalize data and combine the 2012 and 2022 datasets.

    Args:
        source_data_path_2012 (str): Path to the 2012 data file.
        mapping_f_path_2012 (str): Path to the 2012 mapping file.
        source_data_path_2022 (str): Path to the 2022 data file.
        output_path (str): Path to save the combined data file.
    """
    df_2012 = read_and_merge_data(source_data_path_2012, mapping_f_path_2012)
    df_2022 = read_2022_data(source_data_path_2022)
    df_combined = pd.concat([df_2012, df_2022], ignore_index=True)

    # Save to file
    df_combined.to_csv(output_path, index=False)
    print(f"File saved to {output_path}")


def main():
    source_data_path_2012 = r"C:\github\china stats yearbook RAG\data\training data\excel sheet training data yrbk 2012.csv"
    mapping_f_path_2012 = r"C:\github\china stats yearbook RAG\data\training data\section_group_mapping_2012.csv"
    source_data_path_2022 = r"C:\github\china stats yearbook RAG\data\training data\excel sheet training data yrbk 2022 excel version.xlsx"
    output_path = r"C:\github\china stats yearbook RAG\data\excel data\yearbook 2012 and 2022 english tables.csv"

    normalize_and_combine_data(
        source_data_path_2012, mapping_f_path_2012, source_data_path_2022, output_path
    )


if __name__ == "__main__":
    main()
