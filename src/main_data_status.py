from pathlib import Path
import pandas as pd
from utils.file_encoding_detector import detect_encoding

from data_processing.data_processing_utils import get_filtered_files
from utils.read_csv_file import read_csv_file
from project_config import (
    YEARBOOK_2022_DATA_DIR,
    YEARBOOK_2012_DATA_DIR,
    TRAINING_INFERENCE_DATA_FILE,
    TRAINING_DATA_FILE,
)

# Raw Data
filtered_files_2012 = get_filtered_files(
    source_data_dir=YEARBOOK_2012_DATA_DIR,
    filter_criterion=lambda name: name.endswith(("e", "E")),
)
print(f"2012 yearbook filtered_files: {len(filtered_files_2012)} files.")
print()

filtered_files_2022 = get_filtered_files(
    source_data_dir=YEARBOOK_2022_DATA_DIR,
    filter_criterion=lambda name: name.startswith(("e", "E")),
)
print(f"2022 yearbook filtered_files: {len(filtered_files_2022)} files.")


# Preprocessed Data
total_no_of_files = len(filtered_files_2012) + len(filtered_files_2022)
print(f"total: {total_no_of_files}")
print()

df = read_csv_file(TRAINING_INFERENCE_DATA_FILE)
no_of_files_extracted = df["group"].nunique()
print(f"extracted: {no_of_files_extracted}")

# Training data
df = read_csv_file(TRAINING_DATA_FILE)
no_of_tables_trained = df["group"].nunique()
print(f"trained: {no_of_tables_trained}")
