from pathlib import Path
import pandas as pd
from utils.file_encoding_detector import detect_encoding

from src.data_processing.data_processing_utils import get_filtered_files
from project_config import YEARBOOK_2022_DATA_DIR, YEARBOOK_2012_DATA_DIR

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


csv_file = Path(
    r"C:\github\table_lense\input_output\preprocessing\preprocessed_data_2012.csv"
)
encoding, _ = detect_encoding(csv_file)
df = pd.read_csv(csv_file, encoding=encoding)
print(len(filtered_files_2012) - df["group"].nunique())
