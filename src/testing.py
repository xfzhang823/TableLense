from data_processing.preprocessing_utils import get_filtered_files
from project_config import YEARBOOK_2022_DATA_DIR


filtered_files = get_filtered_files(
    source_data_dir=YEARBOOK_2022_DATA_DIR,
    filter_criterion=lambda name: name.startswith(("e", "E")),
)
print(filtered_files)
