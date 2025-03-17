"""__init__ for src.inference directory"""

from inference.table_label_inference import main as run_table_label_inference
from inference.extract_inference_data import (
    extract_and_save_inference_data as run_filter_on_unlabeled_data,
)
