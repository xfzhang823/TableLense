"""TBA"""
from pipelines.preprocessing_pipeline import run_preprocessing_pipeline
from project_config import (
    YEARBOOK_2012_DATA_DIR,
    YEARBOOK_2022_DATA_DIR,
    preprocessed_2012_data_file,
)
from nn_models.training_utils import process_batch

import logging
import logging_config

# Set up logger
logger = logging.getLogger(__name__)


def main():
    run_preprocessing_pipeline(raw_data_file=, processed_data_file=preprocessed_2012_data_file)
