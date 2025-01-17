""" Data Input/Output dir/file configuration 

# example_usagage (from modules)

from config import (
    PREPROCESSED_FILE,
    TRAINING_FILE,
    CLASSES,
    ...
)

"""

# config.py

from pathlib import Path

# Classes for labeling
CLASSES = ["table_data", "title", "metadata", "header", "empty"]


# *Data Input/Output paths
# Raw data directories
YEARBOOK_2012_DATA_DIR = Path(
    r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2012\html"
)
YEARBOOK_2022_DATA_DIR = Path(
    r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2022\zk\html"
)


# Project Base/Root Directory
BASE_DIR = Path(r"C:\github\table_lense")  # base/root directory

# Subdirectories under base directory that are part of the package inclue:
# - input_output
# - src
# rest of the sub directories, such as "data" are for ananlysis and other purposes only
# (not to be accessed programmatically)


# Input/Output directory, sub-directories, and file paths
INPUT_OUTPUT_DIR = BASE_DIR / "input_output"  # input/output data folder

# Sub directories
# Input
INPUT_DIR = INPUT_OUTPUT_DIR / "input"
RAW_DATA_FILE = INPUT_DIR / "raw_data.csv"  # all the original tables in csv format
SECTION_GROUP_MAPPING_FILE = INPUT_DIR / "yearbook_source_section_group_mapping.xlsx"


# Preprocessing Input/Output
PREPROCESSING_DIR = (
    INPUT_OUTPUT_DIR / "preprocessing"
)  # preprocessed data (each row converted to arragy friendly format)
PREPROCESSED_2012_DATA_FILE = (
    PREPROCESSING_DIR / "preprocessed_data_2012.csv"
)  # output of raw data; input of filtered production data
PREPROCESSED_2022_DATA_FILE = (
    PREPROCESSING_DIR / "preprocessed_data_2022.csv"
)  # output of raw data; input of filtered production data
PREPROCESSED_ALL_DATA_FILE = (
    PREPROCESSING_DIR / "preprocessed_data_all.csv"
)  # output of raw data; input of filtered production data
# PREPROCESSED_TEMP_MISSING_DATA_FILE = (
#     PREPROCESSING_DIR / "preprocessed_temp_missing.csv"
# )

FILTERED_PRODUCTION_DATA_FILE = (
    PREPROCESSING_DIR / "filtered_production_data.csv"
)  # production data excluding training data; all unlabeled preprocessed data
# output of production data; input of raw inference data


# Training Input/Output
TRAINING_DIR = INPUT_OUTPUT_DIR / "training"  # final output
TRAINING_INFERENCE_DATA_FILE = TRAINING_DIR / "training_and_inference_data_v2.csv"

# manually labeled data to train the model
TRAINING_DATA_FILE = TRAINING_DIR / "training_data_v1.csv"


# Model Input/Output
# outputs of training; inputs of inference
NN_MODELS_DIR = INPUT_OUTPUT_DIR / "nn_models"  # model output directory
MODEL_PTH_FILE = NN_MODELS_DIR / "simple_nn_model.pth"  # trained model output
TRAINING_EMBEDDINGS_PKL_FILE = (
    NN_MODELS_DIR / "training_embeddings.pkl"
)  # embeddings of training data
INFERENCE_EMBEDDINGS_PKL_FILE = (
    NN_MODELS_DIR / "inference_embeddings.pkl"
)  # embeddings of unlabeled production data (for inference)
TEST_DATA_PTH_FILE = NN_MODELS_DIR / "test_data.pth"  # model data for test data
TRAIN_TEST_IDX_PTH_FILE = (
    NN_MODELS_DIR / "train_test_indices.pth"
)  # index data for train/test data

EVALUATION_DIR = INPUT_OUTPUT_DIR / "evaluation"
EVALUATION_REPORT_FILE = EVALUATION_DIR / "evaluation_report.txt"
CONFUSION_MATRIX_FILE = EVALUATION_DIR / "confusion_matrix.html"

# Inference Input/Outputs
INFERENCE_DIR = INPUT_OUTPUT_DIR / "inference"
RAW_INFERENCE_OUTPUT_DATA_FILE = (
    INFERENCE_DIR / "raw_inference_output.csv"
)  # output of filtered production data; input of cleaned inference data
INFERENCE_INPUT_DATA_FILE = (
    INFERENCE_DIR / "inference_input_data.csv"
)  # preprocesssed and unlabeled data, excluding training data
CLEANED_INFERENCE_OUTPUT_DATA_FILE = INFERENCE_DIR / "cleaned_inference_outputs.csv"  #
COMBINED_INFERENCE_OUTPUT_DATA_FILE = INFERENCE_DIR / "combined_labeled_data.csv"
# inference data & training data (labeled) after data cleansing
# output of cleane inference data + training data; input of database
