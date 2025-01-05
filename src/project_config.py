""" Data Input/Output dir/file configuration 

# example_usagage (from modules)

from config import (
    raw_data_path,
    production_data_path,
    filtered_production_data,
    training_data_path,
    model_path,
    embeddings_path,
    embeddings_inference_path,
    test_data_path,
    indices_path,
    raw_inference_output_data_path,
    cleaned_inference_output_data_path,
    combined_output_data_path,
    dbs_path
)

"""

# config.py

from pathlib import Path

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
TRAINING_DATA_FILE = (
    TRAINING_DIR / "training_data.csv"
)  # manually labeled data to train the model
TRAINING_INFERENCE_DATA_FILE = TRAINING_DIR / "training_and_inference_data_v1.csv"

# Model Input/Output
# outputs of training; inputs of inference
MODELS_INPUT_OUTPUT_DIR = INPUT_OUTPUT_DIR / "models"  # model output directory
model_path = MODELS_INPUT_OUTPUT_DIR / "simple_nn_model.pth"  # trained model output
embeddings_path = (
    MODELS_INPUT_OUTPUT_DIR / "embeddings.pkl"
)  # embeddings of training data
embeddings_inference_path = (
    MODELS_INPUT_OUTPUT_DIR / "inference_embeddings.pkl"
)  # embeddings of unlabeled production data (for inference)
test_data_path = MODELS_INPUT_OUTPUT_DIR / "test_data.pth"  # model data for test data
indices_path = (
    MODELS_INPUT_OUTPUT_DIR / "train_test_indices.pth"
)  # index data for train/test data

# Inference Input/Outputs
INFERENCE_INPUT_OUTPUT_DIR = INPUT_OUTPUT_DIR / "inference"
raw_inference_output_data_path = (
    INFERENCE_INPUT_OUTPUT_DIR / "raw_inference_output.csv"
)  # output of filtered production data; input of cleaned inference data
cleaned_inference_output_data_path = (
    INFERENCE_INPUT_OUTPUT_DIR / "cleaned_inference_outputs.csv"
)  #
combined_output_data_path = INFERENCE_INPUT_OUTPUT_DIR / "combined_labeled_data.csv"
# inference data & training data (labeled) after data cleansing
# output of cleane inference data + training data; input of database

# Database Input/Output Directory
DATABASES_INPUT_OUTPUT_DIR = INPUT_OUTPUT_DIR / "databases"
database_path = (
    DATABASES_INPUT_OUTPUT_DIR / "china_stats_yearbooks.db"
)  # path to the SQL dbs storing processes and formatted tables
# output of combined output data (from inference pipeline)
# input of RAG
database_upload_summary_path = DATABASES_INPUT_OUTPUT_DIR / "upload_summary.txt"

# Output directory: final output data (if necessary)
OUTPUT_DIR = INPUT_OUTPUT_DIR / "output"
# TBA (i.e., summary files)
