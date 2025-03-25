"""Data Input/Output dir/file configuration

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
NUM_TO_LABEL = {idx: label for idx, label in enumerate(CLASSES)}

# * Data file paths
"""
/project_root
├── /pipeline_data               # ✅ Main folder for ML pipeline data (instead of input_output)
│   ├── /input                   # ✅ Raw input files
│   │   ├── raw_data.csv         # Original tables in CSV format
│   │   ├── yearbook_source_section_group_mapping.xlsx
│   ├── /preprocessing           # ✅ Preprocessing step
│   │   ├── preprocessed_data_2012.csv
│   │   ├── preprocessed_data_2022.csv
│   │   ├── preprocessed_data_all.csv    # All preprocessed data (before split)
│   │   ├── filtered_inference_input.csv # Filtered production data (used for inference)
│   ├── /training_inference       # ✅ Holds combined dataset before splitting
│   │   ├── training_and_inference_data_v2.csv  # Combined before split
│   │   ├── /training             # ✅ Training-specific data
│   │   │   ├── training_data_v1.csv     # Raw training data
│   │   │   ├── cleaned_training_data.csv  # Final cleaned training data
│   │   ├── /inference            # ✅ Inference-specific data
│   │   │   ├── inference_input_data.csv  # Preprocessed and unlabeled, excluding training
│   │   │   ├── raw_inference_output_data.csv  # Raw output before cleaning
│   │   │   ├── cleaned_inference_output_data.csv  # Final cleaned inference data
│   ├── /nn_models                # ✅ Stores models and embeddings
│   │   ├── simple_nn_model.pth    # Trained model file
│   │   ├── training_embeddings.pkl
│   │   ├── inference_embeddings.pkl
│   │   ├── test_data.pth
│   │   ├── train_test_indices.pth
│   ├── /evaluation               # ✅ Evaluation reports and metrics
│   │   ├── evaluation_report.txt
│   │   ├── confusion_matrix.html
│   ├── /output                    # ✅ Final combined output after processing
│   │   ├── combined_cleaned_training_and_inference_data.csv
│   ├── /logs                      # ✅ (Optional) Logs for debugging
│   │   ├── pipeline_run.log
│   │   ├── training_log.txt
│   │   ├── inference_log.txt
"""

# Raw data directories
YEARBOOK_2012_DATA_DIR = Path(
    r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2012\html"
)
YEARBOOK_2022_DATA_DIR = Path(
    r"C:\Users\xzhan\Documents\China Related\China Year Books\China Year Book 2022\zk\html"
)


# Project Base/Root Directory
BASE_DIR = Path(r"C:\github\table_lense")  # base/root directory

# Input/Output directory, sub-directories, and file paths
PIPELINE_DATA_DIR = BASE_DIR / "pipeline_data"  # input/output data folder

# Sub directories
# Input
INPUT_DIR = PIPELINE_DATA_DIR / "input"
RAW_DATA_FILE = INPUT_DIR / "raw_data.csv"  # all the original tables in csv format
SECTION_GROUP_MAPPING_FILE = INPUT_DIR / "yearbook_source_section_group_mapping.xlsx"


# Preprocessing Input/Output
PREPROCESSING_DIR = (
    PIPELINE_DATA_DIR / "preprocessing"
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

# FILTERED_PRODUCTION_DATA_FILE = (
#     PREPROCESSING_DIR / "filtered_production_data.csv"
# )  # production data excluding training data; all unlabeled preprocessed data
# # output of production data; input of raw inference data

# Training and Inference
TRAINING_INFERENCE_DIR = PIPELINE_DATA_DIR / "training_inference"
TRAINING_INFERENCE_DATA_FILE = (
    TRAINING_INFERENCE_DIR / "training_and_inference_data_v1.csv"
)

# Training
TRAINING_DIR = TRAINING_INFERENCE_DIR / "training"
TRAINING_DATA_FILE = TRAINING_DIR / "training_data_v1.csv"  # training output file (raw)
CLEANED_TRAINING_OUTPUT_DATA_FILE = TRAINING_DIR / "cleaned_training_data.csv"

# Inference Input/Outputs
INFERENCE_DIR = TRAINING_INFERENCE_DIR / "inference"
INFERENCE_INPUT_DATA_FILE = (
    INFERENCE_DIR / "inference_input_data.csv"
)  # preprocesssed and unlabeled data, excluding training data
RAW_INFERENCE_OUTPUT_DATA_FILE = (
    INFERENCE_DIR / "raw_inference_output_data.csv"
)  # output of filtered production data; input of cleaned inference data
CLEANED_INFERENCE_OUTPUT_DATA_FILE = INFERENCE_DIR / "cleaned_inference_output_data.csv"

# Output
OUTPUT_DIR = PIPELINE_DATA_DIR / "output"
COMBINED_CLEANED_OUTPUT_DATA_FILE = (
    OUTPUT_DIR / "combined_cleaned_training_and_inference_data.csv"
)  # * final output
RECONSTRUCTED_TABLES_DIR = (
    OUTPUT_DIR / "reconstructed_tables"
)  # * final output - reconconstructed table csv files


# Model Input/Output
# outputs of training; inputs of inference
NN_MODELS_DIR = PIPELINE_DATA_DIR / "nn_models"  # model output directory
MODEL_PTH_FILE = NN_MODELS_DIR / "simple_nn_model.pth"  # trained model output
TRAINING_EMBEDDINGS_PKL_FILE = (
    NN_MODELS_DIR / "training_embeddings.pkl"
)  # embeddings of training data
INFERENCE_EMBEDDINGS_PKL_FILE = (
    NN_MODELS_DIR / "inference_embeddings.pkl"
)  # embeddings of unlabeled production data (for inference)
INFERENCE_EMBEDDINGS_CACHE_PKL_FILE = (
    NN_MODELS_DIR / "inference_cache_embeddings.pkl"
)  # cached_embeddings of unlabeled production data (for inference)

TEST_DATA_PTH_FILE = NN_MODELS_DIR / "test_data.pth"  # model data for test data
TRAIN_TEST_IDX_PTH_FILE = (
    NN_MODELS_DIR / "train_test_indices.pth"
)  # index data for train/test data

EVALUATION_DIR = PIPELINE_DATA_DIR / "evaluation"
EVALUATION_REPORT_FILE = EVALUATION_DIR / "evaluation_report.txt"
CONFUSION_MATRIX_FILE = EVALUATION_DIR / "confusion_matrix.html"
