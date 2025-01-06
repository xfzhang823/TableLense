

# Table Lense: Project README

## Project Overview

The **Table Lense** project automates the preprocessing, training, and inference of tabular data extracted from yearbooks (2012 and 2022 editions). It performs tasks such as data cleaning, model training, classification, and reporting. The project leverages neural networks with features like L2 regularization, dropout, and early stopping to improve performance.

## Table of Contents
1. [Features](#features)
2. [Directory Structure](#directory-structure)
3. [Main Components](#main-components)
4. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
5. [Model Training Pipeline](#model-training-pipeline)
6. [Inference Pipeline](#inference-pipeline)
7. [Key Configuration Paths](#key-configuration-paths)
8. [Usage Instructions](#usage-instructions)
9. [Dependencies](#dependencies)
10. [Authors](#authors)

---

## Features

- **Synchronous and Asynchronous Preprocessing**: Supports multi-threaded and asynchronous workflows for handling large Excel files.
- **Data Cleaning and Normalization**: Handles missing values, merges cells intelligently, and normalizes headers.
- **Neural Network Architecture**:
  - 4-layer feed-forward neural network with ReLU activation and dropout layers.
  - Supports class imbalance handling with class weights.
- **Embedding Generation**: Utilizes BERT-based embeddings combined with metadata features (e.g., row positions).
- **Early Stopping**: Prevents overfitting by monitoring validation loss.
- **Inference Pipeline**: Automatically classifies unlabeled tabular data into categories (e.g., table_data, title, metadata, header, empty).
- **Misclassification Reporting**: Generates a report of misclassified samples for further inspection.

---

## Directory Structure

```plaintext
|-- input_output/
|   |-- preprocessing/         # Preprocessed files
|   |-- training/              # Training data and model outputs
|   |-- inference/             # Inference data and results
|-- nn_models/                 # Neural network scripts and model checkpoints
|-- data_processing/           # Preprocessing utilities and core functions
|-- pipelines/                 # High-level pipeline orchestration scripts
|-- utils/                     # Helper functions (e.g., encoding detection)
```

---

## Main Components

### 1. **Preprocessing Modules**
   - `preprocess_data.py` & `preprocess_data_async.py`: Preprocesses Excel files into a structured CSV format synchronously or asynchronously.
   - `preprocessing_utils.py`: Contains utility functions for clearing sheets, copying content, and detecting empty rows.
   - `preprocessing_pipeline_async.py`: Runs the full preprocessing pipeline for both 2012 and 2022 yearbooks, handling missing files iteratively.

### 2. **Model Training**
   - `train_model.py`: Main script for training the neural network.
   - `training_utils.py`: Contains batch processing, model training functions, and utilities for splitting data while preserving table integrity.
   - `simple_nn.py`: Defines the architecture of the neural network, with support for L2 regularization and dropout.

### 3. **Model Evaluation**
   - `evaluate_model.py`: Loads the trained model and evaluates it using classification metrics. Prints misclassified "header" samples for debugging.

### 4. **Inference Pipeline**
   - `inference_pipeline.py`: Runs the full inference process, classifies new data, and saves combined cleaned results.

---

## Data Preprocessing Pipeline

The preprocessing pipeline:
1. **Load and Validate Files**: Loads Excel files and checks for the required format.
2. **File Filtering**: Filters English files based on naming conventions (e.g., suffix/prefix "e").
3. **Asynchronous File Processing**: Reads, cleans, and serializes each row as structured data.
4. **Missing Files Handling**: Iteratively handles missing files until no files remain.
5. **Combines Results**: Saves the processed data for training and inference in CSV format.

---

## Model Training Pipeline

The training pipeline:
1. **Embedding Generation**: Generates text embeddings using BERT and combines them with metadata (row_id, is_title).
2. **Data Splitting**: Splits data into training and testing sets using `GroupShuffleSplit`.
3. **Training**: Trains the neural network with early stopping and L2 regularization.
4. **Checkpointing**: Saves model weights, embeddings, and indices for reproducibility.
5. **Evaluation**: Evaluates the model on the test set and logs classification reports.

Outputs:
- `simple_nn_model.pth`: Trained model checkpoint.
- `test_data.pth`: Test set embeddings and metadata.
- `train_test_indices.pth`: Training/testing indices.
- `evaluation_report.txt`: Classification report.

---

## Inference Pipeline

The inference pipeline:
1. **Filtering**: Removes training data from the full dataset.
2. **Embedding Generation**: Generates embeddings for the unlabeled data.
3. **Classification**: Classifies data using the trained model.
4. **Data Cleaning**: Cleans and relabels the classified data.
5. **Result Combination**: Combines labeled training and inference results into a single CSV file for further analysis.

---

## Key Configuration Paths

Configuration paths are stored in `project_config.py`:
- `YEARBOOK_2012_DATA_DIR`: Path to 2012 yearbook files.
- `YEARBOOK_2022_DATA_DIR`: Path to 2022 yearbook files.
- `TRAINING_DATA_FILE`: Path to manually labeled training data.
- `MODEL_PTH_FILE`: Path to the trained neural network model checkpoint.
- `INFERENCE_INPUT_DATA_FILE`: Path to filtered production data for inference.

---

## Usage Instructions

1. **Run Preprocessing Pipeline**:
   ```bash
   python main.py --task preprocess
   ```

2. **Run Training Pipeline**:
   ```bash
   python main.py --task train
   ```

3. **Run Inference Pipeline**:
   ```bash
   python main.py --task inference
   ```

4. **Combined Execution**:
   All steps can be run in sequence using the main orchestrator:
   ```python
   asyncio.run(main())
   ```

---

## Dependencies

- `pandas`, `numpy`, `torch`, `transformers`, `xlwings`: Core libraries for data handling and model operations.
- `asyncio`, `concurrent.futures`: For asynchronous file processing.
- `tqdm`: Progress bar for data processing loops.

---

## Authors

**Xiao-Fei Zhang**  
This project was developed and maintained by Xiao-Fei Zhang. For questions, suggestions, or contributions, please contact via GitHub or project repository links.

