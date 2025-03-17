"""
File Name: train_pipeline.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Dec

Description:
This script orchestrates the full pipeline for training, evaluating, and reporting results
for a neural network model on text data. The pipeline follows a modular approach by separating
data preparation, model training, and evaluation into distinct functions.

This structure enhances readability, reusability, and maintainability.

Key techniques include:
- Compensating for data imbalance: Class weights (table_data outweighs other classes)
- Preventing overfitting: L2 regularization, dropout, and early stopping
- Combined both text embedding and positional parameters (added row_id and other features)

Key features and techniques include:
- Compensating for data imbalance: Class weights are used to balance the impact of
different classes.
- Preventing overfitting: L2 regularization, dropout, and early stopping.
- Combined both text embeddings and positional parameters
(such as row_id and other features).

Module Structure:
1. prepare_data: Handles embedding generation, data loading, and splitting into training
and testing sets.
2. train_model_nn: Manages the model training process, including defining the neural network
architecture and running the training loop.
3. evaluate_and_report: Handles model evaluation, prints classification metrics,
and lists misclassified samples.
4. run_training_pipeline: Orchestrates the entire process by calling the above functions
sequentially.

Pipeline Outputs:
- simple_nn_model.pth:
- embeddings.pkl:
- test_data.pth:
- train_test_indices.pth:

- simple_nn_model.pth: Contains the state dictionary of the trained neural network model,
which includes weights and biases for all layers. It contains the state dictionary of
the trained neural network model. The state dictionary is a Python dictionary object that
maps each layer to its parameters (e.g., weights and biases). This file will be used to load
the trained model's parameters for inference or further training later on.

- embeddings.pkl: Stores generated embeddings, labels, original indices, and groups for
the dataset, saved as a serialized Python object. These embeddings are used as input features
for model training.

- test_data.pth: Contains the test dataset and metadata such as input dimension
and original indices for further analysis.
  * It includes:
  * X_test: The feature embeddings of the test set.
  * y_test: The labels of the test set.
  * input_dim: The dimension of the input features used to initialize the model.
  * original_indices: The original indices of the test set from the initial dataset,
  *used to map the test data back to the original data for further analysis.

- train_test_indices.pth: Stores the training and testing indices for reference and
reproducibility.
*It includes:
  * train_idx: The indices of the training set.
  * test_idx: The indices of the test set.
  * original_indices: The original indices of the test set from the initial dataset,
  *used to map the test data back to the original data for further analysis.

Training/Testing Process:
1. Load or Generate Embeddings:
   - If embeddings exist, they are loaded from disk.
   - If not, the script reads the data from an Excel file, tokenizes it,
   generates embeddings using a pre-trained BERT model, and saves the embeddings.

2. Split the Data:
   - Combines text embeddings with additional features.
   - Splits the data into training and testing sets using GroupShuffleSplit to ensure that
   samples from the same group are not split across training and testing sets.

3. Train the Model:
   - Defines the neural network architecture with dropout and L2 regularization.
   - Sets up the loss function with class weights to handle class imbalance.
   - Performs forward and backward passes, computes the loss (including L2 regularization),
   and updates model parameters.

4. Evaluate and Save the Model:
   - Evaluates the model on the test set and prints a classification report.
   - Prints misclassified "header" samples for further analysis.
   - Implements early stopping to prevent overfitting and saves the best-performing model.
"""

# Dependancies

# Standard Python libraries
from pathlib import Path
import logging
import time
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
import numpy as np

# User defined
from nn_models.simple_nn import SimpleNN
from nn_models.training_utils import (
    load_data,
    load_or_generate_embeddings,
    split_data,
    train_model,
    generate_embeddings,
)
from nn_models.evaluate_model import (
    evaluate_model,
    load_test_data_and_text,
    print_misclassified_headers,
    plot_confusion_matrix_altair,
)
from project_config import (
    TRAINING_DATA_FILE,
    MODEL_PTH_FILE,
    TRAINING_EMBEDDINGS_PKL_FILE,
    TEST_DATA_PTH_FILE,
    TRAIN_TEST_IDX_PTH_FILE,
    EVALUATION_REPORT_FILE,
    CONFUSION_MATRIX_FILE,
)

logger = logging.getLogger(__name__)


def prepare_data():
    """
    Load or generate embeddings, load data, and split into training and testing sets.

    Returns:
        tuple: X_train, X_test, y_train, y_test (torch tensors), input_dim, and device.

    The primary task is managing the embedding process, which is the most computational
    intensive.

    * Workflow of the embedding process:
    [train_model.py pipeline]
        └─> prepare_data(...)
            └─> load_or_generate_embeddings(
                    data_file=...,
                    embeddings_file=...,
                    generate_embeddings_func=generate_embeddings
                )
                    └─> if not cached -> generate_embeddings(...)
                        └─> dynamic_batch_processing(...)
                                └─> process_batch_for_embeddings(...)
    """
    #! Crucial Step: Set device to use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Skip embedding generation if the embeddings file already exists

    logger.info("Loading or generating embeddings...")
    embeddings, labels, original_indices, groups = load_or_generate_embeddings(
        data_file=TRAINING_DATA_FILE,
        embeddings_file=TRAINING_EMBEDDINGS_PKL_FILE,
        generate_embeddings_func=generate_embeddings,
    )

    logger.info("Loading training data...")
    df = load_data(TRAINING_DATA_FILE)

    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test, _, _, _ = split_data(
        df, embeddings, labels, groups
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    input_dim = X_train.shape[1]

    return X_train, X_test, y_train, y_test, input_dim, device


def train_model_nn(X_train, y_train, X_test, y_test, input_dim, device):
    """
    Train the neural network model.

    Args:
        X_train (torch.Tensor): Training feature data.
        y_train (torch.Tensor): Training labels.
        X_test (torch.Tensor): Test feature data.
        y_test (torch.Tensor): Test labels.
        input_dim (int): Number of input features.
        device (torch.device): Computation device (CPU or GPU).

    Returns:
        SimpleNN: Trained neural network model.
    """
    if MODEL_PTH_FILE.exists():
        logger.info("Model already exists. Skipping training...")
        return SimpleNN(input_dim).to(device)

    # Training model
    logger.info("Start training model...")

    # Specifies the number of neurons (or units) for each hidden layer
    hidden_dims = [128, 64, 32, 16]

    # Ininitate the NN model
    model_nn = SimpleNN(input_dim, hidden_dims).to(device)

    class_counts = np.bincount(y_train.cpu().numpy())
    class_weights = len(y_train) / (
        len(np.unique(y_train.cpu().numpy())) * class_counts
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model_nn.parameters(), lr=0.001)
    # Adam model stands for Adaptive Moment Estimation;
    # it's the most popular go-to model for beginners and even experts,
    # because it's fast and easy (adaptive learning rates, easy to setup, fast convergence...)

    logger.info("Training the model...")
    train_model(
        model=model_nn,
        criterion=criterion,
        optimizer=optimizer,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_epochs=20,
        batch_size=128,
        patience=3,
        model_path=MODEL_PTH_FILE,
        test_data_path=TEST_DATA_PTH_FILE,
        indices_path=TRAIN_TEST_IDX_PTH_FILE,
    )

    return model_nn


def evaluate_and_report():
    """
    Evaluate the trained model and print the classification report and misclassified samples.
    Automatically runs training if the model or test data does not exist.
    """
    if not TEST_DATA_PTH_FILE.exists():
        logger.warning("Model or test data missing. Running the training pipeline...")
        run_model_training_pipeline()  # Automatically run training to create the missing files

    logger.info("Evaluating the model...")

    # Load data for evaluation
    X_test, y_test, input_dim, test_original_indices, text_data = (
        load_test_data_and_text(
            test_data_file=TEST_DATA_PTH_FILE,
            training_data_file=TRAINING_DATA_FILE,
        )
    )

    # The model path is directly taken from project_config
    model_file = MODEL_PTH_FILE

    # Performe evaluation: Predict labels and log/print evaluation report
    predicted, report, confusion_matrix, classes = evaluate_model(
        model_file, X_test, y_test, input_dim
    )
    logger.info(f"Evaluation Report:\n{report}")

    logger.info("Save classification report...")
    with open(EVALUATION_REPORT_FILE, "w") as f:
        f.write("Classification Report:\n")
        f.write(report)

    logger.info("Printing misclassified headers...")
    print_misclassified_headers(y_test, predicted, test_original_indices, text_data)

    logger.info("Plot and save confustion matrix...")
    plot_confusion_matrix_altair(
        cm=confusion_matrix, classes=classes, file_path=CONFUSION_MATRIX_FILE
    )


def run_model_training_pipeline():
    """
    Orchestrate the entire training pipeline.
    """
    # Step 0: check if the pipeline should be skipped

    # Collect all necessary files to check
    required_files = {
        "Model file": MODEL_PTH_FILE,
        "Embeddings file": TRAINING_EMBEDDINGS_PKL_FILE,
        "Test data file": TEST_DATA_PTH_FILE,
        "Train/Test indices file": TRAIN_TEST_IDX_PTH_FILE,
        "Evaluation report file": EVALUATION_REPORT_FILE,
        "Confustion matrix file": CONFUSION_MATRIX_FILE,
    }

    # * Check to initiate or skip pipeline: if all files exist or not
    missing_files = [name for name, path in required_files.items() if not path.exists()]

    if not missing_files:
        logger.info("All necessary files exist. Skipping the entire training pipeline.")
        return  # Early exit if everything exists

    # Step 1. Start pipeline
    start_time = time.time()  # Start timer
    logger.info("Starting training pipeline...")

    # Step 2. Prepare data
    X_train, X_test, y_train, y_test, input_dim, device = prepare_data()

    # Step 3. Train data
    train_model_nn(X_train, y_train, X_test, y_test, input_dim, device)

    # Step 3. Evaluate and Report
    evaluate_and_report()

    end_time = time.time()  # Record end time
    elapsed_time_seconds = end_time - start_time
    elapsed_time_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_seconds))

    logger.info(f"Training pipeline completed.")
    logger.info(f"Pipeline completed in {elapsed_time_hms}.")


if __name__ == "__main__":
    run_model_training_pipeline()
