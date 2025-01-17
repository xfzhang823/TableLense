"""
File Name: train_model.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Jul 31

This script trains a simple 4-hidden-layer neural network model on text data, generates embeddings,
and saves the trained model, test data, and indices for later evaluation and analysis.

Key techniques include:
- Compensating for data imbalance: Class weights (table_data outweighs other classes)
- Preventing overfitting: L2 regularization, dropout, and early stopping
- Combined both text embedding and positional parameter (added row_id feature)

It saves the following files:

- simple_nn_model.pth: 
  This file contains the state dictionary of the trained neural network model.
  The state dictionary is a Python dictionary object that maps each layer to 
  its parameters (e.g., weights and biases).
  This file will be used to load the trained model's parameters for inference 
  or further training later on.

- test_data.pth: 
  This file contains a dictionary with the test dataset and input dimension size 
  used during training. It includes:
  * X_test: The feature embeddings of the test set.
  * y_test: The labels of the test set.
  * input_dim: The dimension of the input features used to initialize the model.
  * original_indices: The original indices of the test set from the initial dataset.
  * These indices are used to map the test data back to the original data for further analysis.

- train_test_indices.pth: 
  This file contains the training and test indices used during the data split.
  It includes:
  * train_idx: The indices of the training set.
  * test_idx: The indices of the test set.
  * original_indices: The original indices of the test set from the initial dataset.
  * These indices are used to map the test data back to the original data for further analysis.

Training/Testing Process:
1. Check if embeddings file already exist: 
   - if yes, then load embeddings and skip to step 3.
   - if no, then go to step 2.

2. Load and preprocess the data: 
   Read the data from an Excel file, tokenize, and generate embeddings 
   using a pre-trained BERT model; 
   save embeddings on disk (w/t picke)

3. Combine text embeddings with additional features, extract labels, 
   and split the data into training and testing sets using GroupShuffleSplit.

4. Manually calculate class weights.

5. Define the neural network architecture with dropout and L2 regularization, 
   and set up the loss function with class weights.

6. Train the model: Perform forward and backward passes, compute the loss 
   (including L2 regularization), and update the model parameters.

7. Validate the model: Evaluate the performance on the test set, 
   and save the best model based on validation loss.

8. Implement early stopping to prevent overfitting and save the test data 
   and indices for further analysis.
"""

import os
import logging
import pickle
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# Import custom classes/functions
from nn_models.simple_nn import SimpleNN
from nn_models.training_utils import (
    process_batch_for_embeddings,
    dynamic_batch_processing,
    train_model,
    load_data,
    split_data,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def generate_embeddings(df):
    """
    Generate embeddings with dynamic_batching/process_batch functions from training_utils.py file
    """
    results = dynamic_batch_processing(df, process_batch_for_embeddings)
    embeddings = np.concatenate([r[0] for r in results], axis=0)
    labels = np.concatenate([r[1] for r in results], axis=0)
    original_indices = np.concatenate([r[2] for r in results], axis=0)
    groups = np.concatenate([r[3] for r in results], axis=0)
    return embeddings, labels, original_indices, groups


def prepare_data(training_data_path, embeddings_save_path):
    """Load data and generate embeddings if they don't already exist."""
    if os.path.exists(embeddings_save_path):
        logging.info("Loading embeddings from disk")
        with open(embeddings_save_path, "rb") as f:
            embeddings, labels, original_indices, groups = pickle.load(f)
    else:
        logging.info("Loading data from Excel file")
        df = pd.read_excel(training_data_path)
        df["original_index"] = df.index
        embeddings, labels, original_indices, groups = generate_embeddings(df)
        with open(embeddings_save_path, "wb") as f:
            pickle.dump((embeddings, labels, original_indices, groups), f)
    return embeddings, labels, original_indices, groups


def main():
    """Main script to train the model"""

    # Set File Paths
    training_data_path = r"C:\github\china stats yearbook RAG\data\training\training data 2024 Jul 31.xlsx"
    model_dir_path = r"C:\github\china stats yearbook RAG\outputs\models"
    model_path = os.path.join(model_dir_path, "simple_nn_model.pth")
    test_data_path = os.path.join(model_dir_path, "test_data.pth")
    indices_path = os.path.join(model_dir_path, "train_test_indices.pth")
    embeddings_path = os.path.join(model_dir_path, "embeddings.pkl")

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Check if embeddings file exists
    if os.path.exists(embeddings_path):
        logging.info("Loading embeddings from disk")
        with open(embeddings_path, "rb") as f:
            embeddings, labels, original_indices, groups = pickle.load(f)
    else:
        logging.info("Loading data from Excel file")
        # Load the Excel file with original indices
        df = pd.read_excel(training_data_path)
        df["original_index"] = df.index

        # Generate embedding using generate_embeddings func and save to disk w/t pickle
        logging.info("Generating embeddings")
        embeddings, labels, original_indices, groups = generate_embeddings(df)
        with open(embeddings_path, "wb") as f:
            pickle.dump((embeddings, labels, original_indices, groups), f)

    # Log the embedding generation/loading step
    logging.info("Embeddings are ready")

    # Ensure embeddings are numeric
    embeddings = embeddings.astype(np.float32)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, train_idx, test_idx, test_original_indices = (
        split_data(df, embeddings, labels, groups)
    )

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # class weighting technique to compensate for label imbalance
    class_counts = np.bincount(y_train.cpu().numpy())
    total_samples = len(y_train)
    class_weights = total_samples / (
        len(np.unique(y_train.cpu().numpy())) * class_counts
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # input dimension and hidden layers of the neural network
    input_dim = X_train.shape[1]
    hidden_dims = [128, 64, 32, 16]

    # Instantiate the NN model
    model_nn = SimpleNN(input_dim, hidden_dims).to(device)
    criterion = CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

    logging.info("Initialized the model")
    # Print unique labels in y_train for diagnostics
    unique_labels = np.unique(y_train.cpu().numpy())
    print("Unique labels in y_train:", unique_labels)

    train_model(
        model_nn,
        criterion,
        optimizer,
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size=100,
        model_path=model_path,
        test_data_path=test_data_path,
        indices_path=indices_path,
    )

    # Define all possible classes
    all_classes = np.array([0, 1, 2, 3, 4])

    # Print all_classes for diagnostics
    print("All classes:", all_classes)


if __name__ == "__main__":
    main()
