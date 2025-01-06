"""
!Older version: less modular, more coherent!

File Name: train_pipeline.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Dec

Description:
This script orchestrates the full pipeline for training a simple 4-hidden-layer 
neural network model on text data. It includes steps for generating or loading embeddings, 
splitting data into training and testing sets, and training the model while handling 
data imbalance, regularization, and early stopping. 

Key techniques include:
- Compensating for data imbalance: Class weights (table_data outweighs other classes)
- Preventing overfitting: L2 regularization, dropout, and early stopping
- Combined both text embedding and positional parameters (added row_id and other features)

The pipeline saves the following files:
- simple_nn_model.pth: 
  This file contains the state dictionary of the trained neural network model. 
  The state dictionary is a Python dictionary object that maps each layer to 
  its parameters (e.g., weights and biases). This file will be used to load 
  the trained model's parameters for inference or further training later on.

- embeddings.pkl: 
  This file contains the generated embeddings, labels, original indices, and groups 
  for the dataset, saved as a serialized Python object using the pickle module. 
  These embeddings are used as input features for model training.

- test_data.pth: 
  This file contains a dictionary with the test dataset and input dimension size used 
  during training. 
  It includes:
  * X_test: The feature embeddings of the test set.
  * y_test: The labels of the test set.
  * input_dim: The dimension of the input features used to initialize the model.
  * original_indices: The original indices of the test set from the initial dataset, 
  *used to map the test data back to the original data for further analysis.

- train_test_indices.pth: 
  This file contains the training and test indices used during the data split.
  
  It includes:
  * train_idx: The indices of the training set.
  * test_idx: The indices of the test set.
  * original_indices: The original indices of the test set from the initial dataset, 
  *used to map the test data back to the original data for further analysis.

Training/Testing Process:
1. Load or generate embeddings:
   The script first checks if the embeddings file already exists. 
   - If yes, it loads the embeddings from disk.
   - If no, it reads the data from an Excel file, tokenizes, and generates embeddings 
   using a pre-trained BERT model, then saves the embeddings to disk.

2. Split the data:
   The script combines text embeddings with additional features, extracts labels, 
   and splits the data into training and testing sets using GroupShuffleSplit, 
   ensuring that samples from the same group are not represented in both sets.

3. Train the model:
   The script defines the neural network architecture with dropout and L2 regularization, 
   sets up the loss function with class weights, and trains the model, performing forward 
   and backward passes, computing the loss (including L2 regularization), and 
   updating the model parameters.

4. Validate and save the model:
   The model is validated on the test set, and the best model is saved based on 
   validation loss. Early stopping is implemented to prevent overfitting, 
   and the test data and indices are saved for further analysis.
"""

# Dependencies
import os
import logging
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# Custom classes/functions
from nn_models.simple_nn import SimpleNN
from nn_models.training_utils import (
    generate_embeddings,
    load_or_generate_embeddings,
    load_data,
    split_data,
    train_model,
)
from nn_models.evaluate_model import (
    evaluate_model,
    load_model_and_test_data,
    print_misclassified_headers,
)
from project_config import (
    TRAINING_DATA_FILE,
    MODEL_PTH_FILE,
    TRAINING_EMBEDDINGS_PKL_FILE,
    TEST_DATA_PTH_FILE,
    TRAIN_TEST_IDX_PTH_FILE,
)
import logging_config


# Set up logging
logger = logging.getLogger(__name__)


def training_pipeline(
    training_data_path,
    model_save_path,
    embeddings_save_path,
    test_data_path,
    indices_path,
):
    """
    Run the entire training pipeline, including data loading, embedding generation,
    model training, and evaluation.
    """
    # Pipeline-level progress bar
    with tqdm(total=4, desc="Pipeline Progress", unit="step") as pbar:

        # Load or generate embeddings
        pbar.set_description("Loading or generating embeddings")  # set progress bar
        embeddings, labels, original_indices, groups = load_or_generate_embeddings(
            training_data_path, embeddings_save_path, generate_embeddings
        )
        pbar.update(1)

        # Load the original data to use for splitting
        pbar.set_description("Loading original data")  # set progress bar
        df = load_data(training_data_path)
        pbar.update(1)

        # Split the data
        pbar.set_description("Splitting data")  # set progress bar
        X_train, X_test, y_train, y_test, train_idx, test_idx, test_original_indices = (
            split_data(df, embeddings, labels, groups)
        )
        pbar.update(1)

        # Convert to torch tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)

        # Train the model using the train_model function from training_utils
        pbar.set_description("Training the model")  # set progress bar
        model_nn = SimpleNN(X_train.shape[1], [128, 64, 32, 16]).to(device)
        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.001)

        train_model(
            model_nn,
            criterion,
            optimizer,
            X_train,
            y_train,
            X_test,
            y_test,
            num_epochs=20,
            batch_size=128,
            patience=3,
            model_path=model_save_path,
            test_data_path=test_data_path,
            indices_path=indices_path,
        )
        pbar.update(1)

        # Evaluate the model after training
        logging.info("Evaluating the model...")
        model_path, X_test, y_test, input_dim, test_original_indices, text_data = (
            load_model_and_test_data(
                os.path.dirname(model_save_path), training_data_path
            )
        )
        predicted, report = evaluate_model(model_path, X_test, y_test, input_dim)
        print(report)

        # Print misclassified headers
        print_misclassified_headers(y_test, predicted, test_original_indices, text_data)


def main():
    """Main script to orchestrate the model training process"""

    # Data path
    training_data_path = TRAINING_DATA_FILE  # Labeled training data

    # Model file paths: Keep model output location constant
    model_save_path = MODEL_PTH_FILE
    embeddings_save_path = TRAINING_EMBEDDINGS_PKL_FILE
    test_data_path = TEST_DATA_PTH_FILE
    indices_path = TRAIN_TEST_IDX_PTH_FILE

    training_pipeline(
        training_data_path,
        model_save_path,
        embeddings_save_path,
        test_data_path,
        indices_path,
    )


if __name__ == "__main__":
    main()
