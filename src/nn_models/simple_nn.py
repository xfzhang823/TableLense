"""
File Name: simple_nn.py
Author: Xiao-Fei Zhang
Date: last updated on 2024 Jul 31

Description:
    Defines a simple neural network (SimpleNN) with 4 hidden layers for text 
    classification tasks.
    
    The model uses dropout for regularization, ReLU activation, and L2 regularization
    to prevent overfitting. It is designed to be used with embeddings generated
    from BERT and additional features like row and title positional indicators.

Module Dependencies:
    - torch: PyTorch library for building and training neural networks.
    - torch.nn: Neural network module in PyTorch.

Classes:
    - SimpleNN: A feed-forward neural network (fully connected) for classification tasks.

Usage Example:
    >>> model = SimpleNN(input_dim=128)
    >>> X = torch.randn(10, 128)  # 10 samples with 128 features
    >>> output = model(X)
    >>> print(output.shape)
    torch.Size([10, 5])
"""

# Dependencies
# From internal and external
from typing import List, Optional
import logging
import torch
import torch.nn as nn

# From project modules
import logging_config


# Set logger
logger = logging.getLogger(__name__)


# Define the neural network architecture with Dropout and L2 Regularization
class SimpleNN(nn.Module):
    """
    A feed-forward neural network for text classification tasks.
    The network has 4 hidden layers and supports dropout and L2 regularization to
    mitigate overfitting.

    Attributes:
        - hidden_layers (torch.nn.ModuleList): A list of linear layers forming the hidden
        layers.
        - l2_lambda (float): The L2 regularization coefficient.
        - output_layer (torch.nn.Linear): The final linear layer for classification.
        - relu (torch.nn.ReLU): The ReLU activation function used after each hidden layer.
        - dropout (torch.nn.Dropout): Dropout layer for regularization.

    Methods:
        forward(x):
            Performs a forward pass through the network.
        get_l2_regularization_loss():
            Computes the L2 regularization loss by summing the squared norms of the model's
            parameters.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        l2_lambda: float = 0.01,
    ) -> None:
        """
        Initializes the SimpleNN with the specified input dimension, hidden layers, and
        L2 regularization.

        Args:
            - input_dim (int): The number of input features.
            - hidden_dims (list of int): A list specifying the number of neurons in
            each hidden layer.
            - l2_lambda (float): The L2 regularization coefficient (default is 0.01).

        Example:
            >>> model = SimpleNN(input_dim=128)
            >>> print(model)
        """
        super(SimpleNN, self).__init__()

        # * Default hidden dimensions inside the method to avoid mutable default argument
        if hidden_dims is None:
            hidden_dims = [128, 64, 32, 16]  # Safe default value ()

        self.hidden_layers = nn.ModuleList()
        self.l2_lambda = l2_lambda  # L2 regularization lambda

        # Create the hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, 5)

        # Activation and dropout
        self.relu = nn.ReLU()  # ReLU activation function
        self.dropout = nn.Dropout(0.2)
        # Dropout rate 0.2 -> there is a 20% probability of dropping out each neuron
        # in the layer during training.

        self.l2_lambda = l2_lambda
        # l2_lambda is regularization parameter:
        # if it -> 1 then regularization is strong;
        # if -> 0, then regularization is weak

        # Log model structure
        logger.info(
            f"SimpleNN initialized with input_dim={input_dim}, hidden_dims={hidden_dims}, output_dim=5"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): A batch of input features (shape: [batch_size, input_dim]).

        Returns:
            torch.Tensor: The output logits (shape: [batch_size, num_classes]).
        """
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)  # apply ReLu activation
            x = self.dropout(x)  # apply dropout
        x = self.output_layer(x)

        logger.debug(f"Output shape after final layer: {x.shape}")

        return x

    def get_l2_regularization_loss(self) -> torch.Tensor:
        """
        Computes the L2 regularization loss.

        The L2 loss is computed as the sum of the squared norms of the model's parameters.

        Returns:
            torch.Tensor: The computed L2 regularization loss.

        L1 and L2 regularization:
        - prevent overfitting by adding a penalty to the loss function.
        - constrains the size of the weights, forcing the model to learn simpler patterns
        that generalize better to unseen data ("dampens" the weights of neurons in
        the network.)

        - L2 Regularization (Ridge): adds the squared sum of all weights to the loss function.
        - L1 Regularization (Lasso): adds the absolute sum of all weights to the loss function.
        """
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param)

        logger.debug(f"L2 regularization loss: {l2_loss.item()}")

        return self.l2_lambda * l2_loss
