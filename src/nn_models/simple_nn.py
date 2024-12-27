import torch
import torch.nn as nn


# Define the neural network architecture with Dropout and L2 Regularization
class SimpleNN(nn.Module):
    """
    A deeper feed-forward neural network with 4 hidden layers
    (also known as a fully connected neural network)
    """

    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16], l2_lambda=0.01):
        super(SimpleNN, self).__init__()
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
        # Dropout rate 0.2 -> there is a 20% probability of dropping out each neuron in the layer during training.
        self.l2_lambda = l2_lambda
        # l2_lambda is regularization parameter:
        # if it -> 1 then regularization is strong;
        # if -> 0, then regularization is weak

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)  # apply ReLu activation
            x = self.dropout(x)  # apply dropout
        x = self.output_layer(x)
        return x

    def get_l2_regularization_loss(self):
        """
        L1 and L2 regularization: prevent overfitting by adding a penalty to the loss function.
        It constrains the size of the weights, forcing the model to learn simpler patterns that
        generalize better to unseen data ("dampens" the weights of neurons in the network.)

        - L2 Regularization (Ridge): adds the squared sum of all weights to the loss function.
        - L1 Regularization (Lasso): adds the absolute sum of all weights to the loss function.
        """
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param)
        return self.l2_lambda * l2_loss
