from functions import create_activation_function
import numpy as np

from functions import Softmax

class Model:
    def __init__(self, layers, activation_function, learning_rate):
        self.weights , self.biases = self.initialize_weights_bias(layers)
        self.activation_function = create_activation_function(activation_function)
        self.learning_rate = learning_rate

    def initialize_weights_bias(self, layers: tuple[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Initializes weights and biases for a fully connected neural network.
        
        Args:
            layers: tuple with the number of neurons in each layer.
        
        Returns:
            (weights, biases):
                weights[i] has shape (layers[i], layers[i-1])
                biases[i]  has shape (layers[i],)
        """
        if len(layers) < 2:
            raise ValueError("initialize_weights_bias: expected at least 2 layers (input + output), got 1.")
        
        weights = [np.random.uniform(-0.5, 0.5, (layers[i], layers[i-1])) for i in range(1, len(layers))]
        biases  = [np.random.uniform(-0.5, 0.5, layers[i]) for i in range(1, len(layers))]
        return weights, biases
    
    def forward(self, X_batch):
        """
        Does a forward pass through the MLP
        ---
        Args: 
            - X_batch: Mini-batch with values for forward pass.
        ---
        Returns:
            (Z, A)
                Tuple with the values form thw forwarding and their
                respective activations.
        """
        # Forward pass
        Z = []
        A = []

        # Only used if there are hidden layers
        if (len(self.weights) > 1):
            Z.append(X_batch @ self.weights[0].T + self.biases[0])
            A.append(self.activation_function.activation(Z[0]))

            # Next layers (Besides last since it will always use SoftMax)
            for l in range(1, len(self.weights) - 1):
                # Use last result for forwarding.
                Z.append(A[-1] @ self.weights[l].T + self.biases[l])
                A.append(self.activation_function.activation(Z[-1]))

            Z.append(A[-1] @ self.weights[-1].T + self.biases[-1])
            A.append(Softmax().activation(Z[-1]))
        # Only input and output layers.
        else:
            Z.append(X_batch @ self.weights[-1].T + self.biases[-1])
            A.append(Softmax().activation(Z[-1]))

        return Z, A
    
    def backward(self, Z, A, x_batch, y_batch, batch_size):
        """
        Does the backward propagation algorithm.
        """
        # For simplicity we assume SoftMax with cross entropy loss for output.
        dZ = (A[-1] - y_batch)  

        for l in reversed(range(len(self.weights))):
            A_prev = x_batch if l == 0 else A[l - 1]
            dW = (dZ.T @ A_prev) / batch_size
            db = np.mean(dZ)

            if l != 0:
                dA_prev = dZ @ self.weights[l] 
                dZ = dA_prev * self.activation_function.activation_derivative(Z[l-1])
            
            # Update Parameters
            self.weights[l] -= self.learning_rate * dW
            self.biases[l] -= self.learning_rate * db


    def one_shot(self, y_batch, output_size):
        one_shot_y = [[0] * output_size for _ in range(len(y_batch))]
        # If output is n for a sample, entry n - 1 is equal to 1.
        for i, output in enumerate(y_batch):
            one_shot_y[i][output-1] = 1

        return one_shot_y

    def train_model(self, train_x, train_y, batch_size):
        # For every batch.
        for i in range(0, len(train_x), batch_size):
            X_batch = train_x[i:i+batch_size]
            output_size = len(self.weights[-1])
            Y_batch = self.one_shot(train_y[i:i+batch_size], output_size)

            # Forward pass
            Z, A = self.forward(X_batch)
            self.backward(Z, A, X_batch, Y_batch, batch_size)

    def test_accuracy(self, test_x, test_y):
        pass
    def predict(self, sample):
        pass




