from functions import create_activation_function
import numpy as np
import pickle

from functions import Activation_Function, Softmax

class MLPModel:
    def __init__(self, 
            activation_function: str | Activation_Function, 
            learning_rate: float, 
            layers: tuple[int] | None = None, 
            weights: list[list[list[float]]] | None = None,
            biases: list[list[float]] | None = None
    ):
        """
        Initializes a new MLP Model.

        Parameters
        ----------
        activation_function : str or Activation_Function
            Activation function for hidden layers.
        learning_rate : float
            Learning rate of the model.
        layers : tuple[int], optional
            Tuple with number of neurons for each layer.
        weights : list[list[list[float]]], optional
            Weights of the model (used when loading an existing model).
        biases : list[list[float]], optional
            Biases of the model (used when loading an existing model).

        Returns
        -------
        Model
            A new initialized Model object.
        """
        if not weights and not biases:
            self.weights, self.biases = self.initialize_weights_bias(layers)
        else:
            self.weights = weights
            self.biases = biases

        if isinstance(activation_function, str):
            self.activation_function = create_activation_function(activation_function)
        elif isinstance(activation_function, Activation_Function):
            self.activation_function = activation_function

        self.learning_rate = learning_rate

    def initialize_weights_bias(self, layers: tuple[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Initializes weights and biases for a fully connected neural network.
        
        Parameters
        -------
        layers : tuple[int] 
            Tuple with the number of neurons in each layer.
        
        Returns
        -------
        (weights, biases) : tuple[list[np.ndarray], list[np.ndarray]]
            weights[i] has shape (layers[i], layers[i-1]),
            biases[i]  has shape (layers[i],)
        """
        if len(layers) < 2:
            raise ValueError("initialize_weights_bias: expected at least 2 layers (input + output), got 1.")
        
        weights = [np.random.uniform(-0.5, 0.5, (layers[i], layers[i-1])) for i in range(1, len(layers))]
        biases  = [np.random.uniform(-0.5, 0.5, layers[i]) for i in range(1, len(layers))]
        return weights, biases
    
    def forward(self, X_batch: list[list[float]]) -> tuple[list[list[float]]]:
        """
        Does a forward pass through the MLP

        Parameters
        ------- 
        X_batch : list[list[float]] 
            Mini-batch with values for forward pass.

        Returns
        -------
        (Z, A) : tuple[list[list[float]]]
            Tuple with the values form the forwarding and their
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
    
    def backward(self, 
                Z: list[list[float]], 
                A: list[list[float]], 
                x_batch: list[list[float]], 
                y_batch: list[float], 
                batch_size: int):
        """
        Does the backward propagation algorithm. Updating
        the models weights and biases.

        Parameters
        -------
        Z : list[list[float]] 
            List with all of the outputs for each layer.
        A : list[list[float]]
            List with all of the activation outputs for each layer.
        x_batch: list[list[float]]
            List with a batch of samples.
        y_batch : list[int]
            List with all the classifications for each sample on the batch.
        batch_size : int:
            Size of the batch.
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


    def one_hot(self, y_batch : list[list[float]], output_size: int) -> list[list[int]]:
        """
        Transforms a batch of class indices into one-hot encoded vectors.

        Example:
            If a sample has class 4 and the output size is 10 (e.g. digits 0-9),
            this function returns:
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

        Args:
            y_batch : list[int]
                List of integer class labels.
            output_size : int
                Total number of output classes.

        Returns:
            list[list[int]]: One-hot encoded vectors for each sample.
        """
        one_hot_y = [[0] * output_size for _ in range(len(y_batch))]
        for i, output in enumerate(y_batch):
            one_hot_y[i][output] = 1
        return one_hot_y

    def train_model(self, 
                train_x: list[list[float]], 
                train_y: list[list[int]], 
                epochs: int, 
                batch_size: int) -> None:
        """
        Trains the neural network using mini-batch gradient descent.

        Parameters
        -------
        train_x : list[list[float]] 
            Training samples, each as a list of features.
        train_y : list[int] 
            Corresponding class labels for each training sample.
        epochs : int 
            Number of times the model will iterate over the entire dataset.
        batch_size : int
            Number of samples per training batch.

        Process
        -------
        - Shuffles the dataset at the start of each epoch.
        - Performs forward and backward passes for each batch.
        - Updates the model's weights and biases.
        - Prints training accuracy after each epoch.
        """
        for epoch in range(1, epochs + 1):
            # Shuffle data for every epoch
            perm = np.random.permutation(len(train_x))
            train_x_shuffled = train_x[perm]
            train_y_shuffled = train_y[perm]
            # For every batch.
            for i in range(0, len(train_x), batch_size):
                X_batch = train_x_shuffled[i:i + batch_size]
                y_batch = train_y_shuffled[i:i + batch_size]
                output_size = len(self.weights[-1])
                Y_batch = self.one_hot(y_batch, output_size)

                # Forward pass
                Z, A = self.forward(X_batch)
                self.backward(Z, A, X_batch, Y_batch, batch_size)

            train_preds = self.predict(train_x)
            train_acc = np.mean(train_preds == train_y)
            print(f"Epoch {epoch}/{epochs} — Train Accuracy: {train_acc*100:.2f}%")
        
    def predict(self, X: list[list[float]] | list[float]) -> int | np.ndarray:
        """
        Predicts the output class(es) for one or more input samples.

        Parameters
        ------
        X : list[list[float]] | list[float]: 
            Input data. Can be a single sample (1D) or multiple samples (2D).

        Returns
        -------
        int | np.ndarray: 
            Predicted class index if a single sample is provided, 
            or an array of predicted classes for multiple samples.

        Notes:
            - Automatically reshapes 1D samples into 2D format for consistency.
            - The predicted class corresponds to the neuron with the highest output probability.
        """
        X = np.array(X)
        # 1 dimensional sample
        if X.ndim == 1:
            # Transforms sample from (sample_size,) to (1, sample_size)
            X = X.reshape(1, -1)  

        _, A = self.forward(X)
        # Class will be entry with max output.
        predictions = np.argmax(A[-1], axis=1) 
        return predictions[0] if predictions.ndim == 1 else predictions
    
    def test_accuracy(self, test_x: list[list[float]], test_y: list[int]) -> float:
        """
        Evaluates the model on a test dataset and computes its accuracy.

        Parameters
        -------
        test_x : list[list[float]])
            Test samples.
        test_y : list[int]
            True labels for the test samples.

        Returns
        -------
        float 
            The model's accuracy over the test dataset.
        """

        # Predict every testing sample at the same time.
        predictions = self.predict(test_x)  
        for i, pred in enumerate(predictions):
            print(f"Model predicted {pred} Correct answer: {test_y[i]}")
        accuracy = np.mean(predictions == test_y)
        return accuracy

    def save_model(self, filename: str="model.pkl"):
        """
        Saves the trained model (weights, biases, activation, and learning rate) to disk.

        Parameters
        -------
        filename : str or optional, default= "model.pkl"
            Path and filename to save the model (default is "model.pkl").
        """
        data = {
            "weights": self.weights,
            "biases": self.biases,                
            "activation_function": self.activation_function,
            "learning_rate": self.learning_rate
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load_model(cls, filename: str="model.pkl"):
        """
        Loads a saved model from disk and returns a new initialized instance.

        Parameters
        -------
        filename : str or optional, default= "model.pkl"
            Path to saved model (default is "model.pkl").

        Returns:
        Model
            A new instance of the Model class with loaded parameters.

        Notes:
        - This is a class method, so it can be called directly as:
            `model = Model.load_model("model.pkl")`
        - Requires that the saved file contains compatible parameters.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        
        # Create model
        model = cls(activation_function= data["activation_function"], 
                    learning_rate=data["learning_rate"],
                    weights= data["weights"],
                    biases= data["biases"]
                    )
        
        return model




