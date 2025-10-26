from abc import ABC, abstractmethod
import numpy as np

class Activation_Function(ABC):
    @abstractmethod
    def activation(self, Z):
        """
        Calculates the activation fucntion for a given matrix.
        """
        pass

    @abstractmethod
    def activation_derivative(self, Z):
        """
        Calculates the derivative of the activation function for a given matrix.
        """
        pass

class ReLu(Activation_Function):
    # Override
    def activation(self, Z: np.ndarray) -> np.ndarray:
        """
        Applies the ReLu function to each entry on the matrix.
        """
        return np.maximum(0, Z)
    
    #Override
    def activation_derivative(self, Z: np.ndarray) -> np.ndarray:
        """
        Returns the derivative (Jacobian) of the ReLu function.
        """
        return (Z > 0).astype(float)
    
class Softmax(Activation_Function):
    def activation(self, Z: np.ndarray) -> np.ndarray:
        """
        Applies the softmax function row-wise (for each sample).
        """
        # Subtracts the max from each line to avoid numeric overflow.
        exp_shifted = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def activation_derivative(self, Z: np.ndarray) -> np.ndarray:
        """
        Returns the derivative (Jacobian) of the softmax function.
        Note: usually combined with cross-entropy loss directly.
        """
        S = self.activation(Z)
        return S * (1 - S)


def create_activation_function(activation_function: str) -> Activation_Function:
    """
    Receives a name of a function and returns the function.
    """
    activation_function = activation_function.lower()
    if activation_function == 'relu':
        return ReLu()
    
    elif activation_function == 'softmax':
        return Softmax()
    
    else:
        raise ValueError('create_activation_function: Function', activation_function, 'not yet implemented (or a typo).')