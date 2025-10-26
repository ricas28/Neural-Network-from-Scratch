from abc import ABC, abstractmethod
import numpy as np

class Activation_Function(ABC):
    @abstractmethod
    def activation(self, Z):
        """
        Calculates the activation function for a given matrix.

        Parameters
        ----------
        Z : numpy.ndarray
            Input matrix (pre-activation values) for which to compute the activation.

        Returns
        -------
        numpy.ndarray
            Activated output after applying the function element-wise.
        """
        pass

    @abstractmethod
    def activation_derivative(self, Z):
        """
        Calculates the derivative of the activation function for a given matrix.

        Parameters
        ----------
        Z : numpy.ndarray
            Input matrix (pre-activation values).

        Returns
        -------
        numpy.ndarray
            Derivative of the activation function evaluated element-wise.
        """
        pass

class ReLu(Activation_Function):
    def activation(self, Z: np.ndarray) -> np.ndarray:
        """
        Applies the ReLU (Rectified Linear Unit) activation function element-wise.

        Parameters
        ----------
        Z : numpy.ndarray
            Input matrix of pre-activation values.

        Returns
        -------
        numpy.ndarray
            Matrix where all negative values are replaced by 0, 
            and positive values remain unchanged.
        """
        return np.maximum(0, Z)
    
    def activation_derivative(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the ReLU activation function element-wise.

        Parameters
        ----------
        Z : numpy.ndarray
            Input matrix of pre-activation values.

        Returns
        -------
        numpy.ndarray
            Matrix of 1s where Z > 0, and 0s elsewhere.
        """
        return (Z > 0).astype(float)
    
class Softmax(Activation_Function):
    def activation(self, Z: np.ndarray) -> np.ndarray:
        """
        Applies the Softmax activation function row-wise.

        Parameters
        ----------
        Z : numpy.ndarray
            Input matrix of shape (n_samples, n_classes).

        Returns
        -------
        numpy.ndarray
            Matrix of the same shape where each row represents 
            a valid probability distribution (sums to 1).

        Notes
        -----
        Subtracts the maximum value in each row before exponentiation 
        to improve numerical stability.
        """
        exp_shifted = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def activation_derivative(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes the derivative (Jacobian) of the Softmax function element-wise.

        Parameters
        ----------
        Z : numpy.ndarray
            Input matrix of pre-activation values.

        Returns
        -------
        numpy.ndarray
            Element-wise derivative approximation of Softmax.
        
        Notes
        -----
        This simplified version assumes independence between classes and is
        typically combined with the cross-entropy loss for efficiency.
        """
        S = self.activation(Z)
        return S * (1 - S)

def create_activation_function(activation_function: str) -> Activation_Function:
    """
    Creates and returns an activation function object based on its name.

    Parameters
    ----------
    activation_function : str
        Name of the activation function. Supported values:
        - 'relu'
        - 'softmax'

    Returns
    -------
    Activation_Function
        An instance of the corresponding activation function class.

    Raises
    ------
    ValueError
        If the provided activation function name is not implemented.
    """
    activation_function = activation_function.lower()
    if activation_function == 'relu':
        return ReLu()
    elif activation_function == 'softmax':
        return Softmax()
    else:
        raise ValueError(
            f"create_activation_function: Function '{activation_function}' not yet implemented (or a typo)."
        )