import numpy as np


class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def find_principal_components(self, training_data: np.array):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        # Compute the mean of data
        self.mean = np.mean(training_data, axis=0)

        # Center the data with the mean
        X_tilde = training_data - self.mean

        # Create the covariance matrix
        C = np.cov(X_tilde, rowvar=False)

        # Compute the eigenvectors and eigenvalues.
        eigvals, eigvecs = np.linalg.eigh(C)

        # Choose the top d eigenvalues and corresponding eigenvectors.
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]

        self.W = eigvecs[:, :self.d]
        eg = eigvals[:self.d]

        # Compute the explained variance
        exvar = 100 * np.sum(eg) / np.sum(eigvals)
        return exvar

    def reduce_dimension(self, data: np.array):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        # Ensure that mean and W have been computed
        if self.mean is None or self.W is None:
            raise ValueError(
                "The principal components have not been computed. Please run find_principal_components() first.")

        return np.dot(data - self.mean, self.W)
