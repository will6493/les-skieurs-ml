import numpy as np
import sys


class LinearRegression(object):
    """
    LinearRegression Object that performs linear or ridge regression depending on the lambda (lmda) parameter.
    When lmda is 0, it performs ordinary least squares Linear Regression.
    """
    def __init__(self, lmda, task_kind = "regression"):
        """
        Initializes the LinearRegression object with regularization strength lmda and task kind.

        Parameters:
            lmda (float): The regularization strength. Zero by default for linear regression.
            task_kind (str): Type of task - "regression" or other kinds defined in extensions.
        """
        self.lmda = lmda
        self.task_kind = task_kind
        self.weights = None


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.
        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,regression_target_size)
        Returns:
            pred_labels (np.array): target of shape (N,regression_target_size)
        """
        inverseMatrix = np.linalg.inv(training_data.T @ training_data + self.lmda * np.eye(training_data.shape[1]))
        self.weights = (inverseMatrix @ training_data.T) @ training_labels

        pred_regression_targets = training_data @ self.weights
        return pred_regression_targets


    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,regression_target_size)
        """

        pred_regression_targets = test_data @ self.weights
        return pred_regression_targets
