import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = None

    def __softmax(self, data: np.array, W: np.array):
        """
          Softmax function for multi-class logistic regression.

          Args:
              data (array): Input data of shape (N, D)
              W (array): Weights of shape (D, C) where C is the number of classes
          Returns:
              array of shape (N, C): Probability array where each value is in the
                  range [0, 1] and each row sums to 1.
                  The row i corresponds to the prediction of the ith data sample, and
                  the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
          """
        expK = np.exp(data @ W)
        sumJ = np.sum(expK, axis = 1, keepdims = True)
        y = expK/sumJ
        return y

    def __gradient_logistic_multi(self, data: np.array, labels: np.array, W: np.array):
        """
        Compute the gradient of the entropy for multi-class logistic regression.

        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
        y = self.__softmax(data, W)
        grad = data.T @ (y - labels)
        return grad

    def __logistic_regression_predict_multi(self, data, W):
        """
        Prediction the label of data for multi-class logistic regression.

        Args:
            data (array): Dataset of shape (N, D).
            W (array): Weights of multi-class logistic regression model of shape (D, C)
        Returns:
            array of shape (N,): Label predictions of data.
        """
        N = len(data)
        y = self.__softmax(data, W)
        predict = []
        for i in range(N):
            predict.append(np.argmax(y[i]))
        return np.array(predict)


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        D = np.array(training_data).shape[1] # number of features
        C = 1 # number of classes
        # Random initialization of the weights
        self.weights = np.random.normal(0, 0.1, (D, C))
        for it in range(self.max_iters):
            self.weights = self.weights - self.lr * self.__gradient_logistic_multi(training_data, training_labels, self.weights)
        pred_labels = self.__logistic_regression_predict_multi(training_data, self.weights)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        pred_labels = self.__logistic_regression_predict_multi(test_data, self.weights)
        return pred_labels
