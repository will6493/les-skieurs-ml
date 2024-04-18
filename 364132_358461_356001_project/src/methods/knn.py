import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        self.training_data = None
        self.training_labels = None

    def __euclidean_dist(self, Itest_data):
        """Compute the Euclidean distance between a single example
        vector and all training_examples.

        Inputs:
            Itest_data: shape (D,)
        Outputs:
            euclidean distances: shape (N,)
        """
        # WRITE YOUR CODE HERE
        return np.sqrt(np.sum((self.training_data - Itest_data) ** 2, axis=1))

    def __find_k_nearest_neighbors(self, distances):
        """ Find the indices of the k smallest distances from a list of distances.

        Inputs:
            k: integer
            distances: shape (N,)
        Outputs:
            indices of the k nearest neighbors: shape (k,)
        """

        return np.argsort(distances)[:self.k]

    def __kNN_one_example(self, unlabeled_example):
        """Returns the label of a single unlabelled example.

        Inputs:
            unlabeled_example: shape (D,)
            training_features: shape (NxD)
            training_labels: shape (N,)
            k: integer
        Outputs:
            predicted label
        """

        # Compute distances
        distances = self.__euclidean_dist(unlabeled_example)

        # Find neighbors
        nn_indices = self.__find_k_nearest_neighbors(self.k, distances)

        # Get neighbors' labels
        neighbor_labels = self.training_labels[nn_indices]

        # Pick the most common
        best_label = self.__predict_label(neighbor_labels)

        return best_label

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        self.training_data = training_data
        self.training_labels = training_labels
        return training_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        N = np.shape(test_data)[0]
        distances = np.array([N,N])
        nein = np.array([N,self.k])
        test_labels = np.array(N)

        for i in range(N):
            distances = self.__euclidean_dist(test_data[i])
            nein = self.__find_k_nearest_neighbors(distances[i])
            test_labels = np.argmax(np.bincount(nein[i]))

        return test_labels