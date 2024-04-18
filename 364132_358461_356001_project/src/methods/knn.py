import numpy as np


def euclidean_dist(example, training_examples):
    """Compute the Euclidean distance between a single example
    vector and all training_examples.

    Inputs:
        example: shape (D,)
        training_examples: shape (NxD)
    Outputs:
        euclidean distances: shape (N,)
    """
    N = training_examples.shape[0]
    dist = np.zeros(N)
    for i in range(N):
        dist[i] = np.sum((example - training_examples[i]) ** 2)

    return np.sqrt(dist)


def predict_label(neighbor_labels):
    """Return the most frequent label in the neighbors'.

    Inputs:
        neighbor_labels: shape (N,)
    Outputs:
        most frequent label
    """

    return np.argmax(np.bincount(neighbor_labels))


class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        self.training_data = None
        self.training_labels = None

    def __find_k_nearest_neighbors(self, distances):
        """ Find the indices of the k smallest distances from a list of distances.

        Inputs:
            k: integer
            distances: shape (N,)
        Outputs:
            indices of the k nearest neighbors: shape (k,)
        """

        return np.argsort(distances)[:self.k]

    def __kNN_one_example(self, unlabeled_example, training_features, training_labels):
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
        distances = euclidean_dist(unlabeled_example, training_features)

        # Find neighbors
        nn_indices = self.__find_k_nearest_neighbors(distances)

        # Get neighbors' labels
        neighbor_labels = training_labels[nn_indices]

        # Pick the most common
        best_label = predict_label(neighbor_labels)

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

        return np.apply_along_axis(self.__kNN_one_example, 1, test_data,
                                   self.training_data, self.training_labels)
