import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


## MS2

#=================== MLP ==============================
class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, activation_function="relu", hidden_layer_sizes: list = None):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
            activation_function (str): activation function to use in the hidden layers (relu, sigmoid, tanh)
            hidden_layer_sizes (list): list of integers, the sizes of the hidden layers
        """
        super(MLP, self).__init__()

        # DEFAULT HIDDEN LAYER SIZES = [64, 64, 64]
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64] * 3  # Default to 64 nodes per layer (3 layer)

        # Define the layers
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_layer_sizes[0]))

        # Hidden layers
        for i in range(1, len(hidden_layer_sizes) - 1):
            self.layers.append(nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))

        # Output layer
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], n_classes))

        ## Setting the activation function
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        if activation_function not in activations:
            raise ValueError(
                f"Activation function {activation_function} not supported. Choose from {list(activations.keys())}")
        self.activation = activations[activation_function]

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        return self.layers[-1](x)


#============= CNN ==============================
class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN, self).__init__()

        # Edit here to modify the model
        self.conv2d1 = nn.Conv2d(input_channels, 6, 3, padding=1)
        self.conv2d2 = nn.Conv2d(6, 16, 3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """

        x = F.max_pool2d(F.relu(self.conv2d1(x)), 2)  # kernel size = 2 --> size of the feature map is reduced by 2
        x = F.max_pool2d(F.relu(self.conv2d2(x)), 2)
        x = x.reshape((x.shape[0], -1))  # or we could use `x.flatten(-3)`
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ============= MyViT ==============================
class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        """
        Initialize the network.
        
        """
        super(MyViT, self).__init__()

        self.chw = chw  # (C = channels (rvb), H = height , W = width)
        self.n_patches = n_patches
        self.n_blocks = n_blocks  # Number of transformer blocks
        self.n_heads = n_heads  # Number of heads in the multi-head attention
        self.hidden_d = hidden_d  # Hidden dimension (size of Q, K, V)

        # Input and patches sizes
        assert chw[1] % n_patches == 0  # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapper
        # dimensionality of each patch after flattening it
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        # linear layer that maps each flattened patch to a vector of dimension self.hidden_d
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        self.positional_embeddings = self.__get_positional_embeddings()

        # Transformer blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def __get_positional_embeddings(self):
        sequence_length = self.n_patches ** 2 + 1
        d = self.hidden_d
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i, j] = np.sin(i / (10000 ** (2 * j / d))) if j % 2 == 0 else np.cos(i / (10000 ** (2 * j / d)))
        return result

    def __patchify(self, images, n_patches):
        n, c, h, w = images.shape

        assert h == w  # We assume square image.

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches  ### WRITE YOUR CODE HERE

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size,
                            j * patch_size: (j + 1) * patch_size]  ### WRITE YOUR CODE HERE

                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten()  ### WRITE YOUR CODE HERE

        return patches


def forward(self, x):
    """
    Predict the class of a batch of samples with the model.

    Arguments:
        x (tensor): input batch of shape (N, Ch, H, W)
    Returns:
        preds (tensor): logits of predictions of shape (N, C)
            Reminder: logits are value pre-softmax.
    """
    ##
    ###
    #### WRITE YOUR CODE HERE!
    ###
    ##
    return preds


#------------ Helper class for the ViT model -----------------
class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        #Lists of linear layers for the query, key, and value mappings for each head.
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        #softmax layer used to calculate the attention scores --> give a weight to each value (distribution)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)  ### WRITE YOUR CODE HERE

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))  ### WRITE YOUR CODE HERE
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)  ### WRITE YOUR CODE HERE
        self.mhsa = MyMSA(hidden_d, n_heads)  ### WRITE YOUR CODE HERE
        self.norm2 = nn.LayerNorm(hidden_d)  ### WRITE YOUR CODE HERE
        self.mlp = nn.Sequential(  ### WRITE YOUR CODE HERE
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x))
        # Feedforward + residual connection
        out = out + self.mlp(self.norm2(out))
        return out


#=================== Trainer ==============================
class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = ...  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader)

            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        return pred_labels

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(),
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()
