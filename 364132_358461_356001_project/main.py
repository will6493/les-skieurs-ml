import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data_path)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set
    if not args.test:
        print("Creating validation test...")
        num_samples = xtrain.shape[0]
        train_part = args.train_part
        r_inds = np.random.permutation(num_samples) # We shuffle the indices to shuffle the data
        i_train = int(num_samples * train_part) # Final index of the total data that is used for training
        
        xtest = xtrain[r_inds[i_train:]]
        ytest = ytrain[r_inds[i_train:]]
        xtrain = xtrain[r_inds[:i_train]]
        ytrain = ytrain[r_inds[:i_train]]



    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data


    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)

    if args.nn_type == "mlp":
        model = MLP(args.nn_batch_size, n_classes, args.act_func, args.h_lay_sizes)
    elif args.nn_type == "cnn":
        model = CNN(args.nn_batch_size, n_classes)
    elif args.nn_type == "transformer":
        model = MyViT(args.chw, args.n_patches, args.n_blocks, args.hidden_d, args.n_heads, args.out_d)
    else:
        print(args.nn_type + " is not a valid network architecture (try 'mlp', 'cnn' or 'transformer')")    
    summary(model)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, xtest)
    macrof1 = macrof1_fn(preds, xtest)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data_path', default="../dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")
    parser.add_argument('--act_func', type=str, default='relu', help="the activation function used for the model")

    ## MLP arguments
    parser.add_argument('--h_lay_sizes', type=list, default=None, help="hidden layers sizes")

    ## Transformer arguments 
    parser.add_argument('--chw', type=list, default=[1, 28, 28], help="C = channels (rvb), H = height , W = width")
    parser.add_argument('--n_patches', type=int, default=7, help="in how much patches is the images divided")
    parser.add_argument('--n_blocks', type=int, default=2, help="number of transformers blocs")
    parser.add_argument('--hidden_d', type=int, default=8, help="number of hidden dimensions (size of Q, K, V)")
    parser.add_argument('--n_heads', type=int, default=2, help="number of heads in the multi-head attention")
    parser.add_argument('--out_d', type=int, default=10, help="number of output dimensions")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--train_part', default=0.8, type=float, help="part of the given data used for training (rest is used for test)")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)