import argparse
import time
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

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set
    if not args.test:
        print("Creating validation test...")
        num_samples = xtrain.shape[0]
        train_part = args.train_part
        r_inds = np.random.permutation(num_samples)  # We shuffle the indices to shuffle the data
        i_train = int(num_samples * train_part)  # Final index of the total data that is used for training

        xtest = xtrain[r_inds[i_train:]]
        ytest = ytrain[r_inds[i_train:]]
        xtrain = xtrain[r_inds[:i_train]]
        ytrain = ytrain[r_inds[:i_train]]

    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        exvar = pca_obj.find_principal_components(xtrain)
        print(f'The total variance explained by the first {args.pca_d} principal components is {exvar:.3f} %')
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)

    if args.nn_type == "mlp":
        hidden_layer_sizes = [int(size) for size in args.h_lay_sizes.split(',')]
        if args.use_pca:
            model = MLP(args.pca_d, n_classes, args.act_func, hidden_layer_sizes)
            summary(model, input_size=(1, args.pca_d))
        else:
            model = MLP(args.nn_batch_size, n_classes, args.act_func, hidden_layer_sizes)
            summary(model, input_size=(1, args.nn_batch_size))
    elif args.nn_type == "cnn":
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)
        model = CNN(1, n_classes)
        summary(model, input_size=(1, 1, 28, 28))
    elif args.nn_type == "transformer":
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)
        model = MyViT(args.chw, args.n_patches, args.n_blocks, args.hidden_d, args.n_heads, args.out_d)
        summary(model, input_size=(1, 1, 28, 28))
    else:
        print(args.nn_type + " is not a valid network architecture (try 'mlp', 'cnn' or 'transformer')")

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)

    ## 4. Train and evaluate the method
    start = time.time()
    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)
    end = time.time()
    print(f"\nTraining time: {end - start:.2f}s")

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
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
    parser.add_argument('--nn_batch_size', type=int, default=784, help="batch size for NN training")
    # TODO delete? should never be used, automatically set by the model
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")
    parser.add_argument('--act_func', type=str, default='relu', help="the activation function used for the model")

    ## MLP arguments
    parser.add_argument('--h_lay_sizes', type=str, default="512, 256, 128, 64", help="hidden layers sizes")

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
    parser.add_argument('--train_part', default=0.8, type=float,
                        help="part of the given data used for training (rest is used for test)")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)

    # ================= Best parameters =================
    # MLP: python main.py --max_iters=80 --lr=1e-4 --h_lay_sizes=512,512,256,256,128,64 --train_part=0.92
    #      -> Train set: accuracy = 99.998% - F1-score = 0.999982
    #      -> Validation set: accuracy = 86.271% - F1-score = 0.861707
    # MLP (w/PCA) : python main.py --max_iters=35 --lr=1e-4 --h_lay_sizes=512,512,256,256,128,64 --train_part=0.92 --use_pca
    #      -> Train set: accuracy = 99.357 % - F1 - score = 0.993571
    #      -> Validation set: accuracy = 85.500 % - F1 - score = 0.855043
    # CNN: python main.py --max_iters=80 --lr=1e-3 --nn_type=cnn
    #      -> Train set: accuracy = 100.000% - F1-score = 1.000000
    #      -> Validation set: accuracy = 89.167% - F1-score = 0.893344
