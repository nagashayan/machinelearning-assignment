from __future__ import print_function
from theano import tensor as T
import pca_for_networks as pca
import dataProcessor as dp
import numpy as np
import theano
import lasagne
import sys
import os
import time
import matplotlib.pyplot as plt
import math




# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():


    #this is transformed pca data pca data
    # We can now download and read the training and test set images and labels.
    X_train = pca.transformed_trainData1_norm.values
    X_train_forTesting = X_train
    X_train_forTesting = np.array(X_train_forTesting,dtype=np.float32)
    y_train = dp.TrainData1_labels.values
    y_train = map(int,y_train.flatten())

    X_test = np.array(X_train,dtype=np.float32)

    #our test set to use for predictions after we can trained and selected a network
    testSet = np.array(pca.transformed_testData1_norm.values, dtype=np.float32)

    y_test = np.array(y_train,dtype=np.uint8)
    y_test = y_test-1

    # We reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-9], X_train[-9:]
    y_train, y_val = y_train[:-9], y_train[-9:]

    X_train = np.array(X_train,dtype=np.float32)
    X_val = np.array(X_val,dtype=np.float32)

    y_train = np.array(y_train,dtype=np.uint8)
    y_train = y_train-1

    y_val = np.array(y_val,dtype=np.uint8)
    y_val = y_val-1

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train , y_train, X_val, y_val, X_test, y_test, testSet, X_train_forTesting
    '''
    #this is raw data
    X_train = dp.TrainData1_normalized.values
    y_train = dp.TrainData1_labels.values
    y_train = map(int,y_train.flatten())
    X_test = dp.TestData1_normalized.values

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10], X_train[-10:]
    y_train, y_val = y_train[:-10], y_train[-10:]

    y_train = np.array(y_train,dtype=np.uint8)
    y_train = y_train-1

    y_val = np.array(y_val,dtype=np.uint8)
    y_val = y_val-1

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return np.array(X_train,dtype=np.float32), y_train, np.array(X_val,dtype=np.float32), y_val, np.array(X_test,dtype=np.float32)
    '''
# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None,7),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.1)

    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=7,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):

    '''
    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=5,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.3)

    # Another 800-unit layer:
    l_hid3 = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=7,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid3_drop = lasagne.layers.DropoutLayer(l_hid3, p=0.2)
    '''



    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=4,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
average_accuracy = 0;
def run_network():
    predictions = []
    while True:
        model='mlp'
        num_epochs=1000
        # Load the dataset
        print("Loading data...")
        X_train, y_train, X_val, y_val, X_test, y_test,testSet, X_train_forTesting = load_dataset()

        # Prepare Theano variables for inputs and targets
        input_var = T.fmatrix(name='inputs')
        test_var = T.fvector(name='testSample')
        target_var = T.ivector(name='targets')


        network = build_mlp(input_var)

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        network_shape = lasagne.layers.get_output_shape(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=0.01, momentum=0.9)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        # Finally, launch the training loop.
        print("Starting training...")
        # We iterate over epochs:
        accuracies = []
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, 12, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, 4, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
            accuracies.append(val_acc / val_batches * 100)
        def runningMeanFast(x, N):
            return np.convolve(x, np.ones((N,))/N)[(N-1):]
        N = num_epochs/10
        SMA = runningMeanFast(accuracies, N)

        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, 7, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1

        accuracy = test_acc / test_batches * 100
        if(accuracy > 94):

            print("model results:")
            print("  model loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  model accuracy:\t\t{:.2f} %".format(
                accuracy))
            accuracies_to_plot = accuracies[0:len(accuracies) - N]
            SMA_to_plot = SMA[0:len(SMA)-N]
            plt.plot(accuracies_to_plot)
            plt.plot(SMA_to_plot)
            plt.ylabel('accuracy')
            plt.xlabel('training epoch')

            print("----------------------------------------------------------")
            print("Accuracy Threshold Achieved, making predictions...")
            print("----------------------------------------------------------")
            #get the predictions for the network
            for sample in testSet:
                outputValue = lasagne.layers.get_output(network, sample)
                p = outputValue.eval()
                predictions.append(p)
            p = np.asarray(predictions)
            fp = []   #final adjusted predictions
            for sample in p:
                for item in sample:
                        Max = 0
                        MaxClass = 1
                        c = 1
                        for element in item:
                            if(element > Max):
                                Max = element
                                MaxClass = c
                            c = c+1
                        fp.append(MaxClass)
            fp = np.asarray(fp)
            f = open('q11predictions_nn.txt', 'w')
            for pred in fp:
                s = str(pred)
                f.write(s)
                f.write("\n")
            f.close()
            print("-----------------------------------------------------------")
            print("Predictions available for viewing in q11predictions_nn.txt")
            print("------------------------------------------------------------")
            plt.show()
            return
        print("Accuracy threshold not achieved, re-training....")
        print("-----------------------------------------------------------------")
        average_accuracy = 0
