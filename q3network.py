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

def load_dataset(classToPredict):


    #this is transformed pca data pca data
    # We can now download and read the training and test set images and labels.
    X_train = pca.transformed_multiLabelTrainData_norm.values
    y_train = dp.MultiLabelTrainData_labels.values
    X_train_forTesting = X_train
    X_train_forTesting = np.array(X_train_forTesting,dtype=np.float32)
    #need to shuffle to account for grouping of classes
    combined = list(zip(X_train, y_train))
    np.random.shuffle(combined)

    X_train[:], y_train[:] = zip(*combined)

    newy = []
    for row in y_train:
        tmp = map(int,row.flatten())
        newy.append(tmp)
    y_train = newy

    #we will train the network on one class at a time, thus y_train will be a 1d vector, not 2d
    class_1 = []
    for row in y_train:
        class_1.append(row[classToPredict])
    class_1 = np.array(class_1,dtype=np.uint8)
    y_train = class_1


    X_test = np.array(X_train,dtype=np.float32)
    modelXtest = X_test

    y_test = np.array(y_train,dtype=np.uint8)
    modelYtest = y_test

    #our test set to use for predictions after we can trained and selected a network
    testSet = np.array(pca.transformed_multiLabelTestData_norm.values, dtype=np.float32)

    # We reserve the last 10000 training examples for validation
    X_train, X_test = X_train[:-50], X_train[-50:]
    y_train, y_test = y_train[:-50], y_train[-50:]

    X_train, X_val = X_train[:-50], X_train[-50:]
    y_train, y_val = y_train[:-50], y_train[-50:]

    X_train = np.array(X_train,dtype=np.float32)
    X_val = np.array(X_val,dtype=np.float32)
    X_test = np.array(X_test,dtype=np.float32)

    y_train = np.array(y_train,dtype=np.uint8)

    y_val = np.array(y_val,dtype=np.uint8)

    y_test = np.array(y_test,dtype=np.uint8)
    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train , y_train, X_val, y_val, X_test, y_test,modelXtest,modelYtest,X_train_forTesting,testSet
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
    l_in = lasagne.layers.InputLayer(shape=(None,20),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=8,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.3)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):

    '''
    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=14,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.4)

    # Another 800-unit layer:
    l_hid3 = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=7,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid3_drop = lasagne.layers.DropoutLayer(l_hid3, p=0.2)
    '''



    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=2,
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
fp0 = []   #final adjusted predictions
fp1 = []
fp2 = []
fp3 = []   #final adjusted predictions
fp4 = []
fp5 = []
fp6 = []   #final adjusted predictions
fp7 = []
fp8 = []
fp9 = []   #final adjusted predictions
fp10 = []
fp11 = []
fp12 = []
fp13 = []
average_accuracy = 0;
def run_network(classToPredict):
    predictions = []
    while True:
        model='mlp'
        num_epochs=500
        # Load the dataset
        print("Loading data...")
        X_train, y_train, X_val, y_val, X_test, y_test,modelXtest,modelYtest,X_train_forTesting,testSet = load_dataset(classToPredict)

        # Prepare Theano variables for inputs and targets
        input_var = T.fmatrix(name='inputs')
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
            for batch in iterate_minibatches(X_train, y_train, 50, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, 25, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("{} {}".format("Training Class: ", classToPredict))
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
        for batch in iterate_minibatches(X_test, y_test, 25, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1

        #then we test fitness over all training data
        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(modelXtest, modelYtest, 50, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        accuracy = test_acc / test_batches * 100
        #86% for c1
        #89% percent achieved for c2
        #95   for c3
        #92   for c4
        #92   for c5
        #77   for c6
        #93   for c7
        #86   for c8
        #92   for c9
        #91   for c10
        #92   for c11
        #91   for c12
        #90   for c13
        #99   for c14
        threshold = 0
        if(classToPredict == 0):
            threshold = 86
        if(classToPredict == 1):
            threshold = 89
        if(classToPredict == 2):
            threshold = 95
        if(classToPredict == 3):
            threshold = 92
        if(classToPredict == 4):
            threshold = 92
        if(classToPredict == 5):
            threshold = 77
        if(classToPredict == 6):
            threshold = 93
        if(classToPredict == 7):
            threshold = 86
        if(classToPredict == 8):
            threshold = 92
        if(classToPredict == 9):
            threshold = 91
        if(classToPredict == 10):
            threshold = 92
        if(classToPredict == 11):
            threshold = 91
        if(classToPredict == 12):
            threshold = 90
        if(classToPredict == 13):
            threshold = 99

        if(accuracy > threshold):
            print("model results:")
            print("  model loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  model accuracy:\t\t{:.2f} %".format(
                accuracy))
            accuracies_to_plot = accuracies[0:len(accuracies) - N]


            print("----------------------------------------------------------")
            print("Accuracy Threshold Achieved, making predictions...")
            print("----------------------------------------------------------")
            #get the predictions for the network
            for sample in testSet:
                outputValue = lasagne.layers.get_output(network, sample)
                p = outputValue.eval()
                predictions.append(p)
            p = np.asarray(predictions)
            if(classToPredict == 0):
                for sample in p:
                    for item in sample:
                            Max = 0
                            MaxClass = 0
                            c = 0
                            for element in item:
                                if(element > Max):
                                    Max = element
                                    MaxClass = c
                                c = c+1
                            fp0.append(MaxClass)

            if(classToPredict == 1):
                for sample in p:
                    for item in sample:
                            Max = 0
                            MaxClass = 0
                            c = 0
                            for element in item:
                                if(element > Max):
                                    Max = element
                                    MaxClass = c
                                c = c+1
                            fp1.append(MaxClass)

            if(classToPredict == 2):
                for sample in p:
                    for item in sample:
                            Max = 0
                            MaxClass = 0
                            c = 0
                            for element in item:
                                if(element > Max):
                                    Max = element
                                    MaxClass = c
                                c = c+1
                            fp2.append(MaxClass)

            if(classToPredict == 3):
                for sample in p:
                    for item in sample:
                            Max = 0
                            MaxClass = 0
                            c = 0
                            for element in item:
                                if(element > Max):
                                    Max = element
                                    MaxClass = c
                                c = c+1
                            fp3.append(MaxClass)

            if(classToPredict == 4):
                for sample in p:
                    for item in sample:
                            Max = 0
                            MaxClass = 0
                            c = 0
                            for element in item:
                                if(element > Max):
                                    Max = element
                                    MaxClass = c
                                c = c+1
                            fp4.append(MaxClass)

            if(classToPredict == 5):
                for sample in p:
                    for item in sample:
                            Max = 0
                            MaxClass = 0
                            c = 0
                            for element in item:
                                if(element > Max):
                                    Max = element
                                    MaxClass = c
                                c = c+1
                            fp5.append(MaxClass)

            if(classToPredict == 6):
                for sample in p:
                    for item in sample:
                            Max = 0
                            MaxClass = 0
                            c = 0
                            for element in item:
                                if(element > Max):
                                    Max = element
                                    MaxClass = c
                                c = c+1
                            fp6.append(MaxClass)

            if(classToPredict == 7):
                for sample in p:
                    for item in sample:
                            Max = 0
                            MaxClass = 0
                            c = 0
                            for element in item:
                                if(element > Max):
                                    Max = element
                                    MaxClass = c
                                c = c+1
                            fp7.append(MaxClass)

            if(classToPredict == 8):
                for sample in p:
                    for item in sample:
                            Max = 0
                            MaxClass = 0
                            c = 0
                            for element in item:
                                if(element > Max):
                                    Max = element
                                    MaxClass = c
                                c = c+1
                            fp8.append(MaxClass)

            if(classToPredict == 9):
                for sample in p:
                    for item in sample:
                            Max = 0
                            MaxClass = 0
                            c = 0
                            for element in item:
                                if(element > Max):
                                    Max = element
                                    MaxClass = c
                                c = c+1
                            fp9.append(MaxClass)

            if(classToPredict == 10):
                for sample in p:
                    for item in sample:
                            Max = 0
                            MaxClass = 0
                            c = 0
                            for element in item:
                                if(element > Max):
                                    Max = element
                                    MaxClass = c
                                c = c+1
                            fp10.append(MaxClass)

            if(classToPredict == 11):
                for sample in p:
                    for item in sample:
                            Max = 0
                            MaxClass = 0
                            c = 0
                            for element in item:
                                if(element > Max):
                                    Max = element
                                    MaxClass = c
                                c = c+1
                            fp11.append(MaxClass)

            if(classToPredict == 12):
                for sample in p:
                    for item in sample:
                            Max = 0
                            MaxClass = 0
                            c = 0
                            for element in item:
                                if(element > Max):
                                    Max = element
                                    MaxClass = c
                                c = c+1
                            fp12.append(MaxClass)

            if(classToPredict == 13):
                for sample in p:
                    for item in sample:
                            Max = 0
                            MaxClass = 0
                            c = 0
                            for element in item:
                                if(element > Max):
                                    Max = element
                                    MaxClass = c
                                c = c+1
                            fp13.append(MaxClass)


                finals = []
                finals.append(fp0)
                finals.append(fp1)
                finals.append(fp2)
                finals.append(fp3)
                finals.append(fp4)
                finals.append(fp5)
                finals.append(fp6)
                finals.append(fp7)
                finals.append(fp8)
                finals.append(fp9)
                finals.append(fp10)
                finals.append(fp11)
                finals.append(fp12)
                finals.append(fp13)
                f = open('q3predictions_nn.txt', 'w')
                for y in range(0,100):
                    for x in finals:
                        s = str(x[y])
                        f.write(s)
                        f.write("\t")
                    f.write("\n")
                f.close()
            return
        print("Accuracy threshold not achieved, re-training....")
        print("-----------------------------------------------------------------")
        average_accuracy = 0
for x in range(0,14):
    run_network(x)
print("-----------------------------------------------------------")
print("Predictions available for viewing in q3predictions_nn.txt")
print("------------------------------------------------------------")
