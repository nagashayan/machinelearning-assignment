
from sklearn.model_selection import cross_val_score
import  pcaTest as pca
import dataProcessor as data
import numpy as np

def KNNTEST(X,y,test):
    print X.shape


    #test = X[24:35, 0:4434]

    X = X[:,0:4434]
    #X = X[0:35,:]
    #expected = y[0:35]
    #y = y[0:35]

    print X
    print y
    print X.shape
    print y.shape
    #values
    print test.shape
    #labels
    #print expected.shape

    from sklearn import metrics
    from sklearn.neighbors import KNeighborsClassifier
    # fit a k-nearest neighbor model to the data

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    print(model)
    # make predictions

    print model.score(X, y) * 100

    print test

    predicted = model.predict(test)
    print "predicted"
    print predicted.ravel()
'''
    print expected

    # summarize the fit of the model

    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))



    print "Cross validated"
    scores = cross_val_score(model, X, y,  cv=7)
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print predicted
    print expected


    print "Root Mean Square Error"

    from sklearn.metrics import mean_squared_error
    RMSE = mean_squared_error(expected, predicted)**0.5
    print RMSE

'''



X = data.TrainData1_df.convert_objects(convert_numeric=True).values
y = data.TrainData1_labels.convert_objects(convert_numeric=True).values.ravel()
test = data.TestData1_df.convert_objects(convert_numeric=True).values
KNNTEST(X,y,test)
#X = data.TrainData1_df.reindex(np.random.permutation(data.TrainData1_df.index))
#X = X.convert_objects(convert_numeric=True).values


#KNNTEST(X)