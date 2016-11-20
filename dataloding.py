import numpy as np
import urllib
#import pcaTest as pca
#import  dataProcessor as data
#import pandas as pd

#example found at http://kukuruku.co/hub/python/introduction-to-machine-learning-with-python-andscikit-learn

# url with dataset
from numpy import genfromtxt

import numpy as np
import urllib
# url with dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file
raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:8]
y = dataset[:,8]
print X
print y


dataset = genfromtxt('trainingdata1.csv', delimiter=',')
# load the CSV file as a numpy matrix
#dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:47]
y = dataset[:,48]
print X
print y


from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)
print "Model"
print(model)
# make predictions
expected = y

predicted = model.predict(X)
# summarize the fit of the model
print "report"
print(metrics.classification_report(expected, predicted))
print "matrix"
print(metrics.confusion_matrix(expected, predicted))

