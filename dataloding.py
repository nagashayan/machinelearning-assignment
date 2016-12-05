import numpy as np
import urllib
#import pcaTest as pca
#import  dataProcessor as data
#import pandas as pd

#example found at http://kukuruku.co/hub/python/introduction-to-machine-learning-with-python-andscikit-learn

# url with dataset
from numpy import genfromtxt

dataset = genfromtxt('trainingdata1.csv', delimiter=',')
# load the CSV file as a numpy matrix
#dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:47]
y = dataset[:,48]
#print X
#print y

