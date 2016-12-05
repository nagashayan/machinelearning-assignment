import pandas as pd
#importing dataProcess file
import dataProcessor as data
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import numpy as np
#import dataloding as datal
import pcaTest as pca
#plotting normalized data
#pca.transformed_trainData1_norm.plot(kind='scatter')
'''
X = pca.transformed_trainData1
y = data.TrainData1_labels.convert_objects(convert_numeric=True).values.ravel()
print X.shape
print y.shape
plt.plot(X, y, "o")

plt.show()

'''

X = pca.transformed_trainData1_norm.convert_objects(convert_numeric=True).values
#X = data.TrainData1_df.convert_objects(convert_numeric=True).values
y = data.TrainData1_labels.convert_objects(convert_numeric=True).values.ravel()

np.set_printoptions(threshold=np.nan)

X = X[:,0:4434]
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)
print np.sort(model.feature_importances_)

print X.shape
print y.shape
print(pca.transformed_trainData1_norm.describe())
print(pca.transformed_trainData1_norm.corr())
#plt.plot(X, y, "o")

#plt.show()


'''
#print knn.knnmatrix

#plotting unnormalized data
print pd.DataFrame(pca.transformed_trainData1)
pd.DataFrame(pca.transformed_trainData1).plot(kind = 'bar')
plt.show()


x = [1,2,3,4]
y = [3,4,8,6]
#print len(data.TrainData1_df.X)
#print len(data.TrainData1_df.y)
print data.TrainData1_df
data.TrainData1_df.hist()
plt.show()
'''