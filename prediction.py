
from sklearn.model_selection import cross_val_score
#import  pcaTest as pca
from sklearn.preprocessing import Imputer
import dataProcessor as data
import numpy as np,collections as coll
import matplotlib.pyplot as plt
import pandas as pd
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, MICE

X = data.Dataset2_df.convert_objects(convert_numeric=True)
y = data.TrainData2_labels.convert_objects(convert_numeric=True)
y = y[0:50]
np.set_printoptions(threshold=np.nan)
#print X
print X.shape
print y.shape
#print "describe"
#print X.describe(include = 'all')


print X.isnull().sum()
print X.isnull().sum().sum()


'''
print "for 3 problem"

nan_rows = X[X.isnull().T.any().T]
print nan_rows

X.dropna(how='all')
print X.shape

'''
#runing classfier
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
X1 = X
print "X1 zero count before"
print (X1.values == 0).sum()
X1 = X1.fillna(0)
print "X1 zero count"
print (X1.values == 0).sum()
#print X1
print "*****"
y = pd.concat([y,y], axis=0)
y = y[0:50]
print y
print y.shape
y = y.values.ravel()

print y.shape
print "print y"
print y
model.fit(X1, y)
print model.score(X1,y)*100

#X.plot(kind = 'hist')
#plt.show()


#using ski-kit imputer
#imp = Imputer(missing_values='NaN', strategy='median', axis=0)
#X = imp.fit_transform(X)

#using fancy impute knn impute
X = MICE().complete(X.values)
'''
columnNames = []
num_features = 242
for x in range(1, num_features + 1):
    columnNames.append("{}_{}".format("Col", x))

X =  pd.DataFrame(X, columns=columnNames)

#print X

model.fit(X, y)
print model.score(X,y)*100

#np.set_printoptions(precision=2)
'''
print X
X = pd.DataFrame(X)
print "converted"
X.to_csv("22.csv")
np.savetxt('22.txt', X.values, fmt='%f')
print "count"
print (X.values == 0).sum()
#X.plot(kind = 'hist')
#plt.show()





