
from sklearn.model_selection import cross_val_score
#import  pcaTest as pca
import dataProcessor as data

X = data.TrainData3_df.convert_objects(convert_numeric=True).values
y = data.TrainData3_labels.convert_objects(convert_numeric=True).values.ravel()
X = X[:,0:9182]
print X
print y
print X.shape
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
# fit a k-nearest neighbor model to the data

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print model.score(X,y)*100

print "Cross validated"
scores = cross_val_score(model, X, y,  cv=7)
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print predicted
print expected





