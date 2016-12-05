import  dataProcessor as data
from sklearn.model_selection import cross_val_score
import pcaTest as pca


X = pca.transformed_trainData3_norm.values
y = data.TrainData3_labels.convert_objects(convert_numeric=True).values.ravel()
X = X[:,0:9182]
print X
print y
print X.shape
from sklearn import metrics
from sklearn.svm import SVC
# fit a SVM model to the data
model = SVC()
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
scores = cross_val_score(model, X, y,  cv=5)
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print predicted
print expected

from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(expected, predicted)**0.5
print RMSE
