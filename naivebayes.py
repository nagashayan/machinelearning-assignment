import  pcaTest as pca
import dataProcessor as data
#import dataloding as data1
from sklearn.model_selection import cross_val_score


#X = pca.transformed_trainData1_norm.values

X = pca.transformed_trainData3_norm.convert_objects(convert_numeric=True).values
y = data.TrainData3_labels.convert_objects(convert_numeric=True).values.ravel()
X = X[:,0:9182]
print X
print y

print X.shape
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X,y)
print "Model"
print(model)
# make predictions
expected = y

predicted = model.predict(X)

predicted1 = model.predict_log_proba(X)
# summarize the fit of the model
print "report"
print(metrics.classification_report(expected, predicted))
print "matrix"
print(metrics.confusion_matrix(expected, predicted))
print model.score(X,y)*100
#k_fold = KFold(len(y), n_splits=3, shuffle=True, random_state=0)

print "Cross validated"
scores = cross_val_score(model, X, y,  cv=5)
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print predicted
print expected
#print predicted1

print "Root Mean Square Error"

from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(expected, predicted)**0.5
print RMSE
