import  dataloding as data

X = data.X
y = data.y


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
