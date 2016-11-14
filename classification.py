import pandas as pd
#importing dataProcess file
import dataProcessor as data
import pcaTest as pca
import knnTest as knn
import matplotlib.pyplot as plt

#plotting normalized data
pca.transformed_trainData1_norm.plot()
plt.show()

print knn.knnmatrix

#plotting unnormalized data
print pd.DataFrame(pca.transformed_trainData1)
pd.DataFrame(pca.transformed_trainData1).plot(kind = 'bar')
plt.show()