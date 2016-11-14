import pcaTest as pca
from sklearn.neighbors import NearestNeighbors
import numpy as np
X = np.array(pca.transformed_trainData1_norm)
kdt =  NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = kdt.kneighbors(X)
knnmatrix = kdt.kneighbors_graph(X).toarray()


