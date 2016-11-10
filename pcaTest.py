from sklearn.decomposition import PCA

#this function can be used to perform the PCA
#it takes as arguments the number of components to test and a dataset of features
#it does not find the optimal number of components by itself, rather you must
#search by trial and error
#after running the function and assigning the result to some varibale, you
#check the total amount of information contained in the components by printing
#     pca.explained_variance_ratio_.sum()
def computePCA(num_components, dataset):
    pca = PCA(n_components = num_components)
    pca.fit(dataset)
    return pca
