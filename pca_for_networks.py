from sklearn.decomposition import PCA
import dataProcessor as dp
import pandas as pd

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

def transformData(pca, trainData, testData):
    transformed_trainData = pca.transform(trainData)
    transformed_testData = pca.transform(testData)
    return transformed_trainData, transformed_testData


#################################################################################
#     FOR QUESTION 1

#     WE NEED TO NORMALIZE THE TRANSFORMED DATA TO BE TRAINED BY THE NETWORKS
################################################################################


#FOR TRAINDATA1, 30 COMPONENTS WILL YIELD 97% OF THE INFORMATION
#the pca is a new mapping for the features
pca_data1 = computePCA(7, dp.TrainData1_normalized.append(dp.TestData1_normalized,ignore_index=True))
#We need to transform the data to these new components
#the data set now has just 30 features instead of the original 4434
transformed_trainData1, transformed_testData1 = transformData(pca_data1, dp.TrainData1_normalized, dp.TestData1_normalized)
#we need to normalize this new dataset as well so that we can train it
#with the neural network.  It is now ready for use by the network
transformed_trainData1_norm, transformed_testData1_norm = dp.normalize(pd.DataFrame(transformed_trainData1), pd.DataFrame(transformed_testData1), 35)



#FOR TRAINDATA2, 125 COMPONENTS WILL YIELD 97% OF THE INFORMATION
#the pca is a new mapping for the features
pca_data2 = computePCA(16, dp.TrainData2_normalized.append(dp.TestData2_normalized,ignore_index=True))
transformed_trainData2, transformed_testData2 = transformData(pca_data2, dp.TrainData2_normalized, dp.TestData2_normalized)
transformed_trainData2_norm, transformed_testData2_norm = dp.normalize(pd.DataFrame(transformed_trainData2), pd.DataFrame(transformed_testData2), 150)



#FOR TRAINDATA1, 90 COMPONENTS WILL YIELD 97% OF THE INFORMATION
#the pca is a new mapping for the features
pca_data3 = computePCA(14, dp.TrainData3_normalized.append(dp.TestData3_normalized,ignore_index=True))
transformed_trainData3, transformed_testData3 = transformData(pca_data3, dp.TrainData3_normalized, dp.TestData3_normalized)
transformed_trainData3_norm, transformed_testData3_norm = dp.normalize(pd.DataFrame(transformed_trainData3), pd.DataFrame(transformed_testData3), 100)


###############################################################################
#     FOR QUESTION 3
###############################################################################
pca_ML_data = computePCA(20, dp.MultiLabelTrainData_normalized.append(dp.MultiLabelTestData_normalized,ignore_index=True))
transformed_multiLabelTrainData, transformed_multiLabelTestData = transformData(pca_ML_data, dp.MultiLabelTrainData_normalized, dp.MultiLabelTestData_normalized)
transformed_multiLabelTrainData_norm, transformed_multiLabelTestData_norm = dp.normalize(pd.DataFrame(transformed_multiLabelTrainData), pd.DataFrame(transformed_multiLabelTestData), 500)
