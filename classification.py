
#importing dataProcess file
import dataProcessor as data


#printing dataset 1
dataset1 = data.TrainData1_df
#printing dataset 1
testset1 = data.TestData1_df


#our dataset is not homogeneous so need to convert to numeric type
dataset1 = data.TrainData1Dateset.convert_objects(convert_numeric=True)

print 'Normalizing data set'

dataset1 =  (dataset1 - dataset1.mean()) / (dataset1.max() - dataset1.min())

print dataset1

#our testset is not homogeneous so need to convert to numeric type
testset1 = testset1.convert_objects(convert_numeric=True)
print 'Normalizing test set'
testset1 =  (testset1 - testset1.mean()) / (testset1.max() - testset1.min())

print testset1


