
#importing dataProcess file
import dataProcessor as data


#printing dataset 1
dataset1 = data.TrainData1_df

#our dataset is not homogeneous so need to convert to numeric type
dataset1 = dataset1.convert_objects(convert_numeric=True).head()

print 'Normalizing data set'
dataset1 =  (dataset1 - dataset1.mean()) / (dataset1.max() - dataset1.min())
print dataset1


