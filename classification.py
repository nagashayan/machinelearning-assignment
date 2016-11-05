
#importing dataProcess file
import dataProcessor as data

#printing dataset 1
dataset1 = data.TrainData1_df
print dataset1
#trying to normalize data by using xi-min /(max-min) formula
print dataset1.max(axis=0)
print dataset1.min(axis=0)
