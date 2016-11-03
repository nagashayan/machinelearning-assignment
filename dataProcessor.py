import urllib2
import pandas as pd
import numpy as np



'''
Datasets for problem 1
'''
TrainData1 = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/TrainData1.txt"
TrainLabel1 = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/TrainLabel1.txt"
TestData1 = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/TestData1.txt"

TrainData2 = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/TrainData2.txt"
TrainLabel2 = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/TrainLabel2.txt"
TestData2 = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/TestData2.txt"

TrainData3 = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/TrainData3.txt"
TrainLabel3 = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/TrainLabel3.txt"
TestData3 = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/TestData3.txt"


'''
Datasets for problem 2
'''
Dataset1 = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/MissingData1.txt"
Dataset2 = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/MissingData2.txt"
Dataset3 = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/FluDataMissingValue.txt"


'''
Datasets for problem 3
'''
MultiLabelTrainData = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/MultLabelTrainData.txt"
MultiLabelTrainLabel = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/MultLabelTrainLabel.txt"
MultiLabelTestData = "http://grid.cs.gsu.edu/zcai/course/8850/Dataset/MultLabelTestData.txt"


#this collects and preprocesses our training data
#it takes as input the url for the data, the number of features and the
#number of samples, and classes.
#It returns a dataframe object for easy manipulation.
#IMPORTANT:  we assume that the first (num_features) constitute one sample. And the second (num_features)
#constitute the second sample, and so on
def Q1_Preprocessor_TrainData(url, labels, num_features, num_samples, num_classes):
    data = urllib2.urlopen(url).read()   #read the data from url
    rawLabels = urllib2.urlopen(labels).read()   #read raw labels
    columnNames = []
    for x in range(1,num_features+1):
        columnNames.append("{}_{}".format("Col", x))
        #print("{}_{}".format("Col", x))
    #df = pd.DataFrame(data.split(), columns=Q1_columns)
    dataSeries = []
    sampleLabels = []
    for line in data.split():
        dataSeries.append(line)                 #build list from raw data
        #print line

    #we need this because there is a strange character at the end of TrainLabel2 that throws off
    #the indexing.
    i = 0
    for label in rawLabels.split('\n'):
        if((num_features ==  3312) and (i < 150)):    #for TrainLabel2
            sampleLabels.append(label)            #build list of labels from rawLabels
            i = i+1
        elif((num_features == 9182) and (i < 100)):     #for TrainLabel3
            sampleLabels.append(label)
            i = i + 1
        elif(num_features == 4434):             #for TrainLabel1
            sampleLabels.append(label)

    a = pd.Series(dataSeries)                        #build series from list
    b = a.values.reshape(num_samples,num_features)       #reshape series to correct 2d dimensions
    df = pd.DataFrame(b, columns=columnNames)            #build dataframe with column names
    df2 = pd.DataFrame({'Class': sampleLabels})
    df = pd.concat([df,df2], axis=1)                    #we add the class labels to the dataframe
    return df

#we need to put our testData in a dataframe as well
#def Q1_Preprocessor_TestData():




'''
- Preprocess TrainData1 for question 1
- 35 rows(samples) and 4434 columns(features) and 4 classes
'''
TrainData1_df = Q1_Preprocessor_TrainData(TrainData1,TrainLabel1, 4434,35,4)
print(TrainData1_df)


'''
- Preprocess TrainData2 for question 1
- 150 rows/samples and 3312 features/columns and 5 classes
'''
TrainData2_df = Q1_Preprocessor_TrainData(TrainData2, TrainLabel2, 3312,150,5)
#print(TrainData2_df)

'''
- Preprocess TrainData3 for question 1
-100 rows/samples and 9182 features/columns and 11 classes
'''
TrainData3_df = Q1_Preprocessor_TrainData(TrainData3,TrainLabel3, 9182,100,11)
#print(TrainData3_df)
