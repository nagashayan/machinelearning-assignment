import urllib2
import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing

#global variable to store training dataset with out class info - helps in normalizing
TrainData1Dateset  = ""

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
    global TrainData1Dateset
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
    TrainData1Dateset = df
    df2 = pd.DataFrame({'Class': sampleLabels})
    df = pd.concat([df,df2], axis=1)                    #we add the class labels to the dataframe
    return df,df2

#we need to put our testData in a dataframe as well
def Q1_Preprocessor_TestData(url, num_features,num_samples):
    data = urllib2.urlopen(url).read()   #read the data from url
    columnNames = []
    for x in range(1,num_features+1):
        columnNames.append("{}_{}".format("Col", x))
        #print("{}_{}".format("Col", x))
    #df = pd.DataFrame(data.split(), columns=Q1_columns)
    dataSeries = []
    for line in data.split():
        dataSeries.append(line)                 #build list from raw data
        #print line

    a = pd.Series(dataSeries)                        #build series from list
    b = a.values.reshape(num_samples,num_features)       #reshape series to correct 2d dimensions
    df = pd.DataFrame(b, columns=columnNames)            #build dataframe with column names
    return df

#we need to put our testData in a dataframe as well
def Q2_Preprocessor_TestData(url, num_features,num_samples):
    data = urllib2.urlopen(url).read()   #read the data from url
    columnNames = []
    for x in range(1,num_features+1):
        columnNames.append("{}_{}".format("Col", x))
        #print("{}_{}".format("Col", x))
    #df = pd.DataFrame(data.split(), columns=Q1_columns)
    dataSeries = []
    for line in data.split():
        dataSeries.append(line)                 #build list from raw data
        #print line


    a = pd.Series(dataSeries)                        #build series from list
    b = a.values.reshape(num_samples,num_features)       #reshape series to correct 2d dimensions
    df = pd.DataFrame(b, columns=columnNames)            #build dataframe with column names
    df2 = df.replace('1.00000000000000e+99', np.nan)       #set missing values to NaN
    return df2


#This is the multi-label problem on a single data set and test set.
#NEED TO FINISH THISSSSSSSS
def Q3_Preprocessor_TrainData(url, labels, num_features, num_samples, num_classes):
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
    for label in rawLabels.split():
        sampleLabels.append(label)

    labelColumns = []
    for y in range(1, num_classes+1):
        labelColumns.append("{}_{}".format("Class", y))

    a = pd.Series(dataSeries)                        #build series from list
    a2 = pd.Series(sampleLabels)
    b = a.values.reshape(num_samples,num_features)       #reshape series to correct 2d dimensions
    b2 = a2.values.reshape(num_samples, num_classes)
    df = pd.DataFrame(b, columns=columnNames)            #build dataframe with column names
    df2 = pd.DataFrame(b2, columns=labelColumns)
    df = pd.concat([df,df2], axis=1)                    #we add the class labels to the dataframe
    return df


done = False
while(done == False):
    sys.stdout.write("\r")
    sys.stdout.write("Retrieving Data...")
    sys.stdout.flush()




    #####################################################################################
    #     FOR QUESTION 1
    ##############################################################################

    '''
    - Preprocess TrainData1 for question 1
    - 35 rows(samples) and 4434 columns(features) and 4 classes
    - the classes are concatenated onto the end of the dataframe
    '''
    TrainData1_df, TrainData1_labels = Q1_Preprocessor_TrainData(TrainData1,TrainLabel1, 4434,35,4) #features and labels
    TrainData1_features = TrainData1_df.ix[:,0:4434].values #just features


    #normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(TrainData1_features)
    TrainData1_normalized = pd.DataFrame(x_scaled) #normalized features

    TrainData1_final = pd.concat([TrainData1_normalized,TrainData1_labels], axis=1)  #contains normalized features and raw labels              #we add the class labels to the dataframe
    #print(TrainData1_df)


    '''
    - Preprocess TrainData2 for question 1
    - 150 rows/samples and 3312 features/columns and 5 classes
    - the classes are concatenated onto the end of the dataframe
    '''
    TrainData2_df, TrainData2_labels = Q1_Preprocessor_TrainData(TrainData2, TrainLabel2, 3312,150,5)
    TrainData2_features = TrainData2_df.ix[:,0:3312].values #just features


    #normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(TrainData2_features)
    TrainData2_normalized = pd.DataFrame(x_scaled) #normalized feature data

    TrainData2_final = pd.concat([TrainData2_normalized,TrainData2_labels], axis=1)  #contains normalized features and raw labels              #we add the class labels to the dataframe
    #print(TrainData2_df)

    '''
    - Preprocess TrainData3 for question 1
    -100 rows/samples and 9182 features/columns and 11 classes
    - the classes are concatenated onto the end of the dataframe
    '''
    TrainData3_df, TrainData3_labels = Q1_Preprocessor_TrainData(TrainData3,TrainLabel3, 9182,100,11)
    TrainData3_features = TrainData3_df.ix[:,0:9182].values #just features


    #normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(TrainData3_features)
    TrainData3_normalized = pd.DataFrame(x_scaled) # normalized features

    TrainData3_final = pd.concat([TrainData3_normalized,TrainData3_labels], axis=1)  #contains normalized features and raw labels              #we add the class labels to the dataframe
    #print(TrainData3_df)





    '''
    - Put TestData1 for question 1 in a dataframe for easy processing
    '''
    TestData1_df = Q1_Preprocessor_TestData(TestData1, 4434, 15)
    #normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(TestData1_df)
    TestData1_normalized = pd.DataFrame(x_scaled) # normalized features
    #print(TestData1_df)

    '''
    - Put TestData2 for question 1 in a dataframe for easy processing
    '''
    TestData2_df = Q1_Preprocessor_TestData(TestData2, 3312, 53)
    #normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(TestData2_df)
    TestData2_normalized = pd.DataFrame(x_scaled) # normalized features
    #print(TestData2_df)

    '''
    - Put TestData3 for question 1 in a dataframe for easy processing
    '''
    TestData3_df = Q1_Preprocessor_TestData(TestData3, 9182, 74)
    #normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(TestData3_df)
    TestData3_normalized = pd.DataFrame(x_scaled) # normalized features
    #print(TestData3_df)

    ############################################################################

    ############################################################################

    sys.stdout.write("\r")
    sys.stdout.write("Processing Data...")
    sys.stdout.flush()


    #####################################################################################
    # FOR QUESTION 2
    ##########################################################################

    '''
    - Put Dataset1 in a dataframe for easy processing
    - 4% missing values
    '''
    Dataset1_df = Q2_Preprocessor_TestData(Dataset1, 242,14)
    #print(Dataset1_df)

    '''
    - Put Dataset2 in a dataframe for easy processing
    - 10% missing values
    '''
    Dataset2_df = Q2_Preprocessor_TestData(Dataset2, 758,50)
    #print(Dataset2_df)

    '''
    - Put Dataset3 in a dataframe for easy processing
    - 83% missing values
    '''
    Dataset3_df = Q2_Preprocessor_TestData(Dataset3, 273,79)
    #print(Dataset3_df)



    #####################################################################################
    # FOR QUESTION 3
    ##########################################################################

    '''
    - our training data for question 3
    - 103 features, 500 samples, 14 classes
    - note: there are 117 columns because 103_featueres + 14_classes = 117
    '''
    MultLabelTrainData_df = Q3_Preprocessor_TrainData(MultiLabelTrainData, MultiLabelTrainLabel, 103,500,14)
    #print(MultLabelTrainData_df)

    '''
    - Our test data for question 3
    - 103 features and 100 samples
    - need to predict the binary values of 14 classes
    '''
    MultiLabelTestData_df = Q1_Preprocessor_TestData(MultiLabelTestData, 103,100)
    #print(MultiLabelTestData_df)

    done = True

sys.stdout.write("\rPreprocessing Complete!         \n")
