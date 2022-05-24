#I need an array of all relevant folders
# I need to be able to interate through the 24 participants
#twelve features (remove index), 6 classes/labels
#test with both sklearn logitstic regression and keras CNN (David M)

import pandas as pd
import numpy as np
import math


from tensorflow.keras.utils import to_categorical

from sklearn import preprocessing

#There are six labels not sure what dws is ? (walking down stairs?)
labels = ['dws', 'jog', 'sit', 'std', 'ups', 'wlk']

# what do these folder names mean? All folders have data for 24 participants
folders = ['dws_1', 'dws_2', 'dws_11', 'jog_9', 'jog_16', 'sit_5', 'sit_13', 'std_6', 'std_14', 'ups_3', 'ups_4', 'ups_12', 'wlk_7', 'wlk_8', 'wlk_15']

# features observed
features = ['attitude.roll', 'attitude.pitch', 'attitude.yaw', 'gravity.x',
       'gravity.y', 'gravity.z', 'rotationRate.x', 'rotationRate.y',
       'rotationRate.z', 'userAcceleration.x', 'userAcceleration.y',
       'userAcceleration.z']

#twenty four participants
numbParts = 24 

#when running dataPreprocess.py from inside modtion-sense folder
path = './archive/A_DeviceMotion_data/A_DeviceMotion_data/'

labelPath = './archive/data_subjects_info.csv'

#based on first three letters of folder name
def getLabel(folderName):
    folderName = folderName[0:3]
    if(folderName == 'dws'):
        return 0
    elif(folderName == 'jog'):
        return 1
    elif(folderName == 'sit'):
        return 2
    elif(folderName == 'std'):
        return 3
    elif(folderName == 'ups'):
        return 4
    elif(folderName == 'wlk'):
        return 5
    else:
        print("There has been an error assigning labels!")


#number should be from 1 to 24
def getPartFile(number):
    return "sub_" + str(number) + ".csv"

#this method creates an array that holds 24 empty data frames -> each participant has [trainDF, testDF]
def initialSets(numbParts):
    arrays = []
    for i in range(0, numbParts):
        arrays.append( [ pd.DataFrame(), np.array([]).astype(np.int64), pd.DataFrame(), np.array([]).astype(np.int64) ] ) 
    return arrays



#split all datasets into train and validation/test .80 and .25
def getIndividualDatasets(numbParts):
    partArrays = initialSets(numbParts)

    #for each folder
    for folder in folders:

        #for each participant
        for part in range(1, numbParts+1):

            pathway = path+ folder + "/" +getPartFile(part)
            partDF = pd.read_csv(pathway)
            partDF = partDF.iloc[:, 1:] #remove index column

            #preprocess partDF -> normalize (EXPERIMENTAL)
            # scaler = preprocessing.StandardScaler().fit(partDF)
            # partDF = scaler.transform(partDF)
            
            #split into train and test (80/20)
            split_val = math.floor(len(partDF) * .80)

            part_train = partDF.iloc[ : split_val   , :]
            part_test = partDF.iloc[ split_val : , :]

            train_label = (np.ones( len(part_train) ) * getLabel(folder)).astype(np.int64)
            test_label = (np.ones( len(part_test) ) * getLabel(folder)).astype(np.int64)   

            #transform to one shot vectors
            train_label = to_categorical(train_label, num_classes=6)
            test_label = to_categorical(test_label, num_classes=6)

            #append to correct participant datapool
            partArrays[part-1][0] = pd.concat( [partArrays[part-1][0], part_train] ) if len(partArrays[part-1][0]) != 0 else part_train
            partArrays[part-1][2] = pd.concat( [partArrays[part-1][2], part_test] ) if len(partArrays[part-1][2]) != 0 else part_test

            #append labels to participant pool
            partArrays[part-1][1] = np.concatenate( (partArrays[part-1][1], train_label) ) if len(partArrays[part-1][1]) != 0 else train_label
            partArrays[part-1][3] = np.concatenate( (partArrays[part-1][3], test_label) ) if len(partArrays[part-1][3]) != 0 else test_label
    
    return partArrays

# I need a function that takes not of training length and testing length
# combines the pandas DF, normalizes the data
# splits the data back into training and testing
#should normalize all HAR data for one participant (both training and testing data)
def normalizeParticipants(partArrays):
    for participant in partArrays:
        allData = pd.concat([participant[0], participant[2]], axis=0)

        scaler = preprocessing.StandardScaler().fit(allData)
        allData = scaler.transform(allData)
        #resplit data into training and testing (back to dataframe)
        train_numpy = allData[:len(participant[0]), :]
        test_numpy = allData[len(participant[0]):, :]
        participant[0] = pd.DataFrame(data=train_numpy, index=None, columns=features)
        participant[2] = pd.DataFrame(data=test_numpy, index=None, columns=features)
    
    

#this combines the participants training and testing data into a pooled dataset
def getCentralDataset(partDataArray):

    #set one dataset for pooled data
    pooledSet = initialSets(1)
    for participant in partDataArray:

        #append to correct participant datapool
        pooledSet[0][0] = pd.concat( [pooledSet[0][0], participant[0]] ) if len(pooledSet[0][0]) != 0 else participant[0]
        pooledSet[0][2] = pd.concat( [pooledSet[0][2], participant[2]] ) if len(pooledSet[0][2]) != 0 else participant[2]

        #append labels to participant pool
        pooledSet[0][1] = np.concatenate( (pooledSet[0][1], participant[1]) ) if len(pooledSet[0][1]) != 0 else participant[1]
        pooledSet[0][3] = np.concatenate( (pooledSet[0][3], participant[3]) ) if len(pooledSet[0][3]) != 0 else participant[3]

    return pooledSet


if __name__ == "__main__":
    partData = getIndividualDatasets(numbParts)
    normalizeParticipants(partData) #HAR data needs to be normalized
    pooledData = getCentralDataset(partData)

#zip files then pull
#scp myfile.zip david@x.x.x.x:/home/david/Data/
