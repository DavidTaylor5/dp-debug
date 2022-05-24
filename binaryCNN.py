from tabnanny import check, verbose
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math

import tensorflow as tf
import os

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import dataPreprocess



# I need to make sure that my federated learning settings are reproducible
# I need to create a 'seed' for both python random and tf random
#####################################################################
RANDOM_SEED = 47568
#seed(47568
# #tf.random.set_random_seed(seed_value))
os.environ['PYTHONHASHSEED']=str(47568)
#random.seed(47568)
tf.random.set_seed(47568)

np.random.seed(RANDOM_SEED)



###################################  Sensitive Attributes   #######################################
#INDEXES OF SENTSITIVE ATTRIBUTES ZERO INDEXED
#the indexes of males "1"
male_indexes = [0, 1, 3, 5, 8, 10, 11, 12, 13, 14, 16, 19, 20, 21]
#the datasets of females "0"
female_indexes = [2, 4, 6, 7, 9, 15, 17, 18, 22, 23]

#old_indexes = [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 15, 18, 21] #greater than 26 (15)

#young_indexes = [4, 7, 10, 16, 17, 19, 20, 22, 23] #less than or equal to 26 (9)




def pool_by_attribute(attr_indexes, participantData):

    attr_train_X = np.array([])
    attr_train_y = np.array([])

    attr_test_X = np.array([])
    attr_test_y = np.array([])

    for p in range(0, len(participantData)):

        # this is an index associated with attribute
        if(p in attr_indexes):

            attr_train_X = np.concatenate( [attr_train_X, participantData[p][0]] ) if len(attr_train_X) != 0 else participantData[p][0]
            attr_train_y = np.concatenate( [attr_train_y, participantData[p][1]] ) if len(attr_train_y) != 0 else participantData[p][1]

            attr_test_X = np.concatenate( [attr_test_X, participantData[p][2]] ) if len( attr_test_X ) != 0 else participantData[p][2]
            attr_test_y = np.concatenate( [attr_test_y, participantData[p][3]] ) if len(attr_test_y) != 0 else participantData[p][3]

    #return training data and labels that have been pooled by an attribute
    return( [attr_train_X, attr_train_y, attr_test_X, attr_test_y] ) 


def check_fairness(model, prot_data, unprot_data): #I might have to change these if I add more data to attribute pooling

    prot_test_pred = model.predict(prot_data[2])
    prot_test_truth = prot_data[3]

    #change back to single value
    prot_test_pred = np.argmax(prot_test_pred, axis=1)
    prot_test_truth = np.argmax(prot_test_truth, axis=1)

    tn, fp, fn, tp = confusion_matrix(y_true=prot_test_truth, y_pred=prot_test_pred, labels=[0, 1]).ravel()

    prot_tpr = tp / (tp + fn)
    prot_fpr = fp/(fp+tn)
    prot_dp = (tp+fp)/len(prot_test_truth)

    unprot_test_pred = model.predict(unprot_data[2])
    unprot_test_truth = unprot_data[3]

    #change back to single value
    unprot_test_pred = np.argmax(unprot_test_pred, axis=1)
    unprot_test_truth = np.argmax(unprot_test_truth, axis=1)


    tn, fp, fn, tp = confusion_matrix(y_true=unprot_test_truth, y_pred=unprot_test_pred, labels=[0, 1]).ravel()
    unprot_tpr = tp / (tp + fn)
    unprot_fpr = fp/(fp+tn)
    unprot_dp = (tp+fp)/len(unprot_test_truth)

    print("DI:", max(prot_tpr / unprot_tpr, unprot_tpr / prot_tpr))
    print("EOP:", abs(prot_tpr-unprot_tpr))
    print("Avg EP diff:", 0.5 * (abs(prot_tpr-unprot_tpr) + abs(prot_fpr-unprot_fpr)))
    print("SPD:", abs(prot_dp-unprot_dp))



################################### PREPROCESSING FUNCTIONS #######################################




# by calculating window size I won't have to contatenate windows/numpy arrays (slow)
def calcWindowAmount(participantData, windowSize):
    trainCounter = 0
    testCounter = 0

    start = 0
    end = start + windowSize

    #loop through training dataset
    while(end < len(participantData[0])):
        x = participantData[1][start]
        y = participantData[1][end-1]
        if participantData[1][start].all() == participantData[1][end-1].all():
            trainCounter +=1

        start = end
        end = end+ windowSize

    start = 0
    end = start + windowSize

    while(end < len(participantData[2])):

        if participantData[3][start].all() == participantData[3][end-1].all():
            testCounter +=1

        start = end
        end = end+ windowSize

    return [trainCounter, testCounter]
    
def windowData(participantData, windowSize):
    #go through the data looking at each 50 window -> creating a window for this if label is same for first and last
    #calculate the number of viable windows -> initialize a dataframe (3 dimensions) / numpy array of zeros (50 reduced -> 1)

    trainSize, testSize = calcWindowAmount(participantData, windowSize)
    trainSet = np.zeros( (trainSize, windowSize, 12) ) 
    trainLabels = np.zeros( (trainSize, 6), dtype= np.int64 )
    testSet = np.zeros( (testSize, windowSize, 12) )
    testLabels = np.zeros( (testSize, 6) , dtype= np.int64) 

    trainCounter = 0
    testCounter = 0

    start = 0
    end = start + windowSize

    #loop through training dataset
    while(end < len(participantData[0])):

        # if the label is consistent throughout the window
        if(participantData[1][start].all() == participantData[1][end-1].all()):
            windowArray = participantData[0].iloc[start:end, :].to_numpy().astype(np.float32)
            trainSet[trainCounter] = windowArray
            trainLabels[trainCounter] = participantData[1][start]
            trainCounter +=1
        start = end
        end = end+ windowSize

    start = 0
    end = start + windowSize

    while(end < len(participantData[2])):

        # if the label is consistent throughout the window
        if(participantData[3][start].all() == participantData[3][end-1].all()):
            windowArray = participantData[2].iloc[start:end, :].to_numpy().astype(np.float32)
            testSet[testCounter] = windowArray
            testLabels[testCounter] = participantData[3][start]
            testCounter +=1

        start = end
        end = end+ windowSize

    return [trainSet, trainLabels, testSet, testLabels]

def getPoolSize(windowDatasets, windowSize):

    pool_train_size = 0
    pool_test_size = 0

    for participant in windowDatasets:
        pool_train_size += len(participant[0])
        pool_test_size += len(participant[2])

    return [pool_train_size, pool_test_size]

def poolWindows(windowDatasets, windowSize):
    
    trainSize, testSize = getPoolSize(windowDatasets, windowSize)

    pool_train_windows = np.zeros( (trainSize, windowSize, 12) )
    pool_train_labels = np.zeros( (trainSize, 2), dtype=np.int64)
    pool_test_windows = np.zeros( (testSize, windowSize, 12) )
    pool_test_labels = np.zeros( (testSize, 2), dtype=np.int64)


    start_train = 0
    start_test = 0

    for participant in windowDatasets:
        xc = len(participant[0])
        pool_train_windows[start_train:(start_train + len(participant[0]))] = participant[0]
        pool_train_labels[start_train:(start_train + len(participant[1]))] = participant[1]
        pool_test_windows[start_test:(start_test + len(participant[2]))] = participant[2]
        pool_test_labels[start_test:(start_test + len(participant[3]))] = participant[3]

        start_train += len(participant[0])
        start_test += len(participant[2])
    
    return[pool_train_windows, pool_train_labels, pool_test_windows, pool_test_labels]

def participantWindows(participantData, windowSize):
    partWindows = []

    for participant in participantData:
        partWindows.append(windowData(participant, windowSize))
    
    return partWindows

########################## CNN / TRAINING FUNCTIONS #####################################################

def sensor_activity_binary( n_timesteps, n_features, n_outputs): #(64, 50, 12) labels should be in form this is what I'm passing into the cnn.
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=2, activation='hard_sigmoid', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=128, kernel_size=8, activation='hard_sigmoid'))
    model.add(MaxPooling1D(pool_size=8))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']) #I HAVE TAKEN OUT COMPILE LINE FOR FL DP #changed to binary_crossentropy

    #print(model.summary())

    return model

def printAverages(acc_list, loss_list):

    sum_acc = 0
    sum_loss = 0

    for acc in acc_list:
        sum_acc += acc
    for loss in loss_list:
        sum_loss += loss
    print("Average Loss -> ", (sum_loss/len(loss_list)))
    print("Average Accuracy -> ", (sum_acc/len(acc_list)))


def individual_cnn(readyData):
    counter = 1
    avg_accuracy = []
    avg_loss = []
    for participant in readyData:
        #print("Participant", counter,  "Local Training ->")
        david_cnn = sensor_activity_binary(n_timesteps=50, n_features=12, n_outputs=2)

        david_cnn.fit(participant[0], participant[1], batch_size=32, epochs=20, verbose=0, validation_data=(participant[2], participant[3]))
        score = david_cnn.evaluate(participant[2], participant[3], verbose=0)
        #print('Test loss:', score[0]) 
        print('participant -> ' + str(counter) +' Test accuracy:', score[1])

        avg_accuracy.append(score[1])
        avg_loss.append(score[0])

        counter +=1
    
    printAverages(avg_accuracy, avg_loss)

#not set up for k fold validation
def centralized_cnn(pooledData):
    # counter = 1
    # avg_accuracy = []
    # avg_loss = []

    #print("Participant", counter,  "Local Training ->")
    david_cnn = sensor_activity_binary(n_timesteps=50, n_features=12, n_outputs=2)

    #my callback function is forcing clients to exit way to early? they don't produce predictive models? I would probably need a validation set? Takes 9 rounds before DI changes?
    my_early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30) #patience 7 instead of 3

    #20% of the 80% of the training data will be used as validation data! #I need to account for this split? Training with less data in centralized compared to FL
    X_train, X_val, y_train, y_val = train_test_split(pooledData[0], pooledData[1], test_size=0.2, shuffle=True, random_state=RANDOM_SEED, stratify=pooledData[1])

    history = david_cnn.fit(X_train, y_train, batch_size=32, epochs=600, verbose=1, validation_data=(X_val, y_val), callbacks=[my_early_stop]) #instead of 20 epochs when should it stop?
    #callbacks=[my_early_stop]

    loss_values = history.history['loss']
    loss_validation = history.history['val_loss']

    epoch_num = range(1, len(loss_values)+1)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(epoch_num, loss_values, label="CL-loss.py")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    fig.savefig('CL.jpg', bbox_inches='tight', dpi=150)

    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(epoch_num, loss_validation, label="CL-validation")
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()

    fig2.savefig('CL-val.jpg', bbox_inches='tight', dpi=150)

    score = david_cnn.evaluate(pooledData[2], pooledData[3], verbose=0)
    #print('Test loss:', score[0]) 
    print('Pooled Test accuracy:', score[1])

    return david_cnn

    # avg_accuracy.append(score[1])
    # avg_loss.append(score[0])

    # counter +=1
    
    #printAverages(avg_accuracy, avg_loss)

#based on  walking upstairs and walking downstairs are now -> [1, 0]
# all other activities are -> [0, 1]
#I'll transform the labels for training and testing for participants to represent new binary
# def transform_to_binary(pooled_windows):
#     for(x in pooled_windows[])
def produce_binary_labels(pooled_labels):
    binary_labels = np.zeros( (len(pooled_labels), 2), dtype=np.int64)

    for i in range(0, len(pooled_labels)):
        binary_labels[i] = to_binary_class(pooled_labels[i])

    return binary_labels

def to_binary_class(oneShot):
    index_val = np.argmax(oneShot)
    if(index_val == 1 or index_val == 5): #testing walking and jogging #sitting standing and walking test (gentle movements) #walking down stairs, sitting, walking
        return [1, 0]
    else:
        return [0, 1]

def participant_list_to_binary(part_windows):
    for part in part_windows:
        part[1] = produce_binary_labels(part[1])
        part[3] = produce_binary_labels(part[3])


if __name__ == "__main__":
    #this is the data for individual training and pooled (hasn't been put into windows yet)

    #to_binary_class_vect = np.vectorize(to_binary_class)


    partData = dataPreprocess.getIndividualDatasets(24)
    dataPreprocess.normalizeParticipants(partData)
    pooledData = dataPreprocess.getCentralDataset(partData)

    part_windows = participantWindows(partData, 50)
    participant_list_to_binary(part_windows) #part windows is still in index order in part_windows
    pooled_windows = poolWindows(part_windows, 50) #pooling all participants

    # This code is for testing gender for discrimination
    pooled_men_test = pool_by_attribute(male_indexes, part_windows)
    pooled_women_test = pool_by_attribute(female_indexes, part_windows)

    # This code seperates data by age range (<= 25) versus (> 25)
    #pooled_old_test = pool_by_attribute(old_indexes, part_windows)
    #pooled_young_test = pool_by_attribute(young_indexes, part_windows)

    #pooled_windows[1] = produce_binary_labels(pooled_windows[1]) 
    #pooled_windows[3] = produce_binary_labels(pooled_windows[3])

    #tests individuals with their own data
    #individual_cnn(part_windows) #I'm not even going to train individuals!


    #tests centralized (all participants pool)
    trained_model = centralized_cnn(pooled_windows) # -> Pooled Test accuracy: 0.7921686768531799
    check_fairness(trained_model, pooled_men_test, pooled_women_test) #5644 = 3237(men) + 2407 (women)


    """
    Fairness for jogging/walking v everything else
    Pooled Test accuracy: 0.9082211256027222
    DI: 1.0573623573437685
    EOP: 0.04895279394863983
    Avg EP diff: 0.024900214483216054
    SPD: 0.043665377691135876
    
    """

    #I might just remove standing and sitting altogether

    """
#training with binary classification -> somehow less! (walking up and walking down stairs)

participant -> 1 Test accuracy: 0.8152610659599304
participant -> 2 Test accuracy: 0.7871485948562622
participant -> 3 Test accuracy: 0.7919999957084656
participant -> 4 Test accuracy: 0.7901785969734192
participant -> 5 Test accuracy: 0.7894737124443054
participant -> 6 Test accuracy: 0.7991266250610352
participant -> 7 Test accuracy: 0.7918367385864258
participant -> 8 Test accuracy: 0.7860082387924194
participant -> 9 Test accuracy: 0.7850877046585083
participant -> 10 Test accuracy: 0.7698744535446167
participant -> 11 Test accuracy: 0.797468364238739
participant -> 12 Test accuracy: 0.7981220483779907
participant -> 13 Test accuracy: 0.8676470518112183
participant -> 14 Test accuracy: 0.8318583965301514
participant -> 15 Test accuracy: 0.8059071898460388
participant -> 16 Test accuracy: 0.7739463448524475
participant -> 17 Test accuracy: 0.7702702879905701
participant -> 18 Test accuracy: 0.7529880404472351
participant -> 19 Test accuracy: 0.8118466734886169
participant -> 20 Test accuracy: 0.7681818008422852
participant -> 21 Test accuracy: 0.7599999904632568
participant -> 22 Test accuracy: 0.8080357313156128
participant -> 23 Test accuracy: 0.7247706651687622
participant -> 24 Test accuracy: 0.8529411554336548
Average Loss ->  0.5056588525573412
Average Accuracy ->  0.792915811141332
Pooled Test accuracy: 0.7921686768531799

binary classification (sitting and standing)
participant -> 1 Test accuracy: 0.6626505851745605
participant -> 2 Test accuracy: 0.8313252925872803
participant -> 3 Test accuracy: 0.7639999985694885
participant -> 4 Test accuracy: 0.59375
participant -> 5 Test accuracy: 0.6746411323547363
participant -> 6 Test accuracy: 0.807860255241394
participant -> 7 Test accuracy: 0.5387755036354065
participant -> 8 Test accuracy: 0.7613168954849243
participant -> 9 Test accuracy: 0.8289473652839661
participant -> 10 Test accuracy: 0.573221743106842
participant -> 11 Test accuracy: 0.49789029359817505
participant -> 12 Test accuracy: 0.577464759349823
participant -> 13 Test accuracy: 0.5686274766921997
participant -> 14 Test accuracy: 0.721238911151886
participant -> 15 Test accuracy: 0.5189873576164246
participant -> 16 Test accuracy: 0.7164750695228577
participant -> 17 Test accuracy: 0.8063063025474548
participant -> 18 Test accuracy: 0.6055777072906494
participant -> 19 Test accuracy: 0.7944250702857971
participant -> 20 Test accuracy: 0.5954545736312866
participant -> 21 Test accuracy: 0.7890909314155579
participant -> 22 Test accuracy: 0.5446428656578064
participant -> 23 Test accuracy: 0.6605504751205444
participant -> 24 Test accuracy: 0.5686274766921997
Average Loss ->  0.6402167206009229
Average Accuracy ->  0.6667436684171358
Pooled Test accuracy: 0.9799787402153015

participant -> 1 Test accuracy: 0.5341365337371826
participant -> 2 Test accuracy: 0.6265060305595398
participant -> 3 Test accuracy: 0.8080000281333923
participant -> 4 Test accuracy: 0.8080357313156128
participant -> 5 Test accuracy: 0.49760764837265015
participant -> 6 Test accuracy: 0.6768559217453003
participant -> 7 Test accuracy: 0.8693877458572388
participant -> 8 Test accuracy: 0.6090534925460815
participant -> 9 Test accuracy: 0.8289473652839661
participant -> 10 Test accuracy: 0.573221743106842
participant -> 11 Test accuracy: 0.8649789094924927
participant -> 12 Test accuracy: 0.7230046987533569
participant -> 13 Test accuracy: 0.8039215803146362
participant -> 14 Test accuracy: 0.6769911646842957
participant -> 15 Test accuracy: 0.5189873576164246
participant -> 16 Test accuracy: 0.7432950139045715
participant -> 17 Test accuracy: 0.5585585832595825
participant -> 18 Test accuracy: 0.6772908568382263
participant -> 19 Test accuracy: 0.6794425249099731
participant -> 20 Test accuracy: 0.5954545736312866
participant -> 21 Test accuracy: 0.581818163394928
participant -> 22 Test accuracy: 0.5446428656578064
participant -> 23 Test accuracy: 0.6605504751205444
participant -> 24 Test accuracy: 0.5686274766921997
Average Loss ->  0.6316871580978235
Average Accuracy ->  0.6678881868720055
Pooled Test accuracy: 0.9698795080184937


binary classification (walking, jogging)
participant -> 1 Test accuracy: 0.718875527381897
participant -> 2 Test accuracy: 0.7349397540092468
participant -> 3 Test accuracy: 0.6639999747276306
participant -> 4 Test accuracy: 0.6160714030265808
participant -> 5 Test accuracy: 0.7129186391830444
participant -> 6 Test accuracy: 0.6550218462944031
participant -> 7 Test accuracy: 0.6693877577781677
participant -> 8 Test accuracy: 0.6790123581886292
participant -> 9 Test accuracy: 0.6710526347160339
participant -> 10 Test accuracy: 0.6569037437438965
participant -> 11 Test accuracy: 0.7046413421630859
participant -> 12 Test accuracy: 0.6244131326675415
participant -> 13 Test accuracy: 0.7009803652763367
participant -> 14 Test accuracy: 0.6681416034698486
participant -> 15 Test accuracy: 0.6751055121421814
participant -> 16 Test accuracy: 0.6666666865348816
participant -> 17 Test accuracy: 0.6711711883544922
participant -> 18 Test accuracy: 0.6414342522621155
participant -> 19 Test accuracy: 0.703832745552063
participant -> 20 Test accuracy: 0.6363636255264282
participant -> 21 Test accuracy: 0.6581818461418152
participant -> 22 Test accuracy: 0.6473214030265808
participant -> 23 Test accuracy: 0.6146789193153381
participant -> 24 Test accuracy: 0.7156862616539001
Average Loss ->  0.6223721702893575
Average Accuracy ->  0.6711167717973391
Pooled Test accuracy: 0.9071580171585083


After I have implemented the no randomness -> 
test 1)
Pooled Test accuracy: 0.9105244278907776
DI: 1.0549090348752685
EOP: 0.0474506647985341
Avg EP diff: 0.027032489107084844
SPD: 0.04013933718960727

test 2) 
Pooled Test accuracy: 0.9105244278907776
DI: 1.0549090348752685
EOP: 0.0474506647985341
Avg EP diff: 0.027032489107084844
SPD: 0.04013933718960727

okay so no more randomness

changing to binary_crossentropy
test 1)
Pooled Test accuracy: 0.8747342228889465
DI: 1.0416950201268644
EOP: 0.034410597406808074
Avg EP diff: 0.02629121002030912
SPD: 0.0389036251105217
test 2) 
Pooled Test accuracy: 0.8747342228889465
DI: 1.0416950201268644
EOP: 0.034410597406808074
Avg EP diff: 0.02629121002030912
SPD: 0.0389036251105217

# 100 epochs early stoppping results test 1) (jogging and walking v everything else)
Pooled Test accuracy: 0.9572997689247131
DI: 1.015175321072779
EOP: 0.014237081081394098
Avg EP diff: 0.007823561254274622
SPD: 0.021688877526019223


#98 epochs test 2 (~100 epochs to get the best centralized model)
Pooled Test accuracy: 0.9578313231468201
DI: 1.0164928750261506
EOP: 0.015473175024533714
Avg EP diff: 0.007807894664374215
SPD: 0.022104332449160102
    """

"""

#sitting standing walking test 1) early stop -> 158 epochs
Pooled Test accuracy: 0.967399001121521
DI: 1.0096009016855372
EOP: 0.009113899426125838
Avg EP diff: 0.009740022422329181
SPD: 0.005848326994982556

#dws, sitting, walking -> 139 epochs
Pooled Test accuracy: 0.9153082966804504
DI: 1.0180729506737105
EOP: 0.017321367618614136
Avg EP diff: 0.015704202526511273
SPD: 0.00869259531494676

################### now I'm testing the attribute age ############################
test (jogging and walking) -> 97 epochs
Pooled Test accuracy: 0.9574769735336304
DI: 1.005887659194739
EOP: 0.005540828696198319
Avg EP diff: 0.009200944313715953
SPD: 0.0012249658813473507

test (downstairs, upstairs)

# how many epochs of centralized learning before the model starts gaining a lot of accuracy? -> at least 25
Epoch 14/20
706/706 [==============================] - 2s 3ms/step - loss: 0.5585 - accuracy: 0.6602 - val_loss: 0.5689 - val_accuracy: 0.6743
Epoch 15/20
706/706 [==============================] - 2s 3ms/step - loss: 0.5499 - accuracy: 0.6685 - val_loss: 0.5553 - val_accuracy: 0.6938
Epoch 16/20
706/706 [==============================] - 2s 3ms/step - loss: 0.5414 - accuracy: 0.6831 - val_loss: 0.5416 - val_accuracy: 0.7365
Epoch 17/20
706/706 [==============================] - 2s 3ms/step - loss: 0.5302 - accuracy: 0.6987 - val_loss: 0.5297 - val_accuracy: 0.7112
Epoch 18/20
706/706 [==============================] - 2s 3ms/step - loss: 0.5095 - accuracy: 0.7274 - val_loss: 0.4828 - val_accuracy: 0.7975
Epoch 19/20
706/706 [==============================] - 2s 3ms/step - loss: 0.4838 - accuracy: 0.7549 - val_loss: 0.4711 - val_accuracy: 0.7741
Epoch 20/20
706/706 [==============================] - 2s 3ms/step - loss: 0.4516 - accuracy: 0.7862 - val_loss: 0.3973 - val_accuracy: 0.8747
Pooled Test accuracy: 0.8747342228889465
DI: 1.0416950201268644
EOP: 0.034410597406808074
Avg EP diff: 0.02629121002030912
SPD: 0.0389036251105217


centralized walking jogging -> paitence 7 -> I SHOULD PROBABLY MAKE THE LOSS GRAPH BASED ON CENTRALIZED LEARNING
Epoch 113/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1370 - accuracy: 0.9521 - val_loss: 0.1336 - val_accuracy: 0.9573
Epoch 114/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1345 - accuracy: 0.9507 - val_loss: 0.1319 - val_accuracy: 0.9585
Epoch 115/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1374 - accuracy: 0.9510 - val_loss: 0.1315 - val_accuracy: 0.9585
Epoch 116/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1356 - accuracy: 0.9508 - val_loss: 0.1217 - val_accuracy: 0.9624
Epoch 117/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1334 - accuracy: 0.9517 - val_loss: 0.1293 - val_accuracy: 0.9598
Epoch 118/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1351 - accuracy: 0.9530 - val_loss: 0.1192 - val_accuracy: 0.9630
Epoch 119/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1325 - accuracy: 0.9526 - val_loss: 0.1228 - val_accuracy: 0.9617
Epoch 120/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1322 - accuracy: 0.9523 - val_loss: 0.1335 - val_accuracy: 0.9585
Epoch 121/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1318 - accuracy: 0.9533 - val_loss: 0.1316 - val_accuracy: 0.9585
Epoch 122/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1310 - accuracy: 0.9535 - val_loss: 0.1290 - val_accuracy: 0.9598
Epoch 123/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1323 - accuracy: 0.9519 - val_loss: 0.1250 - val_accuracy: 0.9612
Epoch 124/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1310 - accuracy: 0.9537 - val_loss: 0.1129 - val_accuracy: 0.9672
Epoch 125/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1298 - accuracy: 0.9542 - val_loss: 0.1322 - val_accuracy: 0.9591
Epoch 126/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1295 - accuracy: 0.9547 - val_loss: 0.1141 - val_accuracy: 0.9663
Epoch 127/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1281 - accuracy: 0.9547 - val_loss: 0.1233 - val_accuracy: 0.9610
Epoch 128/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1285 - accuracy: 0.9556 - val_loss: 0.1233 - val_accuracy: 0.9610
Epoch 129/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1306 - accuracy: 0.9553 - val_loss: 0.1233 - val_accuracy: 0.9614
Epoch 130/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1288 - accuracy: 0.9544 - val_loss: 0.1263 - val_accuracy: 0.9605
Epoch 131/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1285 - accuracy: 0.9554 - val_loss: 0.1176 - val_accuracy: 0.9642
Epoch 132/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1305 - accuracy: 0.9540 - val_loss: 0.1188 - val_accuracy: 0.9633
Epoch 133/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1284 - accuracy: 0.9564 - val_loss: 0.1144 - val_accuracy: 0.9653
Epoch 134/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1254 - accuracy: 0.9578 - val_loss: 0.1355 - val_accuracy: 0.9568
Epoch 135/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1263 - accuracy: 0.9545 - val_loss: 0.1309 - val_accuracy: 0.9592
Epoch 136/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1247 - accuracy: 0.9563 - val_loss: 0.1202 - val_accuracy: 0.9631
Epoch 137/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1247 - accuracy: 0.9554 - val_loss: 0.1128 - val_accuracy: 0.9669
Epoch 138/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1230 - accuracy: 0.9576 - val_loss: 0.1272 - val_accuracy: 0.9605
Epoch 139/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1242 - accuracy: 0.9552 - val_loss: 0.1114 - val_accuracy: 0.9674
Epoch 140/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1228 - accuracy: 0.9570 - val_loss: 0.1246 - val_accuracy: 0.9619
Epoch 141/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1228 - accuracy: 0.9568 - val_loss: 0.1206 - val_accuracy: 0.9630
Epoch 142/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1244 - accuracy: 0.9565 - val_loss: 0.1204 - val_accuracy: 0.9630
Epoch 143/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1264 - accuracy: 0.9570 - val_loss: 0.1195 - val_accuracy: 0.9633
Epoch 144/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1232 - accuracy: 0.9563 - val_loss: 0.1189 - val_accuracy: 0.9633
Epoch 145/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1226 - accuracy: 0.9570 - val_loss: 0.1318 - val_accuracy: 0.9600
Epoch 146/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1229 - accuracy: 0.9579 - val_loss: 0.1196 - val_accuracy: 0.9635
Epoch 147/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1225 - accuracy: 0.9576 - val_loss: 0.1144 - val_accuracy: 0.9646
Epoch 148/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1222 - accuracy: 0.9583 - val_loss: 0.1156 - val_accuracy: 0.9647
Epoch 149/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1192 - accuracy: 0.9589 - val_loss: 0.1148 - val_accuracy: 0.9649
Epoch 150/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1230 - accuracy: 0.9594 - val_loss: 0.1159 - val_accuracy: 0.9647
Epoch 151/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1199 - accuracy: 0.9586 - val_loss: 0.1248 - val_accuracy: 0.9630
Epoch 152/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1193 - accuracy: 0.9585 - val_loss: 0.1363 - val_accuracy: 0.9566
Epoch 153/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1208 - accuracy: 0.9575 - val_loss: 0.1326 - val_accuracy: 0.9594
Epoch 154/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1178 - accuracy: 0.9600 - val_loss: 0.1335 - val_accuracy: 0.9585
Epoch 155/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1182 - accuracy: 0.9594 - val_loss: 0.1197 - val_accuracy: 0.9639
Epoch 156/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1151 - accuracy: 0.9608 - val_loss: 0.1181 - val_accuracy: 0.9640
Epoch 157/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1177 - accuracy: 0.9595 - val_loss: 0.1206 - val_accuracy: 0.9640
Epoch 158/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1166 - accuracy: 0.9597 - val_loss: 0.1110 - val_accuracy: 0.9678
Epoch 159/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1157 - accuracy: 0.9600 - val_loss: 0.1185 - val_accuracy: 0.9642
Epoch 160/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1152 - accuracy: 0.9601 - val_loss: 0.1146 - val_accuracy: 0.9653
Epoch 161/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1176 - accuracy: 0.9591 - val_loss: 0.1155 - val_accuracy: 0.9653
Epoch 162/200
706/706 [==============================] - 2s 3ms/step - loss: 0.1175 - accuracy: 0.9595 - val_loss: 0.1249 - val_accuracy: 0.9626
Epoch 163/200
706/706 [==============================] - 2s 4ms/step - loss: 0.1166 - accuracy: 0.9595 - val_loss: 0.1193 - val_accuracy: 0.9640
Pooled Test accuracy: 0.9640325903892517
DI: 1.016879739607615
EOP: 0.015994245070822144
Avg EP diff: 0.008379535891470684
SPD: 0.022264122804214193


patience value -> 10 centralized learning -> 231 epochs
Pooled Test accuracy: 0.9707654118537903
DI: 1.0089390916516674
EOP: 0.00859157661321841
Avg EP diff: 0.006219356240472971
SPD: 0.016426448499568536


patience value -> 15 centralized learning -> 263 epochs
Epoch 247/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0969 - accuracy: 0.9684 - val_loss: 0.1136 - val_accuracy: 0.9681
Epoch 248/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0925 - accuracy: 0.9703 - val_loss: 0.1110 - val_accuracy: 0.9692
Epoch 249/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0943 - accuracy: 0.9697 - val_loss: 0.1098 - val_accuracy: 0.9701
Epoch 250/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0950 - accuracy: 0.9687 - val_loss: 0.1033 - val_accuracy: 0.9724
Epoch 251/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0952 - accuracy: 0.9678 - val_loss: 0.1048 - val_accuracy: 0.9713
Epoch 252/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0948 - accuracy: 0.9691 - val_loss: 0.1089 - val_accuracy: 0.9702
Epoch 253/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0957 - accuracy: 0.9688 - val_loss: 0.1096 - val_accuracy: 0.9697
Epoch 254/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0966 - accuracy: 0.9683 - val_loss: 0.1156 - val_accuracy: 0.9676
Epoch 255/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0933 - accuracy: 0.9690 - val_loss: 0.1117 - val_accuracy: 0.9686
Epoch 256/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0937 - accuracy: 0.9702 - val_loss: 0.1118 - val_accuracy: 0.9688
Epoch 257/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0944 - accuracy: 0.9685 - val_loss: 0.1073 - val_accuracy: 0.9715
Epoch 258/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0948 - accuracy: 0.9693 - val_loss: 0.1027 - val_accuracy: 0.9725
Epoch 259/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0958 - accuracy: 0.9689 - val_loss: 0.1064 - val_accuracy: 0.9718
Epoch 260/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0927 - accuracy: 0.9703 - val_loss: 0.1288 - val_accuracy: 0.9642
Epoch 261/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0979 - accuracy: 0.9689 - val_loss: 0.1160 - val_accuracy: 0.9681
Epoch 262/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0978 - accuracy: 0.9688 - val_loss: 0.1111 - val_accuracy: 0.9690
Epoch 263/400
706/706 [==============================] - 2s 3ms/step - loss: 0.0949 - accuracy: 0.9691 - val_loss: 0.0999 - val_accuracy: 0.9731
Pooled Test accuracy: 0.9730687737464905
DI: 1.007885643925815
EOP: 0.007616017693222843
Avg EP diff: 0.006005304235338911
SPD: 0.01562749672429764

#run this even more, maybe an anomoly

Epoch 399/400
706/706 [==============================] - 3s 4ms/step - loss: 0.0811 - accuracy: 0.9734 - val_loss: 0.1156 - val_accuracy: 0.9685
Epoch 400/400
706/706 [==============================] - 3s 4ms/step - loss: 0.0789 - accuracy: 0.9743 - val_loss: 0.1031 - val_accuracy: 0.9725
Pooled Test accuracy: 0.9725372195243835
DI: 1.0093541514206075
EOP: 0.009021169918047245
Avg EP diff: 0.006707880347751112
SPD: 0.016554280783611874


Bioinformatics notes, cite the paper,  RPMNormCounts_final.txt -> this is the dataset we should use.
Compare case 18 short and 19 long, I can use their individual dataset?
Low virus load versus high virus load!

#with validation data I get the results
Epoch 316/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0913 - accuracy: 0.9701 - val_loss: 0.0732 - val_accuracy: 0.9796
Epoch 317/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0924 - accuracy: 0.9688 - val_loss: 0.0725 - val_accuracy: 0.9799
Pooled Test accuracy: 0.9714741110801697
DI: 1.0167083928847342
EOP: 0.01602753565711279
Avg EP diff: 0.010391056270587132
SPD: 0.021092326867150346

#I have added reproducible train validation split and I stop when validation stops improving!
#testing to see if no preprocessing is working correctly
Epoch 443/600
565/565 [==============================] - 2s 4ms/step - loss: 0.0804 - accuracy: 0.9740 - val_loss: 0.0735 - val_accuracy: 0.9807
Epoch 444/600
565/565 [==============================] - 2s 4ms/step - loss: 0.0810 - accuracy: 0.9742 - val_loss: 0.0728 - val_accuracy: 0.9805
Epoch 445/600
565/565 [==============================] - 2s 4ms/step - loss: 0.0822 - accuracy: 0.9731 - val_loss: 0.0747 - val_accuracy: 0.9814
Pooled Test accuracy: 0.9700567126274109
DI: 1.0118978573163349
EOP: 0.011401881062867103
Avg EP diff: 0.0078045015186005645
SPD: 0.018173489714827462
#why does no weights run for much more epochs than preprocessed with weights?

#first let me check if this result is reproducible!
Epoch 444/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0810 - accuracy: 0.9742 - val_loss: 0.0728 - val_accuracy: 0.9805
Epoch 445/600
565/565 [==============================] - 2s 3ms/step - loss: 0.0822 - accuracy: 0.9731 - val_loss: 0.0747 - val_accuracy: 0.9814
Pooled Test accuracy: 0.9700567126274109
DI: 1.0118978573163349
EOP: 0.011401881062867103
Avg EP diff: 0.0078045015186005645
SPD: 0.018173489714827462
"""
