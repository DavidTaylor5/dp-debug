from gc import callbacks
import os
import math

"""
This code is just for testing out the DPFL values. I should just be changing the optimizer and loss to be XX and XX
"""

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf
import numpy as np

import binaryCNN
import dataPreprocess

from typing import Optional, Tuple

import matplotlib as plt

import dpCNN

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

NUM_CLIENTS = 24

class FlwerClient(fl.client.NumPyClient):
    def __init__(self, model, cid) -> None:
        super().__init__()
        self.model = model
        self.cid = cid
        self.acc = []
        self.loss = []


    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):

        #print("before mydata? Error?")

        myData = read_part_data_file(self.cid) #this doesn't exist in ray client memeory? read from file!

        print("Before setting model weights")

        self.model.set_weights(parameters)

        print("before fitting my data to my model")


        self.model.fit(myData[0], myData[1], epochs=35, verbose=0) #I could potentially attach a callback function here to make early stopping?


        print("after fitting my data to my model")

        return self.model.get_weights(), len(myData[0]), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        myData = read_part_data_file(self.cid) #this doesn't exist in ray client memeory? read from file!

        loss, acc = self.model.evaluate(myData[2], myData[3], verbose=2)

        return loss, len(myData[2]), {"accuracy":acc}



# necessary data below, window data for each participant
partData = dataPreprocess.getIndividualDatasets(1) #SETUP TO ONLY LOAD IN 1 participant
dataPreprocess.normalizeParticipants(partData)
pooledData = dataPreprocess.getCentralDataset(partData)

part_windows = binaryCNN.participantWindows(partData, 50)

binaryCNN.participant_list_to_binary(part_windows)
pooled_windows = binaryCNN.poolWindows(part_windows, 50) #pooling all participants
pooled_men_test = binaryCNN.pool_by_attribute(binaryCNN.male_indexes, part_windows)
pooled_women_test = binaryCNN.pool_by_attribute(binaryCNN.female_indexes, part_windows)

#I need the correct data!

print(len(part_windows))


def part_windows_to_file(part_windows):

    foldername = './part_windows_folder'

    for i in range (0, len(part_windows)):


        #I create 4 files for each participant
        fileName = '/participantTRAINX' + str(i) + '.npy'
        with open(foldername+fileName, 'wb') as f:
            np.save(f, part_windows[i][0])

        fileName = '/participantTRAINy' + str(i) + '.npy'
        with open(foldername+fileName, 'wb') as f:
            np.save(f, part_windows[i][1])

        fileName = '/participantTESTX' + str(i) + '.npy'
        with open(foldername+fileName, 'wb') as f:
            np.save(f, part_windows[i][2])

        fileName = '/participantTesty' + str(i) + '.npy'
        with open(foldername+fileName, 'wb') as f:
            np.save(f, part_windows[i][3])

def read_part_data_file(participantID): #I STARTED OUT WITH 139 GB of storate -> will this continue to disappear with ray spilled IO objects? After 10 rounds... -> Why is there no change in first 7 rounds?
    #I open 4 files and grab their numpy contents
    foldername = './part_windows_folder'

    with open(foldername + "/participantTRAINX" + str(participantID) + '.npy', 'rb' ) as f:
        train_X = np.load(f)

    with open(foldername + "/participantTRAINy" + str(participantID) + '.npy', 'rb' ) as f:
        train_y = np.load(f)

    with open(foldername + "/participantTESTX" + str(participantID) + '.npy', 'rb' ) as f:
        test_X = np.load(f)

    with open(foldername + "/participantTESTy" + str(participantID) + '.npy', 'rb' ) as f:
        test_y = np.load(f)

    return [train_X, train_y, test_X, test_y]
    



def client_fn(cid: str) -> fl.client.Client:
    # Load model
    model = dpCNN.sensor_activity_binary(n_timesteps=50, n_features=12, n_outputs=2) #I have added a model that uses DP settings for this code!

    # Create and return client
    return FlwerClient(model, cid)

# experiemental evaluate_config for clients
def evaluate_config(rnd: int): #EXPERIMENTAL
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps":val_steps}


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model.set_weights(weights)  # Update model with the latest parameters

        binaryCNN.check_fairness(model, pooled_men_test, pooled_women_test)
        score = model.evaluate(pooled_windows[2], pooled_windows[3], verbose=0) #checking score with pooled test set

        #I need to append my server level model's loss in a file then I can disply it as a graph!
        with open("FederatedLoss/DPFL.txt", "a") as f:
            f.write(str(score[0]) + "\n")
        #print('Test loss:', score[0]) 



        print('-> Pooled Test accuracy:', score[1])


    return evaluate


def main() -> None:

    #send a participant's data into a file!
    print("sending training data to files ...")
    part_windows_to_file(part_windows)
    print("done sending training data to bianry files!")
    # example of one participant getting their data back from files!
    #ready_work = read_part_data_file(0)


    a_model = dpCNN.sensor_activity_binary(n_timesteps=50, n_features=12, n_outputs=1)


    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus":4}, #trying 4
        num_rounds=90,
        strategy=fl.server.strategy.FedAvg(
            #fraction_fit=0.1,
            min_fit_clients=24, #testing, only fitting 4 participants per round #fitting with 10 clients a round
            min_available_clients=NUM_CLIENTS,

            eval_fn=get_eval_fn(a_model)
            # fraction_eval=0.2,
            # min_eval_clients=24,
            # on_evaluate_config_fn=evaluate_config
        )
    )


if __name__ == "__main__":


    my_model = dpCNN.sensor_activity_binary(n_timesteps=50, n_features=12, n_outputs=1)
    some_data = part_windows[0]

    # testing 1d output values
    one_dim = np.argmax(some_data[1], axis=1)
    #one_dim = np.reshape(one_dim, (len(one_dim), 1) ) #reshaping is not helping!



    #dummy_data = np.ones( (996, 64, 64) )
    my_model.fit(some_data[0], one_dim, batch_size=32, epochs=35, verbose=2) #why does this only allow microbatches of 1!

    # Maybe the losses cannot be computed because not 0, 1 normalized?
    #maybe my version of tensorflow, tensorflow privacy, or python may be off?

    # CHANGE THE OUTPUT TO ONE DIMENSION!

    """
    ValueError: Dimension size must be evenly divisible by 32 but is 1 for '{{node Reshape}} = Reshape[T=DT_FLOAT, Tshape=DT_INT32](binary_crossentropy/weighted_loss/value, Reshape/shape)' 
    with input shapes: [], [2] and with input tensors computed as partial shapes: input[1] = [32,?].
    """

    """" I've determined the 2 in minibatch size
    ValueError: Dimension size must be evenly divisible by 2 but is 1
    for 
    '{{node Reshape}} = Reshape[T=DT_FLOAT, Tshape=DT_INT32](binary_crossentropy/weighted_loss/value, Reshape/shape)' with input shapes: [], [2] 
    and with input tensors computed as partial shapes: input[1] = [2,?].

    ValueError: Dimension size must be evenly divisible by 32 but is 1 for '{{node Reshape}} = Reshape[T=DT_FLOAT, Tshape=DT_INT32](binary_crossentropy/weighted_loss/value, Reshape/shape)' 
    with input shapes: [], [2] and with input tensors computed as partial shapes: input[1] = [32,?].
    """

    #main() runs the other clients

    #maybe the microbatch size of 1 is okay, hyperparameter tune
    #n is the size of the dataset -> dustin


"""

Output from DPFL
DI: 1.0266653205067617
EOP: 0.025154077517853923
Avg EP diff: 0.013814562882596566
SPD: 0.029326856497608467
-> Pooled Test accuracy: 0.9606661796569824
INFO flower 2022-05-18 01:07:06,958 | server.py:209 | evaluate_round: no clients selected, cancel
INFO flower 2022-05-18 01:07:06,958 | server.py:182 | FL finished in 3543.5124693999996


Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv1d (Conv1D)             (None, 49, 16)            400

 max_pooling1d (MaxPooling1D  (None, 24, 16)           0
 )

 dropout (Dropout)           (None, 24, 16)            0

 conv1d_1 (Conv1D)           (None, 17, 128)           16512

 max_pooling1d_1 (MaxPooling  (None, 2, 128)           0
 1D)

 dropout_1 (Dropout)         (None, 2, 128)            0

 flatten (Flatten)           (None, 256)               0

 dense (Dense)               (None, 50)                12850

 dense_1 (Dense)             (None, 1)                 51

=================================================================
Total params: 29,813
Trainable params: 29,813
Non-trainable params: 0
_________________________________________________________________
None
Backend TkAgg is interactive backend. Turning interactive mode on.
"""