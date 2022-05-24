from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

import os
import numpy as np
import tensorflow as tf

#####################################################################
RANDOM_SEED = 47568
#seed(47568
# #tf.random.set_random_seed(seed_value))
os.environ['PYTHONHASHSEED']=str(47568)
#random.seed(47568)
tf.random.set_seed(47568)

np.random.seed(RANDOM_SEED)

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import VectorizedDPKerasSGDOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

from tensorflow.python.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
#############################################################################


run_args = {
    'batch_size':32,
    'local_epochs':10,
    'dpsgd':False,
    'microbatches':32, #what is a microbatch #only allows 1!
    'noise_multiplier':1.3, #what is a noise multiplier
    'l2_norm_clip':1.5,
    'learning_rate':.1
}

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

    if 32 % 32 != 0:
        raise ValueError("Number of microbatches should divide evenly batch_size")


    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.1) #What is the issue with microbatches?

    optimizer = DPKerasSGDOptimizer( #optimizer error
    l2_norm_clip=run_args.get("l2_norm_clip"),
    noise_multiplier=run_args.get("noise_multiplier"),
    num_microbatches=32,
    learning_rate=run_args.get("learning_rate")
    )

    #opt DPKerasSGD
    #loss = BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) #change to categorical crossentropy
    loss = BinaryCrossentropy(from_logits=False, reduction=tf.losses.Reduction.NONE, axis=-1, label_smoothing=0.0, name='binary_crossentropy')
    #loss = BinaryCrossentropy(from_logits=False, reduction='none')

    #compile model with Keras #"binary_crossentropy"
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    print(model.summary())
    #having microbatches same size as batch size
    return model

if __name__ == "__main__":
    print("model okay?")
    print(type(run_args.get("l2_norm_clip")))
    print(type(run_args.get("noise_multiplier")))
    print(type(run_args.get("microbatches")))
    print(type(run_args.get("learning_rate")))