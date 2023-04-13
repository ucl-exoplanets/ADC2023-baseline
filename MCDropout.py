
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Dense, Reshape, Input, Concatenate, BatchNormalization, Dropout, Conv1D,Flatten,MaxPooling1D
from tensorflow.keras.models import Model

def MC_trainer(wl_channels,num_targets,p,hidden_units):
    InputLayer = Input(shape=(wl_channels,))
    AddFeat = Input(shape=(1,))

    x = Concatenate(axis=-1)([InputLayer,AddFeat])
    for size in hidden_units:
#         x = BatchNormalization()(x)
        x = Dense(size, activation='relu')(x)
        x = Dropout(p)(x,training=True)
    output = Dense(num_targets, activation=None)(x)
    model = Model(inputs=[InputLayer,AddFeat], outputs=output)
    return model

def MC_Convtrainer(wl_channels,num_targets,p,filters):
    InputLayer = Input(shape=(wl_channels,))
    AddFeat = Input(shape=(1,))
    x = Reshape((-1,1))(InputLayer)
    for f in filters:
        x = Conv1D(f, 3, activation='relu')(x)
        x = Conv1D(f, 3, activation='relu')(x)
        x = MaxPooling1D()(x)
    x = Flatten()(x)
    x = Concatenate(axis=-1)([x,AddFeat])
    d1 = Dense(500, activation= 'relu')(x)
    d1 = Dropout(p)(d1,training=True)
    d2 = Dense(100, activation= 'relu')(d1)
    d2 = Dropout(p)(d2,training=True)
    output = Dense(num_targets, activation=None)(d2)
    model = Model(inputs=[InputLayer,AddFeat], outputs=output)
    return model