from keras.models import Sequential, load_model
from keras.layers import Dense,Conv2D,Flatten
from keras.optimizers import RMSprop
import keras.backend as K
from settings import s

import os

num_actions = len(s.actions)


def build_model():
    model = Sequential()
    model.add(Conv2D(16, 8, 4, activation='relu', input_shape=(17, 17, 4), name='conv1'))
    model.add(Conv2D(32, 4, 2, activation='relu', name='conv2'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', name='dense1'))
    model.add(Dense(num_actions, name='output'))
    optimizer = RMSprop(decay=0.01)
    model.compile(optimizer=optimizer, loss=pure_se)
    return model


def read_model(model_name):
    if not os.path.isfile(model_name):
        raise FileNotFoundError(f'No Model file is found with name: {model_name}')
    model = load_model(model_name)
    return model


def pure_se(y_true, y_pred):
    return K.square(y_pred - y_true)
