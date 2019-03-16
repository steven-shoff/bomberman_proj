from keras.models import Sequential, load_model
from keras.layers import Dense
import keras.backend as K
from settings import s

import os

num_actions = len(s.actions)


def build_model():
    model = Sequential()
    model.add(Dense(44*4, activation='relu',input_shape=(44,), name='input'))
    model.add(Dense(44*2, activation='relu',name='hidden_1'))
    model.add(Dense(44, activation='relu',name='hidden_2'))
    model.add(Dense(num_actions,name='output'))
    model.compile(optimizer='adam',loss=pure_se)
    return model


def read_model(model_name):
    if not os.path.isfile(model_name):
        raise FileNotFoundError(f'No Model file is found with name: {model_name}')
    model = load_model(model_name)
    return model


def pure_se(y_true, y_pred):
    return K.square(y_pred - y_true)
