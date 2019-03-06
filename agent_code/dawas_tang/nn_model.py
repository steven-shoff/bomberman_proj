from keras.models import Sequential,load_model
from keras.layers import Dense

from settings import s

import os

num_actions = len(s.actions)


def build_model():
    model = Sequential()
    model.add(Dense(150, activation='relu',input_shape=(289,),name='input'))
    model.add(Dense(150, activation='relu',name='hidden_1'))
    model.add(Dense(150, activation='relu',name='hidden_2'))
    model.add(Dense(150, activation='relu',name='hidden_3'))
    model.add(Dense(num_actions,name='output'))
    model.compile(optimizer='adam',loss='mse')
    return model


def read_model(model_name):
    if not os.path.isfile(model_name):
        raise FileNotFoundError(f'No Model file is found with name: {model_name}')
    model = load_model(model_name)
    return model
