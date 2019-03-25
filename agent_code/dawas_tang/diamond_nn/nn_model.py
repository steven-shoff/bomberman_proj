import os
from settings import s
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

num_actions = len(s.actions)


def build_model():
    """
    This function builds Multiple Layer Perceptron (MLP) with 3 hidden layers
    """
    model = Sequential()
    model.add(Dense(32, activation='relu',input_shape=(45,), name='input'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu',name='hidden_1'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu',name='hidden_2'))
    model.add(Dropout(0.2))
    model.add(Dense(num_actions,name='output'))
    optimizer = RMSprop()
    model.compile(optimizer=optimizer,loss='mse')
    return model


def read_model(model_name):
    """
    This function read pretrained model from path 'model_name'
    :param model_name   : Path to load pretrained model
    :return             : Pretrained model
    """
    if not os.path.isfile(model_name):
        raise FileNotFoundError(f'No Model file is found with name: {model_name}')
    model = load_model(model_name)
    return model
