from keras.models import Sequential
from keras.layers import Dense
from settings import s

num_actions = len(s.actions)

def build_model():
    model = Sequential()
    model.add(Dense(150,activation='relu'))
    model.add(Dense(150,activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(optimizer='adam',loss='mse')
    return model