
import numpy as np
from time import sleep
from agent_code.dawas_tang.build_model import *


class GainExperience(object):
    def __init__(self, model, memory_size, discount_rate):
        '''

        :param memory_size: amount of states to save at one time for training
        :param discount_rate: discount rate of future rewards in the Q-learning algorithm
        :param eps: Randomness threshold for choosing a new action randomely while training
        :param eps_decay: Decaying rate for eps, we don't want it to be fixed during the entire training process
        '''

        self.memory_size = memory_size
        self.discount_rate = discount_rate
        self.memory = list()
        self.model = model
        self.inputs = list()
        self.targets = list()

    def expand_experience(self):
        pass

    def calculate_predict_target(self):
        predict = self.model.predict(s)
        target = r + self.discount_rate * predict
        self.targets.append(target)
        self.inputs.append(s)
        pass



def setup(agent):
    agent.model = build_model()
    pass

def act(agent):
    agent.logger.info('Pick action according to pressed key')
    agent.next_action = agent.game_state['user_input']

def reward_update(agent):
    pass

def learn(agent):
    pass
