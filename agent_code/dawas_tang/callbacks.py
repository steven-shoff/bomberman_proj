import time
import numpy as np
from time import sleep
from agent_code.dawas_tang.nn_model import *
import json
from settings import s
from random import randint
from .rewards_table import rewards
import sys
import os
import matplotlib.pyplot as plt

class GainExperience(object):
    def __init__(self, model, memory_size, discount_rate):
        '''

        :param memory_size: amount of states to save at one time for training
        :param discount_rate: discount rate of future rewards in the Q-learning algorithm
        :param eps: Randomness threshold for choosing a new action randomely while training
        :param eps_decay: Decaying rate for eps, we don't want it to be fixed during the entire training process
        '''

        self.max_memory_size = memory_size
        self.discount_rate = discount_rate
        self.experience_buffer = list()
        self.model = model
        # self.inputs = np.zeros((self.max_memory_size, 289))
        # self.targets = np.zeros((self.max_memory_size, len(s.actions)))
        self.inputs = list()
        self.targets = list()
        self.current_state = None
        self.experiences_count = -1
        self.rounds_count = 0

    def expand_experience(self, experience, exit_game=False):
        # Recieved experience is: [action_selected, reward_earned, next_state]
        # updating the experience and add the current_state to it
        experience.insert(0,self.current_state)

        assert (not np.array_equal(experience[0], experience[1])),'old experience and new experience are exactly the same'
        # Compute the target value for this state and add it directly into the training data buffers
        self.experiences_count += 1
        self.calculate_targets(experience, exit_game=exit_game)
        if exit_game:
            self.rounds_count += 1
        # if self.rounds_count == 10:
        #     print(self.rounds_count)
        #     print(np.count_nonzero(self.targets,axis=0))
        #     print(self.experiences_count)
        #     print(self.targets[:self.experiences_count+5])

    def calculate_targets(self, experience, exit_game=False):
        current_state, action_selected, rewards_earned, next_state = experience
        target = [0]*len(s.actions)
        if not exit_game:
            max_predict = np.max(self.model.predict(next_state)[0])
            target[action_selected] = rewards_earned + self.discount_rate * max_predict
        else:
            target[action_selected] = rewards_earned
        # self.targets[self.experiences_count, action_selected] = target
        # self.inputs[self.experiences_count] = current_state
        self.targets.append(target)
        self.inputs.append(current_state)

        if len(self.inputs) > self.max_memory_size:
            del self.inputs[0]
            del self.targets[0]



    def predict_action(self,current_state):
        prediction = self.model.predict(current_state)[0]
        action_idx = np.argmax(prediction)
        return action_idx

    def get_inputs(self):
        return self.inputs

    def get_targets(self):
        return self.targets


def setup(agent):

    # load config file that contains paths to the pretrained weights
    with open('agent_code/dawas_tang/config.json') as f:
        config = json.load(f)

    # store that config in the agent to be used later on
    agent.config = config

    # load the flag of using pretrained weights
    load_pretrained = agent.config["training"]["pretrained"]

    agent.pretrained_model_path = os.path.join(agent.config['training']['models_folder'],
                                              agent.config['training']['model_name'])

    # if in training mode, we have two options: train from scratch or load weights
    if agent.config['workflow']['train']:
        if load_pretrained:
            # read the path of the model to be loaded

            try:
                agent.model = read_model(agent.pretrained_model_path)
            # read_model method raises exceptions to be caught here
            except FileNotFoundError:
                agent.logger.info("No model is specified to load")
                sys.exit(-1)
            except Exception:
                agent.logger.info("An error occured in loading the model")
                sys.exit(-1)
        else:
            # build model to be trained from scratch if no pre-trained weights specified
            agent.model = build_model()
    else:
        try:

            agent.model = read_model(agent.pretrained_model_path)
        except FileNotFoundError:
            agent.logger.info(f"Model file is not found, file name: {agent.pretrained_model_path}")
            sys.exit(-1)
        except Exception:
            agent.logger.info("An error occured in loading the model")
            sys.exit(-1)

    experience = GainExperience(model=agent.model,memory_size=s.max_steps,discount_rate=0.99)
    agent.experience = experience


def act(agent):

    # agent.logger.info('Pick action according to pressed key')
    # agent.next_action = agent.game_state['user_input']

    try:
        state = agent.game_state

        current_state = formulate_state(state)
        agent.logger.info(f'current state from act: {current_state}')

        if agent.config['workflow']['train']:
            agent.experience.current_state = current_state
            if state['step'] == 1:
                agent.eps = agent.config["playing"]["eps"]
            rnd = randint(1, 100)
            ths = int(agent.eps * 100)
            if rnd < ths:
                agent.logger.info('Selecting action at Random for exploring...')
                agent.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB','WAIT'])
            else:
                prediction = agent.model.predict(current_state)[0]
                action_idx = np.argmax(prediction)
                agent.next_action = s.actions[action_idx]
        else:
            prediction = agent.model.predict(current_state)[0]
            action_idx = np.argmax(prediction)
            print(action_idx)
            agent.next_action = s.actions[action_idx]

    except Exception as e:
        print(f'Error occured with message: {str(e)}')


def reward_update(agent):

    send_to_experience(agent)



def end_of_episode(agent):

    # Get the last state of the game after the last step
    # add the last experience to the buffer in GainExperience object
    send_to_experience(agent, exit_game=True)

    train(agent)

    agent.eps *= agent.config["playing"]["eps_discount"]


def send_to_experience(agent, exit_game=False):

    last_action = s.actions.index(agent.next_action)

    # Get the new game state, after executing last_action
    state = agent.game_state

    # formulate state in the required shape
    new_state = formulate_state(state)

    # get events happened in the last step
    events = agent.events

    # formulate reward
    reward = compute_reward(events)

    # create one experience and save it into GainExperience object
    new_experience = [last_action, reward, new_state]

    agent.experience.expand_experience(new_experience, exit_game=exit_game)



def formulate_state(state):
    # Extracting info about the game
    arena = state['arena'].copy()
    self_xy = state['self'][0:2]
    others = [(x,y) for (x,y,n,b,_) in state['others']]
    bombs = [(x,y) for (x,y,t) in state['bombs']]
    bombs_times = [t for (x,y,t) in state['bombs']]
    explosions = state['explosions']
    coins = [coin for coin in state['coins']]
    # Enriching the arena with info about own locations, bombs, explosions, opponent positions etc.
    # indicating the location of the player himself
    arena[self_xy] = 3

    # indicating the location of the remaining opponents
    for opponent in others:
        arena[opponent] = 2

    for coin in coins:
        arena[coin] = 4

    for bomb_idx, bomb in enumerate(bombs):
        arena[bomb] = (bombs_times[bomb_idx] + 1) * 10

    explosions_locations = np.nonzero(explosions)
    arena[explosions_locations] = explosions[explosions_locations] * 100

    return arena.T.flatten().reshape((-1,289))


def compute_reward(events):
    total_reward = 0

    for event in events:
        total_reward += list(rewards.values())[event]

    return total_reward


def train(agent):
    inputs = np.array(agent.experience.get_inputs()).reshape((-1,289))
    targets = np.array(agent.experience.get_targets())
    print(f'Start training after round number: {agent.experience.rounds_count}')
    start = time.time()
    agent.training_history = agent.model.fit(x=inputs,y=targets,epochs=30)
    end = time.time()
    print(f'Finish training after round number: {agent.experience.rounds_count}, time lapsed: {end-start}')
    is_save = True if agent.experience.rounds_count % 1000 == 0 else False
    if is_save:
        saved_model_path = os.path.join(agent.config['training']['models_folder'],
                 agent.config['training']['save_model'])
        agent.model.save(saved_model_path)