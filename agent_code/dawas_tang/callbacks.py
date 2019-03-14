import time
import numpy as np
from agent_code.dawas_tang.nn_model import *
import json
from settings import s
from random import randint
import sys
import os
from keras.callbacks import ModelCheckpoint,TensorBoard
from . import reward_fun

class GainExperience(object):
    def __init__(self, train_model, target_model, memory_size, discount_rate):
        '''

        :param memory_size: amount of states to save at one time for training
        :param discount_rate: discount rate of future rewards in the Q-learning algorithm
        :param eps: Randomness threshold for choosing a new action randomely while training
        :param eps_decay: Decaying rate for eps, we don't want it to be fixed during the entire training process
        '''

        self.max_memory_size = memory_size
        self.discount_rate = discount_rate
        self.train_model = train_model
        self.target_model = target_model
        self.inputs = list()
        self.targets = list()
        self.current_state = None
        self.experiences_count = None
        self.experiences = list()
        self.rounds_count = 0
        self.ckpt = ModelCheckpoint('agent_code/dawas_tang/models/ckpt/dawas_tang_model_{epoch:02d}-{val_loss:.2f}.h5',
                                     save_best_only=True, period=1000)
        # self.tb = TensorBoard(log_dir='agent_code/dawas_tang/tensorboard_logs/dawas_tang',update_freq=10000)

        with open('agent_code/dawas_tang/config.json') as f:
            config = json.load(f)

        self.config = config
        self.reset_interval = config["training"]["reset_interval"]

    def expand_experience(self, experience):
        # Recieved experience is: [action_selected, reward_earned, next_state]
        # updating the experience and add the current_state to it
        experience.insert(0,self.current_state)

        self.experiences.append(experience)

        # Compute the target value for this state and add it directly into the training data buffers
        self.experiences_count = len(self.experiences)

        if self.experiences_count > 1000:
            del self.experiences[0]

        if self.experiences_count == 1: return

        idx = list(range(self.experiences_count))
        np.random.shuffle(idx)
        limit = np.min([self.experiences_count,300])
        idx = idx[:limit]
        input_batch = list()
        target_batch = list()
        for i in idx:
            exp = self.experiences[i]
            input_batch.append(exp[0])
            action = exp[1]
            reward = exp[2]
            next_state = exp[3]
            is_terminal = exp[4]
            target_batch.append(self.calculate_targets(action,reward,next_state,is_terminal))

        input_batch = np.array(input_batch)
        target_batch = np.array(target_batch)
        start = time.time()
        self.train_model.fit(x=input_batch, y=target_batch, validation_split=0.1, epochs=10,
                                                     verbose=1, callbacks=[self.ckpt])
        if self.experiences_count % self.reset_interval == 0:
            self.target_model.set_weights(self.train_model.get_weights())
        end = time.time()
        if self.rounds_count % 100 == 0:
            print(f'Finish training after round number: {self.rounds_count}, time elapsed: {end-start}'
                  f'\nSaving model')
            saved_model_path = os.path.join(self.config['training']['models_folder'],
                                            self.config['training']['save_model'])
            self.target_model.save(saved_model_path)
        if experience[-1]:
            print(f'Round # {self.rounds_count} finished\n')


    def calculate_targets(self, action, reward, next_state, is_terminal=False):

        target = [0]*len(s.actions)
        if not is_terminal:
            next_state = np.expand_dims(next_state,axis=0)
            max_predict = np.max(self.target_model.predict(next_state)[0])
            target[action] = reward + self.discount_rate * max_predict
        else:
            target[action] = reward
        return target



    def predict(self,current_state):
        prediction = self.train_model.predict(current_state)[0]
        return prediction

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
    agent.is_conv = agent.config["training"]["conv"]

    # if in training mode, we have two options: train from scratch or load weights
    if agent.config['workflow']['train']:
        if load_pretrained:
            # read the path of the model to be loaded

            try:
                pretrained_model_path = os.path.join(agent.config['training']['models_folder'],
                                                     agent.config['training']['model_name'])
                train_model = read_model(pretrained_model_path)

                if agent.is_conv:
                    target_model = build_conv()
                else:
                    target_model = build_model()

                target_model.set_weights(train_model.get_weights())
            # read_model method raises exceptions to be caught here
            except FileNotFoundError:
                agent.logger.info("No model is specified to load")
                sys.exit(-1)
            except Exception:
                agent.logger.info("An error occured in loading the model")
                sys.exit(-1)
        else:
            # build model to be trained from scratch if no pre-trained weights specified
            if not agent.is_conv:
                train_model = build_model()
                target_model = build_model()
                target_model.set_weights(train_model.get_weights())
            else:
                train_model = build_conv()
                target_model = build_conv()
                target_model.set_weights(train_model.get_weights())

        agent.eps = agent.config["playing"]["eps"]

        max_memory_size = agent.config['training']['max_memory_size']
        experience = GainExperience(train_model=train_model, target_model = target_model, memory_size=max_memory_size, discount_rate=0.9)
        agent.experience = experience
    else:
        try:
            pretrained_model_path = os.path.join(agent.config['training']['models_folder'],
                                                 agent.config['training']['model_name'])
            agent.model = read_model(pretrained_model_path)
        except FileNotFoundError:
            agent.logger.info(f"Model file is not found, file name: {pretrained_model_path}")
            sys.exit(-1)
        except Exception:
            agent.logger.info("An error occured in loading the model")
            sys.exit(-1)
    agent.mybomb = None
    # For reward computation
    agent.last_moves = []



def act(agent):

    # agent.logger.info('Pick action according to pressed key')
    # agent.next_action = agent.game_state['user_input']

    try:
        state = agent.game_state
        current_state = formulate_state(state, agent.is_conv)
        current_pos = self_xy = state['self'][0:2]
        if agent.config['workflow']['train']:
            agent.experience.current_state = current_state
            if state['step'] == 1:
                agent.experience.rounds_count += 1
                agent.last_moves = [current_pos]
            elif len(agent.last_moves) >= 10:
                del agent.last_moves[0]
                agent.last_moves.append(current_pos)
            else:
                agent.last_moves.append(current_pos)

            rnd = randint(1, 100)
            ths = int(agent.eps * 100)
            if rnd < ths:
                agent.logger.info('Selecting action at Random for exploring...')
                choice = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB','WAIT'])
                print(f'Random choice: {choice}')
                agent.next_action = choice
            else:
                current_state = np.expand_dims(current_state,axis=0)
                prediction = agent.experience.predict(current_state)
                action_idx = np.argmax(prediction)
                print(f'prediction: {prediction}, {s.actions[action_idx]}')
                agent.next_action = s.actions[action_idx]
        else:
            current_state = np.expand_dims(current_state, axis=0)
            prediction = agent.model.predict(current_state)[0]
            action_idx = np.argmax(prediction)
            print(f'{prediction}, {s.actions[action_idx]}')
            agent.next_action = s.actions[action_idx]

        if agent.next_action == 'BOMB':
            agent.mybomb = current_pos

    except Exception as e:
        print(f'Error occured with message: {str(e)}')


def reward_update(agent):

    send_to_experience(agent)
    if agent.experience.experiences_count % 100 == 0 and agent.eps > 0.1:
        agent.eps *= agent.config["playing"]["eps_discount"]


def end_of_episode(agent):

    # Get the last state of the game after the last step
    # add the last experience to the buffer in GainExperience object
    send_to_experience(agent, exit_game=True)


def send_to_experience(agent, exit_game=False):

    last_action = s.actions.index(agent.next_action)

    # Get the new game state, after executing last_action
    state = agent.game_state

    # formulate state in the required shape
    new_state = formulate_state(state, agent.is_conv)

    # get events happened in the last step
    events = agent.events

    # formulate reward
    reward = reward_fun.compute_reward(agent)

    # create one experience and save it into GainExperience object
    new_experience = [last_action, reward, new_state, exit_game]
    agent.experience.expand_experience(new_experience)



def formulate_state(state, is_conv):
    # Extracting info about the game
    arena = state['arena'].copy()
    self_xy = state['self'][0:2]
    others = [(x,y) for (x,y,n,b,_) in state['others']]
    bombs = [(x,y) for (x,y,t) in state['bombs']]
    bombs_times = [t for (x,y,t) in state['bombs']]
    explosions = state['explosions']
    coins = state['coins']
    # Enriching the arena with info about own locations, bombs, explosions, opponent positions etc.
    # indicating the location of the player himself

    if not is_conv:
        if self_xy in bombs:
            arena[self_xy] = 5
        else:
            arena[self_xy] = 3

        # indicating the location of the remaining opponents
        for opponent in others:
            arena[opponent] = 2

        for coin in coins:
            arena[coin] = 4

        for bomb_idx, bomb in enumerate(bombs):
            if bomb != self_xy:
                arena[bomb] = (bombs_times[bomb_idx] + 1) * 10

        explosions_locations = np.nonzero(explosions)
        arena[explosions_locations] = explosions[explosions_locations] * 100

        return arena.T.flatten().reshape((-1,289))
    else:
        #layer_1 is the arena
        agents_layer = np.zeros((17,17))
        bombs_layer = np.zeros((17, 17))
        explosions_layer = np.zeros((17, 17))

        for coin in coins:
            arena[coin] = 5

        for other in others:
            agents_layer[other] = 2
        agents_layer[self_xy] = 3

        for bomb_idx, bomb in enumerate(bombs):
            bombs_layer[bomb] = 4

        explosions_locations = np.nonzero(explosions)
        explosions_layer[explosions_locations] = explosions[explosions_locations] * 100

        arena = arena.transpose().copy()
        agents_layer = agents_layer.transpose().copy()
        bombs_layer = bombs_layer.transpose().copy()
        explosions_layer = explosions_layer.transpose().copy()
        new_state = np.stack((arena,agents_layer,bombs_layer,explosions_layer),axis=2)
        return new_state

# def compute_reward(agent):
#     total_reward = 0
#     events = agent.events
#     for event in events:
#         total_reward += list(rewards.values())[event]
#
#     return total_reward
#
#
# def train(agent):
#     inputs = np.array(agent.experience.get_inputs()).reshape((-1,289))
#     targets = np.array(agent.experience.get_targets())
#     print(f'Start training after round number: {agent.experience.rounds_count}')
#     start = time.time()
#     agent.training_history = agent.experience.model.fit(x=inputs,y=targets,validation_split=0.1,batch_size=16,epochs=10,verbose=1,callbacks=[agent.ckpt,agent.tb])
#     end = time.time()
#     print(f'Finish training after round number: {agent.experience.rounds_count}, time elapsed: {end-start}')
#     is_save = True if agent.experience.rounds_count % 1000 == 0 else False
#     if is_save:
#         saved_model_path = os.path.join(agent.config['training']['models_folder'],
#                  agent.config['training']['save_model'])
#         agent.experience.model.save(saved_model_path)