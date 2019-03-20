import time
import numpy as np
from agent_code.dawas_tang.nn_model import *
import json
from settings import s
from random import randint
import sys
import os
from . import reward_fun
from .rewards_table import rewards

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
        self.num_steps = 0
        self.experiences = list()
        self.rounds_count = 1
        self.eps = None
        self.log_file = 'agent_code/dawas_tang/training_log.txt'

        with open('agent_code/dawas_tang/config.json') as f:
            config = json.load(f)

        self.config = config
        self.reset_interval = config["training"]["reset_interval"]

    def expand_experience(self, experience):
        # Recieved experience is: [action_selected, reward_earned, next_state, invalid_action, exit_game]
        # updating the experience and add the current_state to it

        if len(self.experiences) == self.max_memory_size:
            del self.experiences[0]

        experience.insert(0,self.current_state)

        self.experiences.append(experience)

        self.experiences_count = len(self.experiences)
        if self.experiences_count < 2000:
            return

        self.num_steps += 1

        idx = list(range(self.experiences_count))
        np.random.shuffle(idx)
        limit = np.min([len(idx),400]) #Consider skip training until 2000 points are there
        idx = idx[:limit]
        input_batch = list()
        target_batch = list()

        try:
            for i in idx:
                exp = self.experiences[i]
                input_batch.append(exp[0])
                action = exp[1]
                reward = exp[2]
                next_state = exp[3]
                invalid_action = exp[4]
                is_terminal = exp[5]
                target_batch.append(self.calculate_targets(exp[0], action, reward, next_state, invalid_action, is_terminal))
        except:
            print()

        input_batch = np.array(input_batch)
        target_batch = np.array(target_batch)
        start = time.time()
        self.train_model.fit(x=input_batch, y=target_batch, validation_split=0.0, epochs=10, verbose=1)
        if self.num_steps % self.reset_interval == 0:
            self.target_model.set_weights(self.train_model.get_weights())
            with open(self.log_file,'a+') as f:
                f.write(f'Number of steps: {self.num_steps}, model is reset...\n')
        end = time.time()
        if self.rounds_count % 100 == 0 and experience[-1]:
            with open('eps.txt','w') as f:
                f.write(str(self.eps))
            print(f'Finish training after round number: {self.rounds_count}, time elapsed: {end-start}'
                  f'\nSaving model')
            saved_model_path = os.path.join(self.config['training']['models_folder'],
                                            self.config['training']['save_model'])
            self.train_model.save(saved_model_path)

        if experience[-1]:
            print(f'Round # {self.rounds_count} finished\n')
            self.rounds_count += 1


    def calculate_targets(self, current_state, action, reward, next_state, invalid_action, is_terminal=False):

        current_state = np.expand_dims(current_state,axis=0)
        target = self.predict(current_state)
        if is_terminal or invalid_action:
            target[action] = reward
        else:
            next_state = np.expand_dims(next_state, axis=0)
            max_predict_idx = np.argmax(self.predict(next_state))
            max_predict = self.target_model.predict(next_state)[0][max_predict_idx]
            target[action] = reward + self.discount_rate * max_predict
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

        max_memory_size = agent.config['training']['max_memory_size']
        experience = GainExperience(train_model=train_model, target_model = target_model, memory_size=max_memory_size, discount_rate=0.99)
        agent.experience = experience
        agent.experience.eps = agent.config["playing"]["eps"]
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
        current_pos = state['self'][0:2]
        force_to_move = False
        if agent.config['workflow']['train']:
            agent.experience.current_state = current_state
            if state['step'] == 1:
                # agent.experience.rounds_count += 1
                agent.last_moves = [current_pos]
            elif len(agent.last_moves) >= 10:
                del agent.last_moves[0]
                agent.last_moves.append(current_pos)
            else:
                agent.last_moves.append(current_pos)

            rnd = randint(1, 100)
            ths = int(agent.experience.eps * 100)
            if rnd <= ths:
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
            # if len(agent.last_moves) >= 10:
            #     del agent.last_moves[0]
            #     agent.last_moves.append(current_pos)
            #     last_four = agent.last_moves[-4:]
            #     if (last_four[0] == last_four[2] and last_four[1] == last_four[3]) \
            #             or len(np.unique(agent.last_moves)) < 7:
            #         force_to_move = True
            #     else:
            #         force_to_move = False
            # else:
            #     agent.last_moves.append(current_pos)

            current_state = np.expand_dims(current_state, axis=0)
            prediction = agent.model.predict(current_state)[0]
            # valid_actions = reward_fun.det_valid_action(agent.game_state)
            action_idx = np.argmax(prediction)
            # sorted_pred = prediction.argsort()[::-1]
            # for pred in sorted_pred:
            #     if s.actions[pred] not in valid_actions or force_to_move:
            #         force_to_move = False
            #         if randint(1, 100) <= 10:
            #             action_name = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'])
            #         else:
            #             continue
            #     else:
            #         action_name = s.actions[pred]
            #         break
            # print(f'{prediction}, {action_name}')
            action_name = s.actions[action_idx]
            print(f'{prediction}, {action_name}')
            agent.next_action = action_name

        if agent.next_action == 'BOMB' and state['self'][-1] == 1:
            agent.mybomb = current_pos

    except Exception as e:
        print(f'Error occured with message: {str(e)}')


def reward_update(agent):

    send_to_experience(agent)
    if agent.experience.num_steps > 0 and agent.experience.num_steps % 150 == 0 and agent.experience.eps > 0.1:
        agent.experience.eps *= agent.config["playing"]["eps_discount"]
        if agent.experience.eps < 0.1:
            agent.experience.eps = 0.1
            with open(agent.experience.log_file,'a+') as f:
                f.write(f'Number of steps: {agent.experience.num_steps}, eps is annealed...\n')
        else:
            with open(agent.experience.log_file,'a+') as f:
                f.write(f'Number of steps: {agent.experience.num_steps}, eps is decayed to {agent.experience.eps}\n')


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

    invalid_action = True if 6 in events else False

    # formulate reward
    reward = compute_reward(agent)

    # create one experience and save it into GainExperience object
    new_experience = [last_action, reward, new_state, invalid_action, exit_game]
    agent.experience.expand_experience(new_experience)



def formulate_state(state, is_conv):
    # Extracting info about the game
    arena = state['arena'].copy()
    self_x, self_y,_,self_bomb,_ = state['self']
    others = [(x,y) for (x,y,n,b,_) in state['others']]
    bombs = [(x,y) for (x,y,t) in state['bombs']]
    bombs_times = [t for (x,y,t) in state['bombs']]
    explosions = state['explosions']
    coins = state['coins']
    # Enriching the arena with info about own locations, bombs, explosions, opponent positions etc.
    # indicating the location of the player himself

    if not is_conv:
        if (self_x,self_y) in bombs:
            arena[(self_x,self_y)] = 5
        else:
            arena[(self_x,self_y)] = 3

        # indicating the location of the remaining opponents
        for opponent in others:
            arena[opponent] = 2

        for coin in coins:
            arena[coin] = 4

        for bomb_idx, bomb in enumerate(bombs):
            if bomb != (self_x,self_y):
                arena[bomb] = (bombs_times[bomb_idx] + 1) * 10

        explosions_locations = np.nonzero(explosions)
        arena[explosions_locations] = explosions[explosions_locations] * 100

        return arena.T.flatten().reshape((-1,289))
    else:

        fill_value = 17+17
        #layer_1 is the arena
        bombs_layer = np.zeros((17, 17))
        distances_layer = np.full((17,17),fill_value=fill_value)

        for coin in coins:
            arena[coin] = 5
            distances_layer[coin] = np.abs(self_x - coin[0]) + np.abs(self_y - coin[1])

        for other in others:
            arena[other] = 2
            # distances_layer[other] = np.sqrt((self_x-other[0])**2 + (self_y-other[1])**2)
        arena[(self_x,self_y)] = 3 if self_bomb == 0 else 4
        # distances_layer[(self_x,self_y)] = 0

        for bomb_idx, bomb in enumerate(bombs):
            bombs_layer[bomb] = 6 - bombs_times[bomb_idx]
            fill_map = reward_fun.calc_explosion_effect(arena,bomb)
            bombs_layer[fill_map] = 1

        explosions_locations = np.nonzero(explosions)
        bombs_layer[explosions_locations] = 7

        # arena = arena.transpose().copy()
        # bombs_layer = bombs_layer.transpose().copy()
        # distances_layer = distances_layer.transpose().copy()
        new_state = np.stack((arena,bombs_layer,distances_layer),axis=2)
        return new_state

def compute_reward(agent):
    total_reward = 0
    events = agent.events
    for event in events:
        total_reward += list(rewards.values())[event]

    return total_reward
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