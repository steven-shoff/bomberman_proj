import os
import sys
import json
import numpy as np
from agent_code.blind_agent.rewards_table import rewards
from agent_code.blind_agent.qtable.qtable import QTableModel
from settings import s
import time
import math


class ExperienceCache(object):
    def __init__(self, model):
        """
        This class function collects experience from each steps and compute new Q(s, a)
        :param model        : Model used to interact with the agent
        """

        # Class model attributes
        self.model = model

        # class attributes for memory storage
        self.current_state = None

        self.experiences_count = -1     # Experience counter
        self.rounds_count = 0           # Round counter

    def learn_experience(self, experience, exit_game=False):
        """
        This class function receives experience from agent and store s, Q(s,a) into inputs and targets
        :param experience   : tuples containing <a, r, s'> where
                                a - action selected in the current state
                                r - reward of the action
                                s'- new state received
        :param exit_game    : flag indicating the end of game
        """

        # Updating the experience and add the current_state to it
        experience.insert(0, self.current_state)

        # Compute the target value for this state and add it directly into the training data buffers
        self.experiences_count += 1
        self.model.fit(experience)
        if exit_game:
            self.rounds_count += 1


def setup(agent):
    """
    This function performed the following actions in setup:
        - Initialize GainExperience and model
        - Load agent configuration file

    :param agent: Agent used to interact with the Environment
    """

    # Load config file containing game model selection and configuration
    with open('agent_code/tang_agent/config.json') as f:
        config = json.load(f)

    # store that config in the agent to be used later on
    agent.config = config

    # load the flag of using pretrained weights
    load_pretrained = agent.config["workflow"]["pretrained"]

    # load empty model
    agent.model = QTableModel(agent.config['model_config'])

    # load score file (only used in playing)
    agent.score_file = None

    # In training mode, we have two options: train from scratch or load weights
    if agent.config['workflow']['train']:  # Flag to identify training or playing mode
        if load_pretrained:
            # read the path of the model to be loaded
            try:
                all_name = [fname for fname in os.listdir(agent.config['training']['models_folder']) if
                            fname.startswith(agent.config['training']['model_name']) and fname.endswith("_model.csv")]
                if len(all_name) != 0:
                    all_name = sorted(all_name, key=lambda y: int(y[len(agent.config['training']['model_name']):-len("_model.csv")]))
                else:
                    all_name = ['0']
                pretrained_model_path = os.path.join(agent.config['training']['models_folder'], all_name[-1])
                agent.olddig = all_name[-1][len(agent.config['training']['model_name']):-len("_model.csv")]
                agent.model.load_table(pretrained_model_path)
                agent.logger.info('Model in {} successfully loaded'.format(pretrained_model_path))

            # read_model method raises exceptions to be caught here
            except FileNotFoundError:
                agent.logger.info("No model is specified to load")
                sys.exit(-1)
            except Exception:
                agent.logger.info("An error occured in loading the model")
                sys.exit(-1)
        else:
            # build model to be trained from scratch if no pre-trained weights specified
            agent.model.init_table()
            agent.olddig = 0
    else:
        try:
            pretrained_model_path = os.path.join(agent.config['playing']['models_folder'], agent.config['playing']['model_name'])
            agent.model.load_table(pretrained_model_path)

            score_file_path = os.path.join(agent.config['playing']['scores_folder'], agent.config['playing']['scores_name'])
            agent.score_file = open(score_file_path, "a+")

            agent.olddig = agent.config['playing']['model_name'][len(agent.config['training']['model_name']):-len("_model.csv")]
        except FileNotFoundError:
            agent.logger.info("Model file is not found, file name: {}".format(os.path.join(agent.config['playing']['models_folder'], agent.config['playing']['model_name'])))
            sys.exit(-1)
        except Exception as e:
            agent.logger.info("An error occured in loading the model: {}".format(e))
            sys.exit(-1)

    agent.experience = ExperienceCache(model=agent.model)
    agent.mybomb = None
    eps = agent.config["playing"]["eps"]
    agent.eps = eps
    agent.total_reward = 0
    agent.timetick = None

    # For reward computation
    agent.last_moves = []

def act(agent):
    """
    This function defines the action in current step
    :param agent: Agent used to interact with the Environment
    :return:
    """
    try:
        state = agent.game_state.copy()
        (x, y, _, _, score) = state['self']
        if state['step'] == 1:
            agent.total_reward = 0
            agent.timetick = time.strftime('%Y%m%d%H%M%S')
            agent.last_moves = [(x, y)]

        elif len(agent.last_moves) >= 10:
            del agent.last_moves[0]
            agent.last_moves.append((x, y))
        else:
            agent.last_moves.append((x, y))

        if state['step'] % 100 == 0:
            agent.model.eps += (1-agent.model.eps)*agent.config['model_config']['eps_discount']

        # Step 1: Model state creation

        current_state = formulate_state(state)
        agent.logger.info(f'current state from act: {current_state}')

        # Step 2: Model prediction
        if agent.config['workflow']['train']:
            agent.experience.current_state = current_state

            rnd = np.random.rand()
            if rnd < agent.model.eps:
                prediction = agent.model.predict(current_state)[0]
                if len(set(prediction)) == 1:
                    action_idx = np.random.choice(range(6))
                else:
                    action_idx = np.argmax(prediction)
                agent.next_action = s.actions[action_idx]
                #print(prediction)
            else:
                valid_actions = det_valid_action(state)
                agent.next_action = np.random.choice(valid_actions)
                agent.logger.info('Selecting action at Random for exploring...')
        else:
            prediction = agent.model.predict(current_state)[0]
            action_idx = np.argmax(prediction)
            agent.next_action = s.actions[action_idx]
            print(prediction)
            print(agent.next_action)

        # Variable storage for reward calculation
        (x, y, _, _, score) = state['self']
        if agent.next_action == 'BOMB':
            agent.mybomb = (x, y)

        # print(state['self'])
        # print([agent.experience.rounds_count, state['step']])
        if agent.score_file is not None:
            agent.score_file.write('{} {} {} {}\n'.format(agent.timetick, agent.olddig, state['step'], score))

    except Exception as e:
        print(f'Error occured with message: {str(e)}')


def det_valid_action(state, other_loc=None):

    valid_actions = s.actions.copy()
    arena = np.absolute(state['arena'].copy())
    bombs = state['bombs']
    x, y, _, bombs_left, _ = state['self']
    if other_loc is not None:
        x, y = other_loc

    others = [(xo, yo) for (xo, yo, _, _, _) in state['others']]
    bombs_pos = [(xb, yb) for (xb, yb, tb) in bombs]

    if arena[x, y-1] == 1 or (x, y-1) in bombs_pos or (x, y-1) in others:
        valid_actions.remove('UP')
    if arena[x, y+1] == 1 or (x, y+1) in bombs_pos or (x, y+1) in others:
        valid_actions.remove('DOWN')
    if arena[x-1, y] == 1 or (x-1, y) in bombs_pos or (x-1, y) in others:
        valid_actions.remove('LEFT')
    if arena[x+1, y] == 1 or (x+1, y) in bombs_pos or (x+1, y) in others:
        valid_actions.remove('RIGHT')
    if bombs_left == 0:
        valid_actions.remove('BOMB')
    if (x, y) in bombs_pos:
        valid_actions.remove('WAIT')
    return valid_actions


def reward_update(agent):
    send_to_experience(agent)


def end_of_episode(agent):

    # Get the last state of the game after the last step
    # add the last experience to the buffer in GainExperience object
    send_to_experience(agent, exit_game=True)

    if agent.experience.rounds_count == s.n_rounds or agent.experience.rounds_count % 1000 == 0:
        save_model(agent)


def send_to_experience(agent, exit_game=False):

    last_action = s.actions.index(agent.next_action)

    # Get the new game state, after executing last_action
    state = agent.game_state

    # formulate state in the required shape
    new_state = formulate_state(state)

    # get events happened in the last step
    events = agent.events

    # formulate reward
    agent.total_reward = compute_reward(agent)

    new_experience = [last_action, agent.total_reward, new_state]

    agent.experience.learn_experience(new_experience, exit_game)


def formulate_state(state):
    """
    This class function determines states by digesting information
    :param state             : Dictionary contains raw game information
    :return determined_state: determined state
    """
    arena = state['arena'].copy()
    crates_arena = np.maximum(arena, 0)
    for (cx, cy) in state['coins']:
        crates_arena[cx, cy] = 2

    x, y, _, bombs_left, _ = state['self']
    bombs = state['bombs']
    others = [(xo, yo) for (xo, yo, _, _, _) in state['others']]

    diglist = list()
    if len(state['coins']) == 0:
        diglist.append('5')
    else:
        closest_coin = sorted(state['coins'], key=lambda k: abs(k[0] - x) + abs(k[1] - y))[0]
        best_orientation = np.argmin([abs(closest_coin[0] - mx) + abs(closest_coin[1] - my) for (mx, my) in
                                      [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]]) + 1
        diglist.append(str(best_orientation))

    if len(state['bombs']) == 0:
        diglist.append('9')
    else:
        bombs_nearby = [(xb, yb, tb) for (xb, yb, tb) in state['bombs'] if abs(xb -x) + abs(yb-y) <= tb + 4]
        if len(state[bombs_nearby]) == 0:
            diglist.append('9')
        else:
            nearest_bomb = sorted(bombs_nearby, key=lambda k: abs(k[0] - x) + abs(k[1] - y))[0]
            bomb_orientation = np.argmin([abs(nearest_bomb[0] - mx) + abs(nearest_bomb[1] - my) for (mx, my) in
                                          [(x, y-1), (x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1), (x+1, y+1), (x+1, y), (x+1, y+1)]]) + 1
            diglist.append(str(bomb_orientation))

    crates_arena = np.maximum(arena, 0)
    crates_arena = crates_arena.T
    if np.sum(crates_arena) == 0:
        diglist.append('5')
    else:
        q1map = np.sum(crates_arena[:y, :])
        q2map = np.sum(crates_arena[y + 1:, :])
        q3map = np.sum(crates_arena[:, :x])
        q4map = np.sum(crates_arena[:, x + 1:])

        diglist.append(str(np.argmax([q1map, q2map, q3map, q4map]) + 1))

    for (i, j) in [( 0, -1), (-1,  0), (0 ,  0), ( 1,  0), ( 0,  1)]:

        if (x + i) < 0 or (x + i) > 16 or (y + j) < 0 or (y + j) > 16:
            diglist.append('0')
        elif (x + i, y + j) in state['coins']:
            diglist.append('3')
        elif (x + i, y + j, 3) in bombs:
            diglist.append('4')
        elif (x + i, y + j, 2) in bombs:
            diglist.append('5')
        elif (x + i, y + j, 1) in bombs:
            diglist.append('6')
        elif (x + i, y + j, 0) in bombs:
            diglist.append('7')
        elif (x + i, y + j) in others:
            diglist.append('8')
        elif state['explosions'][x + i, y + j] == 2 or state['explosions'][x + i, y + j] == 1:
            diglist.append('9')
        else:
            diglist.append(str(arena[x + i, y + j] + 1))  # 0, 1, 2

    state = str.join('', diglist)
    return state


def compute_reward(agent):

    events = agent.events
    state = agent.game_state.copy()
    mybomb = agent.mybomb
    last_action = agent.next_action

    total_reward = 0
    state = state.copy()
    arena = state['arena'].copy()
    all_bombs = state['bombs']

    for event in events:
        total_reward += list(rewards.values())[event]
    (x, y, _, nb, _) = state['self']

    coord_dict = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0), 'BOMB': (0, 0), 'WAIT': (0, 0)}
    last_x, last_y = np.subtract((x, y), coord_dict[last_action])
    old_coord = None
    all_coord = []
    # Predict future events:
    # t + 1, t + 2, t + 3, t + 4
    for t0 in range(0, 5):
        if t0 == 0:
            next_coord = [(x, y)]
        # Find all possible coordinates in future steps
        else:
            next_coord = []
            for coord in old_coord:
                next_coord += [np.add(coord, coord_dict[i]) for i in
                               det_valid_action(state, other_loc=coord)]  # Possible coordinates in 2nd Step
            if len(next_coord) != 0:
                next_coord = [tuple(i) for i in np.unique(next_coord, axis=0)]
        all_coord.append(next_coord)
        old_coord = next_coord

        # find bombs in the future time step
        bombs_t = [(xb, yb, tb) for (xb, yb, tb) in state['bombs'] if tb == t0 and abs(xb -x) + abs(yb-y) <= 4]
        deatharea = np.minimum(state['explosions'].copy(), 1) * 3 - 2  # -2 or 1
        bombarea = np.zeros(arena.shape)
        for xb, yb, tb in bombs_t:

            bfactor = 2

            if (xb, yb) == mybomb:
                bfactor = 5

            deatharea[xb, yb] = 1
            for c_down in range(1, 4):
                if yb + c_down > 16 or state['arena'][xb, yb + c_down] == -1:
                    break
                deatharea[xb, yb + c_down] = 1
                bombarea[xb, yb + c_down] = bfactor

            for c_up in range(1, 4):
                if yb - c_up < 0 or state['arena'][xb, yb - c_up] == -1:
                    break
                deatharea[xb, yb - c_up] = 1
                bombarea[xb, yb - c_up] = bfactor
            for c_right in range(1, 4):
                if xb + c_right > 16 or state['arena'][xb + c_right, yb] == -1:
                    break
                deatharea[xb + c_right, yb] = 1
                bombarea[xb + c_right, yb] = bfactor
            for c_left in range(1, 4):
                if xb - c_left < 0 or state['arena'][xb - c_left, yb] == -1:
                    break
                deatharea[xb - c_left, yb] = 1
                bombarea[xb - c_left, yb] = bfactor

            if abs(xb -x) + abs(yb-y) < abs(xb -last_x) + abs(yb-last_y):
                total_reward -= 2
        check_map = np.array([deatharea[pos] for pos in next_coord])
        if np.equal(check_map, 1).all():
            total_reward -= 10
        elif nb == 0 and (mybomb[0], mybomb[1], 4) in bombs_t and t0 == 4:
            crates_destroy = np.sum(np.equal(bombarea, state['arena']+4))  # Number of crates the placed bomb can destroy
            if crates_destroy != 0:
                total_reward += 5 + crates_destroy
            else:
                total_reward -= 2
        elif np.sum(check_map) == len(check_map) - 1 and last_action == 'WAIT' and deatharea[(x, y)] == -2:
            total_reward += 4  # good WAIT compensation

        # Predict state in next step
        bombarea = np.minimum(bombarea, 2)
        state['arena'][np.equal(deatharea, arena)] = 0
        state['explosions'] = np.maximum(state['explosions'] - 1, bombarea)

    crates_arena = np.maximum(arena, 0)
    crates_arena = crates_arena.T

    if len(state['coins']) != 0:
        last_closest_coin = sorted(state['coins'], key=lambda k: abs(k[0] - last_x) + abs(k[1] - last_y))[0]
        this_closest_coin = sorted(state['coins'], key=lambda k: abs(k[0] - x) + abs(k[1] - y))[0]

        if abs(last_closest_coin[0] - last_x) + abs(last_closest_coin[1] - last_y) < \
                abs(this_closest_coin[0] - x) + abs(this_closest_coin[1] - y):
            total_reward += 2

    elif len(state['others']) != 0 and np.sum(crates_arena) == 0:
        last_closest_p = sorted(state['others'], key=lambda k: abs(k[0] - last_x) + abs(k[1] - last_y))[0]
        this_closest_p = sorted(state['others'], key=lambda k: abs(k[0] - x) + abs(k[1] - y))[0]

        if abs(last_closest_p[0] - last_x) + abs(last_closest_p[1] - last_y) < \
                abs(this_closest_p[0] - x) + abs(this_closest_p[1] - y):
            total_reward += 2

    elif np.sum(crates_arena) != 0 and len(all_bombs) == 0:
        q1map = np.sum(crates_arena[:y, :])
        q2map = np.sum(crates_arena[y + 1:, :])
        q3map = np.sum(crates_arena[:, :x])
        q4map = np.sum(crates_arena[:, x + 1:])

        if np.argmax([q1map, q2map, q3map, q4map]) == s.actions.index(last_action):
            total_reward += 2
    # print('check: {}, total reward: {}'.format('here', total_reward))
    if len([(xb, yb, tb) for (xb, yb, tb) in state['bombs'] if abs(xb -last_x) + abs(yb-last_y) <= 4]) == 0:

        if len(agent.last_moves) >= 3 and (x, y) == agent.last_moves[-2]:
            total_reward -= 5
    if last_action == 'WAIT':
        total_reward -= 2

    return total_reward


def save_model(agent):
    print('ROUND COUNT: {}'.format(agent.experience.rounds_count))
    newname = agent.config['training']['save_model'] + str(int(agent.olddig) + agent.experience.rounds_count) + '_model.csv'
    saved_model_path = os.path.join(agent.config['training']['models_folder'], newname)
    agent.model.save(saved_model_path)
