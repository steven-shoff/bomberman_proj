import sys
import json
import numpy as np
from random import randint
from agent_code.full_agent.rewards_table import rewards
from agent_code.full_agent.nn_model import *
from settings import s
import time
import keras


class GainExperience(object):
    def __init__(self, model, memory_size, batch_size, discount_rate):
        """
        This class function collects experience from each steps and compute new Q(s, a)
        :param model        : Model used to interact with the agent
        :param memory_size  : Maximum memory loading for training model
        :param batch_size   : Number of records selected for training
        :param discount_rate: Discount rate of future rewards in the Q-learning algorithm
        """

        # Class configuration attributes:
        self.max_memory_size = memory_size
        self.batch_size = batch_size

        # Class model attributes
        self.discount_rate = discount_rate
        self.target_net = model

        # class attributes for memory storage
        self.old_state = list()
        self.reward = list()
        self.actions = list()
        self.new_state = list()
        self.terminal = list()

        self.current_state = None

        self.experiences_count = 1     # Experience counter
        self.rounds_count = 0           # Round counter

    def expand_experience(self, experience, exit_game=False):
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

        self.old_state.append(self.current_state)
        self.actions.append(experience[1])
        self.reward.append(experience[2])
        self.new_state.append(experience[3])
        self.terminal.append(exit_game)

        if len(self.old_state) > self.max_memory_size:
            del self.old_state[0]
            del self.reward[0]
            del self.actions[0]
            del self.new_state[0]
            del self.terminal[0]

        # if self.rounds_count == 10:
        #    print(self.rounds_count)
        #    print(np.count_nonzero(self.targets,axis=0))
        #    print(self.experiences_count)
        #    print(self.targets[:self.experiences_count+5])

    def get_inputs_targets(self, eval_net):
        if len(self.old_state) < self.batch_size:
            selected_index = list(range(len(self.old_state)))
        else:
            selected_index = np.random.choice(list(range(len(self.old_state))), self.batch_size)
        selected_oldstate = np.array(self.old_state)[selected_index]
        selected_actions = np.array(self.actions)[selected_index]
        selected_rewards = np.array(self.reward)[selected_index]
        selected_newstate = np.array(self.new_state)[selected_index]
        selected_terminal = np.array(self.terminal)[selected_index]

        target = eval_net.predict(selected_oldstate)
        renewed_target = selected_rewards + np.max(self.target_net.predict(selected_newstate), axis=1)*selected_terminal*(1 - int(self.discount_rate))

        for idx, a in enumerate(selected_actions):
            target[idx, a] = renewed_target[idx]
        return selected_oldstate, target


def setup(agent):
    """
    This function performed the following actions in setup:
        - Initialize GainExperience and model
        - Load agent configuration file

    :param agent: Agent used to interact with the Environment
    """

    # load config file that contains paths to the pretrained weights
    with open('agent_code/full_agent/config.json') as f:
        config = json.load(f)

    # store that config in the agent to be used later on
    agent.config = config

    # load the flag of using pretrained weights
    load_pretrained = agent.config["workflow"]["pretrained"]

    # load score file (only used in playing=
    agent.score_file = None

    # if in training mode, we have two options: train from scratch or load weights
    if agent.config['workflow']['train']:
        if load_pretrained:
            # read the path of the model to be loaded

            try:
                all_name = [fname for fname in os.listdir(agent.config['training']['models_folder']) if
                            fname.startswith(agent.config['training']['model_name']) and fname.endswith("_model.h5")]
                if len(all_name) != 0:
                    all_name = sorted(all_name, key=lambda y: int(y[len(agent.config['training']['model_name']):-len("_model.h5")]))
                else:
                    all_name = ['0']
                agent.pretrained_model_path = os.path.join(agent.config['training']['models_folder'], all_name[-1])
                agent.olddig = all_name[-1][len(agent.config['training']['model_name']):-len("_model.h5")]
                agent.model = load_model(agent.pretrained_model_path, custom_objects={'pure_se': pure_se})

            # read_model method raises exceptions to be caught here
            except FileNotFoundError:
                agent.logger.info("No model is specified to load")
                sys.exit(-1)
            except Exception as e:
                agent.logger.info("An error occured in loading the model: {}".format(e))
                sys.exit(-1)
        else:
            # build model to be trained from scratch if no pre-trained weights specified
            agent.model = build_model()
            agent.olddig = 0
    else:
        try:
            agent.pretrained_model_path = os.path.join(agent.config['playing']['models_folder'], agent.config['playing']['model_name'])
            agent.model = load_model(agent.pretrained_model_path, custom_objects={'pure_se': pure_se})
            score_file_path = os.path.join(agent.config['playing']['scores_folder'], agent.config['playing']['scores_name'])
            agent.score_file = open(score_file_path, "a+")
            agent.olddig = agent.config['playing']['model_name'][len(agent.config['training']['model_name']):-len("_model.h5")]
        except FileNotFoundError:
            agent.logger.info(f"Model file is not found, file name: {agent.pretrained_model_path}")
            sys.exit(-1)
        except Exception as e:
            agent.logger.info("An error occured in loading the model: {}".format(e))
            sys.exit(-1)

    # experience = GainExperience(model=agent.model,memory_size=s.max_steps,discount_rate=0.99)
    target_net = keras.models.clone_model(agent.model)
    target_net.set_weights(agent.model.get_weights())
    experience = GainExperience(model=target_net, memory_size=32768, batch_size=512, discount_rate=agent.config["model_config"]["gamma"])

    agent.experience = experience
    agent.mybomb = None
    agent.timetick = None
    eps = agent.config["model_config"]["eps"]
    agent.eps = eps
    agent.total_reward = 0

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
            agent.experience.rounds_count += 1
            agent.last_moves = [(x, y)]

        elif len(agent.last_moves) >= 10:
            del agent.last_moves[0]
            agent.last_moves.append((x, y))
        else:
            agent.last_moves.append((x, y))

        current_state = formulate_state(state)
        agent.logger.info(f'current state from act: {current_state}')

        if agent.config['workflow']['train']:
            agent.experience.current_state = current_state

            rnd = randint(1, 100)
            ths = int(agent.eps * 100)
            if rnd < ths:
                agent.logger.info('Selecting action at Random for exploring...')

                #valid_actions = det_valid_action(state)
                agent.next_action = np.random.choice(s.actions[0:-2])
            else:
                prediction = agent.model.predict(np.expand_dims(current_state, axis=0))[0]
                action_idx = np.argmax(prediction)
                agent.next_action = s.actions[action_idx]
                print(prediction)
        else:
            prediction = agent.model.predict(np.expand_dims(current_state, axis=0))[0]
            action_idx = np.argmax(prediction)
            agent.next_action = s.actions[action_idx]
            print(prediction)
            print(agent.next_action)
            print(s.actions)

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

    if agent.experience.rounds_count % agent.config["model_config"]["update_freq"] == 0:
        agent.eps *= agent.config["model_config"]["eps_discount"]
        agent.experience.target_net.set_weights(agent.model.get_weights())


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

    print('LOC; {}, ACTION: {}, REWARD: {}'.format((agent.game_state['self'][0:2]), agent.next_action, agent.total_reward))
    # create one experience and save it into GainExperience object
    new_experience = [last_action, agent.total_reward, new_state]

    agent.experience.expand_experience(new_experience, exit_game=exit_game)
    if agent.experience.experiences_count >= 128:
        train(agent, agent.experience.rounds_count, exit_game)


def formulate_state(state):
    # Extracting info about the game
    arena = state['arena'].copy()
    (xa, ya, _, bomb_a, _) = state['self']
    others = [(x,y) for (x,y,n,b,_) in state['others']]
    bombs = state['bombs'].copy()
    explosions = state['explosions']
    coins = state['coins']

    # Enriching the arena with info about own locations, bombs, explosions, opponent positions etc.
    # indicating the location of the player himself

    #layer_1 is the arena
    agents_layer = np.zeros((17,17))
    bombs_layer = np.zeros((17, 17))
    explosions_layer = np.zeros((17, 17))

    for coin in coins:
        arena[coin] = 50

    for other in others:
        agents_layer[other] = 20
    agents_layer[(xa, ya)] = 10 if bomb_a == 0 else 100

    for xb, yb, tb in bombs:
        bombs_layer[(xb, yb)] = (tb+1)*100

    explosions_locations = np.nonzero(explosions)
    explosions_layer[explosions_locations] = explosions[explosions_locations] * 1000

    arena = arena.transpose().copy()
    agents_layer = agents_layer.transpose().copy()
    bombs_layer = bombs_layer.transpose().copy()
    explosions_layer = explosions_layer.transpose().copy()
    new_state = np.stack((arena,agents_layer,bombs_layer,explosions_layer),axis=2)
    return new_state


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
    last_x, last_y = np.subtract((x, y), coord_dict[last_action]) if 6 not in events else (x, y)
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
                total_reward += 2 + crates_destroy
            else:
                total_reward -= 5
        elif np.sum(check_map) == len(check_map) - 1 and last_action == 'WAIT' and deatharea[(x, y)] == -2:
            total_reward += 6  # good WAIT compensation

        # Predict state in next step
        bombarea = np.minimum(bombarea, 2)
        state['arena'][np.equal(deatharea, arena)] = 0
        state['explosions'] = np.maximum(state['explosions'] - 1, bombarea)

    crates_arena = np.maximum(arena, 0)
    crates_arena = crates_arena.T
    all_bombs_loc = [(xb, yb) for (xb, yb, _) in all_bombs]

    # Coin Environment
    if len(state['coins']) != 0 and 11 not in events:

        arrowcoord = [(last_x, last_y - 1), (last_x, last_y + 1), (last_x - 1, last_y), (last_x + 1, last_y)]
        possible_coord = [(xc, yc) for (xc, yc) in arrowcoord if arena[xc, yc] != -1 and (xc, yc) not in state['others'] and (xc, yc) not in all_bombs_loc]
        sorted_coins = sorted(state['coins'], key=lambda k: abs(k[0] - last_x) + abs(k[1] - last_y))
        check_coin = [sorted_coins[0], sorted_coins[0]] if len(sorted_coins) == 1 else [sorted_coins[0], sorted_coins[1]]
        minarrowcoord = sorted(possible_coord, key=lambda k: ((check_coin[0][0] - k[0]) ** 2 + (
                    check_coin[0][1] - k[1]) ** 2) * 1 + ((check_coin[1][0] - k[0]) ** 2 + (
                    check_coin[1][1] - k[1]) ** 2) * 0.001)[0]

        if (x, y) == minarrowcoord:
            total_reward += 3
        else:
            total_reward -= 3

    # Opponent Environment
    elif len(state['others']) != 0 and np.sum(crates_arena) == 0 and 6 not in events:
        last_closest_p = sorted(state['others'], key=lambda k: abs(k[0] - last_x) + abs(k[1] - last_y))[0]
        this_closest_p = sorted(state['others'], key=lambda k: abs(k[0] - x) + abs(k[1] - y))[0]

        if abs(last_closest_p[0] - last_x) + abs(last_closest_p[1] - last_y) > \
                abs(this_closest_p[0] - x) + abs(this_closest_p[1] - y):
            total_reward += 2

    elif np.sum(crates_arena) != 0 and len(all_bombs) == 0 and 6 not in events:
        q1map = np.sum(crates_arena[:y, :])
        q2map = np.sum(crates_arena[y + 1:, :])
        q3map = np.sum(crates_arena[:, :x])
        q4map = np.sum(crates_arena[:, x + 1:])

        if np.argmax([q1map, q2map, q3map, q4map]) == s.actions.index(last_action):
            total_reward += 2
    # print('check: {}, total reward: {}'.format('here', total_reward))
    # if len([(xb, yb, tb) for (xb, yb, tb) in state['bombs'] if abs(xb -last_x) + abs(yb-last_y) <= 4]) == 0:

        #if len(agent.last_moves) >= 3 and (x, y) == agent.last_moves[-2]:
        #    total_reward += 2

    if last_action == 'WAIT':
        total_reward -= 5

    return total_reward


def train(agent, train_tick=0, exit_game=False):
    inputs, targets = agent.experience.get_inputs_targets(agent.model)
    # Instead of directly passing the losses into the sequential model, we cal
    start = time.time()
    agent.training_loss = agent.model.train_on_batch(x=inputs, y=targets)
    end = time.time()
    # print('Finish training after round number: {}, step number: {}, time elapsed: {}'.format(agent.experience.rounds_count, agent.game_state['step'], end-start))

    is_save = True if agent.experience.rounds_count % agent.config['model_config']['save_freq'] == 0 or agent.experience.rounds_count == s.n_rounds else False
    if is_save and exit_game:
        print('Finish training after round number: {}, step number: {}, time elapsed: {}'.format(agent.experience.rounds_count, agent.game_state['step'], end-start))
        print('last eps: {}'.format(agent.eps))
        newname = agent.config['training']['save_model'] + str(int(agent.olddig) + train_tick) + '_model.h5'
        saved_model_path = os.path.join(agent.config['training']['models_folder'], newname)
        agent.model.save(saved_model_path)
