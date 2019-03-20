import sys
import json
import numpy as np
from random import randint
from agent_code.local_agent.rewards_table import rewards
from agent_code.local_agent.diamond_nn.nn_model import *
from agent_code.local_agent.reward_fun import *
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

        selected_oldstate = selected_oldstate.reshape((-1, 45))
        target = eval_net.predict(selected_oldstate)
        selected_newstate = selected_newstate.reshape((-1, 45))
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
    with open('agent_code/local_agent/config.json') as f:
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
    experience = GainExperience(model=target_net, memory_size=65536, batch_size=1024, discount_rate=agent.config["model_config"]["gamma"])

    agent.experience = experience
    agent.mybomb = None
    agent.timetick = None
    eps = agent.config["model_config"]["eps"]
    agent.eps = eps
    agent.total_reward = 0
    agent.randoms = False

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
                agent.next_action = np.random.choice(s.actions)
                agent.randoms = True
            else:
                prediction = agent.model.predict(current_state)[0]
                action_idx = np.argmax(prediction)
                agent.next_action = s.actions[action_idx]
                # print(prediction)
                agent.randoms = False
        else:
            prediction = agent.model.predict(current_state)[0]
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


def reward_update(agent):
    send_to_experience(agent)


def end_of_episode(agent):

    # Get the last state of the game after the last step
    # add the last experience to the buffer in GainExperience object
    send_to_experience(agent, exit_game=True)

    if agent.experience.rounds_count % 50 == 0:
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
    if agent.experience.experiences_count >= 100:
        train(agent, agent.experience.rounds_count, exit_game)


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
    crates_arena = crates_arena.T

    x, y, _, bombs_left, _ = state['self']
    bombs = state['bombs']
    others = [(xo, yo) for (xo, yo, _, _, _) in state['others']]

    diglist = list()
    if len(state['coins']) == 0:
        diglist.append(5)
    else:
        closest_coin = sorted(state['coins'], key=lambda k: abs(k[0] - x) + abs(k[1] - y))[0]
        best_orientation = np.argmin([(closest_coin[0] - mx)**2 + (closest_coin[1] - my)**2 for (mx, my) in
                                      [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]]) + 1
        diglist.append(best_orientation)

    if np.sum(crates_arena) == 0:
        diglist.append(5)
    else:
        q1map = np.sum(crates_arena[1:6, 1:6])
        q2map = np.sum(crates_arena[1:6, 6:11])
        q3map = np.sum(crates_arena[1:6, 11:16])
        q4map = np.sum(crates_arena[6:11, 1:6])
        q5map = np.sum(crates_arena[6:11, 6:11])
        q6map = np.sum(crates_arena[6:11, 11:16])
        q7map = np.sum(crates_arena[11:16, 1:6])
        q8map = np.sum(crates_arena[11:16, 6:11])
        q9map = np.sum(crates_arena[11:16, 11:16])
        diglist.append(np.argmax([q1map, q2map, q3map, q4map, -1, q5map, q6map, q7map, q8map, q9map]) + 1)

    if len(state['others']) == 0:
        diglist.append(5)
    else:
        closest_p = sorted(state['others'], key=lambda k: abs(k[0] - x) + abs(k[1] - y))[0]
        closest_orientation = np.argmin([abs(closest_p[0] - mx) + abs(closest_p[1] - my) for (mx, my) in
                                      [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]]) + 1
        diglist.append(closest_orientation)

    diglist.append(bombs_left)
    for (i, j) in [( 0, -4), (-1, -3), ( 0, -3), ( 1, -3),
                   (-2, -2), (-1, -2), ( 0, -2), ( 1, -2), ( 2, -2),
                   (-3, -1), (-2, -1), (-1, -1), ( 0, -1), ( 1, -1),
                   ( 2, -1), ( 3, -1), (-4,  0), (-3,  0), (-2,  0),
                   (-1,  0), (0 ,  0), ( 1,  0), ( 2,  0), ( 3,  0),
                   ( 4,  0), (-3,  1), (-2,  1), (-1,  1), ( 0,  1),
                   ( 1,  1), ( 2,  1), ( 3,  1), (-2,  2), (-1,  2),
                   ( 0,  2), ( 1,  2), ( 2,  2), (-1,  3), ( 0,  3),
                   ( 1,  3), ( 0,  4)]:

        if (x + i) < 0 or (x + i) > 16 or (y + j) < 0 or (y + j) > 16:
            diglist.append(0)
        elif (x + i, y + j) in state['coins']:
            diglist.append(300)
        elif state['explosions'][x + i, y + j] == 1:
            diglist.append(10)
        elif state['explosions'][x + i, y + j] == 2:
            diglist.append(9)
        elif (x + i, y + j, 4) in bombs:
            if (x + i, y + j) in others:
                diglist.append(40)
            else:
                diglist.append(4)
        elif (x + i, y + j, 3) in bombs:
            if (x + i, y + j) in others:
                diglist.append(50)
            else:
                diglist.append(5)
        elif (x + i, y + j, 2) in bombs:
            if (x + i, y + j) in others:
                diglist.append(60)
            else:
                diglist.append(6)
        elif (x + i, y + j, 1) in bombs:
            if (x + i, y + j) in others:
                diglist.append(70)
            else:
                diglist.append(7)
        elif (x + i, y + j, 0) in bombs:
            if (x + i, y + j) in others:
                diglist.append(80)
            else:
                diglist.append(8)
        elif (x + i, y + j) in others:
                diglist.append(100)
        else:
            diglist.append(arena[x + i, y + j] + 1)  # 0, 1, 2

    state = np.array(diglist)
    return state.reshape((1, 45))


def train(agent, train_tick=0, exit_game=False):
    inputs, targets = agent.experience.get_inputs_targets(agent.model)
    # Instead of directly passing the losses into the sequential model, we cal
    start = time.time()
    agent.training_loss = agent.model.train_on_batch(x=inputs, y=targets)
    end = time.time()
    # print('Finish training after round number: {}, step number: {}, time elapsed: {}'.format(agent.experience.rounds_count, agent.game_state['step'], end-start))

    is_save = True if agent.experience.rounds_count % 50 == 0 or agent.experience.rounds_count == s.n_rounds else False
    if is_save and exit_game:
        print('Finish training after round number: {}, step number: {}, time elapsed: {}'.format(agent.experience.rounds_count, agent.game_state['step'], end-start))
        print('last eps: {}'.format(agent.eps))
        newname = agent.config['training']['save_model'] + str(int(agent.olddig) + train_tick) + '_model.h5'
        saved_model_path = os.path.join(agent.config['training']['models_folder'], newname)
        agent.model.save(saved_model_path)
