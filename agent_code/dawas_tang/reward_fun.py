import numpy as np
from .rewards_table import rewards
from settings import s
"""
To calculate the reward, you need the following 2 functions
"""

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
                deatharea[yb - c_up, xb] = 1
                bombarea[yb - c_up, xb] = bfactor
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
        q4map = np.sum(crates_arena[x + 1:, ])

        if np.argmax([q1map, q2map, q3map, q4map]) == s.actions.index(last_action):
            total_reward += 2
    # print('check: {}, total reward: {}'.format('here', total_reward))
    if len([(xb, yb, tb) for (xb, yb, tb) in state['bombs'] if abs(xb -last_x) + abs(yb-last_y) <= 4]) == 0:

        if len(agent.last_moves) >= 3 and (x, y) == agent.last_moves[-2]:
            total_reward -= 5
    if last_action == 'WAIT':
        total_reward -= 2

    return total_reward