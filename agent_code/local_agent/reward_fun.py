import numpy as np
from .rewards_table import rewards
from settings import s
"""
To calculate the reward, you need the following 2 functions
"""

class Node:
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.cn_cost = 0
        self.cs_cost = 0
        self.total_cost = 0
        self.last_ac = 'MOVE'
        self.bomb_loc = None

    def __eq__(self, other):
        return self.position == other.position


def find_best_move(arena, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.cn_cost = start_node.cs_cost = start_node.total_cost = 0
    end_node = Node(None, end)
    end_node.cn_cost = end_node.cs_cost = end_node.total_cost = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.total_cost < current_node.total_cost:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            perform = []
            current = current_node
            while current is not None:
                path.append(current.position)
                perform.append(current.last_ac)
                current = current.parent
            ordered_path = path[::-1]
            ordered_act = perform[::-1]
            del ordered_act[0]
            return ordered_path, ordered_act

        # Generate children
        children = []

        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(arena) - 1) or node_position[0] < 0 or node_position[1] > (len(arena[len(arena) - 1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if arena[node_position[0], node_position[1]] <= -1:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            if current_node.bomb_loc is not None :
                diff = np.subtract(child.position, current_node.position)
                if np.equal(diff, 0).any() and abs(np.max(diff)) < 4:
                    factor = -0.5
                    child.bomb_loc = current_node.bomb_loc
                else:
                    factor = arena[child.position]
                child.cn_cost = current_node.cn_cost + 1 + factor

            elif arena[child.position] == 10:
                child.last_ac = 'BOMB'
                child.bomb_loc = child.position
                child.cn_cost = current_node.cn_cost + 1 + arena[child.position]
            else:
                child.cn_cost = current_node.cn_cost + 1 + arena[child.position]
            child.cs_cost = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.total_cost = child.cn_cost + child.cs_cost

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.cn_cost > open_node.cn_cost:
                    continue

            # Add the child to the open list
            open_list.append(child)


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


def calc_explosion_effect(arena, test_loc, fill_map=None, fill_values=0):

    xb, yb = test_loc
    if not isinstance(fill_map, list):
        fill_map = [fill_map]
        fill_values = [fill_values]

    for idx in range(len(fill_map)):
        fill_map[idx][xb, yb] = fill_values[idx]
    for c_down in range(1, 4):
        if yb + c_down > 16 or arena[xb, yb + c_down] == -1: break
        for idx in range(len(fill_map)):
            fill_map[idx][xb, yb + c_down] = fill_values[idx]
    for c_up in range(1, 4):
        if yb - c_up < 0 or arena[xb, yb - c_up] == -1: break
        for idx in range(len(fill_map)):
            fill_map[idx][xb, yb - c_up] = fill_values[idx]
    for c_right in range(1, 4):
        if xb + c_right > 16 or arena[xb + c_right, yb] == -1: break
        for idx in range(len(fill_map)):
            fill_map[idx][xb + c_right, yb] = fill_values[idx]
    for c_left in range(1, 4):
        if xb - c_left < 0 or arena[xb - c_left, yb] == -1: break
        for idx in range(len(fill_map)):
            fill_map[idx][xb - c_left, yb] = fill_values[idx]
    return fill_map


def find_nearest_thing(thing, arena, test_loc, possible_coord):

    tx, ty = test_loc
    sorted_thing = sorted(thing, key=lambda k: abs(k[0] - tx) + abs(k[1] - ty))
    check_thing = [sorted_thing[0], sorted_thing[0]] if len(sorted_thing) == 1 else [sorted_thing[0], sorted_thing[1]]
    nearest_coord = sorted(possible_coord, key=lambda k: ((check_thing[0][0] - k[0]) ** 2 + (
            check_thing[0][1] - k[1]) ** 2) * 1 + ((check_thing[1][0] - k[0]) ** 2 + (
            check_thing[1][1] - k[1]) ** 2) * 0.001)
    return nearest_coord, check_thing[0]


def compute_reward(agent):
    print('started')
    events = agent.events

    last_action = agent.next_action

    state = agent.game_state.copy()
    arena = state['arena'].copy()
    explosions = state['explosions'].copy()
    crates_arena = np.maximum(arena, 0)
    crates_arena = crates_arena.T

    mybomb = agent.mybomb
    all_bombs = state['bombs']
    all_bombs_loc = [(xb, yb) for (xb, yb, _) in all_bombs]
    all_op_loc = [op[0:2] for op in state['others']]

    (x, y, _, nb, _) = state['self']
    coord_dict = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0), 'BOMB': (0, 0), 'WAIT': (0, 0)}
    last_x, last_y = np.subtract((x, y), coord_dict[last_action]) if 6 not in events else (x, y)
    arrowcoord = [(last_x, last_y - 1), (last_x, last_y + 1), (last_x - 1, last_y), (last_x + 1, last_y)]
    possible_coord = [(xc, yc) for (xc, yc) in arrowcoord if
                      arena[xc, yc] != -1 and (xc, yc) not in state['others'] and (xc, yc) not in all_bombs_loc]
    nocrates_coord = [(xc, yc) for (xc, yc) in possible_coord if arena[xc, yc] != 1]

    total_reward = 0
    old_coord = None
    all_coord = []

    # Situation Flag
    escape = False
    place_bomb = False
    no_coins = len(state['coins']) == 0
    no_crates = np.sum(np.maximum(arena, 0)) == 0
    no_opponents = len(state['others']) == 0

    if 6 in events:
        print('Fucked up')
    # Primary Reward
    for event in events:
        total_reward += list(rewards.values())[event]

    # Evaludate Bomb and Escape Policy
    # Estimate future events: t + 1, t + 2, t + 3, t + 4
    for t0 in range(0, 5):

        # Find all possible coordinates after t0 steps
        if t0 == 0:
            next_coord = [(x, y)]
        else:
            next_coord = []
            for coord in old_coord:
                next_coord += [np.add(coord, coord_dict[i]) for i in
                               det_valid_action(state, other_loc=coord)]  # Possible coordinates in 2nd Step
            if len(next_coord) != 0:
                next_coord = [tuple(i) for i in np.unique(next_coord, axis=0)]

        all_coord.append(next_coord)
        old_coord = next_coord


        # find bombs nearby in the future time step
        bombs_t = [(xb, yb, tb) for (xb, yb, tb) in state['bombs'] if tb == t0 and abs(xb - x) + abs(yb - y) <= 4]

        deatharea = np.minimum(state['explosions'].copy(), 1) * 3 - 2  # -2 or 1
        bombarea = np.zeros(arena.shape)
        for xb, yb, tb in bombs_t:

            bfactor = 2
            if len(bombs_t) != 0 and (xb, yb) != mybomb and t0 != 4 not in bombs_t and not escape:
                escape = True
                print('Escape Policy')
            elif (xb, yb) == mybomb and t0 == 4:
                place_bomb = True
                bfactor = 5

            deatharea[xb, yb] = 1
            deatharea, bombarea = calc_explosion_effect(state['arena'], (xb, yb), [deatharea, bombarea], [1, bfactor])

            # Penalize if not escaping
            if (xb - x) ** 2 + (yb - y) ** 2 < (xb - last_x) ** 2 + (yb - last_y) ** 2:
                total_reward -= 2

        # Penalize for entering the death zone
        check_map = np.array([deatharea[pos] for pos in next_coord])
        if np.equal(check_map, 1).all():
            total_reward -= 10

        # Reward for placing good crates
        elif nb == 0 and place_bomb:
            pass
            crates_destroy = np.sum(np.equal(bombarea, state['arena'] + 4))  # Number of crates the placed bomb can destroy
            if crates_destroy != 0:
                total_reward += 2 + crates_destroy
            else:
                total_reward -= 5

        # Good wait compensation
        elif np.sum(check_map) == len(check_map) - 1 and last_action == 'WAIT' and deatharea[(x, y)] == -2:
            total_reward += 6

        # Estimate state in next step
        bombarea = np.minimum(bombarea, 2)
        state['arena'][np.equal(deatharea, state['arena'])] = 0
        state['explosions'] = np.maximum(state['explosions'] - 1, bombarea)

    if not escape and 11 not in events and 6 not in events:

        if not no_coins:
            nearest_coin_coord, nearest_coin = find_nearest_thing(state['coins'], arena, (last_x, last_y), arrowcoord)
            coin_in_sight = True if nearest_coin_coord[0] in all_coord[4] else False
        else:
            coin_in_sight = False

        if not no_opponents:
            nearest_op_coord, nearest_op = find_nearest_thing(all_op_loc, arena, (last_x, last_y), arrowcoord)
            opponent_in_sight = True if nearest_op_coord[0] in all_coord[4] else False
        else:
            opponent_in_sight = False
            nearest_op = None
            nearest_op_coord = None

        print('Finding case:')
        # Case 1 Break Crates
        if not opponent_in_sight and no_coins:

            print('Case 1')
            best_coord = (last_x, last_y)
            explosion_effect = calc_explosion_effect(arena, (last_x, last_y), np.zeros(arena.shape), 1)
            highest_crates_destroyed = np.sum(np.minimum(explosion_effect, np.absolute(arena)))
            for (px, py) in arrowcoord:

                test_effect = calc_explosion_effect(arena, (px, py), np.zeros(arena.shape), 1)
                test_crates_destroyed = np.sum(np.minimum(test_effect, np.absolute(arena)))
                if test_crates_destroyed > highest_crates_destroyed and arena[px, py] != 1:
                    best_coord = (px, py)
                    highest_crates_destroyed = test_crates_destroyed

            if highest_crates_destroyed == 0:
                q1map = np.sum(crates_arena[1:6, 1:6])
                q2map = np.sum(crates_arena[1:6, 6:11])
                q3map = np.sum(crates_arena[1:6, 11:16])
                q4map = np.sum(crates_arena[6:11, 1:6])
                q5map = np.sum(crates_arena[6:11, 6:11])
                q6map = np.sum(crates_arena[6:11, 11:16])
                q7map = np.sum(crates_arena[11:16, 1:6])
                q8map = np.sum(crates_arena[11:16, 6:11])
                q9map = np.sum(crates_arena[11:16, 11:16])

                best_direction = np.argmax([q1map, q2map, q3map, q4map, q5map, q6map, q7map, q8map, q9map])
                center = [(3, 3), (8, 3), (13, 3), (3, 8), (8, 8), (13, 8), (3, 13), (8, 13), (13, 13)]
                best_center = center[best_direction] if isinstance(best_direction, int) else center[4]
                if (x - best_center[0])**2 + (y - best_center[1])**2 < (last_x - best_center[0])**2 + (last_y - best_center[1])**2:
                    total_reward += 3
                else:
                    total_reward -= 3

            elif best_coord == (last_x, last_y):

                if place_bomb and (x, y) == best_coord:
                    total_reward += 3
                else:
                    total_reward -= 3

            elif best_coord == (x, y):
                total_reward += 3

        # Case 2 Attacker
        elif (opponent_in_sight and not coin_in_sight) or (no_crates and no_coins and not no_opponents):

            print('Case 2')
            if nearest_op in nocrates_coord and all_coord[1] <= 2:
                if place_bomb:
                    total_reward += 10
                elif last_action != 'WAIT' or nb == 0:
                    total_reward += 3
                else:
                    total_reward -= 3
            elif (x, y) == nearest_op_coord[0]:
                total_reward += 3

        # Case 3 Break Crates to collect coin:
        else:
            print('Case 3')
            astar_map = np.maximum(arena.copy() * 10, -1)
            explosions_locations = np.nonzero(explosions)
            astar_map[explosions_locations] = 1000
            for (ox,oy) in all_op_loc:
                astar_map[ox,oy] = 1000
            for (bx, by) in all_bombs_loc:
                astar_map[bx, by] = 1000
                astar_map[last_x, last_y] = 0

            print('Passed')
            best_path, best_act = find_best_move(astar_map, (last_x, last_y), nearest_coin) # A* algo
            if best_act[0] == 'MOVE' and best_path[1] == (x, y):
                total_reward += 3
            elif best_act[0] == 'BOMB' and last_action == 'BOMB':
                total_reward += 3
            else:
                total_reward -= 3
    print('Finished')
    return total_reward
