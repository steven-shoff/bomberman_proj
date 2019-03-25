import numpy as np
from agent_code.dawas_tang.rewards_table import rewards
from settings import s


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
    # Add counter for safe cut
    counter = 0
    # Loop until you find the end
    while len(open_list) > 0 and counter < 500:

        counter += 1
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
            if arena[node_position[0], node_position[1]] <= -1 or arena[node_position[0], node_position[1]] > 10:
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

            elif arena[child.position] == 3:
                child.last_ac = 'WAIT'
                child.cn_cost = current_node.cn_cost + 1 + arena[child.position]
            else:
                child.cn_cost = current_node.cn_cost + 1 + arena[child.position]
            child.cs_cost = abs(child.position[0] - end_node.position[0])*11 + abs(child.position[1] - end_node.position[1])*11
            child.total_cost = child.cn_cost + child.cs_cost

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.cn_cost > open_node.cn_cost:
                    continue

            # Add the child to the open list
            open_list.append(child)

    return None, None


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


def calc_explosion(arena, test_loc):

    xb, yb = test_loc
    fill_map = list()

    for c_down in range(1, 4):
        if yb + c_down > 15 or arena[xb, yb + c_down] == -1: break
        fill_map.append([xb, yb + c_down])
    for c_up in range(1, 4):
        if yb - c_up < 0 or arena[xb, yb - c_up] == -1: break
        fill_map.append([xb, yb - c_up])
    for c_right in range(1, 4):
        if xb + c_right > 15 or arena[xb + c_right, yb] == -1: break
        fill_map.append([xb + c_right, yb])
    for c_left in range(1, 4):
        if xb - c_left < 0 or arena[xb - c_left, yb] == -1: break
        fill_map.append([xb - c_left, yb])
    return fill_map

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


def find_nearest_thing(thing, test_loc, possible_coord, reverse=False):

    tx, ty = test_loc
    sorted_thing = sorted(thing, key=lambda k: abs(k[0] - tx) + abs(k[1] - ty), reverse=reverse)
    check_thing = [sorted_thing[0], sorted_thing[0]] if len(sorted_thing) == 1 else [sorted_thing[0], sorted_thing[1]]
    nearest_coord = sorted(possible_coord, key=lambda k: ((check_thing[0][0] - k[0]) ** 2 + (
            check_thing[0][1] - k[1]) ** 2) * 1 + ((check_thing[1][0] - k[0]) ** 2 + (
            check_thing[1][1] - k[1]) ** 2) * 0.001)
    return nearest_coord, check_thing[0]


def compute_reward(agent):
    events = agent.events
    mybomb = agent.mybomb
    last_action = agent.next_action
    state = agent.game_state.copy()

    arena = state['arena'].copy()
    explosions = state['explosions'].copy()
    (x, y, _, nb, _) = state['self']
    all_bombs = state['bombs']

    coord_dict = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0), 'BOMB': (0, 0), 'WAIT': (0, 0)}
    last_x, last_y = np.subtract((x, y), coord_dict[last_action]) if 6 not in events else (x, y)

    no_coins = len(state['coins']) == 0
    no_opponents = len(state['others']) == 0
    all_bombs_loc = [(xb, yb) for (xb, yb, _) in all_bombs]

    if no_opponents:
        all_ops_nearby = []
        closest_op_nearby = None
    else:
        all_ops_nearby = [op[0:2] for op in state['others'] if abs(op[0] - last_x) + abs(op[0] - last_x) <= 3]
        if len(all_ops_nearby) != 0:
            closest_op_nearby = sorted(all_ops_nearby, key=lambda op: abs(op[0] - last_x) + abs(op[0] - last_x))[0]
        else:
            closest_op_nearby = None

    arrowcoord = [(last_x, last_y - 1), (last_x, last_y + 1), (last_x - 1, last_y), (last_x + 1, last_y)]
    total_reward = 0
    all_coord = []
    old_coord = None
    old_op_coord = None
    all_op_coord = []

    # Situation Flag
    escape = False
    place_bomb = False

    # Primary Reward
    for event in events:
        total_reward += list(rewards.values())[event]

    # Evaluate Bomb and Escape Policy

    if 6 in events or 14 in events:
        pass # No extra calculation needed
    elif 8 in events:
        total_reward += 5  # reward for survival
    elif 7 in events or len(all_bombs) != 0:

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

            if closest_op_nearby is not None:
                if t0 == 0:
                    next_op_coord = [closest_op_nearby]
                else:
                    next_op_coord = []
                    for coord in old_op_coord:
                        next_op_coord += [np.add(coord, coord_dict[i]) for i in
                                       det_valid_action(state, other_loc=coord)]  # Possible coordinates in 2nd Step
                    if len(next_op_coord) != 0:
                        next_op_coord = [tuple(i) for i in np.unique(next_op_coord, axis=0)]
                all_op_coord.append(next_op_coord)
                old_op_coord = next_op_coord

            all_coord.append(next_coord)
            old_coord = next_coord

            # find bombs nearby in the future time step
            bombs_t = [(xb, yb, tb) for (xb, yb, tb) in state['bombs'] if tb == t0 and abs(xb - x) + abs(yb - y) <= 4]

            deatharea = np.minimum(state['explosions'].copy(), 1) * 3 - 2  # -2 or 1
            bombarea = np.zeros(arena.shape)
            for xb, yb, tb in bombs_t:

                bfactor = 2
                if len(bombs_t) != 0 and t0 != 4 :
                    escape = True

                if (xb, yb) == mybomb and t0 == 4:
                    place_bomb = True
                    bfactor = 5

                deatharea[xb, yb] = 1
                deatharea, bombarea = calc_explosion_effect(state['arena'], (xb, yb), [deatharea, bombarea], [1, bfactor])

            if place_bomb:
                check_map = np.array([deatharea[pos] for pos in next_coord])
                if np.equal(check_map, 1).all():
                    total_reward -= 5
                else:
                    if closest_op_nearby is not None:
                        op_kill = True
                        check_map = np.array([bombarea[pos] for pos in next_op_coord])
                        if np.equal(check_map, 5).all():
                            kill_score = 5
                        elif abs(closest_op_nearby[0] - last_x) + abs(closest_op_nearby[1] -  last_y) <= 2:
                            kill_score = 3
                        else:
                            op_kill = False
                            kill_score = 0
                    else:
                        op_kill = False
                        kill_score = 0

                    explosion_effect = calc_explosion_effect(arena, (last_x, last_y), np.zeros(arena.shape), 1)
                    crates_destroyed = np.sum(np.minimum(explosion_effect, np.absolute(state['arena'])))
                    if crates_destroyed != 0 or op_kill:
                        total_reward += crates_destroyed + kill_score
                    elif not op_kill:
                        total_reward -= 5

            if escape:

                # Penalize for entering the death zone
                check_map = np.array([deatharea[pos] for pos in next_coord])
                # If agent cannot escape from the bomb zone
                if np.equal(check_map, 1).all():

                    if deatharea[last_x, last_y] == 1 and t0 != 4:
                        sorted_bombs_t = sorted(bombs_t, key=lambda btk: abs(btk[0] - last_x) - abs(btk[1] - last_y))
                        if abs(sorted_bombs_t[0][0] - x) + abs(sorted_bombs_t[0][1] - y) >= abs(sorted_bombs_t[0][0] - last_x) + abs(sorted_bombs_t[0][1] - last_y):
                            total_reward += 0
                        else:
                            total_reward -= 1
            escape = False
            # Estimate state in next step
            bombarea = np.minimum(bombarea, 2)
            state['arena'][np.equal(deatharea, state['arena'])] = 0
            state['explosions'] = np.maximum(state['explosions'] - 1, bombarea)

    # Extra Reward:
    elif not no_coins:
            nearest_coin_coord, nearest_coin = find_nearest_thing(state['coins'], (last_x, last_y), arrowcoord)
            astar_map = np.maximum(arena.copy() * 10, -1)
            explosions_locations = np.nonzero(explosions)
            astar_map[explosions_locations] = 3
            for (ox,oy) in all_ops_nearby:
                astar_map[ox,oy] = 1000
            for (bx, by) in all_bombs_loc:
                astar_map[bx, by] = 1000

            best_path, best_act = find_best_move(astar_map, (last_x, last_y), nearest_coin) # A* algo

            if best_path is not None:
                if len(best_path) == 1 and last_action == 'WAIT':
                    total_reward += 0
                if best_act[0] == 'MOVE' and best_path[1] == (x, y):
                    total_reward += 0
                elif last_action == 'WAIT':
                    if best_act[0] == 'WAIT':
                        total_reward += 0
                    else:
                        total_reward -= 1
                else:
                    nbest_path, nbest_act = find_best_move(astar_map, (x, y), nearest_coin)
                    if nbest_act[0] == 'MOVE' and nbest_path[1] == (last_x, last_y):
                        total_reward -= 1
    return total_reward
