# coded by Jonathan Lewis and Adam Carter

from mcts_node import MCTSNode
from random import choice
from math import sqrt, log

num_nodes = 100
explore_faction = 2.


def ucb(node):
    """
        NOTES: Adversarial planning – the bot will be simulating both players’ turns.
        This requires you to alter the UCT function (during the tree traversal/selection phase)
        on the opponent’s turn. Remember: the opponent’s win rate (Xj) = (1 – bot’s win rate).

        Assuming 1 = MCTS_Vanilla and 2 = bot
        made ucb and ucb2 for each case
    """

    if node.visits == 0:
        return float("inf")
    exploit = node.wins / node.visits
    explore = explore_faction * sqrt(log(node.parent.visits) / node.visits)

    return exploit + explore


def ucb2(node):
    if node.visits == 0:
        return float("inf")
    exploit = 1 - (node.wins / node.visits)
    explore = explore_faction * sqrt(log(node.parent.visits) / node.visits)
    return exploit + explore


def traverse_nodes(node, board, state, identity):
    """ Traverses the tree until the end criterion are met.

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 'red' or 'blue'.

    Returns:        A node from which the next stage of the search can proceed.

    """
    # Hint: return leaf_node

    leaf_node = node
    leaf_state = state
    '''
    # must fully expand the node first
    if node.untried_actions != []:
        #print("expanding ", node.untried_actions)
        leaf_node = expand_leaf(node, board, state)
        return leaf_node
    '''

    turn = True
    # when no actions are left, then we can check ucb against child nodes
    while not leaf_node.untried_actions and leaf_node.child_nodes:
        temp_state = leaf_state
        highscore = float('-inf')
        for curr_node in leaf_node.child_nodes.values():
            score = 0
            if turn:  # in p2_sim.py, red == 1
                score = ucb(curr_node)
            elif not turn:  # in p2_sim.py, blue == 2
                score = ucb2(curr_node)
            if highscore < score:
                leaf_node = curr_node
                temp_state = board.next_state(leaf_state, leaf_node.parent_action)
                highscore = score
        turn = not turn
        leaf_state = temp_state
    return leaf_node, leaf_state


def expand_leaf(node, board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node.

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:    The added child node.

    """
    # Hint: return new_node

    next_action = choice(node.untried_actions)
    node.untried_actions.remove(next_action)
    next_state = board.next_state(state, next_action)
    new_node = MCTSNode(node, next_action, board.legal_actions(next_state))
    node.child_nodes[next_action] = new_node
    return new_node, next_state


def rollout(board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    we referenced this to help get us started:
    https://github.com/SamWool1/GameAI_P3/blob/master/src/mcts_modified.py
    """

    if board.is_ended(state):
        return board.points_values(state)
    # this is where we start modifying
    else:
        # make a heuristic to eval next move
        curr_action = choice(board.legal_actions(state))    # choose a legal action at random
        action_list = board.legal_actions(state)            # make a copy of the current available actions

        boxes_dict = board.owned_boxes(state)               # check which boxes have been played in already
        curr_player = board.current_player(state)           # get the current player's turn
        # a dict of used boxes
        #               draw -- p1 -- p2
        played_boxes = {0: 0, 1: 0, 2: 0}
        for i, j in boxes_dict:                             
            played_boxes[j] = played_boxes[j] + 1

        for action in action_list:                          # tests playing actions from the list
            next_state = board.next_state(state, action)
            next_box = board.owned_boxes(next_state)
            next_played_boxes = {0: 0, 1: 0, 2: 0}

            for i, j in next_box:
                next_played_boxes[j] = next_played_boxes[j] + 1

            if next_played_boxes[curr_player] > played_boxes[curr_player]:  # if the action wins then do it
                curr_action = action
                break

        # if a suitable action not found, just plays a random one from the action list
        new_state = board.next_state(state, curr_action)
        return rollout(board, new_state)


def backpropagate(node, won):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """

    if node.parent is None:
        node.wins += won
        node.visits += 1
        return

    else:
        node.wins += won
    node.visits += 1
    backpropagate(node.parent, won)


def think(board, state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        state:  The state of the game.

    Returns:    The action to be taken.

    """
    identity_of_bot = board.current_player(state)
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(state))

    for step in range(num_nodes):
        # Copy the game for sampling a playthrough
        sampled_game = state

        # Start at root
        node = root_node

        # Do MCTS - This is all you!

        # Selection
        leaf, sample_game = traverse_nodes(node, board, sampled_game, identity_of_bot)
        if len(leaf.untried_actions) == 0:
            win_results = board.points_values(sample_game)[identity_of_bot]
            backpropagate(leaf, win_results)
            continue

        # Expansion
        new_leaf, sample_game = expand_leaf(leaf, board, sample_game)

        # Rollout
        sim_result = rollout(board, sample_game)

        # Backpropagate
        won = False
        if sim_result[identity_of_bot] == 1:
            won = True
        backpropagate(new_leaf, won)

        # Return an action, typically the most frequently used action (from the root) or the action with the best
        # estimated win rate.

    action = None
    win_rate = float('-inf')

    child = root_node.child_nodes
    for moves in child.keys():
        curr_child = child[moves]
        if curr_child.wins > win_rate:
            action = curr_child.parent_action
            win_rate = curr_child.wins

    return action