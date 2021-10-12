#coded by Jonathan Lewis and Adam Carter
from mcts_node import MCTSNode
from random import choice
from math import sqrt, log

num_nodes = 100 #make sure to change back
explore_faction = 2.

def ucb(node):
    if node.visits == 0: 
        return float("inf")
    '''
    NOTES: Adversarial planning – the bot will be simulating both players’ turns.
    This requires you to alter the UCT function (during the tree traversal/selection phase)
    on the opponent’s turn. Remember: the opponent’s win rate (Xj) = (1 – bot’s win rate).

    Assuming 1 = MCTS_Vanilla and 2 = bot
    '''
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
            if turn == True:   # in p2_sim.py, red == 1
                score = ucb(curr_node)
            elif turn == False:  # in p2_sim.py, blue == 2
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

    """

    if board.is_ended(state):
        return board.points_values(state)
    else:
        new_state = board.next_state(state, choice(board.legal_actions(state)))
        return rollout(board, new_state)


def backpropagate(node, won):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    if node.parent is None:
        node.wins += won
        node.visits+=1
        return

    else:
        node.wins += won
    node.visits+=1
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
        sampled_game = state

        node = root_node

        leaf, sample_game = traverse_nodes(node, board, sampled_game, identity_of_bot)
        if len(leaf.untried_actions)==0:
            win_results = board.points_values(sample_game)[identity_of_bot]
            backpropagate(leaf, win_results)
            continue
        new_leaf, sample_game = expand_leaf(leaf, board, sample_game)

        sim_result = rollout(board, sample_game)

        won = False
        if sim_result[identity_of_bot] == 1:
            won = True
        backpropagate(new_leaf, won)
        
    action = None
    win_rate = float('-inf')   # -1 = loss, 0 = draw, 1 = win

    child = root_node.child_nodes
    for moves in child.keys():
        curr_child = child[moves]
        if curr_child.wins > win_rate:
            action = curr_child.parent_action
            win_rate = curr_child.wins

    return action