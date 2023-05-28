# https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo/blob/master/td_prediction.ipynb
# the temporal difference 0 method to find the optimal policy
# only policy evaluation, not optimization

import numpy as np
from gridWorldGame import standard_grid, negative_grid, print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
ALPHA = 0.1


def policyfun():
    # state -> action
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }
    # initial policy
    print("initial policy:")
    print_policy(policy, grid)
    return policy


def random_action(a, eps=0.1):
    # epsilon-soft to ensure all states are visited
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)


def play_game(grid, policy):
    # returns a list of states and corresponding rewards (not returns as in MC)
    # start at the designated start state
    s = (2, 0)
    grid.set_state(s)
    states_and_rewards = [(s, 0)]  # list of tuples of (state, reward)
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))
    return states_and_rewards


def td(gr, policy):
    # repeat until convergence
    for it in range(1000):
        # generate an episode using pi
        states_and_rewards = play_game(grid, policy)
        # the first (s, r) tuple is the state we start in and 0
        # (since we don't get a reward) for simply starting the game
        # the last (s, r) tuple is the terminal state and the final reward
        # the value for the terminal state is by definition 0, so we don't
        # care about updating it.
        for t in range(len(states_and_rewards) - 1):
            s, _ = states_and_rewards[t]
            s2, r = states_and_rewards[t + 1]
            # we will update V(s) AS we experience the episode
            V[s] = V[s] + ALPHA * (r + GAMMA * V[s2] - V[s])


if __name__ == '__main__':
    grid = standard_grid()
    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)
    policy = policyfun()

    # initialize V(s) and returns
    V = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0

    # initial value for all states in grid
    print_values(V, grid)
    td(grid, policy)

    print("final values:")
    print_values(V, grid)
    print("final policy:")
    print_policy(policy, grid)
