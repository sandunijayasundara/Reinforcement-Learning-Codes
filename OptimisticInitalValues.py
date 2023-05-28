import logging as logger
from abc import (
    ABC,
    abstractmethod,
)
from collections import defaultdict
from typing import List
from uuid import uuid4
import numpy as np
import matplotlib.pyplot as plt
from EpsilonGreedy import EpsilonGreedyAgent


class NoBanditsError(Exception):
    ...


class Bandit:
    def __init__(self, m: float, lower_bound: float = None, upper_bound: float = None):
        """
        Simulates bandit.
        Args:
            m (float): True mean.
            lower_bound (float): Lower bound for rewards.
            upper_bound (float): Upper bound for rewards.
        """

        self.m = m
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.id = uuid4()

    def pull(self):
        """
        Simulate pulling the arm of the bandit.
        Normal distribution with mu = self.m and sigma = 1. If lower_bound or upper_bound are defined then the
        distribution will be truncated.
        """
        n = 10
        possible_rewards = np.random.randn(n) + self.m

        allowed = np.array([True] * n)
        if self.lower_bound is not None:
            allowed = possible_rewards >= self.lower_bound
        if self.upper_bound is not None:
            allowed *= possible_rewards <= self.upper_bound

        return possible_rewards[allowed][0]


class BanditRewardsLog:
    def __init__(self):
        self.total_actions = 0
        self.total_rewards = 0
        self.all_rewards = []
        self.record = defaultdict(lambda: dict(actions=0, reward=0))

    def record_action(self, bandit, reward):
        self.total_actions += 1
        self.total_rewards += reward
        self.all_rewards.append(reward)
        self.record[bandit.id]['actions'] += 1
        self.record[bandit.id]['reward'] += reward

    def __getitem__(self, bandit):
        return self.record[bandit.id]


class Agent(ABC):
    def __init__(self):
        self.rewards_log = BanditRewardsLog()
        self._bandits = None

    @property
    def bandits(self) -> List[Bandit]:
        if not self._bandits:
            raise NoBanditsError()
        return self._bandits

    @bandits.setter
    def bandits(self, val: List[Bandit]):
        self._bandits = val

    @abstractmethod
    def take_action(self):
        ...

    def take_actions(self, n: int):
        for _ in range(n):
            self.take_action()


class OptimisticInitialValuesAgent(Agent):
    def __init__(self, max_reward: float):
        super().__init__()
        self.max_reward = max_reward

    def _get_current_best_bandit(self) -> Bandit:
        estimates = []
        for bandit in self.bandits:
            bandit_record = self.rewards_log[bandit]
            if not bandit_record['actions']:
                estimates.append(self.max_reward)
            else:
                estimates.append(
                    (self.max_reward + bandit_record['reward']) / (1 + bandit_record['actions']),
                )

        return self.bandits[np.argmax(estimates)]

    def take_action(self):
        current_bandit = self._get_current_best_bandit()
        reward = current_bandit.pull()
        self.rewards_log.record_action(current_bandit, reward)
        return reward

    def __repr__(self):
        return 'OptimisticInitialValuesAgent(max_reward={})'.format(self.max_reward)


def compare_agents(agents: List[Agent], bandits: List[Bandit], iterations: int, show_plot=True):
    for agent in agents:
        logger.info("Running for agent = %s", agent)
        agent.bandits = bandits
        agent.take_actions(iterations)
        if show_plot:
            plt.plot(np.cumsum(agent.rewards_log.all_rewards), label=str(agent))

    if show_plot:
        plt.xlabel("iteration")
        plt.ylabel("total rewards")
        plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
        plt.show()


if __name__ == '__main__':
    # how optimistic should  it be
    bandits = [
        Bandit(m=mu, lower_bound=0, upper_bound=10)
        for mu in [3, 5, 7, 9]
    ]
    agents = [
        OptimisticInitialValuesAgent(max_reward=r)
        for r in [15, 20, 50, 100, 1000]
    ]
    iterations = 2000
    compare_agents(agents, bandits, iterations)

    # Optimistic is better than Epsilon Greedy
    agents = [
        EpsilonGreedyAgent(bandits),
        OptimisticInitialValuesAgent(max_reward=20),
    ]
    iterations = 2000
    compare_agents(agents, bandits, iterations)