# Safe Opponent Explotation algorithms written in python
# The algorithms are ran on Kuhn Poker

import numpy as np
import itertools
from collections import defaultdict

ROCK, PAPER, SCISSORS = range(3)
A = list(itertools.permutations(range(3), 2))
print(A)

class InformationSet:
    def __init__(self):
        self.strategy_sum = np.zeros(len(Actions))
        self.regret_sum = np.zeros(len(Actions))
        self.num_actions = len(Actions)

    def normalize(self, strategy):
        if sum(strategy) > 0:
            return strategy/sum(strategy)
        else:
            return np.ones(self.num_actions)/self.num_actions

    def get_strategy(self, transition_prob):
        strategy = np.maximum(self.regret_sum, 0)
        strategy = self.normalize(strategy)

        self.strategy_sum += transition_prob * strategy
        return strategy

    def get_average_strategy(self):
        return self.normalize(self.strategy_sum)

class RPS:
    
    A = np.array(list(itertools.product(range(3), repeat=2))) 

    @staticmethod
    def is_terminal():
        #we don't need for RPS
        pass
    
    @staticmethod
    def get_payoff(a1, a2):
        mod3 = (a1 - a2) % 3
        if mod3 == 2:
            return -1
        else:
            return mod3

def expected_utility(s1, s2):
    utility = np.array([RPS.get_payoff(a1, a2) for a1, a2 in RPS.A])
    probs = np.array([s1[a1]*s2[a2] for a1, a2 in RPS.A])
    return sum(utility*probs)

def best_response(s2):
    return np.argmax([expected_utility([1,0,0], s2), expected_utility([0,1,0], s2), expected_utility([0,0,1], s2)])
    

def exploitability(s1):
    return -min([expected_utility(s1, [1,0,0]), expected_utility(s1, [0,1,0]), expected_utility(s1, [0,0,1])])

mixed = np.ones(3) / 3
pure = np.array([1,0,0])
heavy = np.array([.3, .4, .3])

br = best_response(mixed)
print(f"pure: {exploitability(pure)}")
print(f"mixed: {exploitability(mixed)}")
print(f"heavy: {exploitability(heavy)}")

