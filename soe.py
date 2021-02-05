# Safe Opponent Explotation algorithms written in python
# The algorithms are ran on Kuhn Poker

import numpy as np
import sys
import itertools
from collections import defaultdict


class InformationSet:
    def __init__(self):
        self.strategy_sum = np.zeros(len(KuhnPoker.Actions))
        self.regret_sum = np.zeros(len(KuhnPoker.Actions))
        self.num_actions = len(KuhnPoker.Actions)

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

class KuhnPoker:
    cards = ['J', 'Q', 'K']
    Actions = ['B', 'C'] 

    @staticmethod
    def is_terminal(history):
        return history in ['BB', 'BC', 'CC', 'CBB', 'CBC']

    @staticmethod
    def get_payoff(history, cards):
        #get payoff with respect to current player
        if history in ['BC', 'CBC']:
            return 1
        else:
            payoff = 2 if 'B' in history else 1
            active_player = len(history) % 2
            player_card = cards[active_player]
            opp_card = cards[(active_player + 1) % 2]
            if player_card == 'K' or opp_card == 'J':
                return payoff
            else:
                return -payoff

    @staticmethod
    def deal(num_of_cards,players=None):
       cards = np.random.choice(KuhnPoker.cards, size=num_of_cards, replace=False) 
       if players is not None:
           for player, card in zip(players, cards):
               players[player].card = card
       return cards

    @staticmethod
    def play_hand(players, history, cards, current_player):
        if KuhnPoker.is_terminal(history):
            payoff = current_player*KuhnPoker.get_payoff(history, cards)
            return payoff
        
        player = players[current_player]
        action = player.get_action(history)
        return KuhnPoker.play_hand(players, history+action, cards, -current_player)

class KuhnPokerTrainer:
    def __init__(self):
        self.infoset_map = {}

    def get_information_set(self, card_plus_history):
        if card_plus_history not in self.infoset_map:
            self.infoset_map[card_plus_history] = InformationSet()
        return self.infoset_map[card_plus_history]

    def cfr(self, cards, history, transition_probs, active_player):
        if KuhnPoker.is_terminal(history):
            return KuhnPoker.get_payoff(history, cards)

        my_card = cards[active_player]
        info_set = self.get_information_set(my_card+history)

        strategy = info_set.get_strategy(transition_probs[active_player])
        opp = (active_player + 1) % 2
        counterfactual_values = np.zeros(len(KuhnPoker.Actions))

        for i, action in enumerate(KuhnPoker.Actions):
            action_probability = strategy[i]

            new_transition_probs = transition_probs.copy()
            new_transition_probs[active_player] *= action_probability

            counterfactual_values[i] = -self.cfr(cards, history+action, new_transition_probs, opp)

        node_value = counterfactual_values.dot(strategy)
        for i, action in enumerate(KuhnPoker.Actions):
            info_set.regret_sum[i] += transition_probs[opp] * (counterfactual_values[i] - node_value)

        return node_value

    def train(self, iterations):
        util = 0
        for _ in range(iterations):
            cards = KuhnPoker.deal(2)
            history = ''
            transition_probs = np.ones(len(KuhnPoker.Actions))
            util += self.cfr(cards, history, transition_probs, 0)
        return util

class Player:
    def __init__(self, strat_map=None, card=None):
        self.strat_map = strat_map
        self.card = card

    def assign_mixed(self):
        for key in self.strat_map:
            num_actions = len(self.strat_map[key])
            self.strat_map[key] = np.ones(num_actions) / num_actions

    def get_action(self, history):
        return np.random.choice(KuhnPoker.Actions, p=self.strat_map[self.card+history])

def print_tree(history, indent):
    if KuhnPoker.is_terminal(history[1:]):
        return
    player = '+' if indent%2==0 else '-'
    strategy = cfr_trainer.infoset_map[history].get_average_strategy()
    print(player, ' '*indent, history, strategy)
    for action in KuhnPoker.Actions:
        print_tree(history+action, indent+1)

def expected_utility(players, history, cards, current_player):
    if KuhnPoker.is_terminal(history):
        payoff = current_player*KuhnPoker.get_payoff(history, cards)
        return KuhnPoker.get_payoff(history, cards)

    player = players[current_player]
    probs = player.strat_map[player.card+history]
    next_utilities = [expected_utility(players, history+a, cards, -current_player) for a in KuhnPoker.Actions]
    return -np.dot(probs, next_utilities)
        
if __name__ == '__main__':
    if len(sys.argv) < 2:
        iterations = 100000
    else:
        iterations = int(sys.argv[1])
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

    cfr_trainer = KuhnPokerTrainer()
    print(f"\nRunning Kuhn Poker chance sampling CFR for {iterations} iterations")
    #util = cfr_trainer.train(iterations)
    util = 0
    print(f"\nExpected average game value (for player 1): {(-1./18):.3f}")
    print(f"Computed average game value               : {(util / iterations):.3f}\n")
    print("We expect the bet frequency for a Jack to be between 0 and 1/3")
    print("The bet frequency of a King should be three times the one for a Jack\n")
    for card in KuhnPoker.cards:
        #print_tree(card, 0)
        print()

    nash = {'Q': np.array([0.00, 1.00]), 'KB': np.array([1.00, 0.00]), 'KC': np.array([1.00, 0.00]), 'QCB': np.array([0.45, 0.55]), 'K': np.array([0.31, 0.69]), 'QB': np.array([0.33, 0.67]), 'QC': np.array([0.00, 1.00]), 'KCB': np.array([1.00, 0.00]), 'JB': np.array([0.00, 1.00]), 'JC': np.array([0.32, 0.68]), 'J': np.array([0.10, 0.90]), 'JCB': np.array([0.00, 1.00])}

    #strat_map = {history: node.get_average_strategy() for history, node in cfr_trainer.infoset_map.items()}
    strat_map = nash
    opp_map = strat_map.copy()
    me = Player(strat_map)
    opp = Player(opp_map)
    #me.assign_mixed()
    players = {1:me, -1:opp}
    expected_util = 0
    for cards in itertools.permutations(KuhnPoker.cards, 2):
        for player, card in zip(players, cards):
            players[player].card = card
        expected_util += 1/6 * expected_utility(players, '', cards, 1)
    print(f"expected utility: {expected_util}")
        

def best_response(s2):
    return np.argmax([expected_utility([1,0,0], s2), expected_utility([0,1,0], s2), expected_utility([0,0,1], s2)])
    

def exploitability(s1):
    return -min([expected_utility(s1, [1,0,0]), expected_utility(s1, [0,1,0]), expected_utility(s1, [0,0,1])])

