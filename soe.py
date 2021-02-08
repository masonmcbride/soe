# Safe Opponent Explotation algorithms written in python
# The algorithms are ran on Kuhn Poker

import numpy as np
import sys
import itertools
from collections import defaultdict

V_STAR = -1/18

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

    def to_strat_map(self):
        return {history: node.get_average_strategy() for history, node in self.infoset_map.items()}

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

    def br(self, players, cards, history, transition_probs, current_player):
        if KuhnPoker.is_terminal(history):
            return KuhnPoker.get_payoff(history, cards)

        player = players[current_player]
        opp = players[-current_player]

        info_set = self.get_information_set(player.card+history)
        if current_player == -1:
            strategy = player.strat_map[player.card+history]
        else:
            strategy = info_set.get_strategy(transition_probs[current_player])

        counterfactual_values = np.zeros(len(KuhnPoker.Actions))
        for i, action in enumerate(KuhnPoker.Actions):
            action_probability = strategy[i]

            new_transition_probs = transition_probs.copy()
            new_transition_probs[current_player] *= action_probability

            counterfactual_values[i] = -self.br(players, cards, history+action, \
                                                            new_transition_probs, -current_player)
            
        node_value = counterfactual_values.dot(strategy)
        if current_player != -1:
            for i, action in enumerate(KuhnPoker.Actions):
                info_set.regret_sum[i] += transition_probs[-current_player] * \
                                            (counterfactual_values[i] - node_value)
        return node_value

    def train(self, iterations):
        util = 0
        for _ in range(iterations):
            cards = KuhnPoker.deal(2)
            history = ''
            transition_probs = np.ones(len(KuhnPoker.Actions))
            util += self.cfr(cards, history, transition_probs, 0)
        return util

    def best_response(self, players, iterations):
        util = 0
        for _ in range(iterations):
            cards = KuhnPoker.deal(2, players)
            history = ''
            transition_probs = np.ones(len(KuhnPoker.Actions))
            util += self.br(players, cards, history, transition_probs, 1)
        return util


class Player:
    def __init__(self, strat_map=None, card=None):
        self.strat_map = strat_map
        self.card = card

    def assign_mixed(self):
        for key in self.strat_map:
            num_actions = len(self.strat_map[key])
            self.strat_map[key] = np.ones(num_actions) / num_actions

    def assign_flipped(self):
        for key in self.strat_map:
            self.strat_map[key] = np.flip(self.strat_map[key])

    def get_action(self, history):
        return np.random.choice(KuhnPoker.Actions, p=self.strat_map[self.card+history])

    def print_strat_map(self):
        def print_tree(history, indent):
            if KuhnPoker.is_terminal(history[1:]):
                return
            player = '+' if indent%2==0 else '-'
            strategy = self.strat_map[history]
            print(player, ' '*indent, history, strategy)
            for action in KuhnPoker.Actions:
                print_tree(history+action, indent+1)
        for card in KuhnPoker.cards:
            print_tree(card, 0)
            print()

def expected_utility(players, history, cards, current_player):
    if KuhnPoker.is_terminal(history):
        return KuhnPoker.get_payoff(history, cards)

    player = players[current_player]
    probs = player.strat_map[player.card+history]
    next_utilities = [expected_utility(players, history+a, cards, -current_player) for a in KuhnPoker.Actions]
    return -np.dot(probs, next_utilities)

def BEFEWP(players, history, cards, last_action, k):
    tau = 0
    k += 0
    br = KuhnPokerTrainer()
    br.best_response(players, 100000)
    best_response = br.to_strat_map()
    epsilon = V_STAR - exploitability #TODO
    if epsilon <= k:
        pi = best_response
    else:
        pi = player.strat_map

    return np.random.choice(KuhnPoker.Actions, p=pi[player.card+history])

def whole_expected_utility():
    expected_util = 0
    for cards in itertools.permutations(KuhnPoker.cards, 2):
        for player, card in zip(players, cards):
            players[player].card = card
        expected_util += 1/6 * expected_utility(players, '', cards, 1)
    print(f"expected utility: {expected_util}")
        
if __name__ == '__main__':
    if len(sys.argv) < 2:
        iterations = 100000
    else:
        iterations = int(sys.argv[1])
        
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

    """ 
    print(f"\nRunning Kuhn Poker chance sampling CFR for {iterations} iterations")
    cfr_trainer = KuhnPokerTrainer()
    util = cfr_trainer.train(iterations)
    strat_map = {history: node.get_average_strategy() \
            for history, node in cfr_trainer.infoset_map.items()}
    print(cfr_trainer.infoset_map['J'].regret_sum)
    """
   

    nash = {'Q': np.array([0.00, 1.00]), 'KB': np.array([1.00, 0.00]), \
            'KC': np.array([1.00, 0.00]), 'QCB': np.array([0.45, 0.55]), \
            'K': np.array([0.31, 0.69]), 'QB': np.array([0.33, 0.67]), \
            'QC': np.array([0.00, 1.00]), 'KCB': np.array([1.00, 0.00]), \
            'JB': np.array([0.00, 1.00]), 'JC': np.array([0.32, 0.68]), \
            'J': np.array([0.10, 0.90]), 'JCB': np.array([0.00, 1.00])}
    strat_map = nash
    opp_map = strat_map.copy()
    me = Player(strat_map)
    opp = Player(opp_map)
    opp.assign_mixed()
    players = {1:me, -1:opp}

    br = KuhnPokerTrainer()
