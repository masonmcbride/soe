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
    Actions = np.array(['B', 'C'])

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

    def train(self, iterations):
        util = 0
        for _ in range(iterations):
            cards = KuhnPoker.deal(2)
            history = ''
            transition_probs = np.ones(len(KuhnPoker.Actions))
            util += self.cfr(cards, history, transition_probs, 0)
        return util

class Player:
    def __init__(self, strat_map=None, card=None, strat_fn=None):
        self.strat_map = strat_map
        self.card = card
        self.strat_fn=strat_fn

    def assign_mixed(self):
        for key in self.strat_map:
            num_actions = len(self.strat_map[key])
            self.strat_map[key] = np.ones(num_actions) / num_actions

    def assign_flipped(self):
        for key in self.strat_map:
            self.strat_map[key] = np.flip(self.strat_map[key])

    def get_action(self, history):
        return np.random.choice(KuhnPoker.Actions, p=self.strat_map[self.card+history])

    def strat_sample(self, history):
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

def wrap_players(p1, p2):
    return {1:p1, -1:p2}

def tau_player(players, history, card):
    best_r = best_response(players)
    """
    last_action = history[0]
    pure = np.zeros(2)
    index = np.where(KuhnPoker.Actions == last_action)[0][0]
    pure[index] = 1
    best_r.strat_map[card+history] = pure
    """
    return best_r

def calc_best_response(node_map, br_strat_map, br_player, cards, history, active_player, prob):
    """
    after chance node, so only decision nodes and terminal nodes left in game tree
    """
    if KuhnPoker.is_terminal(history):
        return KuhnPoker.get_payoff(history, cards)
    key = cards[active_player] + history
    next_player = (active_player + 1) % 2
    if active_player == br_player:
        vals = [calc_best_response(node_map, br_strat_map, br_player, cards, history + action,
            next_player, prob) for action in KuhnPoker.Actions]
        best_response_value = max(vals)
        if key not in br_strat_map:
            br_strat_map[key] = np.array([0.0, 0.0])
        br_strat_map[key] = br_strat_map[key] + prob * np.array(vals, dtype=np.float64)
        return -best_response_value
    else:
        strategy = node_map[key]
        action_values = [calc_best_response(node_map, br_strat_map, br_player, cards,
            history + action, next_player, prob * strategy[ix])
            for ix, action in enumerate(KuhnPoker.Actions)]
        return -np.dot(strategy, action_values)

def best_response(players):
    player = list(players.values())[0]
    br = {}
    for cards in itertools.permutations(KuhnPoker.cards, 2):
        calc_best_response(player.strat_map, br, 0, cards, '', 0, 1)
        calc_best_response(player.strat_map, br, 1, cards, '', 0, 1)
    for k,v in br.items():
        old = br[k]
        pure = np.zeros_like(old)
        pure[old.argmin()] = 1
        v[:] = pure
    return Player(br)

def calc_ev(p1_strat, p2_strat, cards, history, active_player):
    if KuhnPoker.is_terminal(history):
        return KuhnPoker.get_payoff(history, cards)
    my_card = cards[active_player]
    next_player = (active_player + 1) % 2
    if active_player == 0:
        strat = p1_strat[my_card + history]
    else:
        strat = p2_strat[my_card + history]
    next_utilities = [calc_ev(p1_strat, p2_strat, cards, history + a, next_player) \
            for a in KuhnPoker.Actions]
    return -np.dot(strat, next_utilities) 

def ev(p1_strat, p2_strat):
    expected_util = 0
    for c in itertools.permutations(KuhnPoker.cards, 2):
        expected_util += 1/6 * calc_ev(p1_strat, p2_strat, c, '', 0)
    return expected_util

def calc_expected_utility(players, history, cards, current_player):
    if KuhnPoker.is_terminal(history):
        payoff = KuhnPoker.get_payoff(history, cards)
        return KuhnPoker.get_payoff(history, cards)

    player = players[current_player]
    probs = player.strat_map[player.card+history]
    next_utilities = [calc_expected_utility(players, history+a, cards, -current_player) \
            for a in KuhnPoker.Actions]

    return -np.dot(probs, next_utilities)

def expected_utility(players, current_player=1):
    expected_util = 0
    for cards in itertools.permutations(KuhnPoker.cards, 2):
        players[1].card = cards[0]
        players[-1].card = cards[1]
        util = calc_expected_utility(players, '', cards, current_player)
        expected_util += 1/6 * util
    return expected_util

def BEFEWP(players, history, cards, k):
    me = players[1]
    opp = players[-1]
    brp = best_response(players)

    e = V_STAR - ev(me.strat_map, brp.strat_map)
    if e <= k:
        brp.card = cards[1]
        pi = brp
    else:
        pi = opp
    players = wrap_players(brp, me)
    tau = tau_player(players, history, me.card)
    k += V_STAR - ev(tau.strat_map, brp.strat_map)
    action = pi.get_action(history)
    return k, action

if __name__ == '__main__':
    if len(sys.argv) < 2:
        iterations = 10000
    else:
        iterations = int(sys.argv[1])

    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
    heavy = {'J': np.array([0.16, 0.84]), 'KB': np.array([0.97, 0.03]), \
            'KC': np.array([0.70, 0.30]), 'JCB': np.array([1.0, 0.0]), \
            'Q': np.array([0.25, 0.75]), 'JB': np.array([1., 0.]), \
            'JC': np.array([0.30, 0.70]), 'QCB': np.array([0.98, 0.02]), \
            'QB': np.array([0.59, 0.41]), 'QC': np.array([0.12, 0.88]), \
            'K': np.array([0.90, 0.10]), 'KCB': np.array([0.85, 0.15])}

    nash = {'Q': np.array([0.00, 1.00]), 'KB': np.array([1.00, 0.00]), \
            'KC': np.array([1.00, 0.00]), 'QCB': np.array([.1+1/3, 2/3-.1]), \
            'K': np.array([0.30, 0.70]), 'QB': np.array([1/3, 2/3]), \
            'QC': np.array([0.00, 1.00]), 'KCB': np.array([1.00, 0.00]), \
            'JB': np.array([0.00, 1.00]), 'JC': np.array([1/3, 2/3]), \
            'J': np.array([0.10, 0.90]), 'JCB': np.array([0.00, 1.00])}

    me = Player(nash.copy())
    opp = Player(nash.copy())
    #opp.assign_mixed()
    #me.assign_mixed()
    players = wrap_players(me, opp)

    times = 100000
    total_payoff = 0
    k = -500
    fixed = False
    for _ in range(times):
        history = ''
        cards = KuhnPoker.deal(2, players)
        current_player = 1
        while not KuhnPoker.is_terminal(history):
            if fixed or current_player == 1:
                a = players[current_player].get_action(history)
            else:
                k, a = BEFEWP(players, history, cards, k)
            current_player = -current_player
            history += a
        payoff = current_player*KuhnPoker.get_payoff(history, cards)    
        total_payoff += payoff
    print(f"total payoff {total_payoff}")

"""
-------------------------
nash v. fixed  | -5904  |
mixed v. fixed | -16690 |
heavy v. fixed | -34993 |
nash v. soe    | -6000  |
mixed v. soe   | -41855 |
heavy v. soe   | -14482 |
------------------------


"""
#NOTE EMAIL AUTHOR OF PAPER 
