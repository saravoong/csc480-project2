import random
import math
from itertools import combinations
from collections import Counter
import sys

class Card:
    # Represents a single playing card 
    ranks = "23456789TJQKA"
    suits = "cdhs"
    rank_values = {rank: i for i, rank in enumerate(ranks)}

    def __init__(self, card_str):
        if len(card_str) != 2:
            raise ValueError("Card string must be 2 characters")
        self.rank_str = card_str[0].upper()
        self.suit_str = card_str[1].lower()

        if self.rank_str not in self.ranks or self.suit_str not in self.suits:
            raise ValueError(f"Invalid card: {card_str}")

        self.rank_value = self.rank_values[self.rank_str]

    def __str__(self):
        return self.rank_str + self.suit_str

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, Card) and self.rank_str == other.rank_str and self.suit_str == other.suit_str

    def __hash__(self):
        return hash((self.rank_str, self.suit_str))

# Helper functions for the deck
def get_deck():
    return [Card(r + s) for r in Card.ranks for s in Card.suits]

# Helper to determine if two hole cards are of the same suit
def is_suited(card1, card2):
    return card1.suit_str == card2.suit_str

def evaluate_hand(five_cards):
    # Core hand ranking logic, evaluates a 5-card poker hand and returns a tuple score for comparison
    # Higher tuple means a better hand
    # Tie-breaking is handled by subsequent elements 
    if len(five_cards) != 5:
        raise ValueError("evaluate_hand requires exactly 5 cards.")
    
    # Sort cards by rank value 
    sorted_cards = sorted(five_cards, key=lambda c: c.rank_value, reverse=True)

    ranks = [c.rank_value for c in sorted_cards]
    suits = [c.suit_str for c in sorted_cards]

    rank_counts = Counter(ranks)
    counts_list = sorted(rank_counts.values(), reverse=True)
    unique_ranks = sorted(rank_counts.keys(), reverse=True)

    is_flush = any(suits.count(s) == 5 for s in Card.suits)
    is_straight, straight_high_rank_value = check_straight(ranks)

    # Texas Hold’em hand ranking
    # 9: Straight Flush / Royal Flush
    if is_flush and is_straight:
        return (9, straight_high_rank_value)
    
    # 8: Four of a Kind
    if counts_list[0] == 4:
        quad_rank = unique_ranks[0] if rank_counts[unique_ranks[0]] == 4 else unique_ranks[1]
        kicker = [r for r in unique_ranks if r != quad_rank][0]
        return (8, quad_rank, kicker)

    # 7: Full House
    if counts_list[0] == 3 and counts_list[1] == 2:
        trip_rank = unique_ranks[0] if rank_counts[unique_ranks[0]] == 3 else unique_ranks[1]
        pair_rank = unique_ranks[0] if rank_counts[unique_ranks[0]] == 2 else unique_ranks[1]
        return (7, trip_rank, pair_rank)

    # 6: Flush
    if is_flush:
        return (6, *unique_ranks) # All 5 ranks for tie-breaking

    # 5: Straight
    if is_straight:
        return (5, straight_high_rank_value)

    # 4: Three of a Kind
    if counts_list[0] == 3:
        trip_rank = unique_ranks[0] if rank_counts[unique_ranks[0]] == 3 else unique_ranks[1]
        kickers = [r for r in unique_ranks if r != trip_rank]
        return (4, trip_rank, *kickers)

    # 3: Two Pair
    if counts_list[0] == 2 and counts_list[1] == 2:
        pair_ranks = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        kicker = [r for r in unique_ranks if r not in pair_ranks][0]
        return (3, *pair_ranks, kicker)

    # 2: One Pair
    if counts_list[0] == 2:
        pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
        kickers = [r for r in unique_ranks if r != pair_rank]
        return (2, pair_rank, *kickers)

    # 1: High Card
    return (1, *unique_ranks) # All 5 ranks for tie-breaking

def check_straight(ranks):
    # Identifies straights, returns the high card of the straight comparison
    # Checks for a straight in a list of 5 rank values.
    # Returns (True, high_rank_value) if a straight is found, otherwise (False, None).
    unique_sorted_ranks = sorted(list(set(ranks))) # Ensure unique and sorted
    
    if len(unique_sorted_ranks) < 5:
        return False, None

    # Check for regular straight
    is_regular_straight = True
    for i in range(len(unique_sorted_ranks) - 1):
        if unique_sorted_ranks[i] + 1 != unique_sorted_ranks[i+1]:
            is_regular_straight = False
            break
    if is_regular_straight:
        return True, unique_sorted_ranks[-1] # High card of the straight

    ace_low_values = {Card.rank_values['2'], Card.rank_values['3'],
                      Card.rank_values['4'], Card.rank_values['5'], Card.rank_values['A']}
    if set(unique_sorted_ranks) == ace_low_values:
        return True, Card.rank_values['5'] # High card is 5 for A2345 straight

    return False, None

def find_best_5_card_hand(hole_cards, community_cards):
    all_7_cards = hole_cards + community_cards
    if len(all_7_cards) < 5:
        return None
    
     # Iterate through all combinations of 5 cards from the 7 available
    return max(evaluate_hand(list(combo)) for combo in combinations(all_7_cards, 5))

class MCTSNode:
    # Represents a node in the Monte Carlo Tree Search tree
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = state.get_possible_actions()

    def ucb1(self, c=math.sqrt(2)):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def select_child(self):
        # Prioritize unvisited children first
        unvisited_children = [child for child in self.children if child.visits == 0]
        if unvisited_children:
            return random.choice(unvisited_children) # Randomly pick one unvisited
        
        # If all children visited, choose based on UCB1
        return max(self.children, key=lambda child: child.ucb1())

    def expand(self):
        if not self.untried_actions:
            raise ValueError("No untried actions to expand.")
        
        action = self.untried_actions.pop() # Take one action from untried
        
        next_state = self.state.perform_action(action)
        child = MCTSNode(next_state, parent=self)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

class PokerState:
    # Defines the current state of the poker game for a given MCTS node
    def __init__(self, my_cards, opp_cards=None, board=None, deck=None):
        self.my = my_cards # List of Card objects
        self.opp = opp_cards if opp_cards is not None else [] # List of Card objects
        self.board = board if board is not None else [] # List of Card objects
        
        if deck:
            self.deck = deck 
        else:
            self.deck = get_deck() # Full deck for the root state

        # Remove cards already in play from the deck
        cards_in_play = set(self.my + self.opp + self.board)
        self.deck = [card for card in self.deck if card not in cards_in_play]

    def get_possible_actions(self):
        # Determine the type of cards to deal next
        # Level 1: 1000 sampled opponent hole card combos
        if not self.opp: # Need to deal opponent's 2 hole cards
            if len(self.deck) < 2: return []
            return random.sample(list(combinations(self.deck, 2)), min(1000, len(self.deck) * (len(self.deck) - 1) // 2))
        # Level 2: 1000 sampled flops (3 cards)
        elif len(self.board) == 0: # Need to deal the 3-card flop
            if len(self.deck) < 3: return []
            return random.sample(list(combinations(self.deck, 3)), min(1000, len(self.deck) * (len(self.deck) - 1) * (len(self.deck) - 2) // 6))
        # Level 3: 1000 sampled turn cards (1 card)
        elif len(self.board) == 3: # Need to deal the 1-card turn
            if not self.deck: return []
            return random.sample(self.deck, min(1000, len(self.deck)))
        # Level 4: 1000 sampled river cards (1 card) – Perform full hand evaluation and propagate the result
        elif len(self.board) == 4: # Need to deal the 1-card river
            if not self.deck: return []
            return random.sample(self.deck, min(1000, len(self.deck)))
        else: # Board is full (5 cards) or already dealt opponent cards
            return []

    def perform_action(self, action):
        # Creates a new PokerState object by adding the action (dealt cards) to the current state and updating the remaining deck
        new_my_cards = list(self.my)
        new_opp_cards = list(self.opp)
        new_board = list(self.board)

        # Create a copy of the current deck for the new state
        new_deck_for_child = list(self.deck)
        
        # Action could be a tuple of cards (for opponent, flop) or a single card (for turn, river)
        if isinstance(action, Card):
            cards_dealt = [action]
        else: # Assume tuple/list of cards
            cards_dealt = list(action)

        for card in cards_dealt:
            if card in new_deck_for_child:
                new_deck_for_child.remove(card)

        if not self.opp: # Action is opponent's hole cards
            new_opp_cards = cards_dealt
        elif len(self.board) == 0: # Action is flop cards
            new_board.extend(cards_dealt)
        else: # Action is turn or river card
            new_board.extend(cards_dealt)

        return PokerState(new_my_cards, new_opp_cards, new_board, new_deck_for_child)

def simulate(state):
    # Performs a random rollout from the given state to the river
    current_board = list(state.board)
    current_opp_cards = list(state.opp)
    
    # Create a temporary deck for this simulation run
    temp_deck = list(state.deck) # Start with the remaining cards from the state's deck
    random.shuffle(temp_deck) # Shuffle for random dealing

    # Deal missing opponent hole cards if not already known
    if not current_opp_cards:
        if len(temp_deck) < 2: return 0 # Not enough cards for opponent
        current_opp_cards = [temp_deck.pop() for _ in range(2)]

    # Deal remaining community cards until 5 are on the board
    while len(current_board) < 5:
        if not temp_deck: return 0 # Not enough cards in deck
        current_board.append(temp_deck.pop())

    # Evaluate hands
    my_final_score = find_best_5_card_hand(state.my, current_board)
    opp_final_score = find_best_5_card_hand(current_opp_cards, current_board)

    if my_final_score > opp_final_score:
        return 1  # Win
    elif my_final_score == opp_final_score:
        return 0.5 # Tie
    else:
        return 0 # Loss

def mcts(root_state, n_sim=1000):
    # The main MCTS algorithm loop
    root = MCTSNode(root_state)
    for _ in range(n_sim):
        node = root
        while node.untried_actions == [] and node.children != []:
            node = node.select_child()
        if node.untried_actions:
            node = node.expand()
        result = simulate(node.state)
        while node:
            node.update(result)
            node = node.parent
    return root.wins / root.visits

def lookup_preflop_table(card1, card2):
    # Ensure card ranks are sorted for consistent lookup
    r1, r2 = sorted([card1.rank_str, card2.rank_str], key=lambda r: Card.rank_values[r], reverse=True)
    suited = is_suited(card1, card2)
    if r1 == r2:
        key = (r1, r2, True)  # force suited to match existing table entry
    else:
        key = (r1, r2, suited)

    table = {
        ("A", "A", True): 86, ("K", "K", True): 83, ("Q", "Q", True): 81, ("J", "J", True): 78, ("T", "T", True): 76, ("9", "9", True): 73, ("8", "8", True): 70, ("7", "7", True): 67, ("6", "6", True): 64, ("5", "5", True): 61, ("4", "4", True): 58, ("3", "3", True): 55, ("2", "2", True): 51,
        ("A", "K", True): 68, ("A", "Q", True): 67, ("A", "J", True): 67, ("A", "T", True): 66, ("A", "9", True): 64, ("A", "8", True): 64, ("A", "7", True): 63, ("A", "6", True): 62, ("A", "5", True): 62, ("A", "4", True): 61, ("A", "3", True): 60, ("A", "2", True): 59,
        ("K", "Q", True): 65, ("K", "J", True): 64, ("K", "T", True): 63, ("K", "9", True): 61, ("K", "8", True): 60, ("K", "7", True): 59, ("K", "6", True): 59, ("K", "5", True): 58, ("K", "4", True): 57, ("K", "3", True): 56, ("K", "2", True): 55,
        ("Q", "J", True): 62, ("Q", "T", True): 61, ("Q", "9", True): 59, ("Q", "8", True): 58, ("Q", "7", True): 56, ("Q", "6", True): 56, ("Q", "5", True): 55, ("Q", "4", True): 54, ("Q", "3", True): 53, ("Q", "2", True): 52,
        ("J", "T", True): 59, ("J", "9", True): 57, ("J", "8", True): 56, ("J", "7", True): 54, ("J", "6", True): 53, ("J", "5", True): 52, ("J", "4", True): 51, ("J", "3", True): 50, ("J", "2", True): 49,
        ("T", "9", True): 56, ("T", "8", True): 54, ("T", "7", True): 53, ("T", "6", True): 51, ("T", "5", True): 49, ("T", "4", True): 49, ("T", "3", True): 48, ("T", "2", True): 47,
        ("9", "8", True): 53, ("9", "7", True): 51, ("9", "6", True): 50, ("9", "5", True): 48, ("9", "4", True): 46, ("9", "3", True): 46, ("9", "2", True): 45,
        ("8", "7", True): 50, ("8", "6", True): 49, ("8", "5", True): 47, ("8", "4", True): 45, ("8", "3", True): 43, ("8", "2", True): 43,
        ("7", "6", True): 48, ("7", "5", True): 46, ("7", "4", True): 44, ("7", "3", True): 42, ("7", "2", True): 40,
        ("6", "5", True): 46, ("6", "4", True): 44, ("6", "3", True): 42, ("6", "2", True): 40,
        ("5", "4", True): 44, ("5", "3", True): 42, ("5", "2", True): 40,
        ("4", "3", True): 41, ("4", "2", True): 39,
        ("3", "2", True): 38,
        ("A", "K", False): 67, ("A", "Q", False): 66, ("A", "J", False): 65, ("A", "T", False): 65, ("A", "9", False): 63, ("A", "8", False): 62, ("A", "7", False): 61, ("A", "6", False): 60, ("A", "5", False): 60, ("A", "4", False): 59, ("A", "3", False): 58, ("A", "2", False): 57,
        ("K", "Q", False): 63, ("K", "J", False): 62, ("K", "T", False): 62, ("K", "9", False): 60, ("K", "8", False): 58, ("K", "7", False): 57, ("K", "6", False): 56, ("K", "5", False): 56, ("K", "4", False): 55, ("K", "3", False): 54, ("K", "2", False): 53,
        ("Q", "J", False): 60, ("Q", "T", False): 59, ("Q", "9", False): 57, ("Q", "8", False): 56, ("Q", "7", False): 54, ("Q", "6", False): 53, ("Q", "5", False): 53, ("Q", "4", False): 52, ("Q", "3", False): 51, ("Q", "2", False): 50,
        ("J", "T", False): 57, ("J", "9", False): 55, ("J", "8", False): 54, ("J", "7", False): 52, ("J", "6", False): 50, ("J", "5", False): 50, ("J", "4", False): 49, ("J", "3", False): 48, ("J", "2", False): 47,
        ("T", "9", False): 54, ("T", "8", False): 52, ("T", "7", False): 50, ("T", "6", False): 49, ("T", "5", False): 47, ("T", "4", False): 46, ("T", "3", False): 45, ("T", "2", False): 44,
        ("9", "8", False): 51, ("9", "7", False): 49, ("9", "6", False): 47, ("9", "5", False): 46, ("9", "4", False): 44, ("9", "3", False): 43, ("9", "2", False): 42,
        ("8", "7", False): 48, ("8", "6", False): 46, ("8", "5", False): 44, ("8", "4", False): 42, ("8", "3", False): 40, ("8", "2", False): 40,
        ("7", "6", False): 45, ("7", "5", False): 44, ("7", "4", False): 42, ("7", "3", False): 40, ("7", "2", False): 38,
        ("6", "5", False): 43, ("6", "4", False): 41, ("6", "3", False): 39, ("6", "2", False): 37,
        ("5", "4", False): 41, ("5", "3", False): 39, ("5", "2", False): 37,
        ("4", "3", False): 38, ("4", "2", False): 36,
        ("3", "2", False): 35,
    }
    
    return table.get(key, None)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Wrong number of arguments, run the program: \npython3 planner.py [Card #1 (e.g. Ks)] [Card #1 (e.g. Jd)]")
        sys.exit(1)

    card1 = sys.argv[1]
    card2 = sys.argv[2]
    my_cards = [Card(card1), Card(card2)]
    mcts_solver_state = PokerState(my_cards)
    estimated_win_probability = mcts(mcts_solver_state, n_sim=1000)
    print(f"Estimated Win Probability: {estimated_win_probability:.2%}")
    expected_from_table = lookup_preflop_table(my_cards[0], my_cards[1])
    if expected_from_table is not None:
        print(f"Expected from table: {expected_from_table}%")
    else:
        print("No exact match in table for these cards (check table data or card order).")
