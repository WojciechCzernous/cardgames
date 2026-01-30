#!/usr/bin/env python3
"""
Simple 24-card trick-taking game
Two players: User vs Computer (random)
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui import GameUI


class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"


# ANSI color codes
RED = "\033[91m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[H"

RANKS = [" 9", " J", " Q", " K", "10", " A"]
RANK_VALUES = {" 9": 0, "10": 10, " J": 2, " Q": 3, " K": 4, " A": 11}


@dataclass
class Card:
    rank: str
    suit: Suit

    def __str__(self):
        card_str = f"{self.rank}{self.suit.value}"
        if self.suit in (Suit.HEARTS, Suit.DIAMONDS):
            return f"{RED}{card_str}{RESET}"
        return card_str

    def value(self):
        return RANK_VALUES[self.rank]


class ActionType(Enum):
    """Types of actions a player can take."""
    PLAY_CARD = "play_card"        # Play a card (with optional marriage)
    SWAP_TRUMP = "swap_trump"      # Swap 9-trump with trump card
    CLOSE_GAME = "close_game"      # Close the game early
    PASS = "pass"                  # No action (continue)


@dataclass
class Action:
    """A formal action that can be taken by a player."""
    type: ActionType
    card_index: int | None = None   # Index in hand for PLAY_CARD
    marriage_suit: Suit | None = None  # Suit for marriage announcement
    
    def __repr__(self):
        if self.type == ActionType.PLAY_CARD:
            if self.marriage_suit:
                return f"Action(PLAY_CARD, idx={self.card_index}, marriage={self.marriage_suit.name})"
            return f"Action(PLAY_CARD, idx={self.card_index})"
        return f"Action({self.type.name})"


@dataclass
class GameState:
    """Observable game state for a player."""
    # Player's own hand
    hand: list[Card]
    
    # Visible game info
    trump_suit: Suit
    trump_card: Card | None  # Visible trump card (or None if taken)
    draw_pile_size: int
    phase: int  # 1 or 2
    closed: bool
    closed_by: str | None
    
    # Scores
    my_score: int
    opponent_score: int
    
    # Current trick info
    is_leading: bool
    lead_card: Card | None  # Opponent's card if responding
    
    # Available actions
    valid_actions: list[Action]
    
    # Context
    is_winner_action_phase: bool  # True when choosing swap/close after trick
    
    # Memory of seen cards (for RL - cards observed during round)
    seen_cards: set[tuple[str, str]]  # Set of (rank, suit_value) tuples


def create_deck() -> list[Card]:
    """Create a 24-card deck."""
    deck = []
    for suit in Suit:
        for rank in RANKS:
            deck.append(Card(rank, suit))
    return deck


def card_strength(card: Card, lead_suit: Suit, trump_suit: Suit) -> int:
    """Calculate card strength for comparison."""
    base = RANKS.index(card.rank)
    if card.suit == trump_suit:
        return 100 + base  # Trump cards beat everything
    elif card.suit == lead_suit:
        return 50 + base   # Lead suit cards
    else:
        return base        # Other suits can't win


def display_hand(hand: list[Card], show_numbers: bool = True) -> str:
    """Display a hand of cards."""
    if show_numbers:
        cards_line = "  ".join(str(card) for card in hand)
        numbers_line = "  ".join(f"[{i+1}]" for i in range(len(hand)))
        return f"{cards_line}\n{numbers_line}"
    return "  ".join(str(card) for card in hand)


def colored_suit(suit: Suit) -> str:
    """Return suit symbol with color for red suits."""
    if suit in (Suit.HEARTS, Suit.DIAMONDS):
        return f"{RED}{suit.value}{RESET}"
    return suit.value


def display_hidden_cards(count: int) -> str:
    """Display hidden cards as backs."""
    return " ".join(["[?]"] * count)


class Round:
    """A single round played to 66 points."""
    
    def __init__(self, player_starts: bool = None, ui: "GameUI | None" = None):
        self.ui = ui  # UI can be None for headless/testing
        self.deck = create_deck()
        random.shuffle(self.deck)
        
        # Deal 6 cards to each player
        self.player_hand: list[Card] = []
        self.computer_hand: list[Card] = []
        
        for _ in range(6):
            self.player_hand.append(self.deck.pop())
            self.computer_hand.append(self.deck.pop())
        
        # Trump card (visible) - one card from remaining deck
        self.trump_card = self.deck.pop()
        self.trump_suit = self.trump_card.suit
        
        # Remaining cards form the draw pile (hidden)
        self.draw_pile = self.deck  # 11 cards left
        
        # Scores
        self.player_score = 0
        self.computer_score = 0
        
        # Who leads the next trick
        if player_starts is None:
            self.player_leads = random.choice([True, False])
        else:
            self.player_leads = player_starts
        
        # Track last drawn cards and last trick result
        self.player_last_drawn: Card | None = None
        self.computer_last_drawn: Card | None = None
        self.last_trick_info: str = ""
        
        # Track if round is "closed" (early phase 2 transition)
        self.closed = False
        self.closed_by: str | None = None  # "player" or "computer"
        
        # Track round winner (None until someone wins)
        self.round_winner: str | None = None  # "player" or "computer"
        self.win_score = 66
        
        # Match scores for display
        self.match_scores: dict[str, int] | None = None
        
        # Track cards seen by computer (for RL state)
        # Uses (rank, suit_value) tuples for hashability
        self.computer_seen_cards: set[tuple[str, str]] = set()
        
        # Initialize with computer's starting hand and trump card
        for card in self.computer_hand:
            self.computer_seen_cards.add((card.rank, card.suit.value))
        self.computer_seen_cards.add((self.trump_card.rank, self.trump_card.suit.value))

    def computer_sees_card(self, card: Card):
        """Record that the computer has seen a card."""
        self.computer_seen_cards.add((card.rank, card.suit.value))

    def sort_hand(self, hand: list[Card]):
        """Sort hand by suit, then by rank."""
        hand.sort(key=lambda c: (list(Suit).index(c.suit), RANKS.index(c.rank)))

    @property
    def phase(self) -> int:
        """Return current game phase: 1 = draw pile active, 2 = draw pile empty or closed."""
        if self.closed:
            return 2
        if not self.draw_pile and self.trump_card is None:
            return 2
        return 1

    def clear_screen(self):
        """Clear the terminal screen."""
        if self.ui:
            from ui import TerminalUI
            if isinstance(self.ui, TerminalUI):
                self.ui.clear_screen()

    def display_state(self, computer_card: Card | None = None, player_card: Card | None = None,
                       lead_card: Card | None = None, state: "GameState | None" = None):
        """Display the current game state via UI."""
        if not self.ui:
            return
        if state is None:
            state = self.get_game_state("player", lead_card)
        self.ui.display_state(
            state=state,
            match_scores=self.match_scores,
            computer_card=computer_card,
            player_card=player_card,
            last_trick_info=self.last_trick_info,
            player_last_drawn=self.player_last_drawn
        )

    def get_valid_cards(self, hand: list[Card], lead_card: Card | None) -> list[Card]:
        """Get valid cards that can be played based on current phase."""
        if lead_card is None:
            return hand  # Can play anything when leading
        
        # Phase 1: Any card is allowed
        if self.phase == 1:
            return hand
        
        # Phase 2: Must follow suit -> else must trump -> else any card
        same_suit = [c for c in hand if c.suit == lead_card.suit]
        if same_suit:
            return same_suit
        
        trumps = [c for c in hand if c.suit == self.trump_suit]
        if trumps:
            return trumps
        
        return hand

    def get_valid_actions(self, player: str, lead_card: Card | None = None, 
                          is_winner_action: bool = False) -> list[Action]:
        """Get all valid actions for a player in current state."""
        hand = self.player_hand if player == "player" else self.computer_hand
        actions = []
        
        if is_winner_action:
            # Winner action phase: swap, close, or pass
            if self.has_nine_trump(hand) and self.trump_card:
                actions.append(Action(ActionType.SWAP_TRUMP))
            actions.append(Action(ActionType.CLOSE_GAME))
            actions.append(Action(ActionType.PASS))
            return actions
        
        # Card play phase
        valid_cards = self.get_valid_cards(hand, lead_card)
        marriages = self.get_marriages(hand) if lead_card is None else []
        
        for i, card in enumerate(hand):
            if card in valid_cards:
                # Check if this card can be played with a marriage
                for suit in marriages:
                    if card.suit == suit and card.rank in (" K", " Q"):
                        actions.append(Action(ActionType.PLAY_CARD, card_index=i, marriage_suit=suit))
                # Can always play without announcing marriage
                actions.append(Action(ActionType.PLAY_CARD, card_index=i))
        
        return actions

    def get_game_state(self, player: str, lead_card: Card | None = None,
                       is_winner_action: bool = False) -> GameState:
        """Get observable game state from a player's perspective."""
        if player == "player":
            hand = self.player_hand.copy()
            my_score = self.player_score
            opp_score = self.computer_score
            is_leading = self.player_leads
        else:
            hand = self.computer_hand.copy()
            my_score = self.computer_score
            opp_score = self.player_score
            is_leading = not self.player_leads
        
        valid_actions = self.get_valid_actions(player, lead_card, is_winner_action)
        
        return GameState(
            hand=hand,
            trump_suit=self.trump_suit,
            trump_card=self.trump_card,
            draw_pile_size=len(self.draw_pile),
            phase=self.phase,
            closed=self.closed,
            closed_by=self.closed_by,
            my_score=my_score,
            opponent_score=opp_score,
            is_leading=is_leading and lead_card is None,
            lead_card=lead_card,
            valid_actions=valid_actions,
            is_winner_action_phase=is_winner_action,
            seen_cards=self.computer_seen_cards.copy() if player == "computer" else set()
        )

    def execute_action(self, player: str, action: Action, 
                       lead_card: Card | None = None) -> tuple[Card | None, int]:
        """Execute an action for a player. Returns (card_played, marriage_points)."""
        hand = self.player_hand if player == "player" else self.computer_hand
        
        # Use value comparison to avoid enum identity issues from circular imports
        action_type = action.type.value
        
        if action_type == "swap_trump":
            old_trump = self.trump_card
            self.swap_nine_trump(hand)
            return None, 0
        
        elif action_type == "close_game":
            self.closed = True
            self.closed_by = "you" if player == "player" else "computer"
            return None, 0
        
        elif action_type == "pass":
            return None, 0
        
        elif action_type == "play_card":
            card = hand[action.card_index]
            hand.remove(card)
            
            # Clear last drawn marker
            if player == "player":
                self.player_last_drawn = None
                
                # If player announces marriage, computer sees both K and Q
                if action.marriage_suit:
                    self.computer_seen_cards.add((" K", action.marriage_suit.value))
                    self.computer_seen_cards.add((" Q", action.marriage_suit.value))
            
            marriage_points = 0
            if action.marriage_suit:
                marriage_points = self.marriage_value(action.marriage_suit)
            
            return card, marriage_points
        
        return None, 0

    def player_play(self, lead_card: Card | None, computer_card: Card | None = None) -> tuple[Card, int]:
        """Let the player choose a card to play. Returns (card, marriage_points)."""
        if not self.ui:
            raise RuntimeError("Cannot get player input without UI")
        
        state = self.get_game_state("player", lead_card)
        
        while True:
            # Use the SAME state for display and prompt so indices match
            self.display_state(computer_card=computer_card, lead_card=lead_card, state=state)
            action = self.ui.prompt_card_play(state, computer_card)
            card, marriage_points = self.execute_action("player", action, lead_card)
            if card:
                return card, marriage_points

    def computer_play(self, lead_card: Card | None) -> tuple[Card, int]:
        """Computer plays using action system. Returns (card, marriage_points)."""
        state = self.get_game_state("computer", lead_card)
        action = self.computer_choose_action(state)
        card, marriage_points = self.execute_action("computer", action, lead_card)
        return card, marriage_points

    def computer_choose_action(self, state: GameState) -> Action:
        """Computer's decision logic. Override this for RL agent."""
        valid_actions = state.valid_actions
        
        if state.is_winner_action_phase:
            # Always swap if possible (beneficial)
            swap_actions = [a for a in valid_actions if a.type.value == "swap_trump"]
            if swap_actions:
                return swap_actions[0]
            # Don't close for now
            return Action(ActionType.PASS)
        
        # Card play: prefer marriage if available, else random
        marriage_actions = [a for a in valid_actions if a.marriage_suit]
        if marriage_actions:
            # Prefer trump marriage
            trump_marriages = [a for a in marriage_actions if a.marriage_suit == state.trump_suit]
            if trump_marriages:
                return random.choice(trump_marriages)
            return random.choice(marriage_actions)
        
        # Random card play (no marriage)
        play_actions = [a for a in valid_actions if a.type.value == "play_card"]
        return random.choice(play_actions)

    def play_trick(self) -> bool:
        """Play one trick. Returns True if player won."""
        player_marriage = 0
        computer_marriage = 0
        
        if self.player_leads:
            player_card, player_marriage = self.player_play(None)
            # Computer sees the player's card
            self.computer_sees_card(player_card)
            # If player announced marriage, computer sees both K and Q
            if player_marriage:
                self.player_score += player_marriage
            computer_card, _ = self.computer_play(player_card)
            lead_suit = player_card.suit
        else:
            computer_card, computer_marriage = self.computer_play(None)
            if computer_marriage:
                self.computer_score += computer_marriage
            player_card, _ = self.player_play(computer_card, computer_card)
            # Computer sees the player's response card
            self.computer_sees_card(player_card)
            lead_suit = computer_card.suit
        
        # Show final state with both cards
        self.display_state(computer_card=computer_card, player_card=player_card)
        
        # Check if marriage announcer reached 66 (before trick points)
        if player_marriage and self.player_score >= self.win_score:
            self.round_winner = "player"
        if computer_marriage and self.computer_score >= self.win_score:
            self.round_winner = "computer"
        
        # Determine winner
        player_strength = card_strength(player_card, lead_suit, self.trump_suit)
        computer_strength = card_strength(computer_card, lead_suit, self.trump_suit)
        
        trick_points = player_card.value() + computer_card.value()
        
        if player_strength > computer_strength:
            self.last_trick_info = f"You won +{trick_points} pts ({player_card} beat {computer_card})"
            self.player_score += trick_points
            self.player_leads = True
            winner_is_player = True
        else:
            self.last_trick_info = f"Computer won +{trick_points} pts ({computer_card} beat {player_card})"
            self.computer_score += trick_points
            self.player_leads = False
            winner_is_player = False
        
        # Check if trick winner reached 66
        if self.player_score >= self.win_score:
            self.round_winner = "player"
        elif self.computer_score >= self.win_score:
            self.round_winner = "computer"
        
        # Show result via UI
        if self.ui:
            from ui import TrickResult
            result = TrickResult(
                player_card=player_card,
                computer_card=computer_card,
                winner="player" if winner_is_player else "computer",
                trick_points=trick_points,
                player_marriage=player_marriage,
                computer_marriage=computer_marriage
            )
            self.ui.show_trick_result(result, self.player_score, self.computer_score, self.round_winner)
        
        return winner_is_player

    def has_nine_trump(self, hand: list[Card]) -> Card | None:
        """Check if hand contains 9 of trump suit."""
        for card in hand:
            if card.rank == " 9" and card.suit == self.trump_suit:
                return card
        return None

    def get_marriages(self, hand: list[Card]) -> list[Suit]:
        """Find all marriages (K+Q of same suit) in hand. Returns list of suits."""
        marriages = []
        for suit in Suit:
            has_king = any(c.rank == " K" and c.suit == suit for c in hand)
            has_queen = any(c.rank == " Q" and c.suit == suit for c in hand)
            if has_king and has_queen:
                marriages.append(suit)
        return marriages

    def marriage_value(self, suit: Suit) -> int:
        """Return marriage value: 40 for trump, 20 for others."""
        return 40 if suit == self.trump_suit else 20

    def swap_nine_trump(self, hand: list[Card]) -> bool:
        """Swap 9 of trump with the trump card. Returns True if swapped."""
        nine = self.has_nine_trump(hand)
        if nine and self.trump_card:
            hand.remove(nine)
            hand.append(self.trump_card)
            self.trump_card = nine
            # Computer always sees the new trump card (the 9)
            self.computer_sees_card(nine)
            return True
        return False

    def player_winner_actions(self):
        """Let player perform special actions after winning a trick in phase 1."""
        if not self.ui:
            return
        
        state = self.get_game_state("player", is_winner_action=True)
        
        while True:
            self.display_state()
            action = self.ui.prompt_winner_action(state)
            action_type = action.type.value
            
            if action_type == "swap_trump":
                old_trump = self.trump_card
                self.execute_action("player", action)
                self.ui.show_message(f"Swapped! New trump card: {self.trump_card}")
                # Update state for potential second action
                state = self.get_game_state("player", is_winner_action=True)
            elif action_type == "close_game":
                self.execute_action("player", action)
                self.ui.show_message("Game closed! No more drawing. Phase 2 rules now apply.")
                return
            elif action_type == "pass":
                return

    def computer_winner_actions(self):
        """Computer performs special actions after winning a trick in phase 1."""
        state = self.get_game_state("computer", is_winner_action=True)
        action = self.computer_choose_action(state)
        action_type = action.type.value
        
        if action_type == "swap_trump":
            old_trump = self.trump_card
            self.execute_action("computer", action)
            self.last_trick_info += f" | Swapped 9{colored_suit(self.trump_suit)} for {old_trump}"
        elif action_type == "close_game":
            self.execute_action("computer", action)
            self.last_trick_info += " | Closed the game"

    def draw_cards(self):
        """Both players draw a card if available."""
        self.player_last_drawn = None
        self.computer_last_drawn = None
        
        # No drawing if game is closed
        if self.closed:
            return
        
        if self.draw_pile:
            if self.player_leads:
                drawn = self.draw_pile.pop()
                self.player_hand.append(drawn)
                self.player_last_drawn = drawn
                if self.draw_pile:
                    computer_drawn = self.draw_pile.pop()
                    self.computer_hand.append(computer_drawn)
                    self.computer_sees_card(computer_drawn)
                elif self.trump_card:
                    self.computer_hand.append(self.trump_card)
                    self.computer_sees_card(self.trump_card)
                    self.trump_card = None
            else:
                computer_drawn = self.draw_pile.pop()
                self.computer_hand.append(computer_drawn)
                self.computer_sees_card(computer_drawn)
                if self.draw_pile:
                    drawn = self.draw_pile.pop()
                    self.player_hand.append(drawn)
                    self.player_last_drawn = drawn
                elif self.trump_card:
                    self.player_hand.append(self.trump_card)
                    self.player_last_drawn = self.trump_card
                    self.trump_card = None
        elif self.trump_card:
            # Draw pile empty, trump card goes to winner
            if self.player_leads:
                self.player_hand.append(self.trump_card)
                self.player_last_drawn = self.trump_card
            else:
                self.computer_hand.append(self.trump_card)
                self.computer_sees_card(self.trump_card)
            self.trump_card = None

    def play_round(self, match_scores: dict[str, int] | None = None) -> tuple[str | None, int]:
        """Play a single round. Returns (winner, game_points) or (None, 0) for tie."""
        self.match_scores = match_scores
        
        # Play until someone wins (66 pts) or no cards left
        while self.player_hand and self.computer_hand and not self.round_winner:
            self.play_trick()
            
            # Skip drawing and winner actions if round is over
            if self.round_winner:
                break
                
            self.draw_cards()
            
            # Winner can perform special actions in phase 1 (after drawing)
            if self.phase == 1:
                if self.player_leads:
                    self.player_winner_actions()
                else:
                    self.computer_winner_actions()
        
        # Calculate game points
        return self.calculate_game_points()
    
    def calculate_game_points(self) -> tuple[str | None, int]:
        """Calculate who wins the round and how many game points."""
        # If someone closed the game
        if self.closed:
            if self.closed_by == "you":
                # Player closed - must reach 66 or opponent gets 3 pts
                if self.round_winner == "player":
                    return ("player", 3)
                else:
                    return ("computer", 3)
            else:
                # Computer closed - must reach 66 or player gets 3 pts
                if self.round_winner == "computer":
                    return ("computer", 3)
                else:
                    return ("player", 3)
        
        # Normal finish
        if self.round_winner == "player":
            # Player won - check opponent's score
            if self.computer_score < 33:
                return ("player", 2)  # Opponent < 33
            return ("player", 1)
        elif self.round_winner == "computer":
            # Computer won - check opponent's score
            if self.player_score < 33:
                return ("computer", 2)  # Opponent < 33
            return ("computer", 1)
        else:
            # No one reached 66 - tie (0-0)
            return (None, 0)
    
    def show_round_result(self, winner: str | None, game_points: int, match_scores: dict[str, int]):
        """Display the round result via UI."""
        if not self.ui:
            return
        from ui import RoundResult
        result = RoundResult(
            winner=winner,
            game_points=game_points,
            player_score=self.player_score,
            computer_score=self.computer_score,
            closed=self.closed,
            closed_by=self.closed_by
        )
        self.ui.show_round_result(result, match_scores)


class Match:
    """A match consisting of multiple rounds, first to 7 game points wins."""
    
    def __init__(self, ui: "GameUI | None" = None):
        self.ui = ui
        self.player_game_points = 0
        self.computer_game_points = 0
        self.win_points = 7
        self.round_number = 0
        self.player_starts_next = random.choice([True, False])
    
    def play(self):
        """Play the match until someone reaches 7 game points."""
        if self.ui:
            self.ui.show_welcome()
        
        while self.player_game_points < self.win_points and self.computer_game_points < self.win_points:
            self.round_number += 1
            
            # Create and play a round
            round_game = Round(player_starts=self.player_starts_next, ui=self.ui)
            match_scores = {"player": self.player_game_points, "computer": self.computer_game_points}
            
            winner, points = round_game.play_round(match_scores)
            
            # Update game points
            if winner == "player":
                self.player_game_points += points
            elif winner == "computer":
                self.computer_game_points += points
            
            # Alternate who leads next round
            self.player_starts_next = not self.player_starts_next
            
            # Show round result
            new_scores = {"player": self.player_game_points, "computer": self.computer_game_points}
            round_game.show_round_result(winner, points, new_scores)
            
            # Check if match is over
            if self.player_game_points >= self.win_points or self.computer_game_points >= self.win_points:
                break
            
            if self.ui:
                self.ui.prompt_next_round()
        
        # Match over
        if self.ui:
            from ui import MatchResult
            result = MatchResult(
                winner="player" if self.player_game_points >= self.win_points else "computer",
                player_game_points=self.player_game_points,
                computer_game_points=self.computer_game_points,
                rounds_played=self.round_number
            )
            self.ui.show_match_result(result)


def main():
    from ui import TerminalUI
    ui = TerminalUI()
    
    while True:
        match = Match(ui=ui)
        match.play()
        
        if not ui.prompt_play_again():
            break


if __name__ == "__main__":
    main()
