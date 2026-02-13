"""
Core data types for Sixty-Six card game.
Pure data — no game logic, no UI dependencies.
"""

from dataclasses import dataclass, field
from enum import Enum


class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"


RANKS = [" 9", " J", " Q", " K", "10", " A"]
RANK_VALUES = {" 9": 0, "10": 10, " J": 2, " Q": 3, " K": 4, " A": 11}


@dataclass
class Card:
    rank: str
    suit: Suit

    def value(self) -> int:
        return RANK_VALUES[self.rank]

    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    def __repr__(self):
        return f"Card({self.rank.strip()}, {self.suit.name})"

    def key(self) -> tuple[str, str]:
        """Hashable key for seen-cards sets."""
        return (self.rank, self.suit.value)


class ActionType(Enum):
    """Types of actions a player can take."""
    PLAY_CARD = "play_card"
    SWAP_TRUMP = "swap_trump"
    CLOSE_GAME = "close_game"
    PASS = "pass"


@dataclass
class Action:
    """A formal action that can be taken by a player."""
    type: ActionType
    card_index: int | None = None
    marriage_suit: Suit | None = None

    def __repr__(self):
        if self.type == ActionType.PLAY_CARD:
            if self.marriage_suit:
                return f"Action(PLAY_CARD, idx={self.card_index}, marriage={self.marriage_suit.name})"
            return f"Action(PLAY_CARD, idx={self.card_index})"
        return f"Action({self.type.name})"


@dataclass
class PlayerView:
    """
    Observable game state from one player's perspective.
    This is what gets passed to a Player agent for decision-making.
    """
    seat: int                          # 0 or 1
    hand: list[Card]
    trump_suit: Suit
    trump_card: Card | None
    draw_pile_size: int
    phase: int                         # 1 or 2
    closed: bool
    closed_by: int | None              # seat that closed, or None
    my_score: int
    opponent_score: int
    is_leading: bool
    lead_card: Card | None             # opponent's card if responding
    lead_marriage: Suit | None         # marriage suit announced with lead card
    valid_actions: list[Action]
    is_winner_action_phase: bool
    seen_cards: set[tuple[str, str]]   # cards this player has observed
    played_cards: set[Card] = field(default_factory=set)  # cards played in completed tricks

    # Additional context for display (not used by agents)
    opponent_hand_size: int = 0
    opponent_hand: list[Card] | None = None   # set when hand is revealed
    last_trick_info: str = ""
    last_drawn: Card | None = None
    match_scores: dict[int, int] = field(default_factory=dict)


@dataclass
class TrickResult:
    """Result of a completed trick."""
    cards: dict[int, Card]             # seat -> card played
    winner: int                        # winning seat
    trick_points: int
    marriages: dict[int, int]          # seat -> marriage points announced


@dataclass
class RoundResult:
    """Result of a completed round."""
    winner: int | None                 # winning seat, or None for tie
    game_points: int
    scores: dict[int, int]             # seat -> trick score
    closed: bool
    closed_by: int | None


@dataclass
class MatchResult:
    """Result of a completed match."""
    winner: int
    game_points: dict[int, int]        # seat -> total game points
    rounds_played: int
