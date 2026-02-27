"""
Rules oracle for Sixty-Six.
Stateless pure functions that determine what is legal given a state.
No side effects — the game engine applies the results.
"""

from models import Card, Suit, Action, ActionType, RANKS, RANK_VALUES


# ---------------------------------------------------------------------------
# Card evaluation
# ---------------------------------------------------------------------------

def card_strength(card: Card, lead_suit: Suit, trump_suit: Suit) -> int:
    """Calculate card strength for trick comparison."""
    base = RANKS.index(card.rank)
    if card.suit == trump_suit:
        return 100 + base
    elif card.suit == lead_suit:
        return 50 + base
    return base


def trick_winner(card_a: Card, card_b: Card, lead_suit: Suit, trump_suit: Suit) -> int:
    """Return 0 if card_a wins, 1 if card_b wins."""
    return 0 if card_strength(card_a, lead_suit, trump_suit) > card_strength(card_b, lead_suit, trump_suit) else 1


# ---------------------------------------------------------------------------
# Hand queries
# ---------------------------------------------------------------------------

def find_nine_trump(hand: list[Card], trump_suit: Suit) -> Card | None:
    """Return the 9 of trump from hand, or None."""
    for card in hand:
        if card.rank == " 9" and card.suit == trump_suit:
            return card
    return None


def find_marriages(hand: list[Card]) -> list[Suit]:
    """Find all marriages (K + Q of same suit) in hand."""
    marriages = []
    for suit in Suit:
        has_king = any(c.rank == " K" and c.suit == suit for c in hand)
        has_queen = any(c.rank == " Q" and c.suit == suit for c in hand)
        if has_king and has_queen:
            marriages.append(suit)
    return marriages


def marriage_value(suit: Suit, trump_suit: Suit) -> int:
    """Marriage point value: 40 for trump suit, 20 otherwise."""
    return 40 if suit == trump_suit else 20


# ---------------------------------------------------------------------------
# Valid cards / actions
# ---------------------------------------------------------------------------

def get_valid_cards(hand: list[Card], lead_card: Card | None, trump_suit: Suit,
                    phase: int) -> list[Card]:
    """
    Which cards from *hand* may legally be played?
    Phase 1 (draw pile active): any card.
    Phase 2 (closed / draw pile gone): must follow suit → must trump → any.
    """
    if lead_card is None or phase == 1:
        return list(hand)

    # Phase 2 obligations
    same_suit = [c for c in hand if c.suit == lead_card.suit]
    if same_suit:
        return same_suit

    trumps = [c for c in hand if c.suit == trump_suit]
    if trumps:
        return trumps

    return list(hand)


def get_valid_actions(hand: list[Card], trump_suit: Suit, trump_card: Card | None,
                      lead_card: Card | None, phase: int,
                      is_winner_action: bool = False) -> list[Action]:
    """
    All legal actions for the active player.
    The game engine is responsible for calling this at the right moment.
    """
    actions: list[Action] = []

    if is_winner_action:
        if find_nine_trump(hand, trump_suit) and trump_card:
            actions.append(Action(ActionType.SWAP_TRUMP))
        actions.append(Action(ActionType.CLOSE_GAME))
        actions.append(Action(ActionType.PASS))
        return actions

    valid_cards = get_valid_cards(hand, lead_card, trump_suit, phase)

    for i, card in enumerate(hand):
        if card in valid_cards:
            actions.append(Action(ActionType.PLAY_CARD, card_index=i))

    return actions


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

WIN_SCORE = 66


def compute_game_points(scores: dict[int, int], winner: int | None,
                        closed: bool, closed_by: int | None) -> tuple[int | None, int]:
    """
    Given round outcome, return (winning_seat, game_points).
    """
    if closed:
        closer = closed_by
        opponent = 1 - closer
        if winner == closer:
            return (closer, 3)
        else:
            return (opponent, 3)

    if winner is not None:
        opponent = 1 - winner
        if scores[opponent] < 33:
            return (winner, 2)
        return (winner, 1)

    return (None, 0)


def sort_hand(hand: list[Card]):
    """Sort a hand in-place by suit then rank."""
    hand.sort(key=lambda c: (list(Suit).index(c.suit), RANKS.index(c.rank)))
