"""
Fast conversion from PlayerView to a flat feature tensor (248 dims).
No game logic — pure numeric mapping.
"""

from __future__ import annotations

import torch

from models import (
    Card, Suit, Action, ActionType, PlayerView,
    RANKS, RANK_VALUES,
)

# Pre-computed index maps (module-level, built once)
_SUITS = list(Suit)
_SUIT_IDX = {s: i for i, s in enumerate(_SUITS)}
_RANK_IDX = {r: i for i, r in enumerate(RANKS)}
_NUM_CARDS = len(RANKS) * len(_SUITS)          # 24

def _card_idx(card: Card) -> int:
    """Card → integer in [0, 23].  Order: suit-major, rank-minor."""
    return _SUIT_IDX[card.suit] * len(RANKS) + _RANK_IDX[card.rank]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FEATURE_DIM = 248
ACTION_DIM = 27
VALID_ACTIONS_OFFSET = 87   # position of the 27-bit valid-actions mask in the state tensor


def _card_idx(card: Card) -> int:
    """Card → integer in [0, 23].  Order: suit-major, rank-minor."""
    return _SUIT_IDX[card.suit] * len(RANKS) + _RANK_IDX[card.rank]


def action_to_index(action: Action, hand: list[Card]) -> int:
    """Map an Action to an integer in [0, 26]."""
    if action.type == ActionType.PLAY_CARD and action.card_index is not None:
        return _card_idx(hand[action.card_index])
    if action.type == ActionType.SWAP_TRUMP:
        return 24
    if action.type == ActionType.CLOSE_GAME:
        return 25
    return 26  # PASS


def index_to_action(index: int, view: PlayerView) -> Action:
    """Map an action index back to an Action for the given view."""
    if index == 24:
        return Action(ActionType.SWAP_TRUMP)
    if index == 25:
        return Action(ActionType.CLOSE_GAME)
    if index == 26:
        return Action(ActionType.PASS)
    target_suit = _SUITS[index // len(RANKS)]
    target_rank = RANKS[index % len(RANKS)]
    for i, c in enumerate(view.hand):
        if c.rank == target_rank and c.suit == target_suit:
            return Action(ActionType.PLAY_CARD, card_index=i)
    raise ValueError(f"Card at index {index} not in hand")


# ---------------------------------------------------------------------------
# State tensor
# ---------------------------------------------------------------------------

def player_view_to_tensor(view: PlayerView) -> torch.Tensor:
    """Convert a PlayerView into a flat float32 tensor of shape (248,)."""
    buf = [0.0] * FEATURE_DIM
    pos = 0

    # hand (24)
    for c in view.hand:
        buf[pos + _card_idx(c)] = 1.0
    pos += 24

    # trump_suit (4)
    buf[pos + _SUIT_IDX[view.trump_suit]] = 1.0
    pos += 4

    # trump_card (24, all-zero if None)
    if view.trump_card is not None:
        buf[pos + _card_idx(view.trump_card)] = 1.0
    pos += 24

    # draw_pile_size (1)
    buf[pos] = view.draw_pile_size * 0.1
    pos += 1

    # phase (1)
    buf[pos] = float(view.phase == 2)
    pos += 1

    # closed_by (2): [me, opponent]
    if view.closed_by is not None:
        buf[pos + (0 if view.closed_by == view.seat else 1)] = 1.0
    pos += 2

    # my_score, opponent_score (1 each, ÷66)
    buf[pos] = view.my_score / 66.0
    buf[pos + 1] = view.opponent_score / 66.0
    pos += 2

    # is_leading (1)
    buf[pos] = float(view.is_leading)
    pos += 1

    # lead_card (24, all-zero if None)
    if view.lead_card is not None:
        buf[pos + _card_idx(view.lead_card)] = 1.0
    pos += 24

    # lead_marriage (4, all-zero if None)
    if view.lead_marriage is not None:
        buf[pos + _SUIT_IDX[view.lead_marriage]] = 1.0
    pos += 4

    # valid_actions: 24-bit play mask + 3 bits (swap/close/pass) = 27
    _hand = view.hand
    for a in view.valid_actions:
        at = a.type
        if at == ActionType.PLAY_CARD:
            buf[pos + _card_idx(_hand[a.card_index])] = 1.0
        elif at == ActionType.SWAP_TRUMP:
            buf[pos + 24] = 1.0
        elif at == ActionType.CLOSE_GAME:
            buf[pos + 25] = 1.0
        else:  # PASS
            buf[pos + 26] = 1.0
    pos += 27

    # is_winner_action_phase (1)
    buf[pos] = float(view.is_winner_action_phase)
    pos += 1

    # my_won_cards (24)
    for c in view.my_won_cards:
        buf[pos + _card_idx(c)] = 1.0
    pos += 24

    # opponent_won_cards (24)
    for c in view.opponent_won_cards:
        buf[pos + _card_idx(c)] = 1.0
    pos += 24

    # my_marriages (4)
    for s in view.my_marriages:
        buf[pos + _SUIT_IDX[s]] = 1.0
    pos += 4

    # opponent_marriages (4)
    for s in view.opponent_marriages:
        buf[pos + _SUIT_IDX[s]] = 1.0
    pos += 4

    # opponent_known_cards (24)
    for c in view.opponent_known_cards:
        buf[pos + _card_idx(c)] = 1.0
    pos += 24

    # opponent_void_suits (4)
    for s in view.opponent_void_suits:
        buf[pos + _SUIT_IDX[s]] = 1.0
    pos += 4

    # unknown_cards (24)
    for c in view.unknown_cards:
        buf[pos + _card_idx(c)] = 1.0
    pos += 24

    # card_threats (24 floats, normalized ÷6)
    _inv6 = 1.0 / 6.0
    for card, count in view.card_threats.items():
        buf[pos + _card_idx(card)] = count * _inv6
    pos += 24

    # opponent_hand_size (1)
    buf[pos] = view.opponent_hand_size / 6.0

    return torch.tensor(buf, dtype=torch.float32)


def sample_transition(transitions: list[tuple[PlayerView, Action, int]]
                      ) -> tuple[torch.Tensor, int]:
    """Pick one random (state_tensor, action_index) from a round's transitions."""
    import random
    view, action, _seat = random.choice(transitions)
    return player_view_to_tensor(view), action_to_index(action, view.hand)
