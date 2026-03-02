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


def player_view_to_tensor(view: PlayerView) -> torch.Tensor:
    """Convert a PlayerView into a flat float32 tensor of shape (248,)."""
    buf = torch.zeros(FEATURE_DIM, dtype=torch.float32)
    pos = 0

    def _write_card_vec(cards, offset):
        for c in cards:
            buf[offset + _card_idx(c)] = 1.0

    def _write_suit_vec(suits, offset):
        for s in suits:
            buf[offset + _SUIT_IDX[s]] = 1.0

    # hand (24)
    _write_card_vec(view.hand, pos); pos += 24

    # trump_suit (4)
    buf[pos + _SUIT_IDX[view.trump_suit]] = 1.0; pos += 4

    # trump_card (24, all-zero if None)
    if view.trump_card is not None:
        buf[pos + _card_idx(view.trump_card)] = 1.0
    pos += 24

    # draw_pile_size (1)
    buf[pos] = view.draw_pile_size / 10.0; pos += 1

    # phase (1)
    buf[pos] = float(view.phase == 2); pos += 1

    # closed_by (2): [me, opponent]
    if view.closed_by is not None:
        if view.closed_by == view.seat:
            buf[pos] = 1.0
        else:
            buf[pos + 1] = 1.0
    pos += 2

    # my_score, opponent_score (1 each, ÷66)
    buf[pos] = view.my_score / 66.0; pos += 1
    buf[pos] = view.opponent_score / 66.0; pos += 1

    # is_leading (1)
    buf[pos] = float(view.is_leading); pos += 1

    # lead_card (24, all-zero if None)
    if view.lead_card is not None:
        buf[pos + _card_idx(view.lead_card)] = 1.0
    pos += 24

    # lead_marriage (4, all-zero if None)
    if view.lead_marriage is not None:
        buf[pos + _SUIT_IDX[view.lead_marriage]] = 1.0
    pos += 4

    # valid_actions: 24-bit play mask + 3 bits (swap/close/pass) = 27
    for a in view.valid_actions:
        if a.type == ActionType.PLAY_CARD and a.card_index is not None:
            card = view.hand[a.card_index]
            buf[pos + _card_idx(card)] = 1.0
        elif a.type == ActionType.SWAP_TRUMP:
            buf[pos + 24] = 1.0
        elif a.type == ActionType.CLOSE_GAME:
            buf[pos + 25] = 1.0
        elif a.type == ActionType.PASS:
            buf[pos + 26] = 1.0
    pos += 27

    # is_winner_action_phase (1)
    buf[pos] = float(view.is_winner_action_phase); pos += 1

    # my_won_cards (24)
    _write_card_vec(view.my_won_cards, pos); pos += 24

    # opponent_won_cards (24)
    _write_card_vec(view.opponent_won_cards, pos); pos += 24

    # my_marriages (4)
    _write_suit_vec(view.my_marriages, pos); pos += 4

    # opponent_marriages (4)
    _write_suit_vec(view.opponent_marriages, pos); pos += 4

    # opponent_known_cards (24)
    _write_card_vec(view.opponent_known_cards, pos); pos += 24

    # opponent_void_suits (4)
    _write_suit_vec(view.opponent_void_suits, pos); pos += 4

    # unknown_cards (24)
    _write_card_vec(view.unknown_cards, pos); pos += 24

    # card_threats (24 floats, normalized ÷6)
    for card, count in view.card_threats.items():
        buf[pos + _card_idx(card)] = count / 6.0
    pos += 24

    # opponent_hand_size (1)
    buf[pos] = view.opponent_hand_size / 6.0; pos += 1

    assert pos == FEATURE_DIM
    return buf
