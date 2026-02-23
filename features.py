"""
Canonical suit mapping and card-set encoding for RL state representation.

Exploits suit symmetry: all non-trump suits are equivalent, so we
canonicalise them to reduce the state space.

Canonical suit indices:
    0 — trump suit (always)
    1, 2, 3 — remaining suits sorted by hand-hash (descending)

Hand-hash for a suit (from cards in the player's hand):
    1·[has 9] + 2·[has J] + 4·[has Q] + 8·[has K] + 16·[has 10] + 32·[has A]

This gives a deterministic, player-perspective-invariant ordering.
"""

from __future__ import annotations

import numpy as np

from models import Card, Suit, RANKS

# Bit weight for each rank index: 2^i
_RANK_BIT = {rank: (1 << i) for i, rank in enumerate(RANKS)}
#  " 9" -> 1,  " J" -> 2,  " Q" -> 4,  " K" -> 8,  "10" -> 16,  " A" -> 32


def suit_hash(hand: list[Card], suit: Suit) -> int:
    """Compute the canonical hash of a suit from the cards in hand."""
    h = 0
    for card in hand:
        if card.suit == suit:
            h |= _RANK_BIT[card.rank]
    return h


def build_suit_map(
    hand: list[Card], trump_suit: Suit
) -> tuple[dict[Suit, int], dict[int, Suit]]:
    """
    Build canonical suit mappings from a hand and trump suit.

    Returns:
        m1: Suit -> canonical index (0–3)
        m2: canonical index -> Suit
    """
    m1: dict[Suit, int] = {trump_suit: 0}

    # Remaining suits sorted by hand-hash (descending), with suit enum value
    # as a tiebreaker for determinism when hashes collide.
    others = sorted(
        (s for s in Suit if s != trump_suit),
        key=lambda s: (-suit_hash(hand, s), s.value),
    )
    for i, s in enumerate(others, start=1):
        m1[s] = i

    m2: dict[int, Suit] = {idx: s for s, idx in m1.items()}
    return m1, m2


def cards_to_matrix(
    m1: dict[Suit, int], cards: set[Card] | list[Card]
) -> np.ndarray:
    """
    Encode a set of cards as a 6×4 binary matrix (uint8).

    Rows  = ranks  (in RANKS order: 9, J, Q, K, 10, A)
    Cols  = canonical suit indices (0 = trump, 1–3 = sorted others)

    Returns an ndarray of shape (6, 4), dtype uint8.
    """
    mat = np.zeros((6, 4), dtype=np.uint8)
    for card in cards:
        rank_idx = RANKS.index(card.rank)
        suit_idx = m1[card.suit]
        mat[rank_idx, suit_idx] = 1
    return mat


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def cards_to_flat(m1: dict[Suit, int], cards: set[Card] | list[Card]) -> np.ndarray:
    """Same as cards_to_matrix but flattened to a 24-element vector (uint8)."""
    return cards_to_matrix(m1, cards).ravel()


def _won_plane(
    m1: dict[Suit, int],
    won_cards: list[Card],
    marriages: set[Suit],
    closed_by_me: bool,
) -> np.ndarray:
    """
    Encode one player's won-cards plane: 24 card bits + 4 marriage bits + 1 closed bit = 29.
    """
    plane = np.zeros(29, dtype=np.uint8)
    plane[:24] = cards_to_flat(m1, won_cards)
    for suit in marriages:
        plane[24 + m1[suit]] = 1
    if closed_by_me:
        plane[28] = 1
    return plane


# ---------------------------------------------------------------------------
# Full state encoding
# ---------------------------------------------------------------------------

def encode_state(view: "PlayerView") -> np.ndarray:
    """
    Encode a PlayerView into a fixed-size uint8 vector for RL.

    Layout (all bits use canonical suit mapping):
        [0..28]   — cards won by me         (24) + marriages (4) + closed (1)
        [29..57]  — cards won by opponent   (24) + marriages (4) + closed (1)
        [58..81]  — my hand                 (24)
        [82..105] — visible trump card      (24)  (all zeros if none)
        [106..129]— card on table           (24)  (all zeros if none)

    Total: 130 uint8.

    Cards-not-yet-seen can be recovered as the complement of the union
    of all five 24-bit card planes against the full 24-card deck (all ones).
    """
    from models import PlayerView  # deferred to avoid circular import

    m1, _ = build_suit_map(view.hand, view.trump_suit)

    # Plane 0: my won cards + marriages + closed
    closed_by_me = view.closed and view.closed_by == view.seat
    p0 = _won_plane(m1, view.won_cards_me, view.marriages_me, closed_by_me)

    # Plane 1: opponent won cards + marriages + closed
    closed_by_opp = view.closed and view.closed_by == (1 - view.seat)
    p1 = _won_plane(m1, view.won_cards_opp, view.marriages_opp, closed_by_opp)

    # Plane 2: my hand (24)
    p2 = cards_to_flat(m1, view.hand)

    # Plane 3: visible trump card (24) — zeros when face-down or taken
    if view.trump_card is not None:
        p3 = cards_to_flat(m1, [view.trump_card])
    else:
        p3 = np.zeros(24, dtype=np.uint8)

    # Plane 4: card on the table (24) — only when responding to a lead
    if view.lead_card is not None:
        p4 = cards_to_flat(m1, [view.lead_card])
    else:
        p4 = np.zeros(24, dtype=np.uint8)

    return np.concatenate([p0, p1, p2, p3, p4])


# ---------------------------------------------------------------------------
# Action encoding — 48 flat actions
# ---------------------------------------------------------------------------
#
#   0–23: play card at canonical position (rank_idx * 4 + suit_idx)
#  24–47: close the game, then play card at canonical position
#
# Closing is only legal when leading in phase 1.
# Marriage announcement is implicit — if the played card triggers a marriage,
# it is announced automatically.  Trump-9 swap is always executed if possible.

NUM_ACTIONS = 48


def card_to_action_index(card: Card, m1: dict[Suit, int]) -> int:
    """Map a Card to its canonical position 0–23."""
    return RANKS.index(card.rank) * 4 + m1[card.suit]


def encode_action(action: "Action", hand: list[Card],
                  m1: dict[Suit, int], close: bool = False) -> int:
    """
    Encode an Action into a flat action index (0–47).

    Args:
        action: the Action being taken (must be PLAY_CARD)
        hand:   current hand (to resolve card_index)
        m1:     canonical suit map
        close:  True if the player is closing the game this turn

    Returns:
        int in [0, 47]
    """
    card = hand[action.card_index]
    idx = card_to_action_index(card, m1)
    return idx + (24 if close else 0)


def decode_action_index(action_idx: int) -> tuple[int, bool]:
    """
    Decode a flat action index back to (canonical_card_position, close).

    Returns:
        (card_pos 0–23, close bool)
    """
    close = action_idx >= 24
    card_pos = action_idx - (24 if close else 0)
    return card_pos, close


def valid_action_mask(view: "PlayerView",
                      m1: dict[Suit, int]) -> np.ndarray:
    """
    Build a 48-element boolean mask of valid actions.

    Uses view.valid_actions to determine which card slots and close slots
    are legal.
    """
    from models import ActionType

    mask = np.zeros(NUM_ACTIONS, dtype=np.uint8)

    # Check if closing is among valid actions
    can_close = any(a.type == ActionType.CLOSE_GAME for a in view.valid_actions)

    for action in view.valid_actions:
        if action.type.value != "play_card":
            continue
        card = view.hand[action.card_index]
        idx = card_to_action_index(card, m1)
        mask[idx] = 1          # play without closing
        if can_close:
            mask[idx + 24] = 1  # close then play

    return mask
