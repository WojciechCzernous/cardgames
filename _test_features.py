"""Quick test for features.py suit canonicalization and encode_state."""
from models import Card, Suit, PlayerView, Action, ActionType
from features import (build_suit_map, cards_to_matrix, suit_hash, encode_state,
                      encode_action, decode_action_index, valid_action_mask,
                      card_to_action_index, NUM_ACTIONS)
import numpy as np

hand = [
    Card(" A", Suit.HEARTS), Card(" K", Suit.HEARTS),
    Card(" Q", Suit.SPADES), Card("10", Suit.SPADES),
    Card(" 9", Suit.CLUBS),
    Card(" J", Suit.DIAMONDS),
]
m1, m2 = build_suit_map(hand, Suit.HEARTS)

print("m1 (Suit -> index):")
for s, i in sorted(m1.items(), key=lambda x: x[1]):
    print(f"  {s.value} -> {i}  (hash={suit_hash(hand, s)})")

print("\nm2 (index -> Suit):")
for i in range(4):
    print(f"  {i} -> {m2[i].value}")

print("\nHand matrix (6 ranks x 4 canonical suits):")
mat = cards_to_matrix(m1, hand)
ranks = ["9 ", "J ", "Q ", "K ", "10", "A "]
print("     trump  s1  s2  s3")
for r, row in zip(ranks, mat):
    print(f"  {r}:  {row}")

seen = {Card(" A", Suit.HEARTS), Card(" Q", Suit.SPADES), Card(" 9", Suit.DIAMONDS)}
print("\nSeen cards matrix:")
mat2 = cards_to_matrix(m1, seen)
for r, row in zip(ranks, mat2):
    print(f"  {r}:  {row}")

# ----- encode_state test -----
print("\n" + "=" * 50)
print("encode_state test")
print("=" * 50)

won_cards_me = [Card(" 9", Suit.HEARTS), Card(" J", Suit.HEARTS),
                Card("10", Suit.DIAMONDS), Card(" A", Suit.CLUBS)]
won_cards_opp = [Card(" K", Suit.CLUBS), Card(" Q", Suit.DIAMONDS)]

view = PlayerView(
    seat=0,
    hand=hand,
    trump_suit=Suit.HEARTS,
    trump_card=Card("10", Suit.HEARTS),   # visible trump card
    draw_pile_size=5,
    phase=1,
    closed=True,
    closed_by=0,                             # I closed
    my_score=30,
    opponent_score=12,
    is_leading=False,
    lead_card=Card(" K", Suit.SPADES),       # opponent led this card
    lead_marriage=None,
    valid_actions=[Action(ActionType.PLAY_CARD, 3)],
    is_winner_action_phase=False,
    seen_cards=set(),
    played_cards=set(),
    won_cards_me=won_cards_me,
    won_cards_opp=won_cards_opp,
    marriages_me={Suit.HEARTS},
    marriages_opp={Suit.DIAMONDS},
)

state = encode_state(view)
print(f"State vector shape: {state.shape}, dtype: {state.dtype}")
print(f"Total: {len(state)} floats\n")

labels = [
    ("My won cards (24) + marriages (4) + closed (1)", 0, 29),
    ("Opp won cards (24) + marriages (4) + closed (1)", 29, 58),
    ("My hand (24)", 58, 82),
    ("Trump card (24)", 82, 106),
    ("Table card (24)", 106, 130),
]
ranks = ["9 ", "J ", "Q ", "K ", "10", "A "]
suit_labels = ["trump", "s1", "s2", "s3"]

def show_plane(name, start, end):
    bits = state[start:end]
    ones = int(bits.sum())
    print(f"  {name}: {ones} bit(s) set")
    # Show the 24-bit card portion as a 6x4 matrix
    card_bits = bits[:24].reshape(6, 4)
    print(f"    {'':>3}  " + "  ".join(f"{l:>5}" for l in suit_labels))
    for r, row in zip(ranks, card_bits):
        cols = "  ".join(f"{int(v):>5}" for v in row)
        print(f"    {r:>3}  {cols}")
    # Show extra bits if present
    if end - start > 24:
        extra = bits[24:]
        mar_str = "  ".join(f"{suit_labels[i]}={int(extra[i])}" for i in range(4))
        print(f"    marriages: {mar_str}")
        if len(extra) > 4:
            print(f"    closed:    {int(extra[4])}")
    print()

for name, start, end in labels:
    show_plane(name, start, end)

# Verify complement: union of all 24-bit card planes
all_cards_bits = state[0:24] + state[29:53] + state[58:82] + state[82:106] + state[106:130]
unseen = (1.0 - np.clip(all_cards_bits, 0, 1))
print(f"\nCards not yet seen (complement): {int(unseen.sum())} cards")
print(f"  {unseen}")

# ----- action encoding test -----
print("\n" + "=" * 50)
print("action encoding test")
print("=" * 50)

# Verify round-trip for each card in hand
print("\nCard -> action index -> decode:")
for i, card in enumerate(hand):
    idx = card_to_action_index(card, m1)
    pos, close = decode_action_index(idx)
    print(f"  {card} -> idx {idx:2d} -> pos {pos:2d}, close={close}")
    assert pos == idx and close is False

# Encode with close=True
action = Action(ActionType.PLAY_CARD, card_index=0)  # A♥
idx_no_close = encode_action(action, hand, m1, close=False)
idx_close = encode_action(action, hand, m1, close=True)
print(f"\n  A♥ without close: idx={idx_no_close}")
print(f"  A♥ with close:    idx={idx_close}")
pos_nc, close_nc = decode_action_index(idx_no_close)
pos_c, close_c = decode_action_index(idx_close)
assert close_nc is False and close_c is True
assert pos_nc == pos_c
print("  Round-trip OK ✓")

# Valid action mask — use the existing view (leading=False, so no close)
# Make a leading view to test close availability
view_lead = PlayerView(
    seat=0, hand=hand, trump_suit=Suit.HEARTS,
    trump_card=Card("10", Suit.HEARTS), draw_pile_size=5,
    phase=1, closed=False, closed_by=None,
    my_score=30, opponent_score=12,
    is_leading=True, lead_card=None, lead_marriage=None,
    valid_actions=[
        Action(ActionType.PLAY_CARD, 0),   # A♥
        Action(ActionType.PLAY_CARD, 1),   # K♥
        Action(ActionType.PLAY_CARD, 2),   # Q♠
        Action(ActionType.PLAY_CARD, 3),   # 10♠
        Action(ActionType.PLAY_CARD, 4),   # 9♣
        Action(ActionType.PLAY_CARD, 5),   # J♦
        Action(ActionType.CLOSE_GAME),
    ],
    is_winner_action_phase=False,
    seen_cards=set(), played_cards=set(),
    won_cards_me=[], won_cards_opp=[],
    marriages_me=set(), marriages_opp=set(),
)
mask = valid_action_mask(view_lead, m1)
print(f"\nValid action mask ({int(mask.sum())} valid of {NUM_ACTIONS}):")
print(f"  Play slots (0-23):  {mask[:24].reshape(6,4).astype(int)}")
print(f"  Close slots (24-47):{mask[24:].reshape(6,4).astype(int)}")
