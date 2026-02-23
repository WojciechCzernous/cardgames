"""
Visual test: play one game up to trick 6, show the trick log,
then display the captured sample as matrices.
Run twice: once capturing the leader, once the follower.
"""

import random
import numpy as np

from models import Card, Suit, RANKS, ActionType
from game import RoundState, Round
from agents import SmartPlayer
from sampler import SamplingRound
from features import build_suit_map, decode_action_index

TARGET_TRICK = 6
RANK_LABELS = [" 9", " J", " Q", " K", "10", " A"]
SUIT_HDR = ["trump", "  s1", "  s2", "  s3"]


def card_str(card: Card) -> str:
    return f"{card.rank.strip()}{card.suit.value}"


def show_state_matrices(state: np.ndarray, m2: dict[int, Suit]):
    """Display the 130-float state vector as labelled matrices."""
    planes = [
        ("My won cards", 0, 24),
        ("Opp won cards", 29, 53),
        ("My hand", 58, 82),
        ("Trump card", 82, 106),
        ("Table card", 106, 130),
    ]
    extras = [
        ("My marriages + closed", 24, 29),
        ("Opp marriages + closed", 53, 58),
    ]

    suit_real = "  ".join(f"{m2[i].value:>4}" for i in range(4))

    for name, s, e in planes:
        mat = state[s:e].reshape(6, 4)
        n = int(mat.sum())
        print(f"\n  {name} ({n} card(s)):")
        print(f"    {'':>3}  {' '.join(SUIT_HDR)}  |  {suit_real}")
        for r, row in zip(RANK_LABELS, mat):
            cols = "     ".join(f"{int(v)}" for v in row)
            print(f"    {r}   {cols}")

    for name, s, e in extras:
        bits = state[s:e]
        mar_str = "  ".join(f"{SUIT_HDR[i].strip()}={int(bits[i])}" for i in range(4))
        closed = int(bits[4]) if len(bits) > 4 else 0
        print(f"\n  {name}: {mar_str}  closed={closed}")


def run_game(force_leader: bool):
    """Run games until we get a sample at trick 6, forcing leader or follower."""
    seed = random.randint(0, 999999)
    while True:
        random.seed(seed)
        p0 = SmartPlayer("P0")
        p1 = SmartPlayer("P1")
        rnd = SamplingRound([p0, p1], target_trick=TARGET_TRICK)

        # Monkey-patch to force which side is captured
        orig_play_trick = rnd.play_trick.__func__

        def patched_play_trick(self, _force=force_leader):
            # Replace the random coin flip
            import types
            st = self.state
            capture = (self._trick_num == self.target_trick)
            if capture:
                # Inject our forced choice via monkey-patching random
                import random as _rnd
                _orig_random = _rnd.random
                _rnd.random = lambda: 0.0 if _force else 1.0
                result = orig_play_trick(self)
                _rnd.random = _orig_random
                return result
            return orig_play_trick(self)

        import types
        rnd.play_trick = types.MethodType(patched_play_trick, rnd)

        rnd.play()

        if rnd.sample is not None:
            break
        seed += 1

    return rnd, seed


def show_game(rnd: SamplingRound, force_leader: bool, seed: int):
    """Replay the game with narration up to the captured trick."""
    role = "LEADER" if force_leader else "FOLLOWER"
    print(f"\n{'='*60}")
    print(f"  Capturing {role}'s view at trick #{TARGET_TRICK}  (seed={seed})")
    print(f"{'='*60}")

    # Replay to show tricks — we need to re-run with logging
    random.seed(seed)
    p0 = SmartPlayer("P0")
    p1 = SmartPlayer("P1")
    replay = Round([p0, p1])
    st = replay.state

    print(f"\n  Trump: {card_str(st.trump_card)} ({st.trump_suit.value})")
    print(f"  P0 hand: {', '.join(card_str(c) for c in st.hands[0])}")
    print(f"  P1 hand: {', '.join(card_str(c) for c in st.hands[1])}")
    print()

    replay.match_scores = {0: 0, 1: 0}
    for t in range(min(TARGET_TRICK + 1, 12)):
        if not st.hands[0] or not st.hands[1] or st.round_winner is not None:
            break

        leader = st.leader
        result = replay.play_trick()
        card_l = result.cards[leader]
        card_f = result.cards[1 - leader]
        winner_tag = "→ P" + str(result.winner) + " wins"
        mar_str = ""
        for s in (0, 1):
            if result.marriages[s]:
                mar_str += f" (P{s} marriage +{result.marriages[s]})"

        marker = " ◀◀◀ SAMPLED" if t == TARGET_TRICK else ""
        print(f"  Trick {t:2d}: P{leader} leads {card_str(card_l):>4}  "
              f"P{1-leader} plays {card_str(card_f):>4}  "
              f"{winner_tag}  pts={result.trick_points}{mar_str}{marker}")

        # Notify + draw + winner actions
        for seat in (0, 1):
            replay.players[seat].notify_trick(
                result, st.scores[0], st.scores[1], st.round_winner)
        if st.round_winner is not None:
            break
        replay.draw_cards()
        if st.phase == 1:
            replay.winner_actions(st.leader)

    print(f"\n  Scores after trick {TARGET_TRICK}: P0={st.scores[0]}  P1={st.scores[1]}")

    # Now show the captured sample
    sample = rnd.sample
    seat = sample["_seat"]
    m2 = sample["_m2"]  # actual canonical suit mapping from capture time

    action_idx = sample["action"]
    card_pos, close = decode_action_index(action_idx)
    rank_i, suit_i = divmod(card_pos, 4)

    print(f"\n  Sampled player: P{seat} ({'leader' if rnd.sample.get('_was_leader', force_leader) else 'follower'})")
    print(f"  Action index: {action_idx} → rank={RANK_LABELS[rank_i].strip()}, "
          f"suit=canon_{suit_i}, close={close}")
    print(f"  Result: {sample['result']:+.0f}  (game points: {sample['gp']})")

    print(f"\n  Suit mapping: " + ", ".join(
        f"canon_{i}={m2[i].value}" for i in range(4)))
    print(f"\n  State vector ({len(sample['state'])} floats):")
    show_state_matrices(sample["state"], m2)


# --- Main ---
if __name__ == "__main__":
    # Force same game for both
    rnd_lead, seed_lead = run_game(force_leader=True)
    show_game(rnd_lead, force_leader=True, seed=seed_lead)

    rnd_follow, seed_follow = run_game(force_leader=False)
    show_game(rnd_follow, force_leader=False, seed=seed_follow)
