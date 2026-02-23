"""
Game data sampler for RL training.

Plays Smart vs Smart games, captures one snapshot per game at a
chosen trick number (0-based).  If the game ends before that trick,
nothing is collected.

Each sample is:
    state  — 130-float encoded state (from sampled player's perspective)
    action — int 0–47 (encoded action chosen)
    result — +1 if the sampled player won the round, -1 if lost, 0 tie
    gp     — game points awarded (0 if tie)
"""

from __future__ import annotations

import os
import random as _rnd
import time
from multiprocessing import Pool

import numpy as np

from models import ActionType, Action, TrickResult, RoundResult
from game import RoundState, Round
from agents import SmartPlayer
from features import (
    encode_state, encode_action,
    build_suit_map,
)
import rules


# ---------------------------------------------------------------------------
# Instrumented round — plays trick-by-trick, captures at target trick
# ---------------------------------------------------------------------------

class SamplingRound(Round):
    """
    A Round subclass that captures both the leader's and follower's
    decision-point data at a specific trick number.
    Randomly picks either the leader's or follower's perspective (one sample per game).
    """

    def __init__(self, players, target_trick: int):
        super().__init__(players)
        self.target_trick = target_trick
        self.sample: dict | None = None   # 0 or 1 sample per game
        self._trick_num = 0

    def play_trick(self):
        """Override to intercept one player's view at the target trick."""
        st = self.state
        capture = (self._trick_num == self.target_trick)
        leader = st.leader
        follower = 1 - leader
        marriages: dict[int, int] = {0: 0, 1: 0}

        # Decide up front which side to capture
        if capture:
            capture_leader = _rnd.random() < 0.5

        # --- Leader plays ---
        view_l = st.player_view(leader, lead_card=None,
                                match_scores=self.match_scores)
        if capture and capture_leader:
            m1_l, m2_l = build_suit_map(view_l.hand, view_l.trump_suit)
            state_l = encode_state(view_l)

        action_l = self.players[leader].choose_action(view_l)
        card_l, mar_l = self.execute_action(leader, action_l)
        if mar_l:
            marriages[leader] = mar_l
            st.scores[leader] += mar_l

        if capture and capture_leader:
            idx = None
            for i, c in enumerate(view_l.hand):
                if c == card_l:
                    idx = i
                    break
            if idx is not None:
                enc = encode_action(
                    Action(ActionType.PLAY_CARD, card_index=idx),
                    view_l.hand, m1_l, close=False)
                self.sample = {
                    "state": state_l, "action": enc,
                    "_seat": leader, "_m2": m2_l,
                }

        # --- Follower plays (sees leader's card) ---
        view_f = st.player_view(follower, lead_card=card_l,
                                match_scores=self.match_scores)
        view_f.lead_marriage = action_l.marriage_suit
        if capture and not capture_leader:
            m1_f, m2_f = build_suit_map(view_f.hand, view_f.trump_suit)
            state_f = encode_state(view_f)

        action_f = self.players[follower].choose_action(view_f)
        card_f, mar_f = self.execute_action(follower, action_f)
        if mar_f:
            marriages[follower] = mar_f
            st.scores[follower] += mar_f

        if capture and not capture_leader:
            idx = None
            for i, c in enumerate(view_f.hand):
                if c == card_f:
                    idx = i
                    break
            if idx is not None:
                enc = encode_action(
                    Action(ActionType.PLAY_CARD, card_index=idx),
                    view_f.hand, m1_f, close=False)
                self.sample = {
                    "state": state_f, "action": enc,
                    "_seat": follower, "_m2": m2_f,
                }

        # --- Resolve trick (same as base) ---
        cards = {leader: card_l, follower: card_f}
        lead_suit = card_l.suit
        relative_winner = rules.trick_winner(card_l, card_f, lead_suit, st.trump_suit)
        winner = leader if relative_winner == 0 else follower
        trick_points = card_l.value() + card_f.value()

        st.scores[winner] += trick_points
        st.leader = winner
        st.played_cards.append(card_l)
        st.played_cards.append(card_f)
        st.won_cards[winner].append(card_l)
        st.won_cards[winner].append(card_f)
        for s in (0, 1):
            if marriages[s]:
                suit = action_l.marriage_suit if s == leader else action_f.marriage_suit
                if suit:
                    st.announced_marriages[s].add(suit)

        for s in (0, 1):
            if st.scores[s] >= rules.WIN_SCORE:
                st.round_winner = s

        for seat in (0, 1):
            view = st.player_view(seat, lead_card=None,
                                  match_scores=self.match_scores)
            self.players[seat].notify_trick_cards(view, cards, leader, marriages)

        result = TrickResult(
            cards=cards, winner=winner,
            trick_points=trick_points, marriages=marriages,
        )
        winner_label = self.players[winner].name
        loser = 1 - winner
        st.last_trick_info = (
            f"{winner_label} won +{trick_points} pts "
            f"({cards[winner]} beat {cards[loser]})"
        )

        self._trick_num += 1
        return result

    def play(self, match_scores: dict[int, int] | None = None):
        """Play the full round, attaching outcome to samples."""
        self.match_scores = match_scores or {0: 0, 1: 0}
        st = self.state
        self._trick_num = 0

        while st.hands[0] and st.hands[1] and st.round_winner is None:
            result = self.play_trick()

            for seat in (0, 1):
                self.players[seat].notify_trick(
                    result, st.scores[0], st.scores[1], st.round_winner)

            if st.round_winner is not None:
                break

            self.draw_cards()

            if st.phase == 1:
                self.winner_actions(st.leader)

        winner, game_points = rules.compute_game_points(
            st.scores, st.round_winner, st.closed, st.closed_by)

        # Attach outcome to captured sample
        if self.sample is not None:
            seat = self.sample["_seat"]
            if winner is None:
                self.sample["result"] = 0.0
                self.sample["gp"] = 0
            elif winner == seat:
                self.sample["result"] = 1.0
                self.sample["gp"] = game_points
            else:
                self.sample["result"] = -1.0
                self.sample["gp"] = game_points

        return RoundResult(
            winner=winner, game_points=game_points,
            scores=dict(st.scores), closed=st.closed,
            closed_by=st.closed_by,
        )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def _run_one_game(target_trick: int) -> dict | None:
    """Play one Smart-vs-Smart game, return sample dict or None."""
    p0 = SmartPlayer("S0")
    p1 = SmartPlayer("S1")
    rnd = SamplingRound([p0, p1], target_trick=target_trick)
    rnd.play()
    if rnd.sample is not None and "result" in rnd.sample:
        s = rnd.sample
        return {
            "state": s["state"],
            "action": s["action"],
            "result": s["result"],
            "gp": s["gp"],
        }
    return None


def collect_samples(
    target_trick: int,
    n_samples: int,
    max_attempts: int | None = None,
    n_workers: int | None = None,
) -> list[dict]:
    """
    Collect n_samples from Smart-vs-Smart games at trick `target_trick`.
    Captures one sample per game (leader or follower, 50/50).
    Uses multiprocessing for speed (n_workers defaults to CPU count).

    Returns list of dicts with keys:
        state  (np.ndarray, 130)
        action (int, 0–47)
        result (float: +1, -1, 0)
        gp     (int: game points)
    """
    if max_attempts is None:
        max_attempts = n_samples * 10
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    # Use multiprocessing: submit batches of games, collect results
    samples: list[dict] = []
    remaining = max_attempts

    with Pool(n_workers) as pool:
        while len(samples) < n_samples and remaining > 0:
            batch = min(remaining, (n_samples - len(samples)) * 3, 512)
            remaining -= batch
            results = pool.map(
                _run_one_game, [target_trick] * batch)
            for r in results:
                if r is not None:
                    samples.append(r)
                    if len(samples) >= n_samples:
                        break

    return samples[:n_samples]


# ---------------------------------------------------------------------------
# Main — collect and time
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for stage in [6, 7, 8, 9]:
        t0 = time.time()
        samples = collect_samples(target_trick=stage, n_samples=1000)
        elapsed = time.time() - t0
        results = [s["result"] for s in samples]
        wins = sum(1 for r in results if r > 0)
        losses = sum(1 for r in results if r < 0)
        ties = sum(1 for r in results if r == 0)
        print(f"Stage {stage:2d}: {len(samples):4d} samples in {elapsed:.2f}s "
              f"(W/L/T: {wins}/{losses}/{ties})")

