"""
Player agents for Sixty-Six.
All players implement the same interface — human or AI.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from models import Action, ActionType, Card, PlayerView, TrickResult, RoundResult, MatchResult, RANK_VALUES

if TYPE_CHECKING:
    from ui import TerminalUI


# ---------------------------------------------------------------------------
# Swap-trump probability helper
# ---------------------------------------------------------------------------

def _should_swap_trump(view: PlayerView) -> bool:
    """
    Decide whether to swap 9-of-trump for the face-up trump card.

    Probability depends on the trump card's rank:
      - 10 or A           → 100%
      - K or Q with partner in hand → 100%
      - K or Q without partner       →  80%
      - J or 9                        →  50%
    """
    tc = view.trump_card
    if tc is None:
        return False

    if tc.rank in ("10", " A"):
        p = 1.0
    elif tc.rank in (" K", " Q"):
        partner_rank = " Q" if tc.rank == " K" else " K"
        if any(c.rank == partner_rank and c.suit == tc.suit for c in view.hand):
            p = 1.0
        else:
            p = 0.8
    else:
        p = 0.5

    return random.random() < p


# ---------------------------------------------------------------------------
# Close-game decision helper
# ---------------------------------------------------------------------------

def _should_close(view: PlayerView) -> bool:
    """
    Decide whether to close the game.

    Estimates a conservative lower-bound on points reachable with just
    the cards in hand (no more draws).  We count:
      - marriage points for any K+Q pair we hold (auto-announced on lead)
      - sure trick points for cards with 0 threats (nothing unseen beats them)
      - likely trick points for low-threat trump cards
    Each won trick also captures the opponent's card; we assume a modest
    2-point average capture for sure winners.
    """
    from rules import find_marriages, marriage_value

    score = view.my_score

    # --- marriage points ---
    marriages = find_marriages(view.hand)
    for suit in marriages:
        score += marriage_value(suit, view.trump_suit)

    # --- trick points ---
    threats = view.card_threats
    hand = view.hand

    sure_winners: list[Card] = []
    likely_winners: list[Card] = []

    for card in hand:
        t = threats.get(card, 99)
        if t == 0:
            sure_winners.append(card)
        elif card.suit == view.trump_suit and t <= 1:
            # Low-threat trump (e.g. K of trump with only A unseen)
            likely_winners.append(card)

    # Sure winners: own value + conservative 2 pts captured from opponent
    for card in sure_winners:
        score += card.value() + 2

    # Likely winners: count own value only (conservative)
    for card in likely_winners:
        score += card.value()

    # --- decision ---
    # Under pressure (opponent close to winning), accept more risk
    if view.opponent_score >= 33:
        return score >= 40
    return score >= 46


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Player(ABC):
    """Symmetric player interface — the same for both seats."""

    def __init__(self, name: str = "Player"):
        self.name = name

    @abstractmethod
    def choose_action(self, view: PlayerView) -> Action:
        """Pick an action given the current observable state."""
        ...

    # Optional notification hooks (override as needed)
    def notify_trick_cards(self, view: PlayerView,
                           table_cards: dict[int, Card],
                           leader: int = 0,
                           marriages: dict[int, int] | None = None) -> None:
        """Called after both cards are on the table, before result."""
        pass

    def notify_trick(self, result: TrickResult,
                     score_0: int, score_1: int,
                     round_winner: int | None) -> None:
        pass

    def notify_swap(self, old_trump, new_trump) -> None:
        pass

    def notify_close(self, closed_by: int) -> None:
        pass

    def notify_round_result(self, result: RoundResult,
                            match_scores: dict[int, int]) -> None:
        pass

    def notify_match_start(self) -> None:
        pass

    def notify_match_result(self, result: MatchResult) -> None:
        pass

    def notify_next_round(self, hand: list[Card] | None = None,
                          first_round: bool = False) -> None:
        pass

    def set_opponent_hand(self, hand: list[Card]) -> None:
        """Called before choose_action with the opponent's current hand (for display only)."""
        pass


# ---------------------------------------------------------------------------
# AI agents
# ---------------------------------------------------------------------------

class RandomPlayer(Player):
    """Picks uniformly at random from valid actions (swaps 9-trump if able)."""

    def __init__(self, name: str = "Random"):
        super().__init__(name)

    def choose_action(self, view: PlayerView) -> Action:
        if view.is_winner_action_phase:
            swaps = [a for a in view.valid_actions
                     if a.type.value == "swap_trump"]
            if swaps and _should_swap_trump(view):
                return swaps[0]
            return Action(ActionType.PASS)
        return random.choice(view.valid_actions)


class GreedyPlayer(Player):
    """Prefers marriages (trump first) and plays highest-value cards."""

    def __init__(self, name: str = "Greedy"):
        super().__init__(name)

    def choose_action(self, view: PlayerView) -> Action:
        va = view.valid_actions

        if view.is_winner_action_phase:
            swaps = [a for a in va if a.type.value == "swap_trump"]
            if swaps and _should_swap_trump(view):
                return swaps[0]
            # Consider closing
            if _should_close(view):
                closes = [a for a in va if a.type.value == "close_game"]
                if closes:
                    return closes[0]
            return Action(ActionType.PASS)

        plays = [a for a in va if a.type.value == "play_card"]
        if not plays:
            return random.choice(va)

        # When leading, prefer marriage cards (K or Q from a pair)
        if view.is_leading:
            from rules import find_marriages
            marriages = find_marriages(view.hand)
            if marriages:
                # Exception: if game is closed and we hold A of trump,
                # play the ace first
                has_trump_ace = any(
                    c.rank == " A" and c.suit == view.trump_suit
                    for c in view.hand
                )
                if view.closed and has_trump_ace:
                    ace_plays = [
                        a for a in plays
                        if view.hand[a.card_index].rank == " A"
                        and view.hand[a.card_index].suit == view.trump_suit
                    ]
                    if ace_plays:
                        return ace_plays[0]

                # Play a marriage card (prefer trump marriage)
                trump_mar = [s for s in marriages if s == view.trump_suit]
                mar_suits = trump_mar or marriages
                for suit in mar_suits:
                    mar_plays = [
                        a for a in plays
                        if view.hand[a.card_index].suit == suit
                        and view.hand[a.card_index].rank in (" K", " Q")
                    ]
                    if mar_plays:
                        return mar_plays[0]

        # When following, try to win the trick
        if view.lead_card is not None:
            from rules import trick_winner
            lead = view.lead_card
            lead_is_high = lead.rank in ("10", " A")

            winners = []
            for a in plays:
                c = view.hand[a.card_index]
                if trick_winner(lead, c, lead.suit, view.trump_suit) == 1:
                    is_trump = (c.suit == view.trump_suit
                                and lead.suit != view.trump_suit)
                    # Don't spend a trump unless the lead card is a 10 or A
                    if is_trump and not lead_is_high:
                        continue
                    winners.append(a)

            if winners:
                # Play highest-value winner
                return max(winners,
                           key=lambda a: view.hand[a.card_index].value())

            # Can't win profitably → play lowest
            return min(plays, key=lambda a: view.hand[a.card_index].value())

        # Leading (no marriage to play) → play lowest
        return min(plays, key=lambda a: view.hand[a.card_index].value())


class SmartPlayer(Player):
    """Greedy in phase 1, minimax-optimal in phase 2 (perfect information)."""

    def __init__(self, name: str = "Smart"):
        super().__init__(name)
        self._greedy = GreedyPlayer(name)

    def choose_action(self, view: PlayerView) -> Action:
        # Phase 2 with empty draw pile → perfect info → minimax
        if (view.draw_pile_size == 0
                and not view.is_winner_action_phase
                and view.phase == 2):
            return self._minimax_action(view)

        # Otherwise fall back to greedy heuristic
        return self._greedy.choose_action(view)

    # ------------------------------------------------------------------

    def _minimax_action(self, view: PlayerView) -> Action:
        from solver import EndgameSolver
        from models import Card, Suit, RANKS

        # Derive opponent's hand by elimination
        all_cards = {Card(r, s) for s in Suit for r in RANKS}
        my_hand_set = set(view.hand)
        won_cards = set(view.my_won_cards) | set(view.opponent_won_cards)
        opponent_hand = sorted(
            all_cards - my_hand_set - won_cards,
            key=lambda c: c.key(),
        )

        opp = 1 - view.seat
        hands = {view.seat: list(view.hand), opp: opponent_hand}
        scores = {view.seat: view.my_score, opp: view.opponent_score}
        leader = view.seat if view.is_leading else opp
        lead_card = None if view.is_leading else view.lead_card

        solver = EndgameSolver(
            trump_suit=view.trump_suit,
            my_seat=view.seat,
        )
        idx, mar_suit, _val = solver.best_action(
            hands, scores, leader, lead_card)

        return Action(ActionType.PLAY_CARD, card_index=idx)


# ---------------------------------------------------------------------------
# Neural-network policy player (for RL self-play)
# ---------------------------------------------------------------------------

class PolicyPlayer(Player):
    """
    Plays using a neural network (PolicyNet or ActorCriticNet).
    Optionally records (state_tensor, action_index, log_prob, value) per step
    for PPO training.
    """

    def __init__(self, model, name: str = "Policy",
                 greedy: bool = False, record: bool = False):
        super().__init__(name)
        self.model = model
        self.greedy_mode = greedy
        self.record = record
        # Per-round trajectory: list of (state, action_idx, log_prob, value)
        self.trajectory: list[tuple] = []

    def reset_trajectory(self):
        self.trajectory = []

    def choose_action(self, view: PlayerView) -> Action:
        import torch
        from features import player_view_to_tensor, index_to_action

        state = player_view_to_tensor(view).unsqueeze(0)  # (1, 248)

        with torch.no_grad():
            has_value = hasattr(self.model, 'policy_and_value')

            if has_value:
                masked_logits, value = self.model.policy_and_value(state)
                value = value.item()
            else:
                masked_logits = self.model.masked_logits(state)
                value = 0.0

            probs = torch.softmax(masked_logits, dim=-1)
            log_probs = torch.log_softmax(masked_logits, dim=-1)

            if self.greedy_mode:
                action_idx = masked_logits.argmax(dim=-1).item()
            else:
                # Ensure valid distribution (nan-safe)
                probs = probs.clamp(min=0.0)
                if probs.sum() < 1e-8:
                    # Fallback: uniform over valid actions
                    mask = x[..., 87:87+27]
                    probs = mask / mask.sum()
                action_idx = torch.multinomial(probs, 1).item()

            log_prob = log_probs[0, action_idx].clamp(min=-20.0).item()

        if self.record:
            self.trajectory.append((
                state.squeeze(0),    # (248,)
                action_idx,
                log_prob,
                value,
            ))

        return index_to_action(action_idx, view)


# ---------------------------------------------------------------------------
# Human player (delegates to a TerminalUI for I/O)
# ---------------------------------------------------------------------------

class HumanPlayer(Player):
    """Interactive player — reads from terminal via a UI object."""

    def __init__(self, ui: "TerminalUI", name: str = "You"):
        super().__init__(name)
        self.ui = ui

    def set_opponent_hand(self, hand: list[Card]) -> None:
        self.ui.set_opponent_hand(hand)

    def choose_action(self, view: PlayerView) -> Action:
        if view.is_winner_action_phase:
            self.ui.display_state(view)
            return self.ui.prompt_winner_action(view)
        else:
            self.ui.display_state(view)
            return self.ui.prompt_card_play(view)

    def notify_trick_cards(self, view: PlayerView,
                           table_cards: dict[int, Card],
                           leader: int = 0,
                           marriages: dict[int, int] | None = None) -> None:
        self.ui.show_table(view, table_cards, leader, marriages)

    def notify_trick(self, result: TrickResult,
                     score_0: int, score_1: int,
                     round_winner: int | None) -> None:
        self.ui.show_trick_result(result, score_0, score_1, round_winner)

    def notify_swap(self, old_trump, new_trump) -> None:
        self.ui.show_message(f"Swapped! New trump card: {new_trump}")

    def notify_close(self, closed_by: int) -> None:
        self.ui.show_message("Game closed! No more drawing. Phase 2 rules now apply.")

    def notify_round_result(self, result: RoundResult,
                            match_scores: dict[int, int]) -> None:
        self.ui.show_round_result(result, match_scores)

    def notify_match_start(self) -> None:
        self.ui.show_welcome()

    def notify_match_result(self, result: MatchResult) -> None:
        self.ui.show_match_result(result)

    def notify_next_round(self, hand: list[Card] | None = None,
                          first_round: bool = False) -> None:
        self.ui.show_next_round(hand, first_round=first_round)
