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
# Human player (delegates to a TerminalUI for I/O)
# ---------------------------------------------------------------------------

class HumanPlayer(Player):
    """Interactive player — reads from terminal via a UI object."""

    def __init__(self, ui: "TerminalUI", name: str = "You"):
        super().__init__(name)
        self.ui = ui

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
