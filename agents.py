"""
Player agents for Sixty-Six.
All players implement the same interface — human or AI.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from models import Action, ActionType, Card, PlayerView, TrickResult, RoundResult, MatchResult

if TYPE_CHECKING:
    from ui import TerminalUI


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
            if swaps:
                return swaps[0]
            return Action(ActionType.PASS)
        return random.choice(view.valid_actions)


class GreedyPlayer(Player):
    """
    Heuristic player:
      - Announces marriages (trump first).
      - Leading: plays lowest card, conserving trumps.
      - Following: beats high-value leads (> J) with strongest card;
        uses trumps only if lead > K.  Falls back to lowest card.
    """

    def __init__(self, name: str = "Greedy"):
        super().__init__(name)

    def choose_action(self, view: PlayerView) -> Action:
        va = view.valid_actions

        if view.is_winner_action_phase:
            swaps = [a for a in va if a.type.value == "swap_trump"]
            if swaps:
                return swaps[0]
            return Action(ActionType.PASS)

        # Prefer marriages (trump marriage first)
        marriages = [a for a in va if a.marriage_suit]
        if marriages:
            trump_m = [a for a in marriages
                       if a.marriage_suit == view.trump_suit]
            if trump_m:
                return random.choice(trump_m)
            return random.choice(marriages)

        plays = [a for a in va if a.type.value == "play_card"]
        if not plays:
            return random.choice(va)

        if view.is_leading or view.lead_card is None:
            return self._pick_lead(plays, view)
        return self._pick_response(plays, view)

    # ------------------------------------------------------------------

    @staticmethod
    def _card_sort_key(card: Card, trump_suit) -> tuple[int, int]:
        """(is_trump, value) — for ascending sort: non-trump low first."""
        return (1 if card.suit == trump_suit else 0, card.value())

    def _pick_lead(self, plays: list[Action], view: PlayerView) -> Action:
        """Leading: play the lowest card, conserving trumps."""
        return min(
            plays,
            key=lambda a: self._card_sort_key(view.hand[a.card_index],
                                              view.trump_suit),
        )

    def _pick_response(self, plays: list[Action], view: PlayerView) -> Action:
        """
        Following:
          - If lead card value > J (2): try to beat with strongest card.
            Don't use trumps unless lead value > K (4).
          - Otherwise (or if can't beat): play lowest card.
        """
        from rules import card_strength, RANK_VALUES

        lead = view.lead_card
        trump = view.trump_suit
        lead_suit = lead.suit
        lead_str = card_strength(lead, lead_suit, trump)

        if lead.value() > RANK_VALUES[" J"]:  # worth beating (Q, K, 10, A)
            allow_trump = lead.value() > RANK_VALUES[" K"]  # 10 or A

            beaters = []
            for a in plays:
                card = view.hand[a.card_index]
                if card_strength(card, lead_suit, trump) > lead_str:
                    if card.suit == trump and lead_suit != trump and not allow_trump:
                        continue  # conserve trump
                    beaters.append(a)

            if beaters:
                # Pick strongest beater
                return max(
                    beaters,
                    key=lambda a: card_strength(
                        view.hand[a.card_index], lead_suit, trump),
                )

        # Can't beat or not worth it — play lowest card
        return min(
            plays,
            key=lambda a: self._card_sort_key(view.hand[a.card_index], trump),
        )


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
        opponent_hand = sorted(
            all_cards - my_hand_set - view.played_cards,
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

        return Action(ActionType.PLAY_CARD,
                      card_index=idx, marriage_suit=mar_suit)


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
