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

    def notify_next_round(self) -> None:
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
    """Prefers marriages (trump first) and plays highest-value cards."""

    def __init__(self, name: str = "Greedy"):
        super().__init__(name)

    def choose_action(self, view: PlayerView) -> Action:
        va = view.valid_actions

        if view.is_winner_action_phase:
            swaps = [a for a in va if a.type.value == "swap_trump"]
            if swaps:
                return swaps[0]
            return Action(ActionType.PASS)

        # Prefer marriages
        marriages = [a for a in va if a.marriage_suit]
        if marriages:
            trump_m = [a for a in marriages
                       if a.marriage_suit == view.trump_suit]
            if trump_m:
                return random.choice(trump_m)
            return random.choice(marriages)

        # Highest-value play
        plays = [a for a in va if a.type.value == "play_card"]
        if plays:
            return max(plays, key=lambda a: view.hand[a.card_index].value())

        return random.choice(va)


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

    def notify_next_round(self) -> None:
        self.ui.prompt_next_round()
