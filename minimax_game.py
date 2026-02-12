"""
Minimax framework for an alternating-initiative two-player game.

Each round: the initiator picks an action, then the responder picks a reaction.
The round winner gets initiative for the next round.
The game ends after N rounds or when a terminal condition is met.
All values are from Player A's perspective.

State is fully general: actions may depend on the entire game state, and
the responder's available actions may also depend on the initiator's choice.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Hashable, cast


class Player(Enum):
    A = "A"
    B = "B"

    def opponent(self) -> "Player":
        return Player.B if self is Player.A else Player.A


class GameState(ABC):
    """
    Abstract game state.  Subclass to hold whatever your game needs
    (scores, hands, boards, resources, etc.).

    Must implement:
      - round, initiator  (properties)
      - cache_key()        (for memoization — return a hashable)
    """

    @property
    @abstractmethod
    def round(self) -> int:
        """Current round number (1-based)."""
        ...

    @property
    @abstractmethod
    def initiator(self) -> Player:
        """Who has the initiative this round."""
        ...

    @abstractmethod
    def cache_key(self) -> Hashable:
        """
        Return a hashable representation of this state for memoization.
        Two states that would produce identical subtrees must share a key.
        """
        ...


class GameRules(ABC):
    """
    Abstract specification of the game.
    Subclass this to define your concrete game.
    """

    @abstractmethod
    def max_rounds(self) -> int:
        """Total number of rounds N."""
        ...

    @abstractmethod
    def initial_state(self) -> GameState:
        """Return the starting state (round 1, initial scores, A initiates, etc.)."""
        ...

    @abstractmethod
    def get_initiator_actions(self, state: GameState) -> list[Any]:
        """
        Legal actions for the initiator in the current state.
        May depend on anything inside `state`.
        """
        ...

    @abstractmethod
    def get_responder_actions(
        self, state: GameState, action_init: Any
    ) -> list[Any]:
        """
        Legal actions for the responder, given the current state AND
        the initiator's already-chosen action.
        """
        ...

    @abstractmethod
    def resolve_round(
        self, state: GameState, action_init: Any, action_resp: Any
    ) -> GameState:
        """
        Apply both actions to the current state and return the next GameState.
        The returned state must have:
          - round incremented
          - initiator set to the round winner
          - all other fields updated (scores, hands, etc.)
        """
        ...

    @abstractmethod
    def evaluate(self, state: GameState) -> float:
        """
        Terminal evaluation from A's perspective.
        This is your nonlinear W(stateA, stateB).
        Positive ⇒ A favours; negative ⇒ B favours.
        """
        ...

    def is_terminal(self, state: GameState) -> bool:
        """
        Is the game over?  Default: only when rounds are exhausted.
        Override for early wins, draws, etc.
        """
        return state.round > self.max_rounds()


class MinimaxSolver:
    """
    Solves the alternating-initiative game via minimax with optional memoization.
    """

    def __init__(self, rules: GameRules, *, use_cache: bool = True):
        self.rules = rules
        self._cache: Optional[dict[Hashable, float]] = {} if use_cache else None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def solve(self, state: Optional[GameState] = None) -> float:
        """
        Return the minimax value from A's perspective.
        If no state is given, starts from initial_state().
        """
        if state is None:
            state = self.rules.initial_state()
        if self._cache is not None:
            self._cache.clear()
        return self._minimax(state)

    def best_action(self, state: Optional[GameState] = None) -> tuple[Any, float]:
        """
        Return (best_initiator_action, minimax_value) for the current state.
        """
        if state is None:
            state = self.rules.initial_state()
        if self._cache is not None:
            self._cache.clear()
        return self._best_action(state)

    # ------------------------------------------------------------------ #
    # Core minimax
    # ------------------------------------------------------------------ #

    def _minimax(self, state: GameState) -> float:
        # Terminal check
        if self.rules.is_terminal(state):
            return self.rules.evaluate(state)

        # Cache lookup
        if self._cache is not None:
            key = state.cache_key()
            if key in self._cache:
                return self._cache[key]

        value = self._evaluate_initiator(state)

        if self._cache is not None:
            self._cache[key] = value
        return value

    def _evaluate_initiator(self, state: GameState) -> float:
        """Outer loop: initiator picks their action optimally."""
        initiator = state.initiator

        if initiator is Player.A:
            best = float("-inf")
            for action_init in self.rules.get_initiator_actions(state):
                resp_value = self._evaluate_responder(state, action_init)
                best = max(best, resp_value)
        else:
            best = float("inf")
            for action_init in self.rules.get_initiator_actions(state):
                resp_value = self._evaluate_responder(state, action_init)
                best = min(best, resp_value)
        return best

    def _evaluate_responder(self, state: GameState, action_init: Any) -> float:
        """Inner loop: responder reacts optimally to the initiator's action."""
        responder = state.initiator.opponent()

        if responder is Player.A:
            best = float("-inf")
            for action_resp in self.rules.get_responder_actions(state, action_init):
                val = self._recurse(state, action_init, action_resp)
                best = max(best, val)
        else:
            best = float("inf")
            for action_resp in self.rules.get_responder_actions(state, action_init):
                val = self._recurse(state, action_init, action_resp)
                best = min(best, val)
        return best

    def _recurse(self, state: GameState, action_init: Any, action_resp: Any) -> float:
        """Resolve the round and recurse into the next state."""
        next_state = self.rules.resolve_round(state, action_init, action_resp)
        return self._minimax(next_state)

    # ------------------------------------------------------------------ #
    # Best-action recovery
    # ------------------------------------------------------------------ #

    def _best_action(self, state: GameState) -> tuple[Any, float]:
        """Return the initiator's best action and its minimax value."""
        if self.rules.is_terminal(state):
            return None, self.rules.evaluate(state)

        initiator = state.initiator
        best_act = None

        if initiator is Player.A:
            best_val = float("-inf")
            for action_init in self.rules.get_initiator_actions(state):
                resp_value = self._evaluate_responder(state, action_init)
                if resp_value > best_val:
                    best_val = resp_value
                    best_act = action_init
        else:
            best_val = float("inf")
            for action_init in self.rules.get_initiator_actions(state):
                resp_value = self._evaluate_responder(state, action_init)
                if resp_value < best_val:
                    best_val = resp_value
                    best_act = action_init

        return best_act, best_val


# ====================================================================== #
# Example concrete game (delete or replace with your own)
# ====================================================================== #

from dataclasses import dataclass


@dataclass(frozen=True)
class ExampleState(GameState):
    """
    Toy state: just scores + round + initiator.
    For your real game, add hands, decks, resources, etc.
    """
    _round: int
    _initiator: Player
    score_a: float
    score_b: float

    @property
    def round(self) -> int:
        return self._round

    @property
    def initiator(self) -> Player:
        return self._initiator

    def cache_key(self) -> tuple:
        return (self._round, self._initiator, self.score_a, self.score_b)


class ExampleGame(GameRules):
    """
    Toy game: each player picks 1, 2, or 3 points to add to their score.
    The player who picked more wins the round.
    Responder's available actions shrink if the initiator picked 3
    (just to demonstrate state-dependent responder actions).
    After 3 rounds, evaluate with a nonlinear function.
    """

    def max_rounds(self) -> int:
        return 3

    def initial_state(self) -> ExampleState:
        return ExampleState(_round=1, _initiator=Player.A, score_a=0, score_b=0)

    def get_initiator_actions(self, state: GameState) -> list[int]:
        s = cast(ExampleState, state)
        return [1, 2, 3]

    def get_responder_actions(self, state: GameState, action_init: int) -> list[int]:
        s = cast(ExampleState, state)
        # Example: if initiator went all-in (3), responder can only pick 1 or 2
        if action_init == 3:
            return [1, 2]
        return [1, 2, 3]

    def resolve_round(
        self, state: GameState, action_init: int, action_resp: int
    ) -> GameState:
        s = cast(ExampleState, state)
        initiator = s.initiator
        if initiator is Player.A:
            new_a = s.score_a + action_init
            new_b = s.score_b + action_resp
        else:
            new_a = s.score_a + action_resp
            new_b = s.score_b + action_init

        winner = initiator if action_init >= action_resp else initiator.opponent()

        return ExampleState(
            _round=s.round + 1,
            _initiator=winner,
            score_a=new_a,
            score_b=new_b,
        )

    def evaluate(self, state: GameState) -> float:
        s = cast(ExampleState, state)
        # Nonlinear: A wants (a² − a·b) to be large
        return s.score_a ** 2 - s.score_a * s.score_b

    def is_terminal(self, state: GameState) -> bool:
        return state.round > self.max_rounds()


if __name__ == "__main__":
    game = ExampleGame()
    solver = MinimaxSolver(game)

    value = solver.solve()
    action, val = solver.best_action()

    print(f"Minimax value (A's perspective): {value}")
    print(f"A's best opening action: {action}  (value = {val})")
