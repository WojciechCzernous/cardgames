"""
Endgame solver for Sixty-Six.

Perfect-information minimax for phase 2 (draw pile empty).
When both hands are known the game tree is small enough
(≤6 cards per side → max 720² leaf nodes before pruning)
to solve exactly with memoised minimax.

Note: closing the game is only possible while the draw pile is non-empty,
so the solver only runs when the pile emptied naturally — closed is always False.

Usage:
    solver = EndgameSolver(trump_suit, my_seat)
    idx, marriage, value = solver.best_action(hands, scores, leader, lead_card)
"""

from __future__ import annotations

from models import Card, Suit, RANKS
import rules


class EndgameSolver:
    """Minimax solver for the Sixty-Six endgame with perfect information."""

    def __init__(self, trump_suit: Suit, my_seat: int):
        self.trump_suit = trump_suit
        self.my_seat = my_seat
        self._cache: dict = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def best_action(
        self,
        hands: dict[int, list[Card]],
        scores: dict[int, int],
        leader: int,
        lead_card: Card | None = None,
    ) -> tuple[int, Suit | None, float]:
        """
        Return (card_index_in_hand, marriage_suit | None, minimax_value).

        * If my_seat is leading  → lead_card is None.
        * If my_seat is responding → lead_card is the leader's card.

        The returned card_index refers to ``hands[my_seat]``.
        """
        self._cache.clear()
        seat = self.my_seat
        hand = hands[seat]

        if lead_card is None:
            # --- We are leading ---
            return self._best_lead(hands, scores, seat)
        else:
            # --- We are responding ---
            return self._best_response(hands, scores, leader, lead_card)

    # ------------------------------------------------------------------ #
    # Top-level best-move searches (called once at the root)
    # ------------------------------------------------------------------ #

    def _best_lead(self, hands, scores, seat):
        opp = 1 - seat
        hand = hands[seat]
        best_val = float("-inf")
        best_idx = 0
        best_mar: Suit | None = None

        for card, mar_suit in self._leader_options(hand):
            idx = hand.index(card)
            mar_pts = (rules.marriage_value(mar_suit, self.trump_suit)
                       if mar_suit else 0)

            new_scores = {0: scores[0], 1: scores[1]}
            new_scores[seat] += mar_pts

            # Check immediate 66 from marriage
            rw = self._check_66(new_scores)
            if rw is not None:
                val = self._terminal_value(new_scores, rw)
            else:
                remaining = [c for c in hand if c is not card]
                new_hands = {seat: remaining, opp: list(hands[opp])}
                val = self._solve_response(new_hands, new_scores, seat, card)

            if val > best_val:
                best_val = val
                best_idx = idx
                best_mar = mar_suit

        return best_idx, best_mar, best_val

    def _best_response(self, hands, scores, leader, lead_card):
        seat = self.my_seat
        opp = leader          # the leader is the opponent
        hand = hands[seat]
        valid = rules.get_valid_cards(hand, lead_card, self.trump_suit, phase=2)

        best_val = float("-inf")
        best_idx = 0

        for card in valid:
            idx = hand.index(card)
            winner, trick_pts = self._resolve_trick(lead_card, card, leader, seat)

            new_scores = {0: scores[0], 1: scores[1]}
            new_scores[winner] += trick_pts

            rw = self._check_66(new_scores)
            remaining = [c for c in hand if c is not card]
            new_hands = {seat: remaining, opp: list(hands[opp])}

            if rw is not None or self._hands_empty(new_hands):
                val = self._terminal_value(new_scores, rw)
            else:
                val = self._solve_trick(new_hands, new_scores, winner)

            if val > best_val:
                best_val = val
                best_idx = idx

        return best_idx, None, best_val

    # ------------------------------------------------------------------ #
    # Core minimax (internal recursion)
    # ------------------------------------------------------------------ #

    def _solve_trick(self, hands, scores, leader) -> float:
        """Leader picks a card. Returns value from my_seat's perspective."""
        if self._hands_empty(hands):
            return self._terminal_value(scores, None)

        key = self._state_key(hands, scores, leader)
        if key in self._cache:
            return self._cache[key]

        follower = 1 - leader
        is_max = (leader == self.my_seat)
        best = float("-inf") if is_max else float("inf")

        for card, mar_suit in self._leader_options(hands[leader]):
            mar_pts = (rules.marriage_value(mar_suit, self.trump_suit)
                       if mar_suit else 0)
            new_scores = {0: scores[0], 1: scores[1]}
            new_scores[leader] += mar_pts

            rw = self._check_66(new_scores)
            if rw is not None:
                val = self._terminal_value(new_scores, rw)
            else:
                remaining = [c for c in hands[leader] if c is not card]
                new_hands = {leader: remaining, follower: list(hands[follower])}
                val = self._solve_response(new_hands, new_scores, leader, card)

            best = max(best, val) if is_max else min(best, val)

        self._cache[key] = best
        return best

    def _solve_response(self, hands, scores, leader, lead_card) -> float:
        """Follower responds to lead_card. Returns value from my_seat's perspective."""
        follower = 1 - leader
        valid = rules.get_valid_cards(hands[follower], lead_card,
                                      self.trump_suit, phase=2)
        is_max = (follower == self.my_seat)
        best = float("-inf") if is_max else float("inf")

        for card in valid:
            winner, trick_pts = self._resolve_trick(lead_card, card,
                                                     leader, follower)
            new_scores = {0: scores[0], 1: scores[1]}
            new_scores[winner] += trick_pts

            remaining = [c for c in hands[follower] if c is not card]
            new_hands = {leader: list(hands[leader]), follower: remaining}

            rw = self._check_66(new_scores)
            if rw is not None or self._hands_empty(new_hands):
                val = self._terminal_value(new_scores, rw)
            else:
                val = self._solve_trick(new_hands, new_scores, winner)

            best = max(best, val) if is_max else min(best, val)

        return best

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _leader_options(self, hand: list[Card]):
        """Yield (card, marriage_suit | None) for each legal lead.

        Marriage is auto-detected: if the card is a K or Q and the
        partner is still in hand, the marriage suit is yielded.
        """
        marriages = rules.find_marriages(hand)
        for card in hand:
            mar_suit = None
            for suit in marriages:
                if card.suit == suit and card.rank in (" K", " Q"):
                    mar_suit = suit
                    break
            yield card, mar_suit

    def _resolve_trick(self, lead_card, resp_card, leader, follower):
        """Return (winner_seat, trick_points)."""
        rel = rules.trick_winner(lead_card, resp_card,
                                  lead_card.suit, self.trump_suit)
        trick_pts = lead_card.value() + resp_card.value()
        winner = leader if rel == 0 else follower
        return winner, trick_pts

    def _terminal_value(self, scores, round_winner) -> float:
        """Evaluate terminal position from my_seat's perspective."""
        winner, gp = rules.compute_game_points(
            scores, round_winner, closed=False, closed_by=None)
        if winner == self.my_seat:
            return float(gp)
        elif winner is not None:
            return float(-gp)
        return 0.0

    @staticmethod
    def _check_66(scores) -> int | None:
        """Return the seat that reached 66, or None."""
        for s in (0, 1):
            if scores[s] >= rules.WIN_SCORE:
                return s
        return None

    @staticmethod
    def _hands_empty(hands) -> bool:
        return not hands[0] and not hands[1]

    @staticmethod
    def _state_key(hands, scores, leader):
        return (
            frozenset(c.key() for c in hands[0]),
            frozenset(c.key() for c in hands[1]),
            scores[0], scores[1],
            leader,
        )
