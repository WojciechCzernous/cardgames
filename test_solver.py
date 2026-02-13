#!/usr/bin/env python3
"""
Test & visualize the endgame minimax solver.

Usage:
    python test_solver.py              # run all tests
    python test_solver.py --tree       # also print full game trees
    python test_solver.py --tree -n 1  # tree for test case 1 only
"""

from __future__ import annotations

import sys
from models import Card, Suit, RANKS, RANK_VALUES
import rules


# ──────────────────────────────────────────────────────────────────────
# Pretty-printing helpers
# ──────────────────────────────────────────────────────────────────────

def c(rank: str, suit: str) -> Card:
    """Shorthand card constructor: c("Q", "♠") or c("10", "♥")."""
    suit_map = {"♥": Suit.HEARTS, "♦": Suit.DIAMONDS,
                "♣": Suit.CLUBS,  "♠": Suit.SPADES}
    r = rank.rjust(2)
    return Card(r, suit_map[suit])


def fmt(card: Card) -> str:
    return f"{card.rank.strip()}{card.suit.value}"


def hand_str(hand: list[Card]) -> str:
    return " ".join(fmt(c) for c in hand)


# ──────────────────────────────────────────────────────────────────────
# Tree-building solver (extends EndgameSolver to record the game tree)
# ──────────────────────────────────────────────────────────────────────

class TreeNode:
    """A node in the minimax game tree."""

    def __init__(self, *, kind: str, seat: int, label: str,
                 value: float | None = None):
        self.kind = kind        # "lead", "response", "terminal"
        self.seat = seat        # who is choosing
        self.label = label      # human-readable description
        self.value = value      # minimax value (filled after solve)
        self.children: list[tuple[str, TreeNode]] = []   # (action_label, child)

    def pprint(self, prefix="", is_last=True, best_path: set | None = None):
        """Pretty-print the tree with box-drawing characters."""
        connector = "└── " if is_last else "├── "
        marker = ""
        if best_path and id(self) in best_path:
            marker = " ★"
        val_str = f" → {self.value:+.1f}" if self.value is not None else ""
        seat_tag = f"[s{self.seat}]" if self.kind != "terminal" else ""
        print(f"{prefix}{connector}{seat_tag} {self.label}{val_str}{marker}")

        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, (action_lbl, child) in enumerate(self.children):
            last = (i == len(self.children) - 1)
            act_conn = "└── " if last else "├── "
            print(f"{child_prefix}{act_conn}▸ {action_lbl}")
            child.pprint(child_prefix + ("    " if last else "│   "),
                         is_last=True, best_path=best_path)


class TracingSolver:
    """Minimax solver that builds a game tree for visualization."""

    def __init__(self, trump_suit: Suit, my_seat: int,
                 closed: bool = False, closed_by: int | None = None):
        self.trump_suit = trump_suit
        self.my_seat = my_seat
        self.closed = closed
        self.closed_by = closed_by

    def solve_tree(self, hands, scores, leader) -> TreeNode:
        """Solve and return the root TreeNode."""
        return self._solve_trick_tree(hands, scores, leader)

    # ── internals ─────────────────────────────────────────────────────

    def _solve_trick_tree(self, hands, scores, leader) -> TreeNode:
        """Leader picks a card → response subtree."""
        if not hands[0] and not hands[1]:
            return self._terminal_node(scores, None)

        follower = 1 - leader
        is_max = (leader == self.my_seat)
        node = TreeNode(kind="lead", seat=leader,
                        label=f"Lead (s{leader}, {'MAX' if is_max else 'MIN'})  "
                              f"hand: {hand_str(hands[leader])}  "
                              f"score: {scores[0]}-{scores[1]}")

        best = float("-inf") if is_max else float("inf")

        for card, mar_suit in self._leader_options(hands[leader]):
            mar_pts = rules.marriage_value(mar_suit, self.trump_suit) if mar_suit else 0
            new_scores = {0: scores[0], 1: scores[1]}
            new_scores[leader] += mar_pts

            marriage_tag = f" 💍+{mar_pts}" if mar_suit else ""
            action_lbl = f"{fmt(card)}{marriage_tag}"

            rw = self._check_66(new_scores)
            if rw is not None:
                child = self._terminal_node(new_scores, rw)
            else:
                remaining = [c for c in hands[leader] if c is not card]
                new_hands = {leader: remaining, follower: list(hands[follower])}
                child = self._solve_response_tree(new_hands, new_scores, leader, card)

            node.children.append((action_lbl, child))
            val = child.value
            best = max(best, val) if is_max else min(best, val)

        node.value = best
        return node

    def _solve_response_tree(self, hands, scores, leader, lead_card) -> TreeNode:
        """Follower responds to lead_card."""
        follower = 1 - leader
        valid = rules.get_valid_cards(hands[follower], lead_card, self.trump_suit, phase=2)
        is_max = (follower == self.my_seat)
        node = TreeNode(kind="response", seat=follower,
                        label=f"Respond (s{follower}, {'MAX' if is_max else 'MIN'})  "
                              f"to {fmt(lead_card)}  "
                              f"hand: {hand_str(hands[follower])}  "
                              f"score: {scores[0]}-{scores[1]}")

        best = float("-inf") if is_max else float("inf")

        for card in valid:
            rel = rules.trick_winner(lead_card, card, lead_card.suit, self.trump_suit)
            winner = leader if rel == 0 else follower
            trick_pts = lead_card.value() + card.value()

            new_scores = {0: scores[0], 1: scores[1]}
            new_scores[winner] += trick_pts

            win_lbl = f"→s{winner} +{trick_pts}"
            action_lbl = f"{fmt(card)}  ({win_lbl})"

            remaining = [cc for cc in hands[follower] if cc is not card]
            new_hands = {leader: list(hands[leader]), follower: remaining}

            rw = self._check_66(new_scores)
            if rw is not None or (not new_hands[0] and not new_hands[1]):
                child = self._terminal_node(new_scores, rw)
            else:
                child = self._solve_trick_tree(new_hands, new_scores, winner)

            node.children.append((action_lbl, child))
            val = child.value
            best = max(best, val) if is_max else min(best, val)

        node.value = best
        return node

    def _terminal_node(self, scores, round_winner) -> TreeNode:
        winner, gp = rules.compute_game_points(
            scores, round_winner, self.closed, self.closed_by)
        if winner == self.my_seat:
            val = float(gp)
        elif winner is not None:
            val = float(-gp)
        else:
            val = 0.0
        who = f"s{winner}" if winner is not None else "tie"
        label = (f"Terminal  score: {scores[0]}-{scores[1]}  "
                 f"winner: {who}  gp: {gp}")
        return TreeNode(kind="terminal", seat=-1, label=label, value=val)

    def _leader_options(self, hand):
        marriages = rules.find_marriages(hand)
        for card in hand:
            for suit in marriages:
                if card.suit == suit and card.rank in (" K", " Q"):
                    yield card, suit
            yield card, None

    @staticmethod
    def _check_66(scores):
        for s in (0, 1):
            if scores[s] >= rules.WIN_SCORE:
                return s
        return None


# ──────────────────────────────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────────────────────────────

def test_cases() -> list[dict]:
    """Return a list of test scenarios with expected properties."""
    return [
        # ── Case 1: Trivial 1-card each ──
        {
            "name": "1 card each — A beats 10, trump ♥",
            "trump": Suit.HEARTS,
            "my_seat": 0,
            "hands": {0: [c("A", "♠")], 1: [c("10", "♠")]},
            "scores": {0: 50, 1: 40},
            "leader": 0,
            "expect_card": "A♠",
            "expect_positive": True,   # seat 0 should win
        },
        # ── Case 2: 1 card, trump beats ace ──
        {
            "name": "1 card — trump 9 beats off-suit A",
            "trump": Suit.HEARTS,
            "my_seat": 0,
            "hands": {0: [c("9", "♥")], 1: [c("A", "♠")]},
            "scores": {0: 55, 1: 55},
            "leader": 0,
            "expect_card": "9♥",
            "expect_positive": True,
        },
        # ── Case 3: 2 cards, must-follow forces a loss ──
        {
            "name": "2 cards — must follow suit (phase 2)",
            "trump": Suit.HEARTS,
            "my_seat": 0,
            "hands": {0: [c("Q", "♠"), c("K", "♣")],
                      1: [c("A", "♠"), c("9", "♣")]},
            "scores": {0: 40, 1: 40},
            "leader": 0,
        },
        # ── Case 4: 2 cards with a marriage ──
        {
            "name": "2 cards — marriage available (K+Q♠, trump ♥)",
            "trump": Suit.HEARTS,
            "my_seat": 0,
            "hands": {0: [c("K", "♠"), c("Q", "♠")],
                      1: [c("A", "♠"), c("10", "♣")]},
            "scores": {0: 40, 1: 50},
            "leader": 0,
        },
        # ── Case 5: 3 cards, deeper tree ──
        {
            "name": "3 cards — deeper lookahead",
            "trump": Suit.DIAMONDS,
            "my_seat": 0,
            "hands": {0: [c("A", "♠"), c("10", "♦"), c("J", "♣")],
                      1: [c("K", "♠"), c("Q", "♦"), c("9", "♣")]},
            "scores": {0: 30, 1: 35},
            "leader": 0,
        },
        # ── Case 6: Marriage wins by reaching 66 ──
        {
            "name": "Marriage reaches 66 instantly",
            "trump": Suit.HEARTS,
            "my_seat": 0,
            "hands": {0: [c("K", "♥"), c("Q", "♥")],
                      1: [c("A", "♠"), c("10", "♠")]},
            "scores": {0: 30, 1: 60},
            "leader": 0,
            "expect_positive": True,
        },
    ]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    show_tree = "--tree" in sys.argv
    case_filter = None
    if "-n" in sys.argv:
        idx = sys.argv.index("-n")
        if idx + 1 < len(sys.argv):
            case_filter = int(sys.argv[idx + 1])

    from solver import EndgameSolver

    cases = test_cases()
    print("=" * 65)
    print("  Endgame Solver Tests")
    print("=" * 65)

    for i, tc in enumerate(cases, 1):
        if case_filter is not None and i != case_filter:
            continue

        print(f"\n{'─' * 65}")
        print(f"  Case {i}: {tc['name']}")
        print(f"{'─' * 65}")
        print(f"  Trump: {tc['trump'].value}")
        print(f"  Seat 0 hand: {hand_str(tc['hands'][0])}")
        print(f"  Seat 1 hand: {hand_str(tc['hands'][1])}")
        print(f"  Scores: {tc['scores'][0]}-{tc['scores'][1]}")
        print(f"  Leader: seat {tc['leader']}  |  Solver plays: seat {tc['my_seat']}")

        # Run solver
        solver = EndgameSolver(
            trump_suit=tc["trump"], my_seat=tc["my_seat"],
            closed=tc.get("closed", False),
            closed_by=tc.get("closed_by"),
        )
        idx, mar, val = solver.best_action(
            {k: list(v) for k, v in tc["hands"].items()},
            dict(tc["scores"]), tc["leader"])

        best_card = tc["hands"][tc["my_seat"]][idx]
        mar_str = f" (marriage {mar.value})" if mar else ""
        print(f"\n  ➤ Best play: {fmt(best_card)}{mar_str}")
        print(f"  ➤ Minimax value: {val:+.1f} game points")

        # Validate expectations
        if "expect_card" in tc:
            actual = fmt(best_card)
            if actual == tc["expect_card"]:
                print(f"  ✓ Card check passed ({actual})")
            else:
                print(f"  ✗ Card check FAILED: expected {tc['expect_card']}, got {actual}")

        if "expect_positive" in tc:
            if tc["expect_positive"] == (val > 0):
                print(f"  ✓ Sign check passed (value {'> 0' if val > 0 else '≤ 0'})")
            else:
                print(f"  ✗ Sign check FAILED: expected {'positive' if tc['expect_positive'] else 'non-positive'}, got {val}")

        # Tree visualization
        if show_tree:
            print(f"\n  Game Tree (from seat {tc['my_seat']}'s perspective):")
            print()
            tracer = TracingSolver(
                trump_suit=tc["trump"], my_seat=tc["my_seat"],
                closed=tc.get("closed", False),
                closed_by=tc.get("closed_by"),
            )
            tree = tracer.solve_tree(
                {k: list(v) for k, v in tc["hands"].items()},
                dict(tc["scores"]), tc["leader"])
            tree.pprint(prefix="  ")
            print()

    print(f"\n{'=' * 65}")
    print(f"  Done — {len(cases) if case_filter is None else 1} case(s) run.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
