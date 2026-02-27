"""
Game engine for Sixty-Six.
RoundState is an independent data object.
Round and Match drive the game loop, consulting the rules oracle and
asking symmetric Player agents for decisions.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from models import (
    Card, Suit, Action, ActionType, PlayerView,
    TrickResult, RoundResult, MatchResult, RANKS,
)
import rules

if TYPE_CHECKING:
    from agents import Player


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_deck() -> list[Card]:
    """Create a 24-card deck."""
    return [Card(rank, suit) for suit in Suit for rank in RANKS]


# ---------------------------------------------------------------------------
# RoundState — the independent, observable game state
# ---------------------------------------------------------------------------

class RoundState:
    """
    Full authoritative state of a round.
    Pure data — no decision-making, no UI.
    """

    def __init__(self, first_seat: int = 0, force_marriage_seat: int | None = None):
        deck = create_deck()
        random.shuffle(deck)

        self.hands: dict[int, list[Card]] = {0: [], 1: []}
        for _ in range(6):
            self.hands[0].append(deck.pop())
            self.hands[1].append(deck.pop())

        # If requested, re-deal until the target seat has a marriage
        if force_marriage_seat is not None:
            from rules import find_marriages
            while not find_marriages(self.hands[force_marriage_seat]):
                deck = create_deck()
                random.shuffle(deck)
                self.hands = {0: [], 1: []}
                for _ in range(6):
                    self.hands[0].append(deck.pop())
                    self.hands[1].append(deck.pop())

        self.trump_card: Card | None = deck.pop()
        self.trump_suit: Suit = self.trump_card.suit
        self.draw_pile: list[Card] = deck          # remaining 11 cards

        self.scores: dict[int, int] = {0: 0, 1: 0}
        self.leader: int = first_seat              # who leads the next trick

        self.closed: bool = False
        self.closed_by: int | None = None

        self.round_winner: int | None = None

        # Per-player memory of observed cards
        self.seen_cards: dict[int, set[tuple[str, str]]] = {0: set(), 1: set()}
        for seat in (0, 1):
            for card in self.hands[seat]:
                self.seen_cards[seat].add(card.key())
            self.seen_cards[seat].add(self.trump_card.key())

        # Cards that have been played in tricks (public knowledge)
        self.played_cards: list[Card] = []

        # Cards won (captured) by each player in tricks
        self.won_cards: dict[int, list[Card]] = {0: [], 1: []}

        # Display helpers (not game logic)
        self.last_drawn: dict[int, Card | None] = {0: None, 1: None}
        self.last_trick_info: str = ""

    # ------------------------------------------------------------------
    @property
    def phase(self) -> int:
        if self.closed:
            return 2
        if not self.draw_pile and self.trump_card is None:
            return 2
        return 1

    # ------------------------------------------------------------------
    def player_sees_card(self, seat: int, card: Card):
        """Record that *seat* has observed *card*."""
        self.seen_cards[seat].add(card.key())

    # ------------------------------------------------------------------
    def player_view(self, seat: int, lead_card: Card | None = None,
                    is_winner_action: bool = False,
                    match_scores: dict[int, int] | None = None) -> PlayerView:
        """
        Build a PlayerView for *seat*.
        The rules oracle computes valid_actions from the raw state.
        """
        opp = 1 - seat
        hand = self.hands[seat]
        valid_actions = rules.get_valid_actions(
            hand, self.trump_suit, self.trump_card,
            lead_card, self.phase, is_winner_action,
        )
        return PlayerView(
            seat=seat,
            hand=list(hand),              # copy so agent can't mutate
            trump_suit=self.trump_suit,
            trump_card=self.trump_card,
            draw_pile_size=len(self.draw_pile),
            phase=self.phase,
            closed=self.closed,
            closed_by=self.closed_by,
            my_score=self.scores[seat],
            opponent_score=self.scores[opp],
            is_leading=(self.leader == seat) and (lead_card is None),
            lead_card=lead_card,
            lead_marriage=None,
            valid_actions=valid_actions,
            is_winner_action_phase=is_winner_action,
            seen_cards=set(self.seen_cards[seat]),
            played_cards=set(self.played_cards),
            my_won_cards=list(self.won_cards[seat]),
            opponent_won_cards=list(self.won_cards[opp]),
            opponent_hand_size=len(self.hands[opp]),
            opponent_hand=list(self.hands[opp]),
            last_trick_info=self.last_trick_info,
            last_drawn=self.last_drawn[seat],
            match_scores=match_scores or {},
        )


# ---------------------------------------------------------------------------
# Round — the game-loop engine
# ---------------------------------------------------------------------------

class Round:
    """
    Drives one round of Sixty-Six.
    Both seats are symmetric Player agents.
    """

    def __init__(self, players: dict[int, "Player"], first_seat: int | None = None,
                 force_marriage_seat: int | None = None):
        if first_seat is None:
            first_seat = random.choice([0, 1])
        self.players = players
        self.state = RoundState(first_seat, force_marriage_seat=force_marriage_seat)
        self.match_scores: dict[int, int] = {0: 0, 1: 0}
        self._current_lead_card: Card | None = None  # used by execute_action

    # ------------------------------------------------------------------
    # Action execution (mutates RoundState)
    # ------------------------------------------------------------------

    def execute_action(self, seat: int, action: Action) -> tuple[Card | None, int, Suit | None]:
        """
        Apply *action* by *seat* to the round state.
        Returns (card_played, marriage_points, marriage_suit).
        """
        st = self.state
        hand = st.hands[seat]
        opp = 1 - seat
        at = action.type.value           # string comparison (safe across modules)

        if at == "swap_trump":
            nine = rules.find_nine_trump(hand, st.trump_suit)
            if nine and st.trump_card:
                hand.remove(nine)
                hand.append(st.trump_card)
                st.trump_card = nine
                # Both players see the new face-up trump
                st.player_sees_card(0, nine)
                st.player_sees_card(1, nine)
            return None, 0, None

        if at == "close_game":
            st.closed = True
            st.closed_by = seat
            return None, 0, None

        if at == "pass":
            return None, 0, None

        if at == "play_card":
            card = hand[action.card_index]
            hand.remove(card)
            st.last_drawn[seat] = None    # clear "just drawn" marker

            # Opponent sees the played card
            st.player_sees_card(opp, card)

            # Auto-detect marriage: if leading with K or Q from a pair
            marriage_points = 0
            if action.marriage_suit:
                # Legacy path (e.g. solver still passes explicit marriage)
                mar_suit = action.marriage_suit
            elif (self._current_lead_card is None
                  and card.rank in (" K", " Q")):
                partner_rank = " Q" if card.rank == " K" else " K"
                if any(c.rank == partner_rank and c.suit == card.suit
                       for c in hand):
                    mar_suit = card.suit
                else:
                    mar_suit = None
            else:
                mar_suit = None

            if mar_suit:
                st.player_sees_card(opp, Card(" K", mar_suit))
                st.player_sees_card(opp, Card(" Q", mar_suit))
                marriage_points = rules.marriage_value(mar_suit, st.trump_suit)
            return card, marriage_points, mar_suit

        return None, 0, None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_cards(self):
        """Both players draw a card after a trick (phase 1 only)."""
        st = self.state
        st.last_drawn = {0: None, 1: None}

        if st.closed:
            return

        leader, follower = st.leader, 1 - st.leader

        if st.draw_pile:
            drawn = st.draw_pile.pop()
            st.hands[leader].append(drawn)
            st.last_drawn[leader] = drawn
            st.player_sees_card(leader, drawn)

            if st.draw_pile:
                drawn2 = st.draw_pile.pop()
                st.hands[follower].append(drawn2)
                st.player_sees_card(follower, drawn2)
            elif st.trump_card:
                st.hands[follower].append(st.trump_card)
                st.player_sees_card(follower, st.trump_card)
                st.trump_card = None
        elif st.trump_card:
            st.hands[leader].append(st.trump_card)
            st.last_drawn[leader] = st.trump_card
            st.player_sees_card(leader, st.trump_card)
            st.trump_card = None

    # ------------------------------------------------------------------
    # Trick play
    # ------------------------------------------------------------------

    def play_trick(self) -> TrickResult:
        """Play one trick, asking both agents for their actions."""
        st = self.state
        leader = st.leader
        follower = 1 - leader
        marriages: dict[int, int] = {0: 0, 1: 0}

        # --- Leader plays ---
        self._current_lead_card = None          # signal: this is the lead
        view_l = st.player_view(leader, lead_card=None,
                                match_scores=self.match_scores)
        action_l = self.players[leader].choose_action(view_l)
        card_l, mar_l, mar_suit_l = self.execute_action(leader, action_l)
        if mar_l:
            marriages[leader] = mar_l
            st.scores[leader] += mar_l

        # --- Follower plays (sees leader's card) ---
        self._current_lead_card = card_l        # signal: this is a response
        view_f = st.player_view(follower, lead_card=card_l,
                                match_scores=self.match_scores)
        view_f.lead_marriage = mar_suit_l
        action_f = self.players[follower].choose_action(view_f)
        card_f, mar_f, _mar_suit_f = self.execute_action(follower, action_f)
        if mar_f:
            marriages[follower] = mar_f
            st.scores[follower] += mar_f

        # --- Resolve trick ---
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

        # Check for 66
        for s in (0, 1):
            if st.scores[s] >= rules.WIN_SCORE:
                st.round_winner = s

        # Show both cards on the table before resolving
        for seat in (0, 1):
            view = st.player_view(seat, lead_card=None,
                                  match_scores=self.match_scores)
            self.players[seat].notify_trick_cards(view, cards, leader, marriages)

        result = TrickResult(
            cards=cards,
            winner=winner,
            trick_points=trick_points,
            marriages=marriages,
        )

        # Build info string (for display)
        winner_label = self.players[winner].name
        loser = 1 - winner
        st.last_trick_info = (
            f"{winner_label} won +{trick_points} pts "
            f"({cards[winner]} beat {cards[loser]})"
        )

        return result

    # ------------------------------------------------------------------
    # Winner actions (swap / close, phase 1 only)
    # ------------------------------------------------------------------

    def winner_actions(self, seat: int):
        """
        Let the leading seat perform optional actions (swap 9-trump, close)
        after winning a trick in phase 1.
        """
        st = self.state
        player = self.players[seat]

        while True:
            view = st.player_view(seat, is_winner_action=True,
                                  match_scores=self.match_scores)
            action = player.choose_action(view)
            at = action.type.value

            if at == "swap_trump":
                old_trump = st.trump_card
                self.execute_action(seat, action)
                player.notify_swap(old_trump, st.trump_card)
                # After swap, player might also want to close
                continue
            elif at == "close_game":
                self.execute_action(seat, action)
                player.notify_close(seat)
                return
            else:   # pass
                return

    # ------------------------------------------------------------------
    # Round loop
    # ------------------------------------------------------------------

    def play(self, match_scores: dict[int, int] | None = None) -> RoundResult:
        """Play the full round. Returns RoundResult."""
        self.match_scores = match_scores or {0: 0, 1: 0}
        st = self.state

        while st.hands[0] and st.hands[1] and st.round_winner is None:
            result = self.play_trick()

            # Notify players of trick result
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

        rr = RoundResult(
            winner=winner,
            game_points=game_points,
            scores=dict(st.scores),
            closed=st.closed,
            closed_by=st.closed_by,
        )

        for seat in (0, 1):
            self.players[seat].notify_round_result(rr, self.match_scores)

        return rr


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

class Match:
    """
    Best-of rounds until one seat reaches 7 game points.
    Both seats are symmetric Player agents.
    """

    WIN_POINTS = 7

    def __init__(self, players: dict[int, "Player"],
                 force_marriage_seat: int | None = None):
        self.players = players
        self.game_points: dict[int, int] = {0: 0, 1: 0}
        self.round_number = 0
        self.first_seat: int = random.choice([0, 1])
        self._force_marriage_seat = force_marriage_seat

        # If forcing a marriage, that seat leads first so they can use it
        if force_marriage_seat is not None:
            self.first_seat = force_marriage_seat

        self._next_round: Round | None = None

    def play(self) -> MatchResult:
        for seat in (0, 1):
            self.players[seat].notify_match_start()

        while all(gp < self.WIN_POINTS for gp in self.game_points.values()):
            self.round_number += 1

            if self._next_round is not None:
                rnd = self._next_round
                self._next_round = None
            else:
                rnd = Round(self.players, first_seat=self.first_seat,
                            force_marriage_seat=self._force_marriage_seat)
                # Show dealing animation for the first round
                for seat in (0, 1):
                    self.players[seat].notify_next_round(
                        list(rnd.state.hands[seat]), first_round=True)
            rr = rnd.play(match_scores=dict(self.game_points))

            if rr.winner is not None:
                self.game_points[rr.winner] += rr.game_points

            self.first_seat = 1 - self.first_seat    # alternate

            if all(gp < self.WIN_POINTS for gp in self.game_points.values()):
                # Create next round early so we can show the dealt hand
                self._next_round = Round(self.players, first_seat=self.first_seat,
                                         force_marriage_seat=self._force_marriage_seat)
                for seat in (0, 1):
                    self.players[seat].notify_next_round(
                        list(self._next_round.state.hands[seat]))

        winner = 0 if self.game_points[0] >= self.WIN_POINTS else 1
        mr = MatchResult(
            winner=winner,
            game_points=dict(self.game_points),
            rounds_played=self.round_number,
        )
        for seat in (0, 1):
            self.players[seat].notify_match_result(mr)

        return mr
