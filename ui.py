"""
Terminal UI for Sixty-Six.
Used exclusively by HumanPlayer for display and input.
No game logic — only presentation.
"""

from models import (
    Card, Suit, Action, ActionType, PlayerView,
    TrickResult, RoundResult, MatchResult, RANKS,
)
from rules import marriage_value

# ANSI codes
RED = "\033[91m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[H"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def colored_suit(suit: Suit) -> str:
    if suit in (Suit.HEARTS, Suit.DIAMONDS):
        return f"{RED}{suit.value}{RESET}"
    return suit.value


def card_str(card: Card) -> str:
    s = f"{card.rank}{card.suit.value}"
    if card.suit in (Suit.HEARTS, Suit.DIAMONDS):
        return f"{RED}{s}{RESET}"
    return s


def display_hand(hand: list[Card], show_numbers: bool = True) -> str:
    if show_numbers:
        cards_line = "  ".join(card_str(c) for c in hand)
        nums_line = "  ".join(f"[{i+1}]" for i in range(len(hand)))
        return f"{cards_line}\n{nums_line}"
    return "  ".join(card_str(c) for c in hand)


def display_hidden(count: int) -> str:
    return " ".join(["[?]"] * count)


# ---------------------------------------------------------------------------
# TerminalUI
# ---------------------------------------------------------------------------

class TerminalUI:
    """
    All terminal I/O for the human seat.
    The human's seat number is set when we know it (usually 0).
    """

    def __init__(self, reveal_opponent: bool = False,
                 show_player_view: bool = False,
                 show_hints: bool = False):
        self.seat: int = 0
        self._opponent_name = "Opponent"
        self.reveal_opponent = reveal_opponent
        self.show_player_view = show_player_view
        self.show_hints = show_hints

    def set_context(self, seat: int, opponent_name: str):
        self.seat = seat
        self._opponent_name = opponent_name

    # ------------------------------------------------------------------
    # Screen helpers
    # ------------------------------------------------------------------

    def clear_screen(self):
        print(CLEAR, end="")

    # ------------------------------------------------------------------
    # Welcome
    # ------------------------------------------------------------------

    def show_welcome(self) -> None:
        self.clear_screen()
        print("=" * 50)
        print("         WELCOME TO SIXTY-SIX!")
        print("=" * 50)
        print("\nRound Rules:")
        print("- First to 66 points wins the round")
        print("- Trump suit beats other suits")
        print("- Higher rank wins within same suit")
        print("- Card values: A=11, 10=10, K=4, Q=3, J=2, 9=0")
        print("- Marriage (K+Q same suit): 20 pts, trump marriage: 40 pts")
        print("\nPhases:")
        print("- Phase 1 (draw pile active): play any card")
        print("  Winner can: swap 9-trump for trump card, or close game")
        print("- Phase 2 (draw pile empty or closed): must follow suit,")
        print("          else must trump, else any card")
        print("\nGame Points (first to 7 wins match):")
        print("- Win round: 1 point")
        print("- Opponent < 33 pts: 2 points")
        print("- Closed game: 3 points (winner takes all)")
        print("- Both < 66 (no close): tie, 0 points")
        input("\nPress Enter to start the match...")

    # ------------------------------------------------------------------
    # State display
    # ------------------------------------------------------------------

    def display_state(self, view: PlayerView) -> None:
        self.clear_screen()

        opp = self._opponent_name

        # Header
        print("=" * 50)
        if view.match_scores:
            my_gp = view.match_scores.get(view.seat, 0)
            opp_gp = view.match_scores.get(1 - view.seat, 0)
            print(f"    SIXTY-SIX          Match: You {my_gp} - {opp_gp} {opp}")
        else:
            print("              SIXTY-SIX")
        print("=" * 50)

        # Trump / draw / phase
        trump_display = card_str(view.trump_card) if view.trump_card else f"[{colored_suit(view.trump_suit)}]"
        if view.closed:
            who = "you" if view.closed_by == view.seat else opp.lower()
            phase_info = f"CLOSED by {who}"
        elif view.phase == 1:
            phase_info = "Phase 1 (free play)"
        else:
            phase_info = "Phase 2 (must follow)"
        print(f"Trump: {trump_display}  |  Draw pile: {view.draw_pile_size} cards  |  {phase_info}")
        print(f"Score - You: {view.my_score:3d}  |  {opp}: {view.opponent_score:3d}")
        print("-" * 50)

        # Last trick info
        if view.last_trick_info:
            print(f"Last: {view.last_trick_info}")
        else:
            print()
        print()

        # Opponent hand
        if self.reveal_opponent and view.draw_pile_size == 0 and view.opponent_hand:
            print(f"{opp}: {display_hand(view.opponent_hand, show_numbers=False)}")
        else:
            print(f"{opp}: {display_hidden(view.opponent_hand_size)}")

        # Opponent hand inference hints
        if self.show_hints:
            if view.opponent_known_cards:
                known = sorted(view.opponent_known_cards, key=lambda c: c.key())
                print(f"  Known in hand: {', '.join(card_str(c) for c in known)}")
            if view.opponent_void_suits:
                void = sorted(view.opponent_void_suits, key=lambda s: s.value)
                print(f"  Void in: {', '.join(colored_suit(s) for s in void)}")
        print()

        # Table area
        print("─" * 20 + " TABLE " + "─" * 23)
        if view.lead_card:
            marriage_info = ""
            if view.lead_marriage:
                pts = marriage_value(view.lead_marriage, view.trump_suit)
                marriage_info = f"  💍 +{pts}"
            print(f"  {opp} played: {card_str(view.lead_card)}{marriage_info}")
        else:
            print()
        print()
        print("─" * 50)
        print()

        # Player hand
        drawn = f"  (drew: {card_str(view.last_drawn)})" if view.last_drawn else ""
        print(f"Your hand:{drawn}")
        print(display_hand(view.hand))
        print()

        # --- PlayerView dump (all fields by code name) ---
        if not self.show_player_view:
            return
        print("─" * 50)
        print("PlayerView fields:")
        print(f"  seat               = {view.seat}")
        print(f"  hand               = {[str(c) for c in view.hand]}")
        print(f"  trump_suit         = {view.trump_suit}")
        print(f"  trump_card         = {view.trump_card}")
        print(f"  draw_pile_size     = {view.draw_pile_size}")
        print(f"  phase              = {view.phase}")
        print(f"  closed             = {view.closed}")
        print(f"  closed_by          = {view.closed_by}")
        print(f"  my_score           = {view.my_score}")
        print(f"  opponent_score     = {view.opponent_score}")
        print(f"  is_leading         = {view.is_leading}")
        print(f"  lead_card          = {view.lead_card}")
        print(f"  lead_marriage      = {view.lead_marriage}")
        print(f"  valid_actions      = {view.valid_actions}")
        print(f"  is_winner_action_phase = {view.is_winner_action_phase}")
        print(f"  seen_cards         = {sorted(view.seen_cards)}")
        print(f"  played_cards       = {sorted(str(c) for c in view.played_cards)}")
        print(f"  my_won_cards       = {[str(c) for c in view.my_won_cards]}")
        print(f"  opponent_won_cards = {[str(c) for c in view.opponent_won_cards]}")
        print(f"  opponent_known_cards = {sorted(str(c) for c in view.opponent_known_cards)}")
        print(f"  opponent_void_suits  = {sorted(s.name for s in view.opponent_void_suits)}")
        print(f"  opponent_hand_size = {view.opponent_hand_size}")
        print(f"  opponent_hand      = {view.opponent_hand}")
        print(f"  last_trick_info    = {view.last_trick_info!r}")
        print(f"  last_drawn         = {view.last_drawn}")
        print(f"  match_scores       = {view.match_scores}")
        print("─" * 50)
        print()

    # ------------------------------------------------------------------
    # Card play prompt
    # ------------------------------------------------------------------

    def prompt_card_play(self, view: PlayerView) -> Action:
        from rules import find_marriages, marriage_value
        marriages = find_marriages(view.hand) if view.is_leading else []
        error_msg = ""

        while True:
            if view.lead_card:
                if view.phase == 1:
                    print(
                        f"Lead: {colored_suit(view.lead_card.suit)} (any card allowed)")
                else:
                    valid_cards = [
                        view.hand[a.card_index] for a in view.valid_actions
                        if a.type.value == "play_card"]
                    print(
                        f"Must follow: {colored_suit(view.lead_card.suit)}  |  "
                        f"Valid: {display_hand(valid_cards, show_numbers=False)}")
            else:
                lead_msg = ">>> Your lead!"
                if marriages:
                    parts = [
                        f"{colored_suit(s)} ({marriage_value(s, view.trump_suit)}pts)"
                        for s in marriages]
                    lead_msg += f"  Marriages available: {', '.join(parts)}"
                print(lead_msg)

            if error_msg:
                print(f"\n⚠ {error_msg}")
                error_msg = ""

            try:
                prompt = f"\nPlay card [1-{len(view.hand)}]: "
                choice = input(prompt).strip().lower()

                idx = int(choice) - 1
                if 0 <= idx < len(view.hand):
                    for action in view.valid_actions:
                        if (action.type.value == "play_card"
                                and action.card_index == idx):
                            return action
                    error_msg = "Invalid card for current situation!"
                else:
                    error_msg = "Invalid card number!"
            except ValueError:
                error_msg = "Please enter a number!"

    # ------------------------------------------------------------------
    # Winner action prompt
    # ------------------------------------------------------------------

    def prompt_winner_action(self, view: PlayerView) -> Action:
        has_swap = any(a.type.value == "swap_trump"
                       for a in view.valid_actions)
        while True:
            print("Winner actions (Phase 1):")
            if has_swap:
                nine = next(
                    (c for c in view.hand
                     if c.rank == " 9" and c.suit == view.trump_suit), None)
                if nine and view.trump_card:
                    print(
                        f"  [S] Swap {card_str(nine)} with trump {card_str(view.trump_card)}")
            print("  [C] Close the game (enter Phase 2)")
            print("  [Enter] Continue to play")
            choice = input("\nYour choice: ").strip().lower()
            if choice == 's' and has_swap:
                return Action(ActionType.SWAP_TRUMP)
            elif choice == 'c':
                return Action(ActionType.CLOSE_GAME)
            elif choice == '':
                return Action(ActionType.PASS)

    # ------------------------------------------------------------------
    # Table display (both cards played)
    # ------------------------------------------------------------------

    def show_table(self, view: PlayerView,
                   table_cards: dict[int, Card],
                   leader: int = 0,
                   marriages: dict[int, int] | None = None) -> None:
        """Show the table with both cards (lead first, response second), pause 2 seconds."""
        import time
        self.clear_screen()

        opp = self._opponent_name
        opp_seat = 1 - self.seat
        follower = 1 - leader
        if marriages is None:
            marriages = {}

        # Header
        print("=" * 50)
        if view.match_scores:
            my_gp = view.match_scores.get(view.seat, 0)
            opp_gp = view.match_scores.get(1 - view.seat, 0)
            print(f"    SIXTY-SIX          Match: You {my_gp} - {opp_gp} {opp}")
        else:
            print("              SIXTY-SIX")
        print("=" * 50)

        # Trump / draw / phase
        trump_display = card_str(view.trump_card) if view.trump_card else f"[{colored_suit(view.trump_suit)}]"
        if view.closed:
            who = "you" if view.closed_by == view.seat else opp.lower()
            phase_info = f"CLOSED by {who}"
        elif view.phase == 1:
            phase_info = "Phase 1 (free play)"
        else:
            phase_info = "Phase 2 (must follow)"
        print(f"Trump: {trump_display}  |  Draw pile: {view.draw_pile_size} cards  |  {phase_info}")
        print(f"Score - You: {view.my_score:3d}  |  {opp}: {view.opponent_score:3d}")
        print("-" * 50)
        print()
        print()

        # Opponent hand
        if self.reveal_opponent and view.draw_pile_size == 0 and view.opponent_hand:
            print(f"{opp}: {display_hand(view.opponent_hand, show_numbers=False)}")
        else:
            print(f"{opp}: {display_hidden(view.opponent_hand_size)}")
        print()

        # Table with both cards — lead first, response second
        def seat_label(s: int) -> str:
            return "You" if s == self.seat else opp

        def marriage_tag(s: int) -> str:
            pts = marriages.get(s, 0)
            if pts:
                return f"  💍 +{pts}"
            return ""

        print("─" * 20 + " TABLE " + "─" * 23)
        print(f"  Lead:     {seat_label(leader):>10s}  {card_str(table_cards[leader])}{marriage_tag(leader)}")
        print(f"  Response: {seat_label(follower):>10s}  {card_str(table_cards[follower])}{marriage_tag(follower)}")
        print("─" * 50)
        print()

        # Player hand
        print(f"Your hand:")
        print(display_hand(view.hand))
        print()

        time.sleep(2)

    # ------------------------------------------------------------------
    # Result notifications
    # ------------------------------------------------------------------

    def show_trick_result(self, result: TrickResult,
                          score_0: int, score_1: int,
                          round_winner: int | None) -> None:
        for seat, pts in result.marriages.items():
            if pts:
                who = "You" if seat == self.seat else self._opponent_name
                print(f"💍 {who} announced marriage! +{pts} points")

        if round_winner is not None:
            who = "You" if round_winner == self.seat else self._opponent_name
            print(f"\n🌟 {who} reached 66 points!")
        else:
            who_won = "You win!" if result.winner == self.seat else f"{self._opponent_name} wins!"
            symbol = "✓" if result.winner == self.seat else "✗"
            print(f"\n{symbol} {who_won} (+{result.trick_points} points)")

        input("\nPress Enter to continue...")

    def show_round_result(self, result: RoundResult,
                          match_scores: dict[int, int]) -> None:
        self.clear_screen()
        print("=" * 50)
        print("           ROUND OVER!")
        print("=" * 50)
        print(f"\nRound Score:")
        print(f"  You: {result.scores[self.seat]}")
        print(f"  {self._opponent_name}: {result.scores[1 - self.seat]}")
        print()

        if result.winner == self.seat:
            reason = ""
            if result.closed:
                reason = " (closed game)"
            elif result.scores[1 - self.seat] < 33:
                reason = " (opponent < 33)"
            print(f"🎉 You win this round! +{result.game_points} game point(s){reason}")
        elif result.winner is not None:
            reason = ""
            if result.closed:
                reason = " (closed game)"
            elif result.scores[self.seat] < 33:
                reason = " (opponent < 33)"
            print(f"💻 {self._opponent_name} wins this round! +{result.game_points} game point(s){reason}")
        else:
            print("🤝 Round is a tie! No game points awarded.")

        my_gp = match_scores.get(self.seat, 0)
        opp_gp = match_scores.get(1 - self.seat, 0)
        if result.winner is not None:
            if result.winner == self.seat:
                my_gp += result.game_points
            else:
                opp_gp += result.game_points
        print(f"\nMatch Score: You {my_gp} - {opp_gp} {self._opponent_name}")
        print()

    def show_match_result(self, result: MatchResult) -> None:
        self.clear_screen()
        print("=" * 50)
        print("           MATCH OVER!")
        print("=" * 50)
        print(f"\nFinal Match Score:")
        print(f"  You: {result.game_points[self.seat]}")
        print(f"  {self._opponent_name}: {result.game_points[1 - self.seat]}")
        print(f"\nRounds played: {result.rounds_played}")
        print()
        if result.winner == self.seat:
            print("🏆 CONGRATULATIONS! You win the match! 🏆")
        else:
            print(f"💻 {self._opponent_name} wins the match! Better luck next time!")
        print()

    def prompt_play_again(self) -> bool:
        choice = input("Play another match? (y/n): ").lower().strip()
        if choice != 'y':
            print("Thanks for playing! Goodbye!")
        return choice == 'y'

    def show_next_round(self, hand: list[Card] | None = None,
                        first_round: bool = False) -> None:
        import time
        import sys

        if not first_round:
            input("Press Enter for next round...")

        # "Next round" splash
        self.clear_screen()
        print()
        print()
        print("          ╭─────────────────╮")
        print("          │   Next round    │")
        print("          ╰─────────────────╯")
        print()
        time.sleep(1.0)

        # Deal cards one by one
        if hand:
            self.clear_screen()
            print()
            print("  Dealing...\n")
            dealt: list[Card] = []
            for card in hand:
                dealt.append(card)
                self.clear_screen()
                print()
                print("  Dealing...\n")
                print("  Your hand:  ", end="")
                print("  ".join(card_str(c) for c in dealt))
                print()
                sys.stdout.flush()
                time.sleep(0.5)
            time.sleep(0.5)

    def show_message(self, message: str) -> None:
        print(message)
        input("Press Enter to continue...")
