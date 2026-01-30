"""
UI abstraction for Sixty-Six card game.
Separates game logic from presentation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from card_game import Card, Suit, Action, ActionType, GameState, RANKS, RANK_VALUES

# ANSI color codes
RED = "\033[91m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[H"


@dataclass
class TrickResult:
    """Result of a completed trick."""
    player_card: Card
    computer_card: Card
    winner: str  # "player" or "computer"
    trick_points: int
    player_marriage: int  # Marriage points announced
    computer_marriage: int


@dataclass 
class RoundResult:
    """Result of a completed round."""
    winner: str | None  # "player", "computer", or None for tie
    game_points: int
    player_score: int
    computer_score: int
    closed: bool
    closed_by: str | None


@dataclass
class MatchResult:
    """Result of a completed match."""
    winner: str  # "player" or "computer"
    player_game_points: int
    computer_game_points: int
    rounds_played: int


class GameUI(ABC):
    """Abstract base class for game UI implementations."""
    
    @abstractmethod
    def show_welcome(self) -> None:
        """Display welcome screen and rules."""
        pass
    
    @abstractmethod
    def display_state(self, state: GameState, match_scores: dict[str, int] | None = None,
                      computer_card: Card | None = None, player_card: Card | None = None,
                      last_trick_info: str = "") -> None:
        """Display the current game state."""
        pass
    
    @abstractmethod
    def prompt_card_play(self, state: GameState, computer_card: Card | None = None) -> Action:
        """Prompt player to choose a card to play. Returns the chosen action."""
        pass
    
    @abstractmethod
    def prompt_winner_action(self, state: GameState) -> Action:
        """Prompt player for post-trick action (swap/close/pass)."""
        pass
    
    @abstractmethod
    def show_trick_result(self, result: TrickResult, player_score: int, 
                          computer_score: int, round_winner: str | None) -> None:
        """Display the result of a trick."""
        pass
    
    @abstractmethod
    def show_round_result(self, result: RoundResult, match_scores: dict[str, int]) -> None:
        """Display the result of a round."""
        pass
    
    @abstractmethod
    def show_match_result(self, result: MatchResult) -> None:
        """Display the final match result."""
        pass
    
    @abstractmethod
    def prompt_play_again(self) -> bool:
        """Ask if player wants to play again."""
        pass
    
    @abstractmethod
    def prompt_next_round(self) -> None:
        """Wait for player to continue to next round."""
        pass
    
    @abstractmethod
    def show_message(self, message: str) -> None:
        """Display a message to the player."""
        pass


def colored_suit(suit: Suit) -> str:
    """Return suit symbol with color for red suits."""
    if suit in (Suit.HEARTS, Suit.DIAMONDS):
        return f"{RED}{suit.value}{RESET}"
    return suit.value


def display_hand(hand: list[Card], show_numbers: bool = True) -> str:
    """Display a hand of cards."""
    if show_numbers:
        cards_line = "  ".join(str(card) for card in hand)
        numbers_line = "  ".join(f"[{i+1}]" for i in range(len(hand)))
        return f"{cards_line}\n{numbers_line}"
    return "  ".join(str(card) for card in hand)


def display_hidden_cards(count: int) -> str:
    """Display hidden cards as backs."""
    return " ".join(["[?]"] * count)


class TerminalUI(GameUI):
    """Terminal-based UI implementation."""
    
    def clear_screen(self):
        """Clear the terminal screen."""
        print(CLEAR, end="")
    
    def show_welcome(self) -> None:
        """Display welcome screen and rules."""
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
    
    def display_state(self, state: GameState, match_scores: dict[str, int] | None = None,
                      computer_card: Card | None = None, player_card: Card | None = None,
                      last_trick_info: str = "", player_last_drawn: Card | None = None) -> None:
        """Display the current game state."""
        self.clear_screen()
        
        # Header with match score
        print("=" * 50)
        if match_scores:
            print(f"    SIXTY-SIX          Match: You {match_scores['player']} - {match_scores['computer']} Computer")
        else:
            print("              SIXTY-SIX")
        print("=" * 50)
        
        # Trump and draw pile info
        trump_display = str(state.trump_card) if state.trump_card else f"[{colored_suit(state.trump_suit)}]"
        if state.closed:
            phase_info = f"CLOSED by {state.closed_by}"
        elif state.phase == 1:
            phase_info = "Phase 1 (free play)"
        else:
            phase_info = "Phase 2 (must follow)"
        print(f"Trump: {trump_display}  |  Draw pile: {state.draw_pile_size} cards  |  {phase_info}")
        print(f"Score - You: {state.my_score:3d}  |  Computer: {state.opponent_score:3d}")
        print("-" * 50)
        
        # Last trick result
        if last_trick_info:
            print(f"Last: {last_trick_info}")
        else:
            print()
        print()
        
        # Computer's hand (hidden)
        # We don't have opponent hand size in state, estimate from context
        print(f"Computer: {display_hidden_cards(len(state.hand))}")  # Same size as player
        print()
        
        # Table area (cards played this trick)
        print("─" * 20 + " TABLE " + "─" * 23)
        if computer_card:
            print(f"  Computer played: {computer_card}")
        else:
            print()
        if player_card:
            print(f"  You played:      {player_card}")
        else:
            print()
        print("─" * 50)
        print()
        
        # Your hand
        drawn_info = f"  (drew: {player_last_drawn})" if player_last_drawn else ""
        print(f"Your hand:{drawn_info}")
        print(display_hand(state.hand))
        print()
    
    def prompt_card_play(self, state: GameState, computer_card: Card | None = None) -> Action:
        """Prompt player to choose a card to play."""
        # Find marriages in valid actions
        marriages = list(set(a.marriage_suit for a in state.valid_actions if a.marriage_suit))
        error_msg = ""
        
        while True:
            if state.lead_card:
                if state.phase == 1:
                    print(f"Lead: {colored_suit(state.lead_card.suit)} (any card allowed)")
                else:
                    valid_cards = [state.hand[a.card_index] for a in state.valid_actions 
                                   if a.type.value == "play_card"]
                    print(f"Must follow: {colored_suit(state.lead_card.suit)}  |  Valid: {display_hand(valid_cards, show_numbers=False)}")
            else:
                lead_msg = ">>> Your lead!"
                if marriages:
                    marriage_strs = [f"{colored_suit(s)} ({40 if s == state.trump_suit else 20}pts)" for s in marriages]
                    lead_msg += f"  Marriages: {', '.join(marriage_strs)}"
                print(lead_msg)
            
            if error_msg:
                print(f"\n⚠ {error_msg}")
                error_msg = ""
            
            try:
                prompt = f"\nPlay card [1-{len(state.hand)}]"
                if marriages:
                    prompt += " or [M] to announce marriage"
                prompt += ": "
                choice = input(prompt).strip().lower()
                
                # Handle marriage announcement
                if choice == 'm' and marriages:
                    return self._prompt_marriage(state, marriages)
                
                idx = int(choice) - 1
                
                if 0 <= idx < len(state.hand):
                    # Find matching action - compare by value due to potential module reload issues
                    for action in state.valid_actions:
                        if (action.type.value == "play_card" and 
                            action.card_index == idx and 
                            not action.marriage_suit):
                            return action
                    error_msg = f"Invalid card for current situation! (idx={idx}, looking for PLAY_CARD without marriage)"
                else:
                    error_msg = "Invalid card number!"
            except ValueError:
                error_msg = "Please enter a number!"
    
    def _prompt_marriage(self, state: GameState, marriages: list[Suit]) -> Action:
        """Let player choose which marriage to announce and which card to play."""
        while True:
            self.clear_screen()
            print("Announce marriage:")
            for i, suit in enumerate(marriages):
                pts = 40 if suit == state.trump_suit else 20
                print(f"  [{i+1}] {colored_suit(suit)} marriage (+{pts} points)")
            print("  [0] Cancel")
            
            try:
                choice = input("\nChoose marriage: ").strip()
                idx = int(choice)
                
                if idx == 0:
                    return self.prompt_card_play(state)
                
                if 1 <= idx <= len(marriages):
                    suit = marriages[idx - 1]
                    
                    # Find K and Q indices for this suit
                    king_idx = queen_idx = None
                    for i, card in enumerate(state.hand):
                        if card.suit == suit:
                            if card.rank == " K":
                                king_idx = i
                            elif card.rank == " Q":
                                queen_idx = i
                    
                    # Ask which card to play
                    self.clear_screen()
                    print(f"{colored_suit(suit)} Marriage! Play which card?")
                    print(f"  [1] {state.hand[king_idx]}")
                    print(f"  [2] {state.hand[queen_idx]}")
                    
                    card_choice = input("\nYour choice: ").strip()
                    if card_choice == '1':
                        return Action(ActionType.PLAY_CARD, card_index=king_idx, marriage_suit=suit)
                    elif card_choice == '2':
                        return Action(ActionType.PLAY_CARD, card_index=queen_idx, marriage_suit=suit)
            except (ValueError, TypeError):
                pass
    
    def prompt_winner_action(self, state: GameState) -> Action:
        """Prompt player for post-trick action (swap/close/pass)."""
        has_swap = any(a.type.value == "swap_trump" for a in state.valid_actions)
        
        while True:
            print("Winner actions (Phase 1):")
            
            if has_swap:
                # Find the 9 of trump in hand
                nine_trump = next((c for c in state.hand if c.rank == " 9" and c.suit == state.trump_suit), None)
                if nine_trump and state.trump_card:
                    print(f"  [S] Swap {nine_trump} with trump {state.trump_card}")
            print("  [C] Close the game (enter Phase 2)")
            print("  [Enter] Continue to play")
            
            choice = input("\nYour choice: ").strip().lower()
            
            if choice == 's' and has_swap:
                return Action(ActionType.SWAP_TRUMP)
            elif choice == 'c':
                return Action(ActionType.CLOSE_GAME)
            elif choice == '':
                return Action(ActionType.PASS)
    
    def show_trick_result(self, result: TrickResult, player_score: int,
                          computer_score: int, round_winner: str | None) -> None:
        """Display the result of a trick."""
        # Show marriage announcements
        if result.player_marriage:
            print(f"💍 You announced marriage! +{result.player_marriage} points")
        if result.computer_marriage:
            print(f"💍 Computer announced marriage! +{result.computer_marriage} points")
        
        # Check for marriage win
        if round_winner:
            if result.player_marriage and round_winner == "player":
                print(f"\n🌟 You reached 66 points with marriage!")
            elif result.computer_marriage and round_winner == "computer":
                print(f"\n🌟 Computer reached 66 points with marriage!")
            else:
                print(f"\n{'✓ You win!' if result.winner == 'player' else '✗ Computer wins!'} (+{result.trick_points} points)")
                print(f"\n🌟 {'You' if round_winner == 'player' else 'Computer'} reached 66 points!")
        else:
            print(f"\n{'✓ You win!' if result.winner == 'player' else '✗ Computer wins!'} (+{result.trick_points} points)")
        
        input("\nPress Enter to continue...")
    
    def show_round_result(self, result: RoundResult, match_scores: dict[str, int]) -> None:
        """Display the result of a round."""
        self.clear_screen()
        print("=" * 50)
        print("           ROUND OVER!")
        print("=" * 50)
        print(f"\nRound Score:")
        print(f"  You: {result.player_score}")
        print(f"  Computer: {result.computer_score}")
        print()
        
        if result.winner == "player":
            reason = ""
            if result.closed:
                reason = " (closed game)"
            elif result.computer_score < 33:
                reason = " (opponent < 33)"
            print(f"🎉 You win this round! +{result.game_points} game point(s){reason}")
        elif result.winner == "computer":
            reason = ""
            if result.closed:
                reason = " (closed game)"
            elif result.player_score < 33:
                reason = " (opponent < 33)"
            print(f"💻 Computer wins this round! +{result.game_points} game point(s){reason}")
        else:
            print("🤝 Round is a tie! No game points awarded.")
        
        print(f"\nMatch Score: You {match_scores['player']} - {match_scores['computer']} Computer")
        print()
    
    def show_match_result(self, result: MatchResult) -> None:
        """Display the final match result."""
        self.clear_screen()
        print("=" * 50)
        print("           MATCH OVER!")
        print("=" * 50)
        print(f"\nFinal Match Score:")
        print(f"  You: {result.player_game_points}")
        print(f"  Computer: {result.computer_game_points}")
        print(f"\nRounds played: {result.rounds_played}")
        print()
        
        if result.winner == "player":
            print("🏆 CONGRATULATIONS! You win the match! 🏆")
        else:
            print("💻 Computer wins the match! Better luck next time!")
        print()
    
    def prompt_play_again(self) -> bool:
        """Ask if player wants to play again."""
        choice = input("Play another match? (y/n): ").lower().strip()
        if choice != 'y':
            print("Thanks for playing! Goodbye!")
        return choice == 'y'
    
    def prompt_next_round(self) -> None:
        """Wait for player to continue to next round."""
        input("Press Enter for next round...")
    
    def show_message(self, message: str) -> None:
        """Display a message to the player."""
        print(message)
        input("Press Enter to continue...")


class HeadlessUI(GameUI):
    """Headless UI for RL training - no output, no input prompts."""
    
    def show_welcome(self) -> None:
        pass
    
    def display_state(self, state: GameState, match_scores: dict[str, int] | None = None,
                      computer_card: Card | None = None, player_card: Card | None = None,
                      last_trick_info: str = "", player_last_drawn: Card | None = None) -> None:
        pass
    
    def prompt_card_play(self, state: GameState, computer_card: Card | None = None) -> Action:
        """For headless, this should not be called - use agent instead."""
        raise NotImplementedError("HeadlessUI requires an agent to make decisions")
    
    def prompt_winner_action(self, state: GameState) -> Action:
        """For headless, this should not be called - use agent instead."""
        raise NotImplementedError("HeadlessUI requires an agent to make decisions")
    
    def show_trick_result(self, result: TrickResult, player_score: int,
                          computer_score: int, round_winner: str | None) -> None:
        pass
    
    def show_round_result(self, result: RoundResult, match_scores: dict[str, int]) -> None:
        pass
    
    def show_match_result(self, result: MatchResult) -> None:
        pass
    
    def prompt_play_again(self) -> bool:
        return False
    
    def prompt_next_round(self) -> None:
        pass
    
    def show_message(self, message: str) -> None:
        pass
