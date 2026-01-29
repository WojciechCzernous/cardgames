#!/usr/bin/env python3
"""
Simple 24-card trick-taking game
Two players: User vs Computer (random)
"""

import random
from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"


# ANSI color codes
RED = "\033[91m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[H"

RANKS = [" 9", " J", " Q", " K", "10", " A"]
RANK_VALUES = {" 9": 0, "10": 10, " J": 2, " Q": 3, " K": 4, " A": 11}


@dataclass
class Card:
    rank: str
    suit: Suit

    def __str__(self):
        card_str = f"{self.rank}{self.suit.value}"
        if self.suit in (Suit.HEARTS, Suit.DIAMONDS):
            return f"{RED}{card_str}{RESET}"
        return card_str

    def value(self):
        return RANK_VALUES[self.rank]


def create_deck() -> list[Card]:
    """Create a 24-card deck."""
    deck = []
    for suit in Suit:
        for rank in RANKS:
            deck.append(Card(rank, suit))
    return deck


def card_strength(card: Card, lead_suit: Suit, trump_suit: Suit) -> int:
    """Calculate card strength for comparison."""
    base = RANKS.index(card.rank)
    if card.suit == trump_suit:
        return 100 + base  # Trump cards beat everything
    elif card.suit == lead_suit:
        return 50 + base   # Lead suit cards
    else:
        return base        # Other suits can't win


def display_hand(hand: list[Card], show_numbers: bool = True) -> str:
    """Display a hand of cards."""
    if show_numbers:
        cards_line = "  ".join(str(card) for card in hand)
        numbers_line = "  ".join(f"[{i+1}]" for i in range(len(hand)))
        return f"{cards_line}\n{numbers_line}"
    return "  ".join(str(card) for card in hand)


def colored_suit(suit: Suit) -> str:
    """Return suit symbol with color for red suits."""
    if suit in (Suit.HEARTS, Suit.DIAMONDS):
        return f"{RED}{suit.value}{RESET}"
    return suit.value


def display_hidden_cards(count: int) -> str:
    """Display hidden cards as backs."""
    return " ".join(["[?]"] * count)


class Round:
    """A single round played to 66 points."""
    
    def __init__(self, player_starts: bool = None):
        self.deck = create_deck()
        random.shuffle(self.deck)
        
        # Deal 6 cards to each player
        self.player_hand: list[Card] = []
        self.computer_hand: list[Card] = []
        
        for _ in range(6):
            self.player_hand.append(self.deck.pop())
            self.computer_hand.append(self.deck.pop())
        
        # Trump card (visible) - one card from remaining deck
        self.trump_card = self.deck.pop()
        self.trump_suit = self.trump_card.suit
        
        # Remaining cards form the draw pile (hidden)
        self.draw_pile = self.deck  # 11 cards left
        
        # Scores
        self.player_score = 0
        self.computer_score = 0
        
        # Who leads the next trick
        if player_starts is None:
            self.player_leads = random.choice([True, False])
        else:
            self.player_leads = player_starts
        
        # Track last drawn cards and last trick result
        self.player_last_drawn: Card | None = None
        self.computer_last_drawn: Card | None = None
        self.last_trick_info: str = ""
        
        # Track if round is "closed" (early phase 2 transition)
        self.closed = False
        self.closed_by: str | None = None  # "player" or "computer"
        
        # Track round winner (None until someone wins)
        self.round_winner: str | None = None  # "player" or "computer"
        self.win_score = 66
        
        # Match scores for display
        self.match_scores: dict[str, int] | None = None

    def sort_hand(self, hand: list[Card]):
        """Sort hand by suit, then by rank."""
        hand.sort(key=lambda c: (list(Suit).index(c.suit), RANKS.index(c.rank)))

    @property
    def phase(self) -> int:
        """Return current game phase: 1 = draw pile active, 2 = draw pile empty or closed."""
        if self.closed:
            return 2
        if not self.draw_pile and self.trump_card is None:
            return 2
        return 1

    def clear_screen(self):
        """Clear the terminal screen."""
        print(CLEAR, end="")

    def display_state(self, computer_card: Card | None = None, player_card: Card | None = None):
        """Display the current game state."""
        self.clear_screen()
        
        # Header with match score
        print("=" * 50)
        if self.match_scores:
            print(f"    SIXTY-SIX          Match: You {self.match_scores['player']} - {self.match_scores['computer']} Computer")
        else:
            print("              SIXTY-SIX")
        print("=" * 50)
        
        # Trump and draw pile info
        trump_display = str(self.trump_card) if self.trump_card else f"[{colored_suit(self.trump_suit)}]"
        if self.closed:
            phase_info = f"CLOSED by {self.closed_by}"
        elif self.phase == 1:
            phase_info = "Phase 1 (free play)"
        else:
            phase_info = "Phase 2 (must follow)"
        print(f"Trump: {trump_display}  |  Draw pile: {len(self.draw_pile)} cards  |  {phase_info}")
        print(f"Score - You: {self.player_score:3d}  |  Computer: {self.computer_score:3d}")
        print("-" * 50)
        
        # Last trick result
        if self.last_trick_info:
            print(f"Last: {self.last_trick_info}")
        else:
            print()
        print()
        
        # Computer's hand
        print(f"Computer: {display_hidden_cards(len(self.computer_hand))}")
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
        drawn_info = f"  (drew: {self.player_last_drawn})" if self.player_last_drawn else ""
        print(f"Your hand:{drawn_info}")
        print(display_hand(self.player_hand))
        print()

    def get_valid_cards(self, hand: list[Card], lead_card: Card | None) -> list[Card]:
        """Get valid cards that can be played based on current phase."""
        if lead_card is None:
            return hand  # Can play anything when leading
        
        # Phase 1: Any card is allowed
        if self.phase == 1:
            return hand
        
        # Phase 2: Must follow suit -> else must trump -> else any card
        same_suit = [c for c in hand if c.suit == lead_card.suit]
        if same_suit:
            return same_suit
        
        trumps = [c for c in hand if c.suit == self.trump_suit]
        if trumps:
            return trumps
        
        return hand

    def player_play(self, lead_card: Card | None, computer_card: Card | None = None) -> tuple[Card, int]:
        """Let the player choose a card to play. Returns (card, marriage_points)."""
        valid_cards = self.get_valid_cards(self.player_hand, lead_card)
        error_msg = ""
        
        # Check for marriages if leading
        marriages = self.get_marriages(self.player_hand) if lead_card is None else []
        
        while True:
            self.display_state(computer_card=computer_card)
            
            if lead_card:
                if self.phase == 1:
                    print(f"Lead: {colored_suit(lead_card.suit)} (any card allowed)")
                else:
                    print(f"Must follow: {colored_suit(lead_card.suit)}  |  Valid: {display_hand(valid_cards, show_numbers=False)}")
            else:
                lead_msg = ">>> Your lead!"
                if marriages:
                    marriage_strs = [f"{colored_suit(s)} ({self.marriage_value(s)}pts)" for s in marriages]
                    lead_msg += f"  Marriages: {', '.join(marriage_strs)}"
                print(lead_msg)
            
            if error_msg:
                print(f"\n⚠ {error_msg}")
                error_msg = ""
            
            try:
                prompt = f"\nPlay card [1-{len(self.player_hand)}]"
                if marriages:
                    prompt += " or [M] to announce marriage"
                prompt += ": "
                choice = input(prompt).strip().lower()
                
                # Handle marriage announcement
                if choice == 'm' and marriages:
                    return self.player_announce_marriage(marriages)
                
                idx = int(choice) - 1
                
                if 0 <= idx < len(self.player_hand):
                    card = self.player_hand[idx]
                    if card in valid_cards:
                        self.player_hand.remove(card)
                        self.player_last_drawn = None  # Clear after playing
                        return card, 0
                    else:
                        # Determine why card is invalid
                        same_suit = [c for c in self.player_hand if c.suit == lead_card.suit]
                        if same_suit:
                            error_msg = f"You must follow suit ({colored_suit(lead_card.suit)})!"
                        else:
                            error_msg = f"You must play trump ({colored_suit(self.trump_suit)})!"
                else:
                    error_msg = "Invalid card number!"
            except ValueError:
                error_msg = "Please enter a number!"

    def player_announce_marriage(self, marriages: list[Suit]) -> tuple[Card, int]:
        """Let player choose which marriage to announce and which card to play."""
        while True:
            self.display_state()
            print("Announce marriage:")
            for i, suit in enumerate(marriages):
                pts = self.marriage_value(suit)
                print(f"  [{i+1}] {colored_suit(suit)} marriage (+{pts} points)")
            print("  [0] Cancel")
            
            try:
                choice = input("\nChoose marriage: ").strip()
                idx = int(choice)
                
                if idx == 0:
                    # Return to normal play - re-enter player_play
                    return self.player_play(None)
                
                if 1 <= idx <= len(marriages):
                    suit = marriages[idx - 1]
                    points = self.marriage_value(suit)
                    
                    # Find K and Q of this suit
                    king = next(c for c in self.player_hand if c.rank == " K" and c.suit == suit)
                    queen = next(c for c in self.player_hand if c.rank == " Q" and c.suit == suit)
                    
                    # Ask which card to play
                    self.display_state()
                    print(f"{colored_suit(suit)} Marriage! Play which card?")
                    print(f"  [1] {king}")
                    print(f"  [2] {queen}")
                    
                    card_choice = input("\nYour choice: ").strip()
                    if card_choice == '1':
                        self.player_hand.remove(king)
                        self.player_last_drawn = None
                        return king, points
                    elif card_choice == '2':
                        self.player_hand.remove(queen)
                        self.player_last_drawn = None
                        return queen, points
            except (ValueError, StopIteration):
                pass

    def computer_play(self, lead_card: Card | None) -> tuple[Card, int]:
        """Computer plays randomly from valid cards. Returns (card, marriage_points)."""
        valid_cards = self.get_valid_cards(self.computer_hand, lead_card)
        
        # Check for marriages if leading (computer always announces if possible)
        marriage_points = 0
        if lead_card is None:
            marriages = self.get_marriages(self.computer_hand)
            if marriages:
                # Pick best marriage (trump first, else random)
                if self.trump_suit in marriages:
                    suit = self.trump_suit
                else:
                    suit = random.choice(marriages)
                
                marriage_points = self.marriage_value(suit)
                
                # Must play K or Q of that suit
                king = next((c for c in self.computer_hand if c.rank == " K" and c.suit == suit), None)
                queen = next((c for c in self.computer_hand if c.rank == " Q" and c.suit == suit), None)
                card = random.choice([c for c in [king, queen] if c])
                self.computer_hand.remove(card)
                return card, marriage_points
        
        card = random.choice(valid_cards)
        self.computer_hand.remove(card)
        return card, 0

    def play_trick(self) -> bool:
        """Play one trick. Returns True if player won."""
        player_marriage = 0
        computer_marriage = 0
        
        if self.player_leads:
            player_card, player_marriage = self.player_play(None)
            if player_marriage:
                self.player_score += player_marriage
            computer_card, _ = self.computer_play(player_card)
            lead_suit = player_card.suit
        else:
            computer_card, computer_marriage = self.computer_play(None)
            if computer_marriage:
                self.computer_score += computer_marriage
            player_card, _ = self.player_play(computer_card, computer_card)
            lead_suit = computer_card.suit
        
        # Show final state with both cards
        self.display_state(computer_card=computer_card, player_card=player_card)
        
        # Show marriage announcements
        if player_marriage:
            print(f"💍 You announced marriage! +{player_marriage} points")
        if computer_marriage:
            print(f"💍 Computer announced marriage! +{computer_marriage} points")
        
        # Check if marriage announcer reached 66 (before trick points)
        # Marriage is announced before trick is played, so announcer wins if they hit 66
        if player_marriage and self.player_score >= self.win_score:
            self.round_winner = "player"
            print(f"\n🌟 You reached {self.win_score} points with marriage!")
            input("\nPress Enter to continue...")
            return True
        if computer_marriage and self.computer_score >= self.win_score:
            self.round_winner = "computer"
            print(f"\n🌟 Computer reached {self.win_score} points with marriage!")
            input("\nPress Enter to continue...")
            return False
        
        # Determine winner
        player_strength = card_strength(player_card, lead_suit, self.trump_suit)
        computer_strength = card_strength(computer_card, lead_suit, self.trump_suit)
        
        trick_points = player_card.value() + computer_card.value()
        
        if player_strength > computer_strength:
            self.last_trick_info = f"You won +{trick_points} pts ({player_card} beat {computer_card})"
            self.player_score += trick_points
            self.player_leads = True
            winner_is_player = True
        else:
            self.last_trick_info = f"Computer won +{trick_points} pts ({computer_card} beat {player_card})"
            self.computer_score += trick_points
            self.player_leads = False
            winner_is_player = False
        
        # Check if trick winner reached 66
        if self.player_score >= self.win_score:
            self.round_winner = "player"
        elif self.computer_score >= self.win_score:
            self.round_winner = "computer"
        
        print(f"\n{'✓ You win!' if winner_is_player else '✗ Computer wins!'} (+{trick_points} points)")
        if self.round_winner:
            print(f"\n🌟 {'You' if self.round_winner == 'player' else 'Computer'} reached {self.win_score} points!")
        input("\nPress Enter to continue...")
        return winner_is_player

    def has_nine_trump(self, hand: list[Card]) -> Card | None:
        """Check if hand contains 9 of trump suit."""
        for card in hand:
            if card.rank == " 9" and card.suit == self.trump_suit:
                return card
        return None

    def get_marriages(self, hand: list[Card]) -> list[Suit]:
        """Find all marriages (K+Q of same suit) in hand. Returns list of suits."""
        marriages = []
        for suit in Suit:
            has_king = any(c.rank == " K" and c.suit == suit for c in hand)
            has_queen = any(c.rank == " Q" and c.suit == suit for c in hand)
            if has_king and has_queen:
                marriages.append(suit)
        return marriages

    def marriage_value(self, suit: Suit) -> int:
        """Return marriage value: 40 for trump, 20 for others."""
        return 40 if suit == self.trump_suit else 20

    def swap_nine_trump(self, hand: list[Card]) -> bool:
        """Swap 9 of trump with the trump card. Returns True if swapped."""
        nine = self.has_nine_trump(hand)
        if nine and self.trump_card:
            hand.remove(nine)
            hand.append(self.trump_card)
            self.trump_card = nine
            return True
        return False

    def player_winner_actions(self):
        """Let player perform special actions after winning a trick in phase 1."""
        has_nine = self.has_nine_trump(self.player_hand)
        
        while True:
            self.display_state()
            print("Winner actions (Phase 1):")
            
            options = []
            if has_nine and self.trump_card:
                options.append(f"[S] Swap {has_nine} with trump {self.trump_card}")
            options.append("[C] Close the game (enter Phase 2)")
            options.append("[Enter] Continue to play")
            
            for opt in options:
                print(f"  {opt}")
            
            choice = input("\nYour choice: ").strip().lower()
            
            if choice == 's' and has_nine:
                self.swap_nine_trump(self.player_hand)
                print(f"Swapped! New trump card: {self.trump_card}")
                input("Press Enter to continue...")
                has_nine = None  # Can't swap again
            elif choice == 'c':
                self.closed = True
                self.closed_by = "you"
                print("Game closed! No more drawing. Phase 2 rules now apply.")
                input("Press Enter to continue...")
                return
            elif choice == '':
                return
            # Invalid input just loops

    def computer_winner_actions(self):
        """Computer performs special actions after winning a trick in phase 1."""
        # Computer always swaps if it has nine of trump (always beneficial)
        if self.has_nine_trump(self.computer_hand) and self.trump_card:
            old_trump = self.trump_card
            self.swap_nine_trump(self.computer_hand)
            self.last_trick_info += f" | Swapped 9{colored_suit(self.trump_suit)} for {old_trump}"
        
        # Computer doesn't close for now (could add strategy later)

    def draw_cards(self):
        """Both players draw a card if available."""
        self.player_last_drawn = None
        self.computer_last_drawn = None
        
        # No drawing if game is closed
        if self.closed:
            return
        
        if self.draw_pile:
            if self.player_leads:
                drawn = self.draw_pile.pop()
                self.player_hand.append(drawn)
                self.player_last_drawn = drawn
                if self.draw_pile:
                    self.computer_hand.append(self.draw_pile.pop())
                elif self.trump_card:
                    self.computer_hand.append(self.trump_card)
                    self.trump_card = None
            else:
                self.computer_hand.append(self.draw_pile.pop())
                if self.draw_pile:
                    drawn = self.draw_pile.pop()
                    self.player_hand.append(drawn)
                    self.player_last_drawn = drawn
                elif self.trump_card:
                    self.player_hand.append(self.trump_card)
                    self.player_last_drawn = self.trump_card
                    self.trump_card = None
        elif self.trump_card:
            # Draw pile empty, trump card goes to winner
            if self.player_leads:
                self.player_hand.append(self.trump_card)
                self.player_last_drawn = self.trump_card
            else:
                self.computer_hand.append(self.trump_card)
            self.trump_card = None

    def play_round(self, match_scores: dict[str, int] | None = None) -> tuple[str | None, int]:
        """Play a single round. Returns (winner, game_points) or (None, 0) for tie."""
        self.match_scores = match_scores
        
        # Play until someone wins (66 pts) or no cards left
        while self.player_hand and self.computer_hand and not self.round_winner:
            self.play_trick()
            
            # Skip drawing and winner actions if round is over
            if self.round_winner:
                break
                
            self.draw_cards()
            
            # Winner can perform special actions in phase 1 (after drawing)
            if self.phase == 1:
                if self.player_leads:
                    self.player_winner_actions()
                else:
                    self.computer_winner_actions()
        
        # Calculate game points
        return self.calculate_game_points()
    
    def calculate_game_points(self) -> tuple[str | None, int]:
        """Calculate who wins the round and how many game points."""
        # If someone closed the game
        if self.closed:
            if self.closed_by == "you":
                # Player closed - must reach 66 or opponent gets 3 pts
                if self.round_winner == "player":
                    return ("player", 3)
                else:
                    return ("computer", 3)
            else:
                # Computer closed - must reach 66 or player gets 3 pts
                if self.round_winner == "computer":
                    return ("computer", 3)
                else:
                    return ("player", 3)
        
        # Normal finish
        if self.round_winner == "player":
            # Player won - check opponent's score
            if self.computer_score < 33:
                return ("player", 2)  # Opponent < 33
            return ("player", 1)
        elif self.round_winner == "computer":
            # Computer won - check opponent's score
            if self.player_score < 33:
                return ("computer", 2)  # Opponent < 33
            return ("computer", 1)
        else:
            # No one reached 66 - tie (0-0)
            return (None, 0)
    
    def show_round_result(self, winner: str | None, game_points: int, match_scores: dict[str, int]):
        """Display the round result."""
        self.clear_screen()
        print("=" * 50)
        print("           ROUND OVER!")
        print("=" * 50)
        print(f"\nRound Score:")
        print(f"  You: {self.player_score}")
        print(f"  Computer: {self.computer_score}")
        print()
        
        if winner == "player":
            reason = ""
            if self.closed:
                reason = " (closed game)"
            elif self.computer_score < 33:
                reason = " (opponent < 33)"
            print(f"🎉 You win this round! +{game_points} game point(s){reason}")
        elif winner == "computer":
            reason = ""
            if self.closed:
                reason = " (closed game)"
            elif self.player_score < 33:
                reason = " (opponent < 33)"
            print(f"💻 Computer wins this round! +{game_points} game point(s){reason}")
        else:
            print("🤝 Round is a tie! No game points awarded.")
        
        print(f"\nMatch Score: You {match_scores['player']} - {match_scores['computer']} Computer")
        print()


class Match:
    """A match consisting of multiple rounds, first to 7 game points wins."""
    
    def __init__(self):
        self.player_game_points = 0
        self.computer_game_points = 0
        self.win_points = 7
        self.round_number = 0
        self.player_starts_next = random.choice([True, False])
    
    def show_welcome(self):
        """Display welcome message and rules."""
        print("\033[2J\033[H", end="")  # Clear screen
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
    
    def play(self):
        """Play the match until someone reaches 7 game points."""
        self.show_welcome()
        
        while self.player_game_points < self.win_points and self.computer_game_points < self.win_points:
            self.round_number += 1
            
            # Create and play a round
            round_game = Round(player_starts=self.player_starts_next)
            match_scores = {"player": self.player_game_points, "computer": self.computer_game_points}
            
            winner, points = round_game.play_round(match_scores)
            
            # Update game points
            if winner == "player":
                self.player_game_points += points
            elif winner == "computer":
                self.computer_game_points += points
            
            # Alternate who leads next round
            self.player_starts_next = not self.player_starts_next
            
            # Show round result
            new_scores = {"player": self.player_game_points, "computer": self.computer_game_points}
            round_game.show_round_result(winner, points, new_scores)
            
            # Check if match is over
            if self.player_game_points >= self.win_points or self.computer_game_points >= self.win_points:
                break
            
            input("Press Enter for next round...")
        
        # Match over
        self.show_match_result()
    
    def show_match_result(self):
        """Display the final match result."""
        print("\033[2J\033[H", end="")  # Clear screen
        print("=" * 50)
        print("           MATCH OVER!")
        print("=" * 50)
        print(f"\nFinal Match Score:")
        print(f"  You: {self.player_game_points}")
        print(f"  Computer: {self.computer_game_points}")
        print(f"\nRounds played: {self.round_number}")
        print()
        
        if self.player_game_points >= self.win_points:
            print("🏆 CONGRATULATIONS! You win the match! 🏆")
        else:
            print("💻 Computer wins the match! Better luck next time!")
        print()


def main():
    while True:
        match = Match()
        match.play()
        
        play_again = input("Play another match? (y/n): ").lower().strip()
        if play_again != 'y':
            print("Thanks for playing! Goodbye!")
            break


if __name__ == "__main__":
    main()
