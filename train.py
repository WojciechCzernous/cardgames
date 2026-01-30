#!/usr/bin/env python3
"""
Headless training module for Sixty-Six card game.
Runs games without UI for RL training purposes.
"""

import random
import time
from dataclasses import dataclass
from typing import Callable

from card_game import (
    Round, Match, Action, ActionType, GameState, Suit, Card
)
from ui import HeadlessUI


@dataclass
class TrainingStats:
    """Statistics from a training run."""
    games_played: int
    player_wins: int
    computer_wins: int
    total_rounds: int
    avg_rounds_per_game: float
    elapsed_time: float
    games_per_second: float


# Type alias for agent policy function
AgentPolicy = Callable[[GameState], Action]


def random_policy(state: GameState) -> Action:
    """Random agent - picks a random valid action."""
    valid_actions = state.valid_actions
    
    if state.is_winner_action_phase:
        # Always swap if possible, else pass
        swap_actions = [a for a in valid_actions if a.type.value == "swap_trump"]
        if swap_actions:
            return swap_actions[0]
        return Action(ActionType.PASS)
    
    # Random card play
    return random.choice(valid_actions)


def greedy_policy(state: GameState) -> Action:
    """Greedy agent - prefers marriages and high-value cards."""
    valid_actions = state.valid_actions
    
    if state.is_winner_action_phase:
        swap_actions = [a for a in valid_actions if a.type.value == "swap_trump"]
        if swap_actions:
            return swap_actions[0]
        return Action(ActionType.PASS)
    
    # Prefer marriages (especially trump marriages)
    marriage_actions = [a for a in valid_actions if a.marriage_suit]
    if marriage_actions:
        trump_marriages = [a for a in marriage_actions if a.marriage_suit == state.trump_suit]
        if trump_marriages:
            return random.choice(trump_marriages)
        return random.choice(marriage_actions)
    
    # Play highest value card
    play_actions = [a for a in valid_actions if a.type.value == "play_card"]
    if play_actions:
        best_action = max(play_actions, key=lambda a: state.hand[a.card_index].value())
        return best_action
    
    return random.choice(valid_actions)


class HeadlessRound(Round):
    """Round that uses agent policies instead of UI prompts."""
    
    def __init__(self, player_policy: AgentPolicy, computer_policy: AgentPolicy | None = None,
                 player_starts: bool | None = None):
        super().__init__(player_starts=player_starts, ui=HeadlessUI())
        self.player_policy = player_policy
        self.computer_policy = computer_policy
    
    def player_play(self, lead_card: Card | None, computer_card: Card | None = None) -> tuple[Card, int]:
        """Player uses policy to choose action."""
        state = self.get_game_state("player", lead_card)
        action = self.player_policy(state)
        card, marriage_points = self.execute_action("player", action, lead_card)
        return card, marriage_points
    
    def player_winner_actions(self):
        """Player uses policy for winner actions."""
        state = self.get_game_state("player", is_winner_action=True)
        
        while True:
            action = self.player_policy(state)
            action_type = action.type.value
            
            if action_type == "swap_trump":
                self.execute_action("player", action)
                state = self.get_game_state("player", is_winner_action=True)
            elif action_type in ("close_game", "pass"):
                if action_type == "close_game":
                    self.execute_action("player", action)
                return
    
    def computer_choose_action(self, state: GameState) -> Action:
        """Use custom policy if provided, else default."""
        if self.computer_policy:
            return self.computer_policy(state)
        return super().computer_choose_action(state)


class HeadlessMatch:
    """Match that uses agent policies for both players."""
    
    def __init__(self, player_policy: AgentPolicy, computer_policy: AgentPolicy | None = None):
        self.player_policy = player_policy
        self.computer_policy = computer_policy
        self.player_game_points = 0
        self.computer_game_points = 0
        self.win_points = 7
        self.round_number = 0
        self.player_starts_next = random.choice([True, False])
    
    def play(self) -> str:
        """Play the match. Returns winner ('player' or 'computer')."""
        while self.player_game_points < self.win_points and self.computer_game_points < self.win_points:
            self.round_number += 1
            
            round_game = HeadlessRound(
                player_policy=self.player_policy,
                computer_policy=self.computer_policy,
                player_starts=self.player_starts_next
            )
            match_scores = {"player": self.player_game_points, "computer": self.computer_game_points}
            
            winner, points = round_game.play_round(match_scores)
            
            if winner == "player":
                self.player_game_points += points
            elif winner == "computer":
                self.computer_game_points += points
            
            self.player_starts_next = not self.player_starts_next
        
        return "player" if self.player_game_points >= self.win_points else "computer"


def train(num_games: int = 1000,
          player_policy: AgentPolicy = random_policy,
          computer_policy: AgentPolicy | None = None,
          verbose: bool = True) -> TrainingStats:
    """
    Run headless training games.
    
    Args:
        num_games: Number of matches to play
        player_policy: Policy function for player
        computer_policy: Policy function for computer (None = use default)
        verbose: Print progress updates
    
    Returns:
        TrainingStats with results
    """
    player_wins = 0
    computer_wins = 0
    total_rounds = 0
    
    start_time = time.time()
    
    for i in range(num_games):
        match = HeadlessMatch(player_policy, computer_policy)
        winner = match.play()
        
        if winner == "player":
            player_wins += 1
        else:
            computer_wins += 1
        
        total_rounds += match.round_number
        
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Games: {i + 1}/{num_games} | "
                  f"Player: {player_wins} | Computer: {computer_wins} | "
                  f"Speed: {(i + 1) / elapsed:.1f} games/sec")
    
    elapsed_time = time.time() - start_time
    
    return TrainingStats(
        games_played=num_games,
        player_wins=player_wins,
        computer_wins=computer_wins,
        total_rounds=total_rounds,
        avg_rounds_per_game=total_rounds / num_games,
        elapsed_time=elapsed_time,
        games_per_second=num_games / elapsed_time
    )


def main():
    """Run training with different policies."""
    print("=" * 60)
    print("Sixty-Six Headless Training")
    print("=" * 60)
    
    # Test 1: Random vs Random
    print("\n[Test 1] Random vs Random (1000 games)")
    stats = train(num_games=1000, player_policy=random_policy, computer_policy=random_policy)
    print(f"\nResults:")
    print(f"  Player wins: {stats.player_wins} ({100*stats.player_wins/stats.games_played:.1f}%)")
    print(f"  Computer wins: {stats.computer_wins} ({100*stats.computer_wins/stats.games_played:.1f}%)")
    print(f"  Avg rounds/game: {stats.avg_rounds_per_game:.2f}")
    print(f"  Speed: {stats.games_per_second:.1f} games/sec")
    
    # Test 2: Greedy vs Random
    print("\n[Test 2] Greedy vs Random (1000 games)")
    stats = train(num_games=1000, player_policy=greedy_policy, computer_policy=random_policy)
    print(f"\nResults:")
    print(f"  Player (greedy) wins: {stats.player_wins} ({100*stats.player_wins/stats.games_played:.1f}%)")
    print(f"  Computer (random) wins: {stats.computer_wins} ({100*stats.computer_wins/stats.games_played:.1f}%)")
    print(f"  Avg rounds/game: {stats.avg_rounds_per_game:.2f}")
    
    # Test 3: Random vs Greedy
    print("\n[Test 3] Random vs Greedy (1000 games)")
    stats = train(num_games=1000, player_policy=random_policy, computer_policy=greedy_policy)
    print(f"\nResults:")
    print(f"  Player (random) wins: {stats.player_wins} ({100*stats.player_wins/stats.games_played:.1f}%)")
    print(f"  Computer (greedy) wins: {stats.computer_wins} ({100*stats.computer_wins/stats.games_played:.1f}%)")
    print(f"  Avg rounds/game: {stats.avg_rounds_per_game:.2f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")


if __name__ == "__main__":
    main()
