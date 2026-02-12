#!/usr/bin/env python3
"""
Headless training module for Sixty-Six.
Runs matches without UI for RL training purposes.
"""

import time
from dataclasses import dataclass

from agents import Player, RandomPlayer, GreedyPlayer
from game import Match
from models import MatchResult


@dataclass
class TrainingStats:
    """Statistics from a training run."""
    games_played: int
    wins: dict[int, int]          # seat -> win count
    total_rounds: int
    avg_rounds_per_game: float
    elapsed_time: float
    games_per_second: float


def train(num_games: int = 1000,
          player_0: Player | None = None,
          player_1: Player | None = None,
          verbose: bool = True) -> TrainingStats:
    """
    Run headless training games between two agents.

    Args:
        num_games:  Number of matches to play.
        player_0:   Agent for seat 0 (default: RandomPlayer).
        player_1:   Agent for seat 1 (default: RandomPlayer).
        verbose:    Print progress every 100 games.

    Returns:
        TrainingStats with results.
    """
    if player_0 is None:
        player_0 = RandomPlayer("Agent-0")
    if player_1 is None:
        player_1 = RandomPlayer("Agent-1")

    players = {0: player_0, 1: player_1}
    wins = {0: 0, 1: 0}
    total_rounds = 0

    start = time.time()

    for i in range(num_games):
        match = Match(players)
        result = match.play()

        wins[result.winner] += 1
        total_rounds += result.rounds_played

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - start
            print(
                f"Games: {i+1}/{num_games} | "
                f"{player_0.name}: {wins[0]} | "
                f"{player_1.name}: {wins[1]} | "
                f"Speed: {(i+1)/elapsed:.1f} games/sec"
            )

    elapsed = time.time() - start

    return TrainingStats(
        games_played=num_games,
        wins=wins,
        total_rounds=total_rounds,
        avg_rounds_per_game=total_rounds / num_games,
        elapsed_time=elapsed,
        games_per_second=num_games / elapsed,
    )


def main():
    print("=" * 60)
    print("Sixty-Six Headless Training")
    print("=" * 60)

    # Test 1: Random vs Random
    print("\n[1] Random vs Random (1000 games)")
    s = train(1000, RandomPlayer("Random-0"), RandomPlayer("Random-1"))
    print(f"  Results: {s.wins[0]} - {s.wins[1]}  "
          f"({100*s.wins[0]/s.games_played:.1f}% / {100*s.wins[1]/s.games_played:.1f}%)  "
          f"avg rounds: {s.avg_rounds_per_game:.2f}  speed: {s.games_per_second:.0f}/s")

    # Test 2: Greedy vs Random
    print("\n[2] Greedy vs Random (1000 games)")
    s = train(1000, GreedyPlayer("Greedy"), RandomPlayer("Random"))
    print(f"  Results: Greedy {s.wins[0]} - {s.wins[1]} Random  "
          f"({100*s.wins[0]/s.games_played:.1f}% / {100*s.wins[1]/s.games_played:.1f}%)")

    # Test 3: Random vs Greedy
    print("\n[3] Random vs Greedy (1000 games)")
    s = train(1000, RandomPlayer("Random"), GreedyPlayer("Greedy"))
    print(f"  Results: Random {s.wins[0]} - {s.wins[1]} Greedy  "
          f"({100*s.wins[0]/s.games_played:.1f}% / {100*s.wins[1]/s.games_played:.1f}%)")

    # Test 4: Greedy vs Greedy
    print("\n[4] Greedy vs Greedy (1000 games)")
    s = train(1000, GreedyPlayer("Greedy-0"), GreedyPlayer("Greedy-1"))
    print(f"  Results: {s.wins[0]} - {s.wins[1]}  "
          f"({100*s.wins[0]/s.games_played:.1f}% / {100*s.wins[1]/s.games_played:.1f}%)")

    print("\n" + "=" * 60)
    print("Training complete!")


if __name__ == "__main__":
    main()
