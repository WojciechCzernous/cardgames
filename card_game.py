#!/usr/bin/env python3
"""
Sixty-Six card game — interactive entry point.
Usage:
    python card_game.py              # play vs random bot
    python card_game.py greedy       # play vs greedy bot
    python card_game.py --reveal     # show bot hand when draw pile is empty
"""

import sys

from agents import HumanPlayer, RandomPlayer, GreedyPlayer, SmartPlayer
from game import Match
from ui import TerminalUI


BOT_TYPES = {
    "random": lambda: RandomPlayer("Computer"),
    "greedy": lambda: GreedyPlayer("Computer"),
    "smart":  lambda: SmartPlayer("Computer"),
}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    bot_name = args[0].lower() if args else "random"
    reveal = "--reveal" in flags

    if bot_name not in BOT_TYPES:
        print(f"Unknown bot: {bot_name}")
        print(f"Available: {', '.join(BOT_TYPES)}")
        sys.exit(1)

    ui = TerminalUI(reveal_opponent=reveal)
    human = HumanPlayer(ui, name="You")
    bot = BOT_TYPES[bot_name]()

    # Human is seat 0, bot is seat 1
    ui.set_context(seat=0, opponent_name=bot.name)
    players = {0: human, 1: bot}

    print(f"Playing against: {bot_name} bot")
    print()

    while True:
        match = Match(players)
        match.play()
        if not ui.prompt_play_again():
            break


if __name__ == "__main__":
    main()
