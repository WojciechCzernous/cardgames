#!/usr/bin/env python3
"""
Sixty-Six card game — interactive entry point.
Usage:
    python card_game.py              # play vs random bot
    python card_game.py greedy       # play vs greedy bot
    python card_game.py --reveal     # show bot hand when draw pile is empty
    python card_game.py --PlayerView # show raw PlayerView fields
    python card_game.py --marriage   # force human's hand to include a marriage
    python card_game.py --nine-trump # force human's hand to include 9 of trump
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
    show_player_view = "--PlayerView" in flags
    force_marriage = "--marriage" in flags
    force_nine_trump = "--nine-trump" in flags

    if bot_name not in BOT_TYPES:
        print(f"Unknown bot: {bot_name}")
        print(f"Available: {', '.join(BOT_TYPES)}")
        sys.exit(1)

    ui = TerminalUI(reveal_opponent=reveal, show_player_view=show_player_view)
    human = HumanPlayer(ui, name="You")
    bot = BOT_TYPES[bot_name]()

    # Human is seat 0, bot is seat 1
    ui.set_context(seat=0, opponent_name=bot.name)
    players = {0: human, 1: bot}

    print(f"Playing against: {bot_name} bot")
    print()

    while True:
        match = Match(players,
                      force_marriage_seat=0 if force_marriage else None,
                      force_nine_trump_seat=0 if force_nine_trump else None)
        match.play()
        if not ui.prompt_play_again():
            break


if __name__ == "__main__":
    main()
