#!/usr/bin/env python3
"""
Sixty-Six card game — interactive entry point.
Usage:
    python card_game.py              # play vs random bot
    python card_game.py greedy       # play vs greedy bot
    python card_game.py ppo          # play vs PPO-trained neural net
    python card_game.py --PlayerView # show raw PlayerView fields
    python card_game.py --marriage   # force human's hand to include a marriage
    python card_game.py --marriage-bot   # force bot's hand to include a marriage
    python card_game.py --nine-trump     # force human's hand to include 9 of trump
    python card_game.py --nine-trump-bot # force bot's hand to include 9 of trump
    python card_game.py --hints          # show inference hints and opponent hand when derivable
"""

import sys

import torch

from agents import HumanPlayer, RandomPlayer, GreedyPlayer, SmartPlayer, PolicyPlayer
from game import Match
from net import ActorCriticNet
from ui import TerminalUI


def _make_ppo_player() -> PolicyPlayer:
    """Load the PPO-trained actor-critic and wrap it in a greedy PolicyPlayer."""
    import os
    path = os.path.join(os.path.dirname(__file__), "policy_ppo_final.pt")
    if not os.path.exists(path):
        # fall back to best-checkpoint
        path = os.path.join(os.path.dirname(__file__), "policy_ppo.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = ActorCriticNet()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return PolicyPlayer(model, name="PPO-Bot", greedy=True)


BOT_TYPES = {
    "random": lambda: RandomPlayer("Computer"),
    "greedy": lambda: GreedyPlayer("Computer"),
    "smart":  lambda: SmartPlayer("Computer"),
    "ppo":   _make_ppo_player,
}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    bot_name = args[0].lower() if args else "random"
    show_player_view = "--PlayerView" in flags
    show_hints = "--hints" in flags
    force_marriage = "--marriage" in flags
    force_marriage_bot = "--marriage-bot" in flags
    force_nine_trump = "--nine-trump" in flags
    force_nine_trump_bot = "--nine-trump-bot" in flags

    # 9-of-trump is unique — can't be forced into both hands
    if force_nine_trump and force_nine_trump_bot:
        print("Error: --nine-trump and --nine-trump-bot are mutually exclusive")
        print("       (there is only one 9 of trump in the deck)")
        sys.exit(1)

    if bot_name not in BOT_TYPES:
        print(f"Unknown bot: {bot_name}")
        print(f"Available: {', '.join(BOT_TYPES)}")
        sys.exit(1)

    ui = TerminalUI(show_player_view=show_player_view,
                     show_hints=show_hints)
    human = HumanPlayer(ui, name="You")
    bot = BOT_TYPES[bot_name]()

    # Human is seat 0, bot is seat 1
    ui.set_context(seat=0, opponent_name=bot.name)
    players = {0: human, 1: bot}

    print(f"Playing against: {bot_name} bot")
    print()

    # Build force-deal parameters
    marriage_seats: set[int] | None = None
    if force_marriage or force_marriage_bot:
        marriage_seats = set()
        if force_marriage:
            marriage_seats.add(0)
        if force_marriage_bot:
            marriage_seats.add(1)

    nine_trump_seat: int | None = None
    if force_nine_trump:
        nine_trump_seat = 0
    elif force_nine_trump_bot:
        nine_trump_seat = 1

    while True:
        match = Match(players,
                      force_marriage_seats=marriage_seats,
                      force_nine_trump_seat=nine_trump_seat)
        match.play()
        if not ui.prompt_play_again():
            break


if __name__ == "__main__":
    main()
