#!/usr/bin/env python3
"""Quick smoke test: MCTS bot plays 20 matches vs Greedy."""

import random
import torch
from agents import GreedyPlayer
from net import ActorCriticNet
from game import Match
from ismcts import ISMCTSPlayer

random.seed(42)

ckpt = torch.load("policy_ppo_final.pt", map_location="cpu", weights_only=True)
model = ActorCriticNet()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

N = 20
wins = losses = draws = 0

for i in range(N):
    mcts_player = ISMCTSPlayer(model, name="MCTS",
                                n_determinizations=8, n_simulations=50)
    greedy_player = GreedyPlayer("Greedy")

    # Alternate seats
    if i % 2 == 0:
        players = {0: mcts_player, 1: greedy_player}
        mcts_seat = 0
    else:
        players = {0: greedy_player, 1: mcts_player}
        mcts_seat = 1

    m = Match(players)
    mr = m.play()

    if mr.winner == mcts_seat:
        wins += 1
        res = "W"
    elif mr.winner is None:
        draws += 1
        res = "D"
    else:
        losses += 1
        res = "L"

    print(f"  Game {i+1:2d}: {res}  GP: {mr.game_points[mcts_seat]} - {mr.game_points[1-mcts_seat]}")

print(f"\nMCTS vs Greedy ({N} games): {wins}W / {draws}D / {losses}L  "
      f"({wins/N:.0%} win rate)")
