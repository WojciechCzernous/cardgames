#!/usr/bin/env python3
"""Full eval: MCTS vs Random, MCTS vs Greedy, MCTS vs PPO."""

import random
import torch
from agents import RandomPlayer, GreedyPlayer, PolicyPlayer
from net import ActorCriticNet
from game import Match
from ismcts import ISMCTSPlayer

random.seed(42)

ckpt = torch.load("policy_ppo_final.pt", map_location="cpu", weights_only=True)
model = ActorCriticNet()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


def run_matches(p0_fn, p1_fn, n, label):
    wins = losses = 0
    gp_for = gp_against = 0
    for i in range(n):
        # Alternate seats to reduce first-player bias
        if i % 2 == 0:
            p0, p1 = p0_fn(), p1_fn()
            seat = 0
        else:
            p0, p1 = p1_fn(), p0_fn()
            seat = 1

        mr = Match({0: p0, 1: p1}).play()
        if mr.winner == seat:
            wins += 1
        elif mr.winner is not None:
            losses += 1
        gp_for += mr.game_points[seat]
        gp_against += mr.game_points[1 - seat]

        if (i + 1) % 10 == 0:
            print(f"  {label}: {i+1}/{n}  W={wins} L={losses}")

    draws = n - wins - losses
    print(f"\n{label} ({n} games):")
    print(f"  Win {wins/n:.1%}  Draw {draws/n:.1%}  Loss {losses/n:.1%}")
    print(f"  Avg GP: {gp_for/n:.2f} for, {gp_against/n:.2f} against\n")


def make_mcts():
    return ISMCTSPlayer(model, name="MCTS",
                        n_determinizations=16, n_simulations=100)


N = 100  # 100 matches each (MCTS is slow, ~5s/match)

print("=== ISMCTS Evaluation ===\n")

run_matches(make_mcts, lambda: RandomPlayer("R"), N, "MCTS vs Random")
run_matches(make_mcts, lambda: GreedyPlayer("G"), N, "MCTS vs Greedy")
run_matches(
    make_mcts,
    lambda: PolicyPlayer(model, name="PPO", greedy=True),
    N,
    "MCTS vs PPO",
)
