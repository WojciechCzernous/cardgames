#!/usr/bin/env python3
"""
Generate a supervised-learning dataset by self-play.

Each round produces one randomly-sampled (state, action) pair.
Output: an .npz file with:
  - states:  float32 array of shape (N, 248)
  - actions: int64 array of shape (N,) with values in [0, 26]

Usage:
    python generate_data.py                  # 2M samples, greedy vs greedy
    python generate_data.py --n 500000       # custom sample count
    python generate_data.py --bot smart      # smart vs smart
    python generate_data.py --out data.npz   # custom output path
"""

import argparse
import random
import time

import numpy as np
from tqdm import tqdm

from agents import GreedyPlayer, SmartPlayer, RandomPlayer
from features import player_view_to_tensor, action_to_index, sample_transition
from game import Round

BOT_TYPES = {
    "random": RandomPlayer,
    "greedy": GreedyPlayer,
    "smart": SmartPlayer,
}


def main():
    parser = argparse.ArgumentParser(description="Generate SL training data")
    parser.add_argument("--n", type=int, default=2_000_000,
                        help="Number of (state, action) samples (default: 2M)")
    parser.add_argument("--bot", type=str, default="greedy",
                        choices=BOT_TYPES.keys(),
                        help="Bot type for self-play (default: greedy)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output .npz path (default: data/sl_{bot}_{n}.npz)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    n = args.n
    bot_cls = BOT_TYPES[args.bot]
    out_path = args.out or f"data/sl_{args.bot}_{n}.npz"

    print(f"Generating {n:,} samples via {args.bot} self-play → {out_path}")

    states = np.empty((n, 248), dtype=np.float32)
    actions = np.empty(n, dtype=np.int64)

    t0 = time.perf_counter()
    for i in tqdm(range(n), desc="Sampling", unit="game", smoothing=0.01):
        rnd = Round({0: bot_cls("A"), 1: bot_cls("B")}, record=True)
        rnd.play()
        state_tensor, action_idx = sample_transition(rnd.transitions)
        states[i] = state_tensor.numpy()
        actions[i] = action_idx

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s ({n/elapsed:.0f} samples/sec)")

    np.savez_compressed(out_path, states=states, actions=actions)
    print(f"Saved to {out_path}")

    # Quick stats
    print(f"  states shape:  {states.shape}")
    print(f"  actions shape: {actions.shape}")
    unique, counts = np.unique(actions, return_counts=True)
    print(f"  action distribution: {dict(zip(unique.tolist(), counts.tolist()))}")


if __name__ == "__main__":
    main()
