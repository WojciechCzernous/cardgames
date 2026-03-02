#!/usr/bin/env python3
"""
PPO self-play training for Sixty-Six.

Initialises from an SL-pretrained policy, then improves via self-play
using Proximal Policy Optimisation with a shared actor-critic network.

Opponent is a frozen snapshot of the policy, refreshed periodically.
Evaluation is run against the GreedyPlayer baseline.

Usage:
    python train_ppo.py                                    # defaults
    python train_ppo.py --sl policy_sl.pt                  # warm-start from SL
    python train_ppo.py --iterations 200 --games 512       # more games per batch
    python train_ppo.py --resume policy_ppo.pt             # continue training
"""

from __future__ import annotations

import argparse
import copy
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from features import FEATURE_DIM, ACTION_DIM
from net import ActorCriticNet
from agents import PolicyPlayer, GreedyPlayer
from game import Round


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def play_batch(model: ActorCriticNet, opponent_model: ActorCriticNet,
               n_games: int) -> tuple[list[dict], dict]:
    """
    Play n_games rounds of self-play.
    The learning agent (seat 0) uses *model* with recording.
    The opponent (seat 1) uses *opponent_model* without recording.

    Returns:
        trajectories: list of dicts, one per game, with keys:
            states, actions, log_probs, values  (lists per timestep)
            reward: float (game outcome for seat 0)
        stats: dict with win counts and score info
    """
    agent = PolicyPlayer(model, name="Agent", record=True)
    opponent = PolicyPlayer(opponent_model, name="Opponent", greedy=False, record=False)

    trajectories = []
    stats = {"wins": 0, "losses": 0, "draws": 0,
             "reward_sum": 0.0, "games": n_games,
             "close_count": 0, "close_success": 0}

    for _ in range(n_games):
        agent.reset_trajectory()

        # Randomise seats to avoid positional bias
        if random.random() < 0.5:
            seat_agent, seat_opp = 0, 1
        else:
            seat_agent, seat_opp = 1, 0

        players = {seat_agent: agent, seat_opp: opponent}
        rnd = Round(players)
        result = rnd.play()

        # Compute reward for the agent
        if result.winner is None:
            reward = 0.0
            stats["draws"] += 1
        elif result.winner == seat_agent:
            reward = float(result.game_points)     # +1, +2, or +3
            stats["wins"] += 1
        else:
            reward = -float(result.game_points)    # -1, -2, or -3
            stats["losses"] += 1

        stats["reward_sum"] += reward
        if result.closed:
            stats["close_count"] += 1
            if result.winner == result.closed_by:
                stats["close_success"] += 1

        traj = agent.trajectory
        if not traj:
            continue

        states = torch.stack([t[0] for t in traj])       # (T, 248)
        actions = torch.tensor([t[1] for t in traj])      # (T,)
        log_probs = torch.tensor([t[2] for t in traj])    # (T,)
        values = torch.tensor([t[3] for t in traj])       # (T,)

        trajectories.append({
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "reward": reward,
        })

    return trajectories, stats


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------

def compute_gae(trajectories: list[dict], gamma: float = 0.99,
                lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor,
                                             torch.Tensor]:
    """
    Compute Generalised Advantage Estimation across all trajectories.

    Each trajectory's reward is the terminal reward for the whole episode.
    Intermediate rewards are 0 (reward only at end of round).

    Returns: (states, actions, old_log_probs, returns, advantages)
    """
    all_states = []
    all_actions = []
    all_log_probs = []
    all_returns = []
    all_advantages = []

    for traj in trajectories:
        T = len(traj["actions"])
        values = traj["values"]              # (T,)
        terminal_reward = traj["reward"]

        # Intermediate rewards are 0; final step gets the episode reward
        rewards = torch.zeros(T)
        rewards[-1] = terminal_reward

        # GAE
        advantages = torch.zeros(T)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0   # terminal
            else:
                next_value = values[t + 1].item()
            delta = rewards[t] + gamma * next_value - values[t].item()
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        all_states.append(traj["states"])
        all_actions.append(traj["actions"])
        all_log_probs.append(traj["log_probs"])
        all_returns.append(returns)
        all_advantages.append(advantages)

    return (torch.cat(all_states),
            torch.cat(all_actions),
            torch.cat(all_log_probs),
            torch.cat(all_returns),
            torch.cat(all_advantages))


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(model: ActorCriticNet, optimizer: torch.optim.Optimizer,
               states: torch.Tensor, actions: torch.Tensor,
               old_log_probs: torch.Tensor, returns: torch.Tensor,
               advantages: torch.Tensor,
               clip_eps: float = 0.2, vf_coef: float = 0.5,
               ent_coef: float = 0.01, max_grad_norm: float = 0.5,
               ppo_epochs: int = 4, batch_size: int = 256,
               device: torch.device = torch.device("cpu"),
               ) -> dict[str, float]:
    """
    Run PPO update epochs over the collected batch.
    Returns dict of loss statistics.
    """
    states = states.to(device)
    actions = actions.to(device)
    old_log_probs = old_log_probs.to(device)
    returns = returns.to(device)
    advantages = advantages.to(device)

    # Normalise advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    # Normalise returns (value head learns relative scale)
    ret_mean = returns.mean()
    ret_std = returns.std() + 1e-8
    returns = (returns - ret_mean) / ret_std

    n = len(states)
    total_pg_loss = 0.0
    total_vf_loss = 0.0
    total_ent = 0.0
    total_clip_frac = 0.0
    n_updates = 0

    for _ in range(ppo_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            b_states = states[idx]
            b_actions = actions[idx]
            b_old_lp = old_log_probs[idx]
            b_returns = returns[idx]
            b_advantages = advantages[idx]

            masked_logits, values = model.policy_and_value(b_states)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            new_lp = log_probs.gather(1, b_actions.unsqueeze(1)).squeeze(1)

            # Entropy bonus (nan-safe: avoid 0 * -inf for masked actions)
            probs = torch.exp(log_probs)
            ent_per_action = -(probs * log_probs)
            entropy = ent_per_action.nan_to_num(0.0).sum(dim=-1).mean()

            # PPO clipped objective (clamp log-ratio to prevent exp overflow)
            log_ratio = (new_lp - b_old_lp).clamp(-20.0, 20.0)
            ratio = torch.exp(log_ratio)
            pg_loss1 = -b_advantages * ratio
            pg_loss2 = -b_advantages * ratio.clamp(1 - clip_eps, 1 + clip_eps)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss (clamp returns to prevent extreme targets)
            vf_loss = F.mse_loss(values, b_returns)

            loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy

            # Skip update if loss is NaN (defensive)
            if torch.isnan(loss):
                continue

            loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            clip_frac = ((ratio - 1).abs() > clip_eps).float().mean().item()
            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_ent += entropy.item()
            total_clip_frac += clip_frac
            n_updates += 1

    return {
        "pg_loss": total_pg_loss / max(n_updates, 1),
        "vf_loss": total_vf_loss / max(n_updates, 1),
        "entropy": total_ent / max(n_updates, 1),
        "clip_frac": total_clip_frac / max(n_updates, 1),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_vs_greedy(model: ActorCriticNet, n_games: int = 500) -> dict:
    """Play n_games as greedy-mode PolicyPlayer vs GreedyPlayer."""
    agent = PolicyPlayer(model, name="Policy", greedy=True, record=False)
    greedy = GreedyPlayer("Greedy")

    wins = losses = draws = 0
    gp_for = gp_against = 0

    for i in range(n_games):
        seat_agent = i % 2  # alternate seats
        seat_opp = 1 - seat_agent
        players = {seat_agent: agent, seat_opp: greedy}
        rnd = Round(players)
        result = rnd.play()

        if result.winner is None:
            draws += 1
        elif result.winner == seat_agent:
            wins += 1
            gp_for += result.game_points
        else:
            losses += 1
            gp_against += result.game_points

    return {
        "win_rate": wins / n_games,
        "loss_rate": losses / n_games,
        "draw_rate": draws / n_games,
        "gp_for": gp_for,
        "gp_against": gp_against,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PPO self-play training")
    parser.add_argument("--sl", type=str, default="policy_sl.pt",
                        help="SL checkpoint to warm-start from")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a PPO checkpoint")
    parser.add_argument("--out", type=str, default="policy_ppo.pt")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Number of PPO iterations")
    parser.add_argument("--games", type=int, default=512,
                        help="Self-play games per iteration")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="PPO update epochs per iteration")
    parser.add_argument("--batch", type=int, default=256,
                        help="Mini-batch size for PPO updates")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--opponent-refresh", type=int, default=10,
                        help="Refresh opponent snapshot every N iterations")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluate vs greedy every N iterations")
    parser.add_argument("--eval-games", type=int, default=500)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cpu")   # game loop is CPU-bound anyway

    hidden = tuple(args.hidden)
    model = ActorCriticNet(hidden_dims=hidden).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_iter = ckpt.get("iteration", 0)
        print(f"Resumed from {args.resume} (iteration {start_iter})")
    elif args.sl:
        try:
            model.load_sl_policy(args.sl)
            print(f"Warm-started from SL policy: {args.sl}")
        except Exception as e:
            print(f"Warning: could not load SL policy ({e}), starting fresh")
        start_iter = 0
    else:
        start_iter = 0

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {' → '.join(map(str, [FEATURE_DIM] + list(hidden) + [ACTION_DIM]))}+V  ({params:,} params)")

    # Opponent is a frozen snapshot
    opponent = copy.deepcopy(model)
    opponent.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Initial evaluation
    print("\nInitial evaluation vs greedy ...")
    ev = evaluate_vs_greedy(model, args.eval_games)
    print(f"  win={ev['win_rate']:.1%}  loss={ev['loss_rate']:.1%}  draw={ev['draw_rate']:.1%}")

    print(f"\nStarting PPO training: {args.iterations} iterations × {args.games} games")
    header = (f"{'It':>4}  {'reward':>7}  {'win%':>5}  {'pg':>7}  {'vf':>7}  "
              f"{'ent':>6}  {'clip':>5}  {'steps':>6}  {'t':>5}")
    print(header)
    print("-" * len(header))

    best_win_rate = ev["win_rate"]

    for iteration in range(start_iter + 1, start_iter + args.iterations + 1):
        t0 = time.perf_counter()

        # --- Collect trajectories ---
        model.eval()
        trajectories, stats = play_batch(model, opponent, args.games)

        if not trajectories:
            print(f"{iteration:4d}  (no trajectories collected)")
            continue

        # --- Compute GAE ---
        states, actions, old_lp, returns, advantages = compute_gae(
            trajectories, gamma=args.gamma, lam=args.lam)
        n_steps = len(states)

        # --- PPO update ---
        model.train()
        losses = ppo_update(
            model, optimizer, states, actions, old_lp, returns, advantages,
            clip_eps=args.clip_eps, vf_coef=args.vf_coef,
            ent_coef=args.ent_coef, max_grad_norm=args.max_grad_norm,
            ppo_epochs=args.ppo_epochs, batch_size=args.batch,
            device=device,
        )

        elapsed = time.perf_counter() - t0
        avg_reward = stats["reward_sum"] / stats["games"]
        win_pct = stats["wins"] / stats["games"]

        print(f"{iteration:4d}  {avg_reward:+7.3f}  {win_pct:5.1%}  "
              f"{losses['pg_loss']:7.4f}  {losses['vf_loss']:7.4f}  "
              f"{losses['entropy']:6.3f}  {losses['clip_frac']:5.2f}  "
              f"{n_steps:6d}  {elapsed:5.1f}s")

        # --- Refresh opponent snapshot ---
        if iteration % args.opponent_refresh == 0:
            opponent = copy.deepcopy(model)
            opponent.eval()
            print(f"      ↻ opponent snapshot refreshed")

        # --- Evaluate vs greedy ---
        if iteration % args.eval_interval == 0:
            model.eval()
            ev = evaluate_vs_greedy(model, args.eval_games)
            marker = " ★" if ev["win_rate"] > best_win_rate else ""
            print(f"      eval vs greedy: win={ev['win_rate']:.1%}  "
                  f"loss={ev['loss_rate']:.1%}  draw={ev['draw_rate']:.1%}{marker}")

            if ev["win_rate"] > best_win_rate:
                best_win_rate = ev["win_rate"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "hidden_dims": hidden,
                    "iteration": iteration,
                    "win_rate_vs_greedy": ev["win_rate"],
                }, args.out)
                print(f"      saved best → {args.out}")

    # Final save
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_dims": hidden,
        "iteration": start_iter + args.iterations,
        "win_rate_vs_greedy": best_win_rate,
    }, args.out.replace(".pt", "_final.pt"))
    print(f"\nFinal model saved to {args.out.replace('.pt', '_final.pt')}")

    # Final evaluation
    model.eval()
    ev = evaluate_vs_greedy(model, args.eval_games * 2)
    print(f"Final eval vs greedy ({args.eval_games * 2} games): "
          f"win={ev['win_rate']:.1%}  loss={ev['loss_rate']:.1%}  "
          f"draw={ev['draw_rate']:.1%}")


if __name__ == "__main__":
    main()
