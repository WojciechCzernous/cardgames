# cardgames

"66" for two players (here: human vs robot) - rules as described by Lech Pijanowski in "Przewodnik gier", ed. Iskry, 1973, Warszawa.

## Architecture

### Core game engine

| File | Role |
|---|---|
| `models.py` | Pure data types: `Card`, `Suit`, `Action`, `PlayerView`, result dataclasses |
| `rules.py` | Stateless rules oracle: `get_valid_actions()`, `trick_winner()`, `find_marriages()`, `compute_game_points()` |
| `game.py` | `RoundState` (independent state object), `Round` and `Match` (game loop engines) |
| `agents.py` | `Player` ABC and all agent implementations (see *Agents* below) |
| `ui.py` | `TerminalUI` — display/input only, used by `HumanPlayer` |
| `card_game.py` | Interactive entry point (`python card_game.py <bot> [--flags]`) |

### Agents (`agents.py` + `ismcts.py`)

| Agent | Description |
|---|---|
| `RandomPlayer` | Uniform random over legal actions |
| `GreedyPlayer` | Hand-crafted heuristic: prefers marriages, high-value wins, trump economy |
| `SmartPlayer` | Greedy in phase 1, exact minimax in phase 2 (via `EndgameSolver`) |
| `PolicyPlayer` | Neural-network policy (PPO-trained `ActorCriticNet`); supports greedy or stochastic play |
| `ISMCTSPlayer` | **Strongest.** Information-Set MCTS with neural-network policy priors and value evaluation — AlphaGo-style search at inference time |
| `HumanPlayer` | Interactive terminal player (delegates to `TerminalUI`) |

### ML pipeline

| File | Role |
|---|---|
| `features.py` | Converts `PlayerView` → 248-dim float32 tensor; maps actions ↔ 27-dim index |
| `net.py` | `PolicyNet` (100K params) and `ActorCriticNet` (133K params, shared trunk + policy/value heads) |
| `generate_data.py` | Self-play dataset generator (greedy/smart bots → `.npz` files) |
| `train_sl.py` | Supervised learning: cross-entropy on SL dataset → `policy_sl.pt` |
| `train_ppo.py` | PPO self-play RL: warm-starts from SL, frozen-opponent snapshots → `policy_ppo_final.pt` |
| `ismcts.py` | Determinized MCTS engine: determinizer, MCTS tree search, IS-MCTS aggregation, `ISMCTSPlayer` |

### Solvers & utilities

| File | Role |
|---|---|
| `solver.py` | `EndgameSolver` — memoised minimax for phase 2 (perfect information, ≤6 cards/side) |
| `minimax_game.py` | Generic alternating-initiative minimax framework |
| `test_solver.py` | Solver test harness with game-tree visualization |
| `train.py` | Headless match runner (any agent vs any agent) |

**Key design choices:**
- **Symmetric players** — both seats are `Player` agents (seat 0, seat 1). No "player"/"computer" asymmetry.
- **Independent state** — `RoundState` holds all authoritative game state; `PlayerView` is the per-seat observable projection.
- **Rules oracle** — `rules.py` contains pure functions that answer "what's legal?" given state, completely separated from the engine.
- **Any matchup** — plug any `Player` subclass (including an RL agent) into either seat.

## Usage

```bash
python card_game.py              # play vs random bot
python card_game.py greedy       # play vs greedy bot
python card_game.py smart        # play vs smart bot (minimax endgame)
python card_game.py ppo          # play vs PPO-trained neural net
python card_game.py mcts         # play vs IS-MCTS (strongest, ~5s/move)
python card_game.py mcts --hints # strongest bot + show inference hints
python card_game.py smart --PlayerView  # show raw PlayerView fields each turn
python card_game.py smart --marriage        # force human's hand to include a marriage
python card_game.py smart --marriage-bot    # force bot's hand to include a marriage
python card_game.py smart --nine-trump      # force human's hand to include 9 of trump
python card_game.py smart --nine-trump-bot  # force bot's hand to include 9 of trump
python train.py                  # run headless training
```

## Endgame Solver

`SmartPlayer` plays greedy heuristics in phase 1, then switches to **exact minimax** in phase 2 once the draw pile empties and full information is available. The opponent's hand is derived by elimination: `all_24_cards − my_hand − won_cards`.

`EndgameSolver` (in `solver.py`) uses memoised minimax over the phase 2 game tree. With ≤6 cards per side the search space is small and solves instantly. The solver handles:
- Must-follow-suit constraints (phase 2 rules)
- Marriage announcements (including instant-66 wins)
- `compute_game_points` scoring (1/2/3 game points depending on opponent's score and closed state)

### Testing & Visualization

```bash
python test_solver.py              # run all test cases
python test_solver.py --tree       # print full game trees
python test_solver.py --tree -n 4  # tree for a specific case only
```

The tree shows each decision node with the deciding seat (`[sN]`), MAX/MIN role, current hand & scores, and the minimax value. Terminal nodes display final scores, winner, and game points. Marriage announcements are marked with 💍.

## ML Pipeline (AlphaGo-inspired)

The AI training follows an AlphaGo-inspired three-stage pipeline:

### Stage 1 — Supervised Learning

Train a policy network to imitate greedy self-play:

```bash
python generate_data.py                 # 2M greedy self-play samples → data/sl_greedy_2000000.npz
python train_sl.py                      # cross-entropy training → policy_sl.pt (89.6% val accuracy)
```

**PolicyNet** (100K params): 248 → 256 → 128 → 27 MLP with valid-action masking.

### Stage 2 — PPO Self-Play (Reinforcement Learning)

Improve beyond imitation via self-play with Proximal Policy Optimisation:

```bash
python train_ppo.py --iterations 200 --games 512    # → policy_ppo_final.pt
```

**ActorCriticNet** (133K params): shared trunk (248 → 256) + policy head (256 → 128 → 27) + value head (256 → 128 → 1). Warm-started from the SL checkpoint. Opponent is a frozen snapshot refreshed every 10 iterations.

### Stage 3 — Information-Set MCTS (Search at Inference Time)

The key AlphaGo insight: use the trained policy + value networks to guide tree search at play time, yielding much stronger decisions than the raw network.

**Determinized MCTS** handles imperfect information (can't see opponent's cards):

1. **Determinize** — sample 16 plausible opponent hands consistent with observations (known cards, void suits, hand size)
2. **Search** — run 100 MCTS simulations per world, guided by:
   - Policy prior P(a|s) for PUCT exploration
   - Value V(s) for leaf evaluation (no rollouts)
3. **Aggregate** — merge visit counts across all determinizations, pick the most-visited action

### Agent Strength Comparison

Evaluated over 2000 matches (PPO, Greedy) and 100 matches (MCTS):

| Agent | vs Random | vs Greedy | vs PPO |
|---|---|---|---|
| **MCTS** (16 det × 100 sim) | **92.0%** | **65.0%** | **62.0%** |
| **PPO** (greedy policy) | 97.5% | 52.3% | — |
| **Greedy** (heuristic) | 90.3% | — | 47.6% |

MCTS is the strongest agent, with a clear edge over all others. The tradeoff is speed: ~5 seconds per move vs instant for PPO/Greedy.

## Feature Encoding

`features.py` converts a `PlayerView` into a **248-dimensional** float32 tensor for neural-network input, and maps actions to a **27-dimensional** discrete action space.

### State vector (248 floats)

| Feature | Encoding | Dims |
|---|---|---|
| `hand` | 24-bit card vector (1 = in hand) | 24 |
| `trump_suit` | 4-bit one-hot | 4 |
| `trump_card` | 24-bit one-hot (all zeros if face-down trump taken) | 24 |
| `draw_pile_size` | scalar (raw count) | 1 |
| `phase` | 1 = phase 2, 0 = phase 1 | 1 |
| `closed_by` | 2 bits: [me, opponent] | 2 |
| `my_score` | scalar ÷ 66 | 1 |
| `opponent_score` | scalar ÷ 66 | 1 |
| `is_leading` | 1 bit | 1 |
| `lead_card` | 24-bit one-hot (zeros when leading) | 24 |
| `lead_marriage` | 4-bit one-hot suit (zeros if no marriage) | 4 |
| `valid_actions` | 24-bit play mask + swap + close + pass | 27 |
| `is_winner_action_phase` | 1 bit | 1 |
| `my_won_cards` | 24-bit card vector | 24 |
| `opponent_won_cards` | 24-bit card vector | 24 |
| `my_marriages` | 4-bit suit vector | 4 |
| `opponent_marriages` | 4-bit suit vector | 4 |
| `opponent_known_cards` | 24-bit card vector (cards seen played/inferred) | 24 |
| `opponent_void_suits` | 4-bit suit vector | 4 |
| `unknown_cards` | 24-bit card vector (could be in opponent hand or pile) | 24 |
| `card_threats` | 24 floats (count of unseen cards that beat each card) | 24 |
| `opponent_hand_size` | scalar | 1 |
| | **Total** | **248** |

Card indexing is **suit-major, rank-minor**: card index = `suit_idx × 6 + rank_idx`, where suits follow enum order (♥ ♣ ♦ ♠) and ranks are ordered 9, J, Q, K, 10, A.

### Action space (27 discrete actions)

| Index | Action |
|---|---|
| 0–23 | Play card (same suit-major rank-minor indexing) |
| 24 | Swap trump 9 |
| 25 | Close the game |
| 26 | Pass (end winner-action phase) |

### Dataset generation

```bash
python generate_data.py                    # 2M greedy self-play samples (default)
python generate_data.py --n 500000         # custom count
python generate_data.py --bot smart        # smart self-play
python generate_data.py --out data.npz     # custom output path
python generate_data.py --seed 42          # reproducible
```

Each round produces one randomly-sampled `(state, action)` pair. Output is a compressed `.npz` file with `states` (N×248 float32) and `actions` (N, int64).

## Rules of Sixty-Six (Sześćdziesiąt sześć)

Rules as described by Lech Pijanowski in *Przewodnik gier*, ed. Iskry, 1973, Warszawa.

### Deck & Deal

- **24 cards**: 9, J, Q, K, 10, A in each of four suits (♥ ♦ ♣ ♠).
- **Card point values**: 9 = 0, J = 2, Q = 3, K = 4, 10 = 10, A = 11. Total in the deck: 120 points.
- Each player is dealt **6 cards**. The next card is placed face-up as the **trump card** (determining the trump suit). The remaining 11 cards form the face-down **draw pile**.

### Trick Play

1. The **leader** (trick winner or first dealer's opponent) plays any card from hand.
2. The **follower** plays one card in response.
3. The higher card wins the trick and its player collects both cards' point values.

**Trick resolution:**
- A trump card beats any non-trump card.
- If both cards are the same suit, the higher rank wins.
- If the follower plays a different (non-trump) suit, the leader's card wins regardless of rank.

**Rank order** (low → high): 9, J, Q, K, 10, A.

### Two Phases

**Phase 1** — while the draw pile is non-empty:
- The follower may play **any card** (no obligation to follow suit or trump).
- After each trick, the winner draws first from the pile, then the loser. Each player's hand stays at 6 cards.
- The trick winner may optionally perform **winner actions** (see below) before the next trick.

**Phase 2** — once the draw pile and trump card are exhausted (or the game is closed):
- The follower **must follow suit** if possible.
- If unable to follow suit, the follower **must play a trump** if possible.
- If unable to do either, any card may be played.
- No more drawing occurs.

### Marriage (Meld)

A player holding both the **King and Queen of the same suit** may announce a **marriage** when leading a trick, by playing the K or Q of that suit:
- **Trump marriage** (K + Q of trump suit): **40 points** added immediately.
- **Non-trump marriage**: **20 points** added immediately.

Marriage can only be announced by the **leader** of a trick (not the follower).

### Special Actions (Phase 1 only, trick winner)

After winning a trick in phase 1, the leader may:

- **Swap the trump card**: If the leader holds the **9 of trump**, they may exchange it for the face-up trump card under the draw pile.
- **Close the game**: The leader declares the game closed. No more cards are drawn; phase 2 rules apply immediately. Closing is a strategic gamble — see scoring below.

### Winning a Round

A round ends when:
- A player's trick points (including marriages) reach **66 or more** — that player wins immediately.
- All cards are played without anyone reaching 66 — a tie.
- If the game was **closed**, special scoring applies (see below).

### Game Points (Round Scoring)

Each round awards **game points** toward the match:

| Situation | Game points |
|---|---|
| All cards played, both failed to reach 66 - a tie | **0** both |
| Winner reached 66, opponent has ≥ 33 points | **1** |
| Winner reached 66, opponent has < 33 points | **2** |
| Game was **closed** and the closer reached 66 | **3** |
| Game was **closed** and the closer failed to reach 66 | **3** to the opponent |

### Match

The first player to accumulate **7 game points** (across multiple rounds) wins the match. The first leader alternates between rounds.




