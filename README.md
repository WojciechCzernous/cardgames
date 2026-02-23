# cardgames

"66" for two players (here: human vs robot) - rules as described by Lech Pijanowski in "Przewodnik gier", ed. Iskry, 1973, Warszawa.

## Architecture

| File | Role |
|---|---|
| `models.py` | Pure data types: `Card`, `Suit`, `Action`, `PlayerView`, result dataclasses |
| `rules.py` | Stateless rules oracle: `get_valid_actions()`, `trick_winner()`, `find_marriages()`, `compute_game_points()` |
| `game.py` | `RoundState` (independent state object), `Round` and `Match` (game loop engines) |
| `agents.py` | `Player` ABC, `RandomPlayer`, `GreedyPlayer`, `SmartPlayer`, `HumanPlayer` |
| `ui.py` | `TerminalUI` — display/input only, used by `HumanPlayer` |
| `solver.py` | `EndgameSolver` — minimax endgame solver for phase 2 (perfect information) |
| `train.py` | Headless training (any agent vs any agent) |
| `card_game.py` | Interactive entry point |
| `test_solver.py` | Solver test harness with game-tree visualization |

**Key design choices:**
- **Symmetric players** — both seats are `Player` agents (seat 0, seat 1). No "player"/"computer" asymmetry.
- **Independent state** — `RoundState` holds all authoritative game state; `PlayerView` is the per-seat observable projection.
- **Rules oracle** — `rules.py` contains pure functions that answer "what's legal?" given state, completely separated from the engine.
- **Seen cards per seat** — both players track their own `seen_cards` (set of observed cards throughout the round).
- **Any matchup** — plug any `Player` subclass (including an RL agent) into either seat.

## Usage

```bash
python card_game.py              # play vs random bot
python card_game.py greedy       # play vs greedy bot
python card_game.py smart        # play vs smart bot (minimax endgame)
python card_game.py smart --reveal  # smart bot + show opponent hand in phase 2
python train.py                  # run headless training
```

## Endgame Solver

`SmartPlayer` plays greedy heuristics in phase 1, then switches to **exact minimax** in phase 2 once the draw pile empties and full information is available. The opponent's hand is derived by elimination: `all_24_cards − my_hand − played_cards`.

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




