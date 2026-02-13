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




