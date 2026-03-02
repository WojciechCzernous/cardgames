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

#### Mathematical Foundations of PPO

##### The problem as a Markov Decision Process

We model each round of Sixty-Six as a finite-horizon MDP $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$, where:

- $\mathcal{S}$: the set of all possible game states (encoded as 248-dim vectors from `PlayerView`)
- $\mathcal{A} = \{0, 1, \ldots, 26\}$: the 27 discrete actions (24 card plays + swap + close + pass)
- $\mathcal{A}(s) \subseteq \mathcal{A}$: the subset of legal actions at state $s$, determined by the rules oracle
- $P(s' | s, a)$: transition kernel (deterministic given both players' actions, stochastic from the agent's perspective due to the opponent's policy and card draws)
- $r$: reward, non-zero only at the terminal state — $r_T \in \{-3, -2, -1, 0, +1, +2, +3\}$ (game points won or lost)
- $\gamma = 0.99$: discount factor

A *policy* is a map $\pi_\theta : \mathcal{S} \to \Delta(\mathcal{A})$ parameterised by neural network weights $\theta$. In our case, $\pi_\theta$ is realised by computing logits $f_\theta(s) \in \mathbb{R}^{27}$, masking illegal actions to $-10^9$, and applying softmax:

$$\pi_\theta(a \mid s) = \frac{\exp(f_\theta(s)_a)}{\sum_{a' \in \mathcal{A}(s)} \exp(f_\theta(s)_{a'})} \quad \text{for } a \in \mathcal{A}(s), \quad 0 \text{ otherwise.}$$

The objective is to find $\theta^*$ that maximises expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ is a trajectory sampled by playing the policy.

##### Policy gradient theorem

Define the *state-value* and *action-value* functions:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t = s\right], \qquad Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t = s, a_t = a\right]$$

and the *advantage function* $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$, which measures how much better action $a$ is compared to the average under $\pi$.

The **policy gradient theorem** (Sutton et al., 1999) states:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A^{\pi_\theta}(s_t, a_t)\right]$$

*Proof sketch.* Write $J(\theta) = \mathbb{E}_{s_0}[V^{\pi_\theta}(s_0)]$. By the chain rule applied recursively through the Bellman equation, the gradient decomposes into a sum over timesteps. Each term factors into $\nabla_\theta \log \pi_\theta(a_t|s_t)$ (the *score function*, from the log-derivative trick $\nabla_\theta \pi = \pi \cdot \nabla_\theta \log \pi$) weighted by $A^{\pi_\theta}(s_t, a_t)$.

The advantage function serves as a *control variate*: since $\mathbb{E}_{a \sim \pi}[A^\pi(s, a)] = 0$, using $A^\pi$ instead of $Q^\pi$ does not change the expected gradient but reduces its variance.

##### From policy gradient to trust regions

The vanilla policy gradient estimator $\hat{g} = \frac{1}{N}\sum_{t} \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t$ suffers from two problems:

1. **High variance** in the gradient estimate, requiring many samples.
2. **Catastrophic updates**: a single large gradient step can collapse the policy to a near-deterministic distribution from which recovery is difficult.

TRPO (Schulman et al., 2015) addresses (2) by constraining the policy update to a *trust region*:

$$\max_\theta \; \hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t\right] \qquad \text{subject to} \quad \hat{\mathbb{E}}_t\left[D_{\mathrm{KL}}\!\left(\pi_{\theta_{\text{old}}}(\cdot|s_t) \;\|\; \pi_\theta(\cdot|s_t)\right)\right] \leq \delta$$

The ratio $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_\text{old}}(a_t|s_t)$ is the *importance sampling correction* that allows reuse of data collected under $\theta_\text{old}$. Observe that $r_t(\theta_\text{old}) = 1$ and $\nabla_\theta r_t(\theta)\big|_{\theta=\theta_\text{old}} = \nabla_\theta \log \pi_\theta(a_t|s_t)\big|_{\theta_\text{old}}$, so the first-order behaviour is identical to the policy gradient. The KL constraint bounds how far the new policy can deviate.

##### PPO's clipped surrogate

PPO (Schulman et al., 2017) replaces the hard KL constraint with a simpler *clipped objective* that achieves a similar effect using only first-order optimisation:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[\min\!\left(r_t(\theta)\,\hat{A}_t, \;\operatorname{clip}\!\left(r_t(\theta),\; 1-\varepsilon,\; 1+\varepsilon\right)\hat{A}_t\right)\right]$$

where $\varepsilon = 0.2$ in our implementation.

**Intuition.** Consider the two cases:

- If $\hat{A}_t > 0$ (action was better than average): the objective wants to *increase* $r_t$. But the clip caps it at $1 + \varepsilon$, so there is no incentive to push the ratio beyond $1.2$ — the policy cannot change too aggressively toward this action.
- If $\hat{A}_t < 0$ (action was worse than average): the objective wants to *decrease* $r_t$. The clip caps it at $1 - \varepsilon$, preventing the policy from moving too far away in a single update.

In both cases, the $\min$ selects the more pessimistic (conservative) of the unclipped and clipped terms.

##### Generalised Advantage Estimation (GAE)

The advantage $A^\pi(s_t, a_t)$ is unknown and must be estimated. Define the one-step *TD residual*:

$$\delta_t = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)$$

where $V_\theta$ is the value head of our network. The GAE estimator (Schulman et al., 2016) is an exponentially-weighted sum of multi-step TD errors:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{\ell=0}^{T-t-1} (\gamma \lambda)^\ell \, \delta_{t+\ell}$$

with $\lambda \in [0,1]$ controlling a bias-variance tradeoff:

- $\lambda = 0$: $\hat{A}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ — the one-step TD estimate. Low variance (uses the value function) but biased (if $V_\theta$ is inaccurate).
- $\lambda = 1$: $\hat{A}_t = \sum_\ell \gamma^\ell r_{t+\ell} - V(s_t)$ — the Monte Carlo return minus the baseline. Unbiased but high variance.

We use $\lambda = 0.95$, which leans toward the Monte Carlo end — appropriate for our setting where episodes are short (~5–10 decisions per player) and the terminal reward carries the primary signal.

**In our implementation**, intermediate rewards are $r_t = 0$ for $t < T$ and $r_T = \pm\{1,2,3\}$. The GAE computation therefore simplifies: non-terminal $\delta_t = \gamma V(s_{t+1}) - V(s_t)$ and terminal $\delta_{T} = r_T - V(s_T)$.

##### The combined objective

The full PPO loss we optimise is:

$$L(\theta) = -L^{\text{CLIP}}(\theta) + c_1 \, L^{\text{VF}}(\theta) - c_2 \, H[\pi_\theta]$$

with three terms:

1. **Policy loss** $L^{\text{CLIP}}$: the clipped surrogate (negated because we minimise).
2. **Value loss** $L^{\text{VF}} = \hat{\mathbb{E}}_t\!\left[(V_\theta(s_t) - \hat{R}_t)^2\right]$: MSE between the value head's prediction and the GAE return target $\hat{R}_t = \hat{A}_t + V_{\theta_\text{old}}(s_t)$. Weighted by $c_1 = 0.5$.
3. **Entropy bonus** $H[\pi_\theta] = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$: encourages exploration by penalising premature convergence to a deterministic policy. Weighted by $c_2 = 0.01$.

Additional implementation details:

- Advantages are normalised to zero mean and unit variance before the update: $\hat{A}_t \leftarrow (\hat{A}_t - \bar{A}) / (\sigma_A + 10^{-8})$.
- The log-ratio $\log r_t = \log \pi_\theta(a_t|s_t) - \log \pi_{\theta_\text{old}}(a_t|s_t)$ is clamped to $[-20, 20]$ before exponentiation to prevent numerical overflow.
- Gradients are clipped to global norm $\leq 0.5$.
- 4 epochs of minibatch SGD (batch size 256) per PPO iteration.

##### Self-play and opponent snapshots

A subtlety of two-player self-play: the agent's environment is *non-stationary* because the opponent's policy changes as training progresses. Naïvely training against the current policy can lead to cyclic strategies (A beats B beats C beats A).

Our mitigation follows AlphaGo's approach: the opponent is a *frozen snapshot* $\pi_{\theta_\text{frozen}}$ of the learning agent's policy, refreshed every 10 iterations. This stabilises the reward signal. Seats are randomised each game to prevent positional overfitting.

### Stage 3 — Information-Set MCTS (Search at Inference Time)

The key AlphaGo insight: use the trained policy + value networks to guide tree search at play time, yielding much stronger decisions than the raw network.

**Determinized MCTS** handles imperfect information (can't see opponent's cards):

1. **Determinize** — sample 16 plausible opponent hands consistent with observations (known cards, void suits, hand size)
2. **Search** — run 100 MCTS simulations per world, guided by:
   - Policy prior P(a|s) for PUCT exploration
   - Value V(s) for leaf evaluation (no rollouts)
3. **Aggregate** — merge visit counts across all determinizations, pick the most-visited action

#### Mathematical Foundations of MCTS

##### Game trees and position values

A two-player zero-sum game defines a tree where nodes are game states, edges are actions, and terminal nodes carry payoffs. The *minimax value* of a position $s$ for the maximising player is defined recursively:

$$V^*(s) = \begin{cases} u(s) & \text{if } s \text{ is terminal} \\ \max_{a \in \mathcal{A}(s)} V^*(s') & \text{if it is MAX's turn (} s' = T(s,a) \text{)} \\ \min_{a \in \mathcal{A}(s)} V^*(s') & \text{if it is MIN's turn} \end{cases}$$

Our `EndgameSolver` computes $V^*$ exactly for phase 2 (≤6 cards per side, small tree). But for the full game tree (phase 1 + drawing + unknown opponent cards), exact minimax is intractable. MCTS approximates it by *sampling*.

##### The bandit analogy and UCB1

At each internal node of the search tree, the agent must choose which subtree to explore — an instance of the *multi-armed bandit problem*.

Given $K$ actions (arms) with unknown expected rewards $\mu_1, \ldots, \mu_K$, after $n$ total trials with $n_i$ trials of arm $i$ yielding empirical mean $\bar{X}_i$, the **UCB1** (Upper Confidence Bound) strategy selects:

$$a^* = \arg\max_i \left[\bar{X}_i + c\sqrt{\frac{\ln n}{n_i}}\right]$$

The first term (exploitation) favours arms with high observed reward. The second term (exploration) favours arms that have been tried less — its form follows from the **Hoeffding inequality**, which guarantees that the true mean $\mu_i$ lies within $\bar{X}_i \pm c\sqrt{\ln n / n_i}$ with high probability. UCB1 achieves cumulative regret $O(\sqrt{K n \ln n})$ — asymptotically near-optimal.

##### The MCTS algorithm

MCTS builds a partial game tree incrementally. Each *simulation* consists of four phases:

1. **Select.** Starting from the root, repeatedly apply the bandit policy (UCB or PUCT) to descend through existing nodes until reaching a leaf (unexpanded node or terminal state).

2. **Expand.** Add the leaf's children to the tree. Each child corresponds to one legal action.

3. **Evaluate.** Estimate the value of the newly expanded leaf. Classical MCTS uses random rollouts (play randomly to termination); AlphaGo replaces this with the value network $V_\theta(s)$, avoiding the high variance of random play.

4. **Backpropagate.** Walk back up the root→leaf path, updating each node's visit count $N(s)$ and cumulative value $W(s)$:
   $$N(s) \leftarrow N(s) + 1, \qquad W(s) \leftarrow W(s) + v$$
   where $v$ is the value from step 3 (or the terminal payoff). The mean value is $Q(s,a) = W(s,a) / N(s,a)$.

After $M$ simulations, the action at the root is chosen by the highest visit count: $a^* = \arg\max_a N(\text{root}, a)$. Visit counts are preferred over $Q$-values because they are more robust — a high-$Q$, low-$N$ action may simply be under-explored.

**Convergence.** As $M \to \infty$, the MCTS value estimate converges to the minimax value (Kocsis & Szepesvári, 2006). In practice, even a few hundred simulations produce strong play.

##### PUCT: neural-network-guided search

AlphaGo (Silver et al., 2016) replaces UCB1 with **PUCT** (Predictor + Upper Confidence bound for Trees), which incorporates a learned prior $P(a|s) = \pi_\theta(a|s)$:

$$a^* = \arg\max_a \left[Q(s,a) + c_{\text{puct}} \cdot P(a|s) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}\right]$$

Compared to UCB1, the exploration term is *weighted by the policy prior* $P(a|s)$. This has a profound effect:

- **Early simulations** ($N(s,a)$ small): the prior dominates. The search concentrates on moves the policy network considers promising, dramatically reducing the effective branching factor.
- **Many simulations** ($N(s,a)$ large): $P(a|s) \cdot \sqrt{N(s)} / (1 + N(s,a)) \to 0$, so $Q(s,a)$ dominates. The search converges to the true minimax value regardless of the prior's quality.

In our implementation, $c_{\text{puct}} = 1.5$, and $P(a|s)$ comes from our PPO-trained `ActorCriticNet`.

The value network $V_\theta(s)$ replaces rollouts at leaf nodes. Since the value head was trained by PPO to predict expected game-point rewards, it provides a reasonable position evaluation without the noise of random play.

##### Imperfect information: information sets

Sixty-Six is an *imperfect-information* game: the agent cannot see the opponent's hand or the order of the draw pile. Let $\mathcal{I}$ denote an **information set** — the equivalence class of all game states consistent with the agent's observations. In our encoding, $\mathcal{I}$ corresponds to a `PlayerView`: the agent knows its own hand, won cards, trump, scores, and some inferred constraints (known opponent cards, void suits), but multiple possible opponent hands and pile orderings are compatible with this view.

Standard MCTS assumes perfect information: it operates on a *single* game tree with known states. In imperfect-information games, naïvely running MCTS on an arbitrary guess of the hidden state produces unreliable results. Two adaptations exist:

1. **Information-Set MCTS (IS-MCTS)**: maintain a single tree over information sets rather than states. Theoretically clean but complex to implement.
2. **Determinized MCTS**: sample multiple *determinizations* (concrete states compatible with the information set), run standard MCTS on each, and aggregate. Simpler, and empirically strong.

We use approach (2).

##### Determinization

Given the agent's `PlayerView`, the **determinizer** samples a concrete opponent hand $\mathbf{h}_{\text{opp}}$ by:

1. **Start with certainties**: include all cards in `opponent_known_cards` (e.g., cards inferred from marriages or trump swaps).
2. **Respect void constraints**: exclude cards from suits in `opponent_void_suits` (inferred when opponent failed to follow suit in phase 2).
3. **Sample the rest**: from `unknown_cards` $\setminus$ `opponent_known_cards` $\setminus$ void-suit cards, sample uniformly at random to fill the opponent's hand to `opponent_hand_size` cards.
4. **Remaining unknowns** become the draw pile (shuffled randomly).

Formally, let $\mathcal{U}$ be the set of unknown cards, $\mathcal{K}$ the known opponent cards, $\mathcal{V}$ the void suits, and $n$ the opponent hand size. The determinizer samples:

$$\mathbf{h}_{\text{opp}} = \mathcal{K} \cup \text{Sample}\!\left(n - |\mathcal{K}|, \;\;\{\,c \in \mathcal{U} \setminus \mathcal{K} : \text{suit}(c) \notin \mathcal{V}\,\}\right)$$

##### IS-MCTS by aggregation

Given $D$ determinizations, each producing root action visit counts $\{N_d(a)\}_{a \in \mathcal{A}(s)}$ from $M$ simulations, the final decision aggregates:

$$a^* = \arg\max_{a \in \mathcal{A}(s)} \sum_{d=1}^{D} N_d(a)$$

This is equivalent to weighting each determinization equally. The intuition: an action that is good across *many* possible worlds is *robustly* good — it does not depend on a lucky guess about the hidden state. Actions that are brilliant in one world but terrible in another accumulate fewer total visits.

In our implementation: $D = 16$ determinizations, $M = 100$ simulations each, yielding $16 \times 100 = 1{,}600$ total simulations per decision.

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




