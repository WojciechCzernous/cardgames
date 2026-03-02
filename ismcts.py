"""
Determinized MCTS (Information-Set MCTS) for Sixty-Six.

Combines:
  - Determinization: sample plausible opponent hands compatible with
    observations (known cards, void suits, hand size).
  - MCTS: UCB1 tree search with neural-network policy prior (P) and
    value evaluation (V) at leaves — à la AlphaGo.
  - Aggregation: merge visit counts across determinizations to pick
    the most robust action.

Usage:
    from ismcts import ISMCTSPlayer
    player = ISMCTSPlayer(model, n_determinizations=16, n_simulations=100)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import torch

from models import Card, Suit, Action, ActionType, PlayerView, RANKS, RANK_VALUES
import rules
from features import (
    FEATURE_DIM, ACTION_DIM, VALID_ACTIONS_OFFSET,
    player_view_to_tensor, action_to_index, index_to_action,
)


# =========================================================================
# Determinizer — sample plausible worlds
# =========================================================================

def determinize(view: PlayerView, rng: random.Random | None = None) -> list[Card]:
    """
    Sample a plausible opponent hand consistent with observations.

    Constraints respected:
      - must contain all cards in view.opponent_known_cards
      - must NOT contain any suit in view.opponent_void_suits
      - must have exactly view.opponent_hand_size cards
      - drawn from view.unknown_cards (opponent hand + draw pile)
    
    Returns a list of Cards representing the sampled opponent hand.
    """
    if rng is None:
        rng = random

    opp_size = view.opponent_hand_size
    known = set(view.opponent_known_cards)
    void_suits = view.opponent_void_suits

    # Pool = unknown cards minus those known to be in opponent's hand
    pool = [c for c in view.unknown_cards if c not in known]

    # Filter pool: remove cards whose suit is voided for the opponent
    eligible = [c for c in pool if c.suit not in void_suits]

    need = opp_size - len(known)

    if need <= 0:
        return list(known)

    if len(eligible) < need:
        # Relax void-suit constraint if we can't fill the hand
        eligible = pool

    sampled = rng.sample(eligible, min(need, len(eligible)))
    return list(known) + sampled


# =========================================================================
# Lightweight forward model — advance game state from a PlayerView
# =========================================================================

def _build_hands_from_determinization(
    view: PlayerView, opp_hand: list[Card]
) -> dict[int, list[Card]]:
    """Build both hands: mine from view, opponent from determinization."""
    opp = 1 - view.seat
    return {view.seat: list(view.hand), opp: list(opp_hand)}


def _build_draw_pile(view: PlayerView, opp_hand: list[Card]) -> list[Card]:
    """Remaining draw pile = unknown_cards − opponent_hand."""
    opp_set = set(opp_hand)
    pile = [c for c in view.unknown_cards if c not in opp_set]
    random.shuffle(pile)
    return pile


# =========================================================================
# MCTS Node
# =========================================================================

@dataclass
class MCTSNode:
    """A node in the search tree."""
    # Game state at this node
    parent: Optional["MCTSNode"] = None
    action_idx: int = -1          # action that led here
    seat: int = 0                 # which seat is to act

    # Statistics
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0            # P(a|s) from the policy network

    # Children (action_idx → child node)
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    is_expanded: bool = False


# =========================================================================
# Single-determinization MCTS
# =========================================================================

class MCTS:
    """
    Runs MCTS on a single determinized game state.

    Uses the neural network for:
      - Policy prior P(a|s) to guide exploration (PUCT)
      - Value V(s) to evaluate leaf nodes (no rollouts)
    """

    def __init__(self, model, c_puct: float = 1.5, 
                 use_value: bool = True):
        self.model = model
        self.c_puct = c_puct
        self.use_value = use_value

    # ------------------------------------------------------------------
    # Core MCTS loop
    # ------------------------------------------------------------------

    def search(self, root_view: PlayerView, opp_hand: list[Card],
               n_simulations: int) -> dict[int, int]:
        """
        Run n_simulations from root_view with a determinized opponent hand.
        Returns {action_idx: visit_count} for the root's legal actions.
        """
        root = MCTSNode(seat=root_view.seat)

        # Expand root
        self._expand(root, root_view)

        hands = _build_hands_from_determinization(root_view, opp_hand)
        draw_pile = _build_draw_pile(root_view, opp_hand)

        for _ in range(n_simulations):
            # Clone state for this simulation
            sim_hands = {s: list(h) for s, h in hands.items()}
            sim_pile = list(draw_pile)
            sim_scores = {root_view.seat: root_view.my_score,
                          1 - root_view.seat: root_view.opponent_score}
            sim_leader = root_view.seat if root_view.is_leading else (1 - root_view.seat)
            sim_lead_card = root_view.lead_card
            sim_phase = root_view.phase
            sim_closed = root_view.closed
            sim_closed_by = root_view.closed_by
            sim_trump_suit = root_view.trump_suit
            sim_trump_card = root_view.trump_card
            sim_marriages = {root_view.seat: list(root_view.my_marriages),
                             1 - root_view.seat: list(root_view.opponent_marriages)}

            node = root
            path: list[MCTSNode] = [node]

            # --- SELECT: walk down the tree ---
            while node.is_expanded and node.children:
                node = self._select_child(node)
                path.append(node)

                # Apply the action to our simulation state
                action_idx = node.action_idx
                acting_seat = node.parent.seat if node.parent else root_view.seat

                sim_hands, sim_scores, sim_lead_card, sim_leader, \
                    sim_phase, sim_closed, sim_closed_by, sim_trump_card, \
                    sim_pile, sim_marriages, terminal, winner = \
                    self._apply_action(
                        action_idx, acting_seat,
                        sim_hands, sim_scores, sim_lead_card, sim_leader,
                        sim_phase, sim_closed, sim_closed_by, sim_trump_suit,
                        sim_trump_card, sim_pile, sim_marriages,
                    )

                if terminal:
                    # Backprop terminal value
                    gp = rules.compute_game_points(sim_scores, winner,
                                                   sim_closed, sim_closed_by)
                    value = self._terminal_reward(gp, root_view.seat)
                    self._backprop(path, value)
                    break

            else:
                # --- EXPAND & EVALUATE ---
                if not node.children and not self._is_terminal(sim_hands):
                    # Build a synthetic PlayerView for the current sim state
                    current_seat = self._next_to_act(
                        sim_lead_card, sim_leader, path, root_view.seat)
                    node.seat = current_seat

                    sim_view = self._make_view(
                        current_seat, sim_hands, sim_scores, sim_lead_card,
                        sim_leader, sim_phase, sim_closed, sim_closed_by,
                        sim_trump_suit, sim_trump_card, sim_pile, sim_marriages,
                    )

                    if sim_view.valid_actions:
                        self._expand(node, sim_view)

                        # Neural value estimate
                        value = self._evaluate(sim_view)
                    else:
                        value = 0.0

                    self._backprop(path, value)
                else:
                    # Terminal or no actions
                    gp = rules.compute_game_points(sim_scores, 
                                                   self._find_winner(sim_scores),
                                                   sim_closed, sim_closed_by)
                    value = self._terminal_reward(gp, root_view.seat)
                    self._backprop(path, value)

        # Collect root visit counts
        return {a: child.visit_count for a, child in root.children.items()}

    # ------------------------------------------------------------------
    # PUCT selection
    # ------------------------------------------------------------------

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest PUCT score."""
        best_score = -float("inf")
        best_child = None
        sqrt_parent = math.sqrt(max(node.visit_count, 1))

        for child in node.children.values():
            q = child.value_sum / max(child.visit_count, 1)
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _expand(self, node: MCTSNode, view: PlayerView):
        """Create children for all valid actions, with policy priors."""
        state_t = player_view_to_tensor(view).unsqueeze(0)

        with torch.no_grad():
            if hasattr(self.model, "policy_and_value"):
                masked_logits, _ = self.model.policy_and_value(state_t)
            else:
                masked_logits = self.model.masked_logits(state_t)

            probs = torch.softmax(masked_logits, dim=-1).squeeze(0)

        for action in view.valid_actions:
            a_idx = action_to_index(action, view.hand)
            child = MCTSNode(
                parent=node,
                action_idx=a_idx,
                seat=1 - node.seat,  # seat alternates (will be corrected)
                prior=probs[a_idx].item(),
            )
            node.children[a_idx] = child

        node.is_expanded = True

    # ------------------------------------------------------------------
    # Leaf evaluation via value network
    # ------------------------------------------------------------------

    def _evaluate(self, view: PlayerView) -> float:
        """Return V(s) from the acting player's perspective, mapped to root's perspective."""
        if not self.use_value:
            return 0.0

        state_t = player_view_to_tensor(view).unsqueeze(0)

        with torch.no_grad():
            if hasattr(self.model, "policy_and_value"):
                _, value = self.model.policy_and_value(state_t)
                return value.item()
            else:
                return 0.0

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backprop(self, path: list[MCTSNode], value: float):
        """Update visit counts and value sums along the path."""
        for node in reversed(path):
            node.visit_count += 1
            # Value is from root player's perspective
            node.value_sum += value

    # ------------------------------------------------------------------
    # Forward model helpers
    # ------------------------------------------------------------------

    def _apply_action(self, action_idx, acting_seat,
                      hands, scores, lead_card, leader,
                      phase, closed, closed_by, trump_suit,
                      trump_card, draw_pile, marriages):
        """
        Apply an action and return updated state.
        Returns: (hands, scores, lead_card, leader, phase, closed,
                  closed_by, trump_card, draw_pile, marriages, terminal, winner)
        """
        terminal = False
        winner = None

        # Special actions
        if action_idx == 24:  # swap trump
            nine = rules.find_nine_trump(hands[acting_seat], trump_suit)
            if nine and trump_card:
                hands[acting_seat].remove(nine)
                hands[acting_seat].append(trump_card)
                trump_card = nine
            return (hands, scores, lead_card, leader, phase, closed,
                    closed_by, trump_card, draw_pile, marriages, False, None)

        if action_idx == 25:  # close game
            closed = True
            closed_by = acting_seat
            phase = 2
            return (hands, scores, lead_card, leader, phase, closed,
                    closed_by, trump_card, draw_pile, marriages, False, None)

        if action_idx == 26:  # pass
            return (hands, scores, lead_card, leader, phase, closed,
                    closed_by, trump_card, draw_pile, marriages, False, None)

        # Play card (action_idx 0-23)
        card_suit_idx = action_idx // len(RANKS)
        card_rank_idx = action_idx % len(RANKS)
        target_suit = list(Suit)[card_suit_idx]
        target_rank = RANKS[card_rank_idx]

        card = None
        for c in hands[acting_seat]:
            if c.rank == target_rank and c.suit == target_suit:
                card = c
                break

        if card is None:
            # Card not in hand — shouldn't happen but be safe
            return (hands, scores, lead_card, leader, phase, closed,
                    closed_by, trump_card, draw_pile, marriages, False, None)

        hands[acting_seat].remove(card)

        # Check marriage (only when leading)
        marriage_pts = 0
        if lead_card is None and card.rank in (" K", " Q"):
            partner_rank = " Q" if card.rank == " K" else " K"
            if any(c.rank == partner_rank and c.suit == card.suit
                   for c in hands[acting_seat]):
                mar_suit = card.suit
                marriage_pts = rules.marriage_value(mar_suit, trump_suit)
                scores[acting_seat] += marriage_pts
                marriages[acting_seat].append(mar_suit)

                if scores[acting_seat] >= rules.WIN_SCORE:
                    return (hands, scores, None, acting_seat, phase, closed,
                            closed_by, trump_card, draw_pile, marriages, True, acting_seat)

        if lead_card is None:
            # This is the lead — wait for follower
            return (hands, scores, card, leader, phase, closed,
                    closed_by, trump_card, draw_pile, marriages, False, None)
        else:
            # This is the response — resolve trick
            follower_card = card
            lead_suit = lead_card.suit
            rel_winner = rules.trick_winner(lead_card, follower_card,
                                            lead_suit, trump_suit)
            trick_winner_seat = leader if rel_winner == 0 else (1 - leader)
            trick_pts = lead_card.value() + follower_card.value()
            scores[trick_winner_seat] += trick_pts

            if scores[trick_winner_seat] >= rules.WIN_SCORE:
                return (hands, scores, None, trick_winner_seat, phase, closed,
                        closed_by, trump_card, draw_pile, marriages, True,
                        trick_winner_seat)

            # Draw cards (phase 1 only)
            if phase == 1 and not closed:
                w, l = trick_winner_seat, 1 - trick_winner_seat
                if draw_pile:
                    hands[w].append(draw_pile.pop())
                    if draw_pile:
                        hands[l].append(draw_pile.pop())
                    elif trump_card:
                        hands[l].append(trump_card)
                        trump_card = None
                elif trump_card:
                    hands[w].append(trump_card)
                    trump_card = None

                # Update phase
                if not draw_pile and trump_card is None:
                    phase = 2

            # Check if hands are empty → terminal
            if not hands[0] and not hands[1]:
                winner = self._find_winner(scores)
                terminal = True

            return (hands, scores, None, trick_winner_seat, phase, closed,
                    closed_by, trump_card, draw_pile, marriages, terminal, winner)

    def _is_terminal(self, hands):
        return not hands[0] and not hands[1]

    def _find_winner(self, scores):
        if scores[0] >= rules.WIN_SCORE:
            return 0
        if scores[1] >= rules.WIN_SCORE:
            return 1
        # No one reached 66 — higher score wins
        if scores[0] > scores[1]:
            return 0
        elif scores[1] > scores[0]:
            return 1
        return None

    def _terminal_reward(self, gp_result, root_seat):
        """Convert (winner, game_points) to a value in [-1, 1] for root_seat."""
        winner, gp = gp_result
        if winner is None:
            return 0.0
        sign = 1.0 if winner == root_seat else -1.0
        # Scale: 1 GP → 0.33, 2 GP → 0.67, 3 GP → 1.0
        return sign * (gp / 3.0)

    def _next_to_act(self, lead_card, leader, path, root_seat):
        """Determine which seat acts next."""
        if lead_card is not None:
            # A lead card is on the table → follower responds
            return 1 - leader
        return leader

    def _make_view(self, seat, hands, scores, lead_card, leader,
                   phase, closed, closed_by, trump_suit, trump_card,
                   draw_pile, marriages):
        """Build a synthetic PlayerView for MCTS internal simulation."""
        opp = 1 - seat
        hand = hands[seat]

        valid_actions = rules.get_valid_actions(
            hand, trump_suit, trump_card,
            lead_card, phase,
            is_winner_action=False,
        )

        # Build unknown_cards (not visible to this seat)
        all_cards_set = {Card(r, s) for s in Suit for r in RANKS}
        known = set(hands[seat])
        # In simulation, we know both hands, but the view shouldn't
        # For the policy network, unknown = what that seat can't see
        visible = set(hands[seat])
        if trump_card:
            visible.add(trump_card)
        unknown = sorted(
            all_cards_set - visible,
            key=lambda c: (list(Suit).index(c.suit), RANK_VALUES[c.rank])
        )
        # Remove won cards from unknown (they're visible)
        # In our simplified sim we don't track won cards explicitly,
        # so approximate: unknown = opp hand + draw pile
        opp_hand_set = set(hands[opp])
        pile_set = set(draw_pile)

        return PlayerView(
            seat=seat,
            hand=list(hand),
            trump_suit=trump_suit,
            trump_card=trump_card,
            draw_pile_size=len(draw_pile),
            phase=phase,
            closed=closed,
            closed_by=closed_by,
            my_score=scores[seat],
            opponent_score=scores[opp],
            is_leading=(leader == seat) and (lead_card is None),
            lead_card=lead_card,
            lead_marriage=None,
            valid_actions=valid_actions,
            is_winner_action_phase=False,
            my_won_cards=[],       # not tracked in sim
            opponent_won_cards=[],
            my_marriages=list(marriages.get(seat, [])),
            opponent_marriages=list(marriages.get(opp, [])),
            opponent_known_cards=set(hands[opp]),  # in sim we know
            opponent_void_suits=set(),
            unknown_cards=list(opp_hand_set | pile_set),
            card_threats={},
            opponent_hand_size=len(hands[opp]),
        )


# =========================================================================
# IS-MCTS: aggregate across determinizations
# =========================================================================

def ismcts_action_counts(
    model,
    view: PlayerView,
    n_determinizations: int = 16,
    n_simulations: int = 100,
    c_puct: float = 1.5,
    seed: int | None = None,
) -> dict[int, int]:
    """
    Run IS-MCTS: sample N worlds, run MCTS on each, aggregate visit counts.
    Returns {action_idx: total_visit_count}.
    """
    rng = random.Random(seed)
    mcts = MCTS(model, c_puct=c_puct)
    total_counts: dict[int, int] = {}

    for _ in range(n_determinizations):
        opp_hand = determinize(view, rng)
        counts = mcts.search(view, opp_hand, n_simulations)
        for a, n in counts.items():
            total_counts[a] = total_counts.get(a, 0) + n

    return total_counts


# =========================================================================
# IS-MCTS Player agent
# =========================================================================

class ISMCTSPlayer:
    """
    Player agent that uses Information-Set MCTS with neural network guidance.

    For winner-action phase (swap/close/pass), falls back to the GreedyPlayer
    heuristic since MCTS over those is overkill.

    Implements the full Player interface so it plugs into Match / Round
    without inheriting from Player (avoids circular imports).
    """

    def __init__(self, model, name: str = "ISMCTS",
                 n_determinizations: int = 16,
                 n_simulations: int = 100,
                 c_puct: float = 1.5,
                 greedy_fallback: bool = True):
        from agents import GreedyPlayer
        self.model = model
        self.name = name
        self.n_determinizations = n_determinizations
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self._greedy = GreedyPlayer(name) if greedy_fallback else None

    def choose_action(self, view: PlayerView) -> Action:
        # Winner-action phase → heuristic
        if view.is_winner_action_phase:
            if self._greedy:
                return self._greedy.choose_action(view)
            # Simple fallback
            for a in view.valid_actions:
                if a.type == ActionType.SWAP_TRUMP:
                    return a
            return Action(ActionType.PASS)

        # Only one legal move → play it instantly
        if len(view.valid_actions) == 1:
            return view.valid_actions[0]

        # Run IS-MCTS
        counts = ismcts_action_counts(
            self.model, view,
            n_determinizations=self.n_determinizations,
            n_simulations=self.n_simulations,
            c_puct=self.c_puct,
        )

        if not counts:
            return view.valid_actions[0]

        # Pick action with most visits
        best_action_idx = max(counts, key=counts.get)
        return index_to_action(best_action_idx, view)

    # ------------------------------------------------------------------
    # Player interface stubs (so it works with the game engine)
    # ------------------------------------------------------------------

    def notify_trick_cards(self, view, table_cards, leader=0, marriages=None):
        pass

    def notify_trick(self, result, score_0, score_1, round_winner):
        pass

    def notify_swap(self, old_trump, new_trump):
        pass

    def notify_close(self, closed_by):
        pass

    def notify_round_result(self, result, match_scores):
        pass

    def notify_match_start(self):
        pass

    def notify_match_result(self, result):
        pass

    def notify_next_round(self, hand=None, first_round=False):
        pass

    def set_opponent_hand(self, hand):
        pass
