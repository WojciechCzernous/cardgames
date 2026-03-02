"""Benchmark sampling speed."""
import random, time
from game import Round
from agents import GreedyPlayer
from features import sample_transition, player_view_to_tensor, action_to_index

random.seed(42)

# Pre-record 100 rounds
rounds = []
for _ in range(100):
    rnd = Round({0: GreedyPlayer('A'), 1: GreedyPlayer('B')}, record=True)
    rnd.play()
    rounds.append(rnd)

total_tr = sum(len(r.transitions) for r in rounds)
print(f"Pool: {len(rounds)} rounds, {total_tr} transitions")

# Full pipeline: play + sample
N = 1000
random.seed(0)
t0 = time.perf_counter()
for _ in range(N):
    rnd = Round({0: GreedyPlayer('A'), 1: GreedyPlayer('B')}, record=True)
    rnd.play()
    s, a = sample_transition(rnd.transitions)
t1 = time.perf_counter()
print(f"Full pipeline: {N/(t1-t0):.0f}/sec  ({(t1-t0)/N*1e3:.2f} ms each)")

# Sample only from pre-recorded
N2 = 10000
t0 = time.perf_counter()
for _ in range(N2):
    rnd = random.choice(rounds)
    s, a = sample_transition(rnd.transitions)
t1 = time.perf_counter()
print(f"Sample only:   {N2/(t1-t0):.0f}/sec  ({(t1-t0)/N2*1e6:.1f} µs each)")

# Breakdown
view, action, seat = rounds[0].transitions[0]
N3 = 10000
t0 = time.perf_counter()
for _ in range(N3):
    player_view_to_tensor(view)
t1 = time.perf_counter()
print(f"  tensor:      {(t1-t0)/N3*1e6:.1f} µs")

t0 = time.perf_counter()
for _ in range(N3):
    action_to_index(action, view.hand)
t1 = time.perf_counter()
print(f"  action_idx:  {(t1-t0)/N3*1e6:.1f} µs")
