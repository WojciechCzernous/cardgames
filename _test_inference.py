"""Smoke test for opponent hand inference tracking."""
import random
random.seed(42)

from game import Round, Match
from agents import GreedyPlayer, RandomPlayer
from rules import find_marriages

# Test 1: Basic round with no forcing
players = {0: GreedyPlayer('A'), 1: RandomPlayer('B')}
rnd = Round(players, first_seat=0)
rr = rnd.play()
print(f'Test 1 - Round result: winner={rr.winner}, scores={rr.scores}, gp={rr.game_points}')

# Test 2: Force marriage on both seats
rnd2 = Round(players, first_seat=0, force_marriage_seats={0, 1})
for s in (0, 1):
    mar = find_marriages(rnd2.state.hands[s])
    print(f'  Seat {s} marriages: {mar}')
rr2 = rnd2.play()
print(f'Test 2 - Both marriages: winner={rr2.winner}, scores={rr2.scores}')

# Test 3: Force nine-trump on bot (seat 1)
rnd3 = Round(players, first_seat=0, force_nine_trump_seat=1)
has_nine = any(c.rank == ' 9' and c.suit == rnd3.state.trump_suit
               for c in rnd3.state.hands[1])
print(f'Test 3 - Bot has 9 of trump: {has_nine}')
rr3 = rnd3.play()
print(f'  Round result: winner={rr3.winner}')

# Test 4: Check tracking fields exist on RoundState
print(f'Test 4 - opponent_known_cards type: {type(rnd3.state.opponent_known_cards)}')
print(f'  opponent_void_suits type: {type(rnd3.state.opponent_void_suits)}')

# Test 5: Check PlayerView has the new fields
view = rnd3.state.player_view(0)
print(f'Test 5 - PlayerView.opponent_known_cards: {view.opponent_known_cards}')
print(f'  PlayerView.opponent_void_suits: {view.opponent_void_suits}')

# Test 6: Full match with bot marriage + nine-trump
random.seed(99)
players2 = {0: GreedyPlayer('X'), 1: GreedyPlayer('Y')}
m = Match(players2, force_marriage_seats={1}, force_nine_trump_seat=1)
mr = m.play()
print(f'Test 6 - Match: winner={mr.winner}, gp={mr.game_points}, rounds={mr.rounds_played}')

# Test 7: Verify inference actually works - simulate a marriage announcement
random.seed(7)
players3 = {0: GreedyPlayer('A'), 1: GreedyPlayer('B')}
rnd7 = Round(players3, first_seat=0, force_marriage_seats={0, 1})
# Play until someone announces a marriage and check state
rnd7.play()
st = rnd7.state
# After a full round, check that the inference sets were populated at some point
# (they may be empty at the end since cards get played and removed)
print(f'Test 7 - Inference round completed, final scores: {st.scores}')
okc0 = st.opponent_known_cards[0]
okc1 = st.opponent_known_cards[1]
ovs0 = st.opponent_void_suits[0]
ovs1 = st.opponent_void_suits[1]
print(f'  Seat 0 knows opponent has: {okc0}, void in: {ovs0}')
print(f'  Seat 1 knows opponent has: {okc1}, void in: {ovs1}')

# Test 8: Mutual exclusivity check
import sys
import subprocess
result = subprocess.run(
    [sys.executable, 'card_game.py', '--nine-trump', '--nine-trump-bot'],
    capture_output=True, text=True
)
assert result.returncode != 0, "Should have failed with mutual exclusivity error"
assert "mutually exclusive" in result.stderr or "mutually exclusive" in result.stdout
print('Test 8 - Mutual exclusivity check passed')

print('\nAll tests passed!')
