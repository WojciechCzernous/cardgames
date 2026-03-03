"""
app.py — Streamlit web UI for Sixty-Six
Human (seat 0) plays against AI (seat 1).

Run locally:
    streamlit run app.py

Share on a local network:
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
    Then open http://<your-ip>:8501 in any browser on the same network.
    Find your IP with: ifconfig | grep "inet "
"""

import streamlit as st

from game import RoundState
from models import Card, Suit, Action, ActionType, RANK_VALUES
from agents import GreedyPlayer, SmartPlayer
import rules
from rules import WIN_SCORE, compute_game_points, find_marriages, marriage_value

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
HUMAN     = 0
AI        = 1
MATCH_WIN = 7

RANK_DISP = {" 9": "9", "10": "10", " J": "J", " Q": "Q", " K": "K", " A": "A"}
FACE_DOWN = "🂠"

# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

def clabel(c: Card) -> str:
    return f"{RANK_DISP[c.rank]}{c.suit.value}"

def is_red(c: Card) -> bool:
    return c.suit in (Suit.HEARTS, Suit.DIAMONDS)

def card_html(c: Card, size: str = "2em", extra_style: str = "") -> str:
    color = "#cc0000" if is_red(c) else "#111"
    return (f'<span style="font-size:{size}; color:{color}; font-weight:bold;'
            f' {extra_style}">{clabel(c)}</span>')

def karta(n: int) -> str:
    """Polish plural for 'karta'."""
    if n == 1: return "karta"
    if 2 <= n <= 4: return "karty"
    return "kart"


def pkt_meczu(n: int) -> str:
    """Polish plural for 'punkt meczu'."""
    if n == 1: return "punkt meczu"
    if 2 <= n <= 4: return "punkty meczu"
    return "punktów meczu"


def card_btn_label(c: Card) -> str:
    pts = c.value()
    if pts > 0:
        return f"{RANK_DISP[c.rank]}{c.suit.value}\n({pts} pkt)"
    return f"{RANK_DISP[c.rank]}{c.suit.value}"

# ──────────────────────────────────────────────────────────────────────────────
# Game-logic helpers (inlined from game.py to allow step-by-step control)
# ──────────────────────────────────────────────────────────────────────────────

def exec_action(state: RoundState, seat: int, action: Action,
                is_leading: bool = False):
    """Apply action to state. Returns (card | None, marriage_pts, marriage_suit | None)."""
    hand = state.hands[seat]
    opp  = 1 - seat
    at   = action.type.value

    if at == "swap_trump":
        nine = rules.find_nine_trump(hand, state.trump_suit)
        if nine and state.trump_card:
            old = state.trump_card
            hand.remove(nine)
            hand.append(old)
            state.trump_card = nine
            state.opponent_known_cards[opp].add(old)
            return nine, 0, None          # nine was swapped out
        return None, 0, None

    if at == "close_game":
        state.closed    = True
        state.closed_by = seat
        return None, 0, None

    if at == "pass":
        return None, 0, None

    if at == "play_card":
        card = hand[action.card_index]
        hand.remove(card)
        state.last_drawn[seat] = None
        state.opponent_known_cards[opp].discard(card)

        # Auto-detect marriage when leading with K or Q that has its partner
        mar_suit = action.marriage_suit
        if mar_suit is None and is_leading and card.rank in (" K", " Q"):
            partner = " Q" if card.rank == " K" else " K"
            if any(c.rank == partner and c.suit == card.suit for c in hand):
                mar_suit = card.suit

        mar_pts = 0
        if mar_suit:
            mar_pts = rules.marriage_value(mar_suit, state.trump_suit)
            state.scores[seat] += mar_pts
            state.marriages[seat].append(mar_suit)
            partner = " Q" if card.rank == " K" else " K"
            state.opponent_known_cards[opp].add(Card(partner, mar_suit))

        return card, mar_pts, mar_suit

    return None, 0, None


def do_draw(state: RoundState):
    """Draw cards for both players after a trick (phase 1 only)."""
    state.last_drawn = {0: None, 1: None}
    if state.closed:
        return
    leader, follower = state.leader, 1 - state.leader
    if state.draw_pile:
        d = state.draw_pile.pop()
        state.hands[leader].append(d)
        state.last_drawn[leader] = d
        if state.draw_pile:
            state.hands[follower].append(state.draw_pile.pop())
        elif state.trump_card:
            state.hands[follower].append(state.trump_card)
            state.trump_card = None
    elif state.trump_card:
        state.hands[leader].append(state.trump_card)
        state.last_drawn[leader] = state.trump_card
        state.trump_card = None


def do_resolve(state: RoundState, leader_s: int, card_l: Card, card_f: Card):
    """Resolve the trick. Returns (winner_seat, trick_pts)."""
    follower_s = 1 - leader_s
    rel    = rules.trick_winner(card_l, card_f, card_l.suit, state.trump_suit)
    winner = leader_s if rel == 0 else follower_s
    pts    = card_l.value() + card_f.value()

    state.scores[winner] += pts
    state.leader = winner
    state.won_cards[winner].extend([card_l, card_f])

    # Phase 2 void tracking
    if state.phase == 2 and card_f.suit != card_l.suit:
        state.opponent_void_suits[leader_s].add(card_l.suit)
        if card_f.suit != state.trump_suit:
            state.opponent_void_suits[leader_s].add(state.trump_suit)

    for s in (0, 1):
        if state.scores[s] >= WIN_SCORE:
            state.round_winner = s

    return winner, pts


def get_ai_action(state: RoundState, seat: int, lead_card=None,
                  is_wa: bool = False, match_scores: dict | None = None) -> Action:
    agent = GreedyPlayer("AI")
    view  = state.player_view(seat, lead_card=lead_card,
                              is_winner_action=is_wa,
                              match_scores=match_scores or {})
    return agent.choose_action(view)


# ──────────────────────────────────────────────────────────────────────────────
# Session-state management
# ──────────────────────────────────────────────────────────────────────────────

def reset_match():
    s = st.session_state
    s.match_scores = {0: 0, 1: 0}
    s.first_seat   = 0
    s.game_log     = []
    reset_round()


def reset_round():
    s = st.session_state
    s.rs            = RoundState(first_seat=s.first_seat)
    s.lead_card     = None      # card played by trick leader
    s.lead_marriage = None      # marriage announced with lead card
    s.trick_leader  = s.rs.leader
    s.trick_info    = None      # filled after both cards played
    s.ai_msg        = ""        # one-line AI status message
    s.drawn_msg     = ""        # message about cards just drawn
    s.stage         = 'pre_turn'


def advance(state: RoundState, ms: dict):
    """
    From 'pre_turn': decide the next stage.
    - If round is over → 'round_over'
    - If human leads   → 'human_lead'
    - If AI leads      → process AI lead, then → 'human_follow'
    """
    s = st.session_state

    # Round may have ended via trick resolution
    all_empty = not state.hands[0] and not state.hands[1]
    if state.round_winner is not None or all_empty:
        winner, gpts = compute_game_points(
            state.scores, state.round_winner, state.closed, state.closed_by)
        s.round_result = {
            'winner': winner, 'gpts': gpts, 'scores': dict(state.scores)}
        s.stage = 'round_over'
        return

    s.trick_leader = state.leader

    if state.leader == HUMAN:
        s.stage = 'human_lead'
    else:
        # AI leads
        act = get_ai_action(state, AI, match_scores=ms)
        card, mar_pts, mar_suit = exec_action(state, AI, act, is_leading=True)
        s.lead_card     = card
        s.lead_marriage = mar_suit
        if mar_pts:
            s.game_log.append(f"AI ogłosiło meldunek {mar_suit.value} +{mar_pts} pkt")
            s.ai_msg = f"AI zagrało {clabel(card)} i ogłosiło meldunek {mar_suit.value} (+{mar_pts} pkt)!"
        else:
            s.ai_msg = ""
        s.stage = 'human_follow'


def after_trick_continue(state: RoundState, ms: dict):
    """
    Called when user clicks 'Next Trick'.
    Draws cards (phase 1), handles winner actions, advances to next stage.
    """
    s = st.session_state
    s.drawn_msg = ""

    # Draw (phase 1 only)
    if state.phase == 1 and not state.closed:
        do_draw(state)
        # Report what human drew
        if state.last_drawn.get(HUMAN):
            s.drawn_msg = f"Dobrałeś: {clabel(state.last_drawn[HUMAN])}"

    trick_winner = s.trick_info['winner']

    # Winner actions (phase 1 only)
    if state.phase == 1 and not state.closed:
        if trick_winner == HUMAN:
            # Check if there are any special actions available
            view = state.player_view(HUMAN, is_winner_action=True, match_scores=ms)
            has_swap  = any(a.type.value == "swap_trump" for a in view.valid_actions)
            can_close = True  # always available if phase 1 and not closed
            if has_swap or can_close:
                s.stage = 'human_winner_action'
                return
        else:
            # AI winner actions
            act = get_ai_action(state, AI, is_wa=True, match_scores=ms)
            at  = act.type.value
            if at == "swap_trump":
                old = state.trump_card
                exec_action(state, AI, act)
                s.ai_msg = f"AI wymieniło 9 za {clabel(old)}."
                s.game_log.append(f"AI wymieniło 9 za {clabel(old)}")
            elif at == "close_game":
                exec_action(state, AI, act)
                s.ai_msg = "AI zamknęło grę!"
                s.game_log.append("AI zamknęło grę!")
            else:
                s.ai_msg = ""

    s.lead_card = None
    s.lead_marriage = None
    s.stage = 'pre_turn'


# ──────────────────────────────────────────────────────────────────────────────
# Page setup & state init
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sixty-Six", page_icon="🃏", layout="centered")

if 'stage' not in st.session_state:
    st.session_state.stage         = 'welcome'
    st.session_state.match_scores  = {0: 0, 1: 0}
    st.session_state.game_log      = []
    st.session_state.first_seat    = 0

s = st.session_state

# Auto-advance pre_turn (no user input needed)
if s.stage == 'pre_turn':
    advance(s.rs, s.match_scores)
    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Welcome screen
# ──────────────────────────────────────────────────────────────────────────────

if s.stage == 'welcome':
    st.title("🃏 Sześćdziesiąt Sześć")
    st.markdown("""
Witaj! **Sześćdziesiąt Sześć** to jedna ze starszych gier karcianych. 
Do Polski gra ta dotrała pod koniec XVII wieku, we francuskiej formie i pod nazwą mariasza; jej warianty znane były także jako gaigel i sznaps. 
(patrz: Lech Pijanowski, "Przewodnik gier", wyd. Iskry, 1973, Warszawa; str. 281-283).

Grasz przeciwko komputerowi.

### Zasady
| Karta | Punkty |
|-------|--------|
| As    | 11     |
| Dziesiątka | 10 |
| Król  | 4      |
| Dama  | 3      |
| Walet | 2      |
| Dziewiątka | 0 |

- **Atut** bije wszystkie inne kolory; wyższa figura wygrywa w tym samym kolorze.
- **Meldunek** (Król + Dama w tym samym kolorze): 20 pkt (meldunek atutowy: 40 pkt).
  Ogłoś meldunek, wychodząc Królem lub Damą, gdy trzymasz oboje.
- **Faza 1** (stos aktywny): zagraj dowolną kartę; obaj gracze dobierają kartę po każdej lewie.
  Zwycięzca lewy może wymienić dziewiątkę atutową za odkrytą kartę atu lub zamknąć grę.
- **Faza 2** (stos wyczerpany lub gra zamknięta): obowiązek koloru → obowiązek atu → dowolna.
- Pierwszy do **66 punktów** wygrywa rundę. Zwycięzca rundy zdobywa 1–3 **punkty meczu**.
- Pierwszy do **7 punktów meczu** wygrywa mecz!
""")
    if st.button("▶ Rozpocznij grę", type="primary"):
        reset_match()
        st.rerun()
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Active game — header always visible
# ──────────────────────────────────────────────────────────────────────────────

rs = s.rs

# Score row
c1, c2, c3 = st.columns([2, 1, 2])
with c1:
    st.metric("🧑 Ty  (pkt meczu)", s.match_scores[HUMAN],
              delta=f"wynik rundy: {rs.scores[HUMAN]}")
with c2:
    st.markdown("<div style='text-align:center; padding-top:1.8em;'>vs</div>",
                unsafe_allow_html=True)
with c3:
    st.metric("🤖 AI  (pkt meczu)", s.match_scores[AI],
              delta=f"wynik rundy: {rs.scores[AI]}")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Round-over / match-over screens
# ──────────────────────────────────────────────────────────────────────────────

if s.stage == 'round_over':
    rr = s.round_result
    gpts = rr['gpts']
    if rr['winner'] == HUMAN:
        round_summary = "Ty wygrałeś rundę!"
    elif rr['winner'] == AI:
        round_summary = "AI wygrało rundę!"
    else:
        round_summary = "Remis!"
    st.subheader(f"Koniec rundy — {round_summary}")
    st.write(f"Wyniki lew: **Ty {rr['scores'][HUMAN]}** — **AI {rr['scores'][AI]}**")
    if gpts:
        earner = "Ty zdobywasz" if rr['winner'] == HUMAN else "AI zdobywa"
        st.write(f"{earner} **{gpts} {pkt_meczu(gpts)}**.")

    if st.button("▶ Następna runda", type="primary"):
        if rr['winner'] is not None:
            s.match_scores[rr['winner']] += gpts
        who_log = "Ty" if rr['winner'] == HUMAN else ("AI" if rr['winner'] == AI else "Remis")
        s.game_log.append(
            f"Runda: {who_log} +{gpts} pkt  (Ty {rr['scores'][HUMAN]}–{rr['scores'][AI]} AI)")
        if s.match_scores[HUMAN] >= MATCH_WIN or s.match_scores[AI] >= MATCH_WIN:
            s.stage = 'match_over'
        else:
            s.first_seat = 1 - s.first_seat   # alternate first player
            reset_round()
        st.rerun()
    st.stop()


if s.stage == 'match_over':
    winner = 0 if s.match_scores[HUMAN] >= MATCH_WIN else 1
    st.balloons()
    if winner == HUMAN:
        st.subheader("🏆 Koniec meczu — Ty wygrałeś!")
    else:
        st.subheader("🏆 Koniec meczu — AI wygrało!")
    st.write(f"Wynik końcowy: **Ty {s.match_scores[HUMAN]}** — **AI {s.match_scores[AI]}** pkt meczu")

    with st.expander("Dziennik gry"):
        for line in s.game_log:
            st.write(line)

    if st.button("🔄 Nowy mecz", type="primary"):
        reset_match()
        st.rerun()
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Trump card & game info
# ──────────────────────────────────────────────────────────────────────────────

info_col, trump_col = st.columns([3, 1])
with info_col:
    phase_label = "Faza 2 — obowiązek koloru" if rs.phase == 2 else "Faza 1 — otwarta"
    closed_tag  = " · **ZAMKNIĘTA** 🔒" if rs.closed else ""
    st.markdown(f"**{phase_label}**{closed_tag} &nbsp;·&nbsp; Stos: **{len(rs.draw_pile)}**")
    if rs.last_trick_info:
        st.caption(f"Ostatnia lewa: {rs.last_trick_info}")
with trump_col:
    if rs.trump_card:
        st.markdown(
            f"**Karta atu:**<br>{card_html(rs.trump_card, size='1.8em')}",
            unsafe_allow_html=True)
    else:
        st.markdown(f"**Kolor atu:** {rs.trump_suit.value}")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# AI hand (face down)
# ──────────────────────────────────────────────────────────────────────────────

ai_cards = len(rs.hands[AI])
st.markdown(f"**🤖 Ręka AI** ({ai_cards} {karta(ai_cards)}): "
            + "  ".join([FACE_DOWN] * ai_cards))

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Table (trick area)
# ──────────────────────────────────────────────────────────────────────────────

# ── trick_shown ────────────────────────────────────────────────────────────────
if s.stage == 'trick_shown':
    ti   = s.trick_info
    ldr  = s.trick_leader       # who led
    flwr = 1 - ldr              # who followed

    leader_card   = ti['cards'][ldr]
    follower_card = ti['cards'][flwr]
    you_card      = ti['cards'][HUMAN]
    ai_card       = ti['cards'][AI]

    won_str = "Ty wygrałeś!" if ti['winner'] == HUMAN else "AI wygrało!"

    tl, tc_col, tr = st.columns([2, 1, 2])
    with tl:
        st.markdown(card_html(you_card, "3em"), unsafe_allow_html=True)
        st.caption("Ty")
    with tc_col:
        st.markdown(
            f"<div style='text-align:center; padding-top:.6em;'>"
            f"<b>{won_str}<br>+{ti['pts']} pkt</b></div>",
            unsafe_allow_html=True)
        for seat, mar_pts in ti['marriages'].items():
            if mar_pts:
                who_mar = "Ty" if seat == HUMAN else "AI"
                st.caption(f"💍 {who_mar} +{mar_pts}")
    with tr:
        st.markdown(card_html(ai_card, "3em"), unsafe_allow_html=True)
        st.caption("AI")

    if s.get('ai_msg'):
        st.info(s.ai_msg)

    if st.button("▶ Następna lewa", type="primary"):
        after_trick_continue(rs, s.match_scores)
        st.rerun()
    st.stop()

# ── human_winner_action ────────────────────────────────────────────────────────
if s.stage == 'human_winner_action':
    st.success("🏆 Wygrałeś lewę!")
    if s.get('drawn_msg'):
        st.info(s.drawn_msg)

    view      = rs.player_view(HUMAN, is_winner_action=True,
                               match_scores=s.match_scores)
    has_swap  = any(a.type.value == "swap_trump" for a in view.valid_actions)

    st.markdown("**Opcjonalne akcje przed wyjściem:**")
    wa1, wa2, wa3 = st.columns(3)

    acted = False
    with wa1:
        if has_swap and st.button("🔄 Wymień 9 za atu", use_container_width=True):
            old = rs.trump_card
            exec_action(rs, HUMAN, Action(ActionType.SWAP_TRUMP))
            s.game_log.append(f"Wymieniłeś 9 → {clabel(old)}")
            acted = True
    with wa2:
        if not acted and st.button("🔒 Zamknij grę", use_container_width=True):
            exec_action(rs, HUMAN, Action(ActionType.CLOSE_GAME))
            s.game_log.append("Zamknąłeś grę!")
            acted = True
    with wa3:
        if not acted and st.button("⏭ Pas", use_container_width=True,
                                   type="primary"):
            acted = True

    if acted:
        s.lead_card     = None
        s.lead_marriage = None
        s.stage = 'pre_turn'
        st.rerun()
    st.stop()

# ── Show lead card if human is following ──────────────────────────────────────
if s.stage == 'human_follow' and s.lead_card is not None:
    st.markdown("**AI zagrało:**")
    st.markdown(card_html(s.lead_card, size="3em"), unsafe_allow_html=True)
    if s.lead_marriage:
        mar_pts = marriage_value(s.lead_marriage, rs.trump_suit)
        st.caption(f"💍 AI ogłasza meldunek {s.lead_marriage.value} +{mar_pts} pkt!")
    st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Human hand — card buttons
# ──────────────────────────────────────────────────────────────────────────────

if s.stage in ('human_lead', 'human_follow'):
    hand     = rs.hands[HUMAN]
    is_lead  = (s.stage == 'human_lead')
    lc       = s.lead_card if not is_lead else None

    view       = rs.player_view(HUMAN, lead_card=lc, match_scores=s.match_scores)
    valid_idxs = {a.card_index for a in view.valid_actions
                  if a.type.value == 'play_card'}

    # Marriage hints
    if is_lead:
        marriages = find_marriages(hand)
        if marriages:
            hints = []
            for ms in marriages:
                pts = marriage_value(ms, rs.trump_suit)
                hints.append(f"{ms.value} +{pts} pkt (zagraj K lub D, żeby ogłosić)")
            st.caption("💍 Dostępne meldunki: " + " | ".join(hints))

    if s.get('drawn_msg'):
        st.info(s.drawn_msg)

    action_text = ("**🧑 Twoja ręka** — kliknij kartę, żeby wyjść:"
                   if is_lead
                   else "**🧑 Twoja ręka** — kliknij kartę, żeby odpowiedzieć:")
    st.markdown(action_text)

    cols = st.columns(len(hand))
    played_action = None

    for i, card in enumerate(hand):
        with cols[i]:
            valid   = (i in valid_idxs)
            label   = card_btn_label(card)

            # Colour styling via markdown — button text is plain
            color = "#cc0000" if is_red(card) else "#111"
            st.markdown(
                f'<div style="text-align:center; font-size:1.4em; '
                f'color:{color}; {"" if valid else "opacity:0.35;"}"> '
                f'{clabel(card)}</div>',
                unsafe_allow_html=True)

            if st.button(
                label,
                key=f"hand_{i}_{clabel(card)}",
                disabled=not valid,
                use_container_width=True,
            ):
                played_action = (i, card, is_lead)

    if played_action is not None:
        idx, card_played, leading = played_action
        act = Action(ActionType.PLAY_CARD, card_index=idx)
        human_card, h_mar_pts, h_mar_suit = exec_action(
            rs, HUMAN, act, is_leading=leading)

        if h_mar_pts:
            s.game_log.append(
                f"Ogłosiłeś meldunek {h_mar_suit.value} +{h_mar_pts} pkt")

        if leading:
            # Human led — now AI follows
            s.lead_card     = human_card
            s.lead_marriage = h_mar_suit

            ai_act  = get_ai_action(rs, AI, lead_card=human_card,
                                    match_scores=s.match_scores)
            ai_card, ai_mar_pts, ai_mar_suit = exec_action(rs, AI, ai_act,
                                                            is_leading=False)
            if ai_mar_pts:
                s.game_log.append(
                    f"AI ogłosiło meldunek {ai_mar_suit.value} +{ai_mar_pts} pkt")

            winner, pts = do_resolve(rs, HUMAN, human_card, ai_card)
            s.trick_info = {
                'cards':     {HUMAN: human_card, AI: ai_card},
                'winner':    winner,
                'pts':       pts,
                'marriages': {HUMAN: h_mar_pts, AI: ai_mar_pts},
            }
        else:
            # Human followed — resolve (AI led)
            leader_card   = s.lead_card
            follow_card   = human_card
            winner, pts   = do_resolve(rs, AI, leader_card, follow_card)
            s.trick_info  = {
                'cards':     {AI: leader_card, HUMAN: follow_card},
                'winner':    winner,
                'pts':       pts,
                'marriages': {HUMAN: h_mar_pts, AI: 0},
            }

        s.stage = 'trick_shown'
        st.rerun()
