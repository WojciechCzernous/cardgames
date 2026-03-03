"""
app.py — Streamlit web UI for Sixty-Six
Human (seat 0) plays against AI (seat 1).

Run locally:
    streamlit run app.py

Share on a local network:
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.serverAddress localhost
    Then open http://<your-ip>:8501 in any browser on the same network.
    Find your IP with: ifconfig | grep "inet "
"""

import time

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
    """Polish plural for 'duży punkt'."""
    if n == 1: return "duży punkt"
    if 2 <= n <= 4: return "duże punkty"
    return "dużych punktów"


def card_btn_label(c: Card) -> str:
    return f"{RANK_DISP[c.rank]}{c.suit.value}"


def render_table_cards(cards: dict, result_text: str = "", prefix: str = "tbl"):
    """Show cards on the table as disabled buttons with result text."""
    # CSS to color red-suit table buttons
    css = ""
    col_idx = 0
    entries = []  # list of (label, col_position)
    for who, card in cards.items():
        col_idx += 1
        label = f"{who}: {card_btn_label(card)}"
        entries.append((label, card, col_idx))

    n_cols = len(entries) + (1 if result_text else 0)
    cols = st.columns(n_cols)
    css_rules = []
    for i, (label, card, _) in enumerate(entries):
        with cols[i]:
            st.button(label, key=f"{prefix}_{i}_{clabel(card)}",
                      disabled=True, use_container_width=True)
        if is_red(card):
            # This is approximate — targets the nth column's disabled button
            css_rules.append(
                f'div[data-testid="stColumns"]:has(button[data-testid="stBaseButton-secondary"][disabled]) '
                f'> div[data-testid="stColumn"]:nth-of-type({i+1}) button p {{ color: #cc0000 !important; }}')
    if result_text:
        with cols[-1]:
            st.markdown(result_text)
    if css_rules:
        st.markdown(f"<style>{''.join(css_rules)}</style>", unsafe_allow_html=True)

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
            # Instant win if marriage pushes score to 66+
            if state.scores[seat] >= WIN_SCORE:
                state.round_winner = seat

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
            d2 = state.draw_pile.pop()
            state.hands[follower].append(d2)
            state.last_drawn[follower] = d2
        elif state.trump_card:
            state.hands[follower].append(state.trump_card)
            state.last_drawn[follower] = state.trump_card
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
        # Marriage may have won the round instantly
        if state.round_winner is not None:
            winner, gpts = compute_game_points(
                state.scores, state.round_winner, state.closed, state.closed_by)
            s.round_result = {
                'winner': winner, 'gpts': gpts, 'scores': dict(state.scores)}
            s.stage = 'round_over'
            return
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
                s.lead_card = None
                s.lead_marriage = None
                s.stage = 'ai_closed'
                return
            else:
                s.ai_msg = ""

    s.lead_card = None
    s.lead_marriage = None
    s.stage = 'pre_turn'


# ──────────────────────────────────────────────────────────────────────────────
# Page setup & state init
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sześćdziesiąt Sześć", page_icon="🃏", layout="centered")

# Reduce default Streamlit padding for a more compact layout
st.markdown("""
<style>
    .block-container { padding-top: 2.5rem; padding-bottom: 0rem; }
    [data-testid="stMetric"] { padding: 0; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    [data-testid="stMetricDelta"] { font-size: 0.85rem; }
    .stDivider { margin: 0.3rem 0; }
    /* Larger card button text */
    div[data-testid="stColumn"] button p { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)

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
    if st.button("▶ Rozpocznij grę", type="primary"):
        reset_match()
        st.rerun()
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
- Pierwszy zdobywca **66 punktów** wygrywa rundę. Zwycięzca rundy zdobywa 1–3 **duże punkty**.
- Pierwszy zdobywca **7 dużych punktów** wygrywa mecz!
""")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Active game — header always visible
# ──────────────────────────────────────────────────────────────────────────────

rs = s.rs

# Score + game info — single compact row
c1, c2 = st.columns([3, 5])
with c1:
    st.markdown(
        f'<span style="font-size:1.6em; font-weight:bold;">{rs.scores[HUMAN]} – {rs.scores[AI]}</span>'
        f'&nbsp; <span style="font-size:0.9em; color:#666;">'
        f'(duże punkty: {s.match_scores[HUMAN]} – {s.match_scores[AI]})</span>',
        unsafe_allow_html=True)
with c2:
    n_pile = len(rs.draw_pile)
    trump_color = "#cc0000" if rs.trump_suit in (Suit.HEARTS, Suit.DIAMONDS) else "#111"
    trump_html = card_html(rs.trump_card, size='1.4em') if rs.trump_card else f'<span style="font-size:1.4em; color:{trump_color}; font-weight:bold;">{rs.trump_suit.value}</span>'
    if rs.closed or rs.phase == 2:
        closed_tag = " 🔒" if rs.closed else ""
        suit_html = f'<span style="font-size:1.4em; color:{trump_color}; font-weight:bold;">{rs.trump_suit.value}</span>'
        st.markdown(f"Talon zamknięty{closed_tag} — {n_pile} {karta(n_pile)}. Atu: {suit_html}", unsafe_allow_html=True)
    else:
        st.markdown(f"Talon otwarty — {n_pile} {karta(n_pile)} + atu: {trump_html}", unsafe_allow_html=True)

if rs.last_trick_info:
    st.caption(rs.last_trick_info)

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
    st.write(f"Wynik końcowy: **Ty {s.match_scores[HUMAN]}** — **AI {s.match_scores[AI]}** duże punkty")

    with st.expander("Dziennik gry"):
        for line in s.game_log:
            st.write(line)

    if st.button("🔄 Nowy mecz", type="primary"):
        reset_match()
        st.rerun()
    st.stop()



# ── ai_closed — flash AI-closed message for 1 second ─────────────────────────
if s.stage == 'ai_closed':
    st.markdown(
        '<div style="text-align:center; padding:28px 8px;">'
        '<div style="font-size:2.4em; font-weight:bold;">🔒</div>'
        '<div style="font-size:1.4em; font-weight:bold; margin-top:8px;">'
        'AI zamknęło talon!</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    time.sleep(1)
    s.stage = 'pre_turn'
    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Table (trick area)
# ──────────────────────────────────────────────────────────────────────────────

# ── trick_cards — show both cards for 3 seconds with winner highlighted ─────────
if s.stage == 'trick_cards':
    ti       = s.trick_info
    you_card = ti['cards'][HUMAN]
    ai_card  = ti['cards'][AI]
    winner   = ti['winner']

    def _card_box(card: Card, is_winner: bool, label: str) -> str:
        color  = "#cc0000" if is_red(card) else "#111"
        bg     = "#fffde7" if is_winner else "#f0f0f0"
        border = "2px solid #f9a825" if is_winner else "1px solid #ccc"
        badge  = "&nbsp;★" if is_winner else ""
        opacity = "1.0" if is_winner else "0.6"
        return (
            f'<div style="text-align:center; padding:12px; background:{bg}; '
            f'border-radius:10px; border:{border}; opacity:{opacity};">'
            f'<div style="font-size:0.85em; color:#666; margin-bottom:6px;">{label}{badge}</div>'
            f'<div style="font-size:3em; color:{color}; font-weight:bold;">{clabel(card)}</div>'
            f'</div>'
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(_card_box(you_card, winner == HUMAN, "Ty"), unsafe_allow_html=True)
    with col2:
        st.markdown(_card_box(ai_card, winner == AI, "AI"), unsafe_allow_html=True)

    time.sleep(3)
    s.stage = 'trick_points'
    st.rerun()

# ── trick_points — show trick result for 2 seconds ───────────────────────────
if s.stage == 'trick_points':
    ti     = s.trick_info
    winner = ti['winner']
    pts    = ti['pts']

    mar_parts = []
    for seat, mar_pts in ti['marriages'].items():
        if mar_pts:
            who = "Ty" if seat == HUMAN else "AI"
            mar_parts.append(f"💍 {who} +{mar_pts}")
    mar_line = ("<br><span style='font-size:1em;'>" + " &nbsp;·&nbsp; ".join(mar_parts) + "</span>") if mar_parts else ""

    won_label = "Zabierasz lewę!" if winner == HUMAN else "AI zabiera lewę."
    pts_color = "#2e7d32" if winner == HUMAN else "#c62828"
    st.markdown(
        f'<div style="text-align:center; padding:24px 8px;">'
        f'<div style="font-size:1.2em; font-weight:bold; margin-bottom:8px;">{won_label}</div>'
        f'<div style="font-size:2.8em; font-weight:bold; color:{pts_color};">+{pts} pkt{mar_line}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    time.sleep(2)
    after_trick_continue(rs, s.match_scores)
    st.rerun()

# ── human_winner_action ── show table cards + action buttons + hand ─────────────
if s.stage == 'human_winner_action':
    # Show trick summary as table cards
    if s.get('trick_info'):
        ti = s.trick_info
        you_card = ti['cards'][HUMAN]
        ai_card  = ti['cards'][AI]
        won_str = "Wygrana!" if ti['winner'] == HUMAN else "Przegrana"
        render_table_cards({"Ty": you_card, "AI": ai_card},
                           result_text=f"**{won_str}** +{ti['pts']} pkt",
                           prefix="wa_tbl")

    if s.get('drawn_msg'):
        st.caption(s.drawn_msg)

    view      = rs.player_view(HUMAN, is_winner_action=True,
                               match_scores=s.match_scores)
    has_swap  = any(a.type.value == "swap_trump" for a in view.valid_actions)

    # Action buttons in a compact row
    acols = []
    if has_swap:
        acols.append('swap')
    acols.append('close')
    acols.append('pass')
    cols_wa = st.columns(len(acols))
    acted = False
    for ci, atype in enumerate(acols):
        with cols_wa[ci]:
            if atype == 'swap' and st.button("🔄 Wymień 9↔atu", use_container_width=True):
                old = rs.trump_card
                exec_action(rs, HUMAN, Action(ActionType.SWAP_TRUMP))
                s.game_log.append(f"Wymieniłeś 9 → {clabel(old)}")
                # Stay on this screen so player can still close or pass
                st.rerun()
            elif atype == 'close' and not acted and st.button("🔒 Zamknij talon", use_container_width=True):
                exec_action(rs, HUMAN, Action(ActionType.CLOSE_GAME))
                s.game_log.append("Zamknąłeś grę!")
                acted = True
            elif atype == 'pass' and not acted and st.button("▶ Graj dalej", use_container_width=True, type="primary"):
                acted = True

    if acted:
        s.lead_card     = None
        s.lead_marriage = None
        s.stage = 'pre_turn'
        st.rerun()

    # Show the hand (read-only)
    hand = rs.hands[HUMAN]
    hand_cols = st.columns(len(hand))
    for i, card in enumerate(hand):
        with hand_cols[i]:
            st.button(card_btn_label(card), key=f"wah_{i}_{clabel(card)}",
                      disabled=True, use_container_width=True)
    st.stop()

# ── Show lead card if human is following ──────────────────────────────────────
if s.stage == 'human_follow' and s.lead_card is not None:
    card = s.lead_card
    color = "#cc0000" if is_red(card) else "#111"
    mar_extra = ""
    if s.lead_marriage:
        mar_pts = marriage_value(s.lead_marriage, rs.trump_suit)
        mar_extra = f'<div style="font-size:0.85em; margin-top:4px;">💍+{mar_pts} pkt</div>'
    st.markdown(
        f'<div style="text-align:center; padding:12px; background:#f0f0f0; '
        f'border-radius:10px; border:1px solid #ccc;">'
        f'<div style="font-size:0.85em; color:#666; margin-bottom:6px;">AI</div>'
        f'<div style="font-size:3em; color:{color}; font-weight:bold;">{clabel(card)}</div>'
        f'{mar_extra}</div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Human hand — card buttons
# ──────────────────────────────────────────────────────────────────────────────

if s.stage in ('human_lead', 'human_follow'):
    hand     = rs.hands[HUMAN]
    is_lead  = (s.stage == 'human_lead')
    lc       = s.lead_card if not is_lead else None

    if s.get('ai_msg'):
        st.caption(s.ai_msg)
    if s.get('drawn_msg'):
        st.caption(s.drawn_msg)

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

    action_text = ("**🧑 Twoje karty** — kliknij kartę, żeby wyjść:"
                   if is_lead
                   else "**🧑 Twoje karty** — kliknij kartę, żeby odpowiedzieć:")
    st.markdown(action_text)

    # Inject per-button CSS to color red suits and highlight new card
    # Only style valid (enabled) cards; disabled ones stay default grey
    new_card = rs.last_drawn.get(HUMAN)
    css_rules = []
    for i, card in enumerate(hand):
        is_valid = (i in valid_idxs)
        if is_valid and is_red(card):
            css_rules.append(
                f'div[data-testid="stColumn"]:nth-of-type({i+1}) button p {{\n'
                f'  color: #cc0000 !important;\n'
                f'}}'
            )
        if is_valid and new_card and card == new_card:
            css_rules.append(
                f'div[data-testid="stColumn"]:nth-of-type({i+1}) button {{\n'
                f'  background-color: #e3f2fd !important;\n'
                f'  border: 2px solid #42a5f5 !important;\n'
                f'}}'
            )
    if css_rules:
        st.markdown(f"<style>{''.join(css_rules)}</style>", unsafe_allow_html=True)

    cols = st.columns(len(hand))
    played_action = None

    for i, card in enumerate(hand):
        with cols[i]:
            valid   = (i in valid_idxs)
            label   = card_btn_label(card)

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

        # Marriage may have won the round instantly
        if rs.round_winner is not None:
            winner, gpts = compute_game_points(
                rs.scores, rs.round_winner, rs.closed, rs.closed_by)
            s.round_result = {
                'winner': winner, 'gpts': gpts, 'scores': dict(rs.scores)}
            s.trick_info = {
                'cards': {HUMAN: human_card, AI: None},
                'winner': rs.round_winner,
                'pts': 0,
                'marriages': {HUMAN: h_mar_pts, AI: 0},
            }
            s.stage = 'round_over'
            st.rerun()

        if leading:
            # Human led — show card on table, AI responds, resolve
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

            # Build table summary: your card shown first ("on the table")
            mar_info = ""
            if h_mar_pts:
                mar_info += f" · 💍Ty+{h_mar_pts}"
            if ai_mar_pts:
                mar_info += f" · 💍AI+{ai_mar_pts}"
            won_str = "**Ty wygrałeś!**" if winner == HUMAN else "**AI wygrało!**"
            s.last_trick_summary = (
                f"Zagrałeś: {card_html(human_card, '1.2em')} &nbsp;↔&nbsp; "
                f"AI: {card_html(ai_card, '1.2em')} &nbsp;→&nbsp; "
                f"{won_str} +{pts} pkt{mar_info}")

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

            mar_info = ""
            if h_mar_pts:
                mar_info += f" · 💍Ty+{h_mar_pts}"
            won_str = "**Ty wygrałeś!**" if winner == HUMAN else "**AI wygrało!**"
            s.last_trick_summary = (
                f"AI: {card_html(leader_card, '1.2em')} &nbsp;↔&nbsp; "
                f"Ty: {card_html(follow_card, '1.2em')} &nbsp;→&nbsp; "
                f"{won_str} +{pts} pkt{mar_info}")

            s.trick_info  = {
                'cards':     {AI: leader_card, HUMAN: follow_card},
                'winner':    winner,
                'pts':       pts,
                'marriages': {HUMAN: h_mar_pts, AI: 0},
            }

        s.stage = 'trick_cards'
        st.rerun()
