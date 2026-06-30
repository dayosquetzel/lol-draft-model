"""
LoL Draft Model — Streamlit UI v2
Tabs: Draft | Meta | Tier List | Accuracy | Settings
"""

import sys
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from model import (
    DraftModel, ModelConfig, build_model,
    ModelEvaluator, DEFAULT_LEAGUES, MAJOR_LEAGUES, INTERNATIONAL,
    LEAGUE_REGION_TIER,
)

st.set_page_config(page_title="LoL Draft Model", page_icon="⚔️", layout="wide")

DB_PATH = str(Path(__file__).parent / "opinions.db")

# ---------------------------------------------------------------------------
# Default training data — auto-combine 2025 + 2026 if both present
# ---------------------------------------------------------------------------
import tempfile as _tempfile

_APP_DIR  = Path(__file__).parent
_CSV_2025 = _APP_DIR / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
_CSV_2026 = _APP_DIR / "2026_LoL_esports_match_data_from_OraclesElixir.csv"

@st.cache_resource(show_spinner="Combining 2025 + 2026 data…")
def _make_default_csv() -> tuple:
    """Combine CSVs at startup once; cached so it only runs once per session."""
    have_25 = _CSV_2025.exists()
    have_26 = _CSV_2026.exists()
    if have_25 and have_26:
        combined = pd.concat(
            [pd.read_csv(_CSV_2025, low_memory=False, encoding='latin-1'),
             pd.read_csv(_CSV_2026, low_memory=False, encoding='latin-1')],
            ignore_index=True,
        )
        tmp = _tempfile.NamedTemporaryFile(suffix=".csv", delete=False,
                                           dir=_tempfile.gettempdir())
        combined.to_csv(tmp.name, index=False)
        tmp.close()
        return tmp.name, "2025_OraclesElixir.csv + 2026_OraclesElixir.csv"
    if have_26:
        return str(_CSV_2026), _CSV_2026.name
    if have_25:
        return str(_CSV_2025), _CSV_2025.name
    return str(_CSV_2026), _CSV_2026.name   # fallback — will error if missing

_DEFAULT_DATA_PATH, _DEFAULT_DATA_LABEL = _make_default_csv()
DATA_PATH = _DEFAULT_DATA_PATH
ROLES       = ["top", "jng", "mid", "bot", "sup"]
ROLE_LABELS = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support"}

ALL_LEAGUES = sorted(LEAGUE_REGION_TIER.keys())
MAJOR_DEFAULT = sorted(lg for lg in DEFAULT_LEAGUES if lg in LEAGUE_REGION_TIER)


# ---------------------------------------------------------------------------
# SQLite opinion log
# ---------------------------------------------------------------------------

def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS opinions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            blue_picks  TEXT,
            red_picks   TEXT,
            model_prob  REAL,
            user_pick   TEXT,
            user_conf   TEXT,
            agree       INTEGER,
            notes       TEXT
        )
    """)
    con.commit()
    con.close()

def _log_opinion(blue: list, red: list, model_prob: float,
                 user_pick: str, user_conf: str, notes: str):
    agree = 1 if (user_pick == "Blue" and model_prob >= 0.5) or \
                 (user_pick == "Red"  and model_prob <  0.5) else 0
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO opinions (timestamp,blue_picks,red_picks,model_prob,"
        "user_pick,user_conf,agree,notes) VALUES (?,?,?,?,?,?,?,?)",
        (
            datetime.now().isoformat(),
            ",".join(b for b in blue if b),
            ",".join(r for r in red  if r),
            round(model_prob, 4),
            user_pick, user_conf, agree, notes,
        )
    )
    con.commit()
    con.close()

def _load_opinions() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql("SELECT * FROM opinions ORDER BY id DESC", con)
    con.close()
    return df

_init_db()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "model":              None,
        "config_hash":        None,
        "_clear_draft_flag":  False,
        "w_base":             0.25,
        "w_synergy":          0.40,
        "w_counter":          0.25,
        "w_fight":            0.10,
        "w_ctr_wr":           0.70,
        "w_syn_wr":           0.60,
        "overrides":          [],
        "blue_picks":         [""] * 5,
        "red_picks":          [""] * 5,
        "bans":               [],
        "min_games":          15,
        "selected_leagues":   MAJOR_DEFAULT,
        "main_data_path":     _DEFAULT_DATA_PATH,
        "main_data_label":    _DEFAULT_DATA_LABEL,
        "synergy_patch_decay": False,
        "counter_patch_decay": False,
        "use_all_leagues":    False,
        "meta_filter_mode":   "2026",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _make_config() -> ModelConfig:
    cfg = ModelConfig()
    cfg.WEIGHT_BASE             = st.session_state.w_base
    cfg.WEIGHT_SYNERGY          = st.session_state.w_synergy
    cfg.WEIGHT_COUNTER          = st.session_state.w_counter
    cfg.WEIGHT_FIGHT            = st.session_state.w_fight
    cfg.WEIGHT_COUNTER_WR       = st.session_state.w_ctr_wr
    cfg.WEIGHT_COUNTER_15       = round(1 - st.session_state.w_ctr_wr, 2)
    cfg.WEIGHT_SYNERGY_WR       = st.session_state.w_syn_wr
    cfg.WEIGHT_SYNERGY_15       = round(1 - st.session_state.w_syn_wr, 2)
    cfg.MIN_CHAMP_GAMES         = st.session_state.min_games
    cfg.SYNERGY_PATCH_DECAY     = st.session_state.get("synergy_patch_decay", False)
    cfg.COUNTER_PATCH_DECAY     = st.session_state.get("counter_patch_decay", False)
    cfg.USE_ALL_LEAGUES         = st.session_state.get("use_all_leagues", False)
    cfg.LEAGUES                 = st.session_state.selected_leagues or None
    cfg.MATCHUP_OVERRIDES       = {
        (o["champ_a"], o["champ_b"]): o["delta"]
        for o in st.session_state.overrides
    }
    return cfg


@st.cache_resource(show_spinner="Building model (~30s)…")
def _load_model(data_path: str, leagues_key: str, min_games: int,
                synergy_decay: bool, counter_decay: bool,
                use_all_leagues: bool,
                _cfg: ModelConfig) -> DraftModel:
    return build_model(data_path, config=_cfg)


def get_model() -> DraftModel:
    leagues_key     = ",".join(sorted(st.session_state.selected_leagues))
    data_path       = st.session_state.get("main_data_path", DATA_PATH)
    synergy_decay   = st.session_state.get("synergy_patch_decay", False)
    counter_decay   = st.session_state.get("counter_patch_decay", False)
    use_all_leagues = st.session_state.get("use_all_leagues", False)
    key = (
        data_path, leagues_key, st.session_state.min_games,
        synergy_decay, counter_decay, use_all_leagues,
    )
    if st.session_state.model is None or st.session_state.config_hash != key:
        cfg = _make_config()
        st.session_state.model = _load_model(
            data_path, leagues_key, st.session_state.min_games,
            synergy_decay, counter_decay, use_all_leagues, cfg,
        )
        st.session_state.config_hash = key

    m = st.session_state.model
    m.config.WEIGHT_BASE             = st.session_state.w_base
    m.config.WEIGHT_SYNERGY          = st.session_state.w_synergy
    m.config.WEIGHT_COUNTER          = st.session_state.w_counter
    m.config.WEIGHT_FIGHT            = st.session_state.w_fight
    m.config.WEIGHT_COUNTER_WR       = st.session_state.w_ctr_wr
    m.config.WEIGHT_COUNTER_15       = round(1 - st.session_state.w_ctr_wr, 2)
    m.config.WEIGHT_SYNERGY_WR       = st.session_state.w_syn_wr
    m.config.WEIGHT_SYNERGY_15       = round(1 - st.session_state.w_syn_wr, 2)
    m.config.MATCHUP_OVERRIDES       = {
        (o["champ_a"], o["champ_b"]): o["delta"]
        for o in st.session_state.overrides
    }
    return m


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _data_banner():
    label = st.session_state.get("main_data_label", Path(DATA_PATH).name)
    st.caption(f"📂 Training data: **{label}** — change in ⚙️ Settings")


@st.cache_data(show_spinner=False)
def _compute_priority_scores(raw_df_hash: str, _raw_df: pd.DataFrame, roles: tuple) -> dict:
    """
    Compute (champ, role) -> P+B% priority scores from raw_df.
    Cached by a hash of the data so it only recomputes when data changes.
    Returns dict keyed by (champ, role).
    """
    priority_scores: dict[tuple, float] = {}
    if _raw_df is None or _raw_df.empty:
        return priority_scores

    player_rows = _raw_df[_raw_df["position"] != "team"].copy()
    player_rows = player_rows[player_rows["position"].isin(set(roles))]
    player_rows = player_rows[player_rows["champion"].notna() & (player_rows["champion"] != "")]
    total_games = player_rows["gameid"].nunique()
    if total_games == 0:
        return priority_scores

    priority_picks: dict[tuple, int] = {}
    for (champ, role), grp in player_rows.groupby(["champion", "position"]):
        priority_picks[(champ, role)] = len(grp)

    ban_cols  = [c for c in _raw_df.columns if c.startswith("ban") and c[3:].isdigit()]
    team_rows = _raw_df[_raw_df["position"] == "team"].copy()
    priority_bans: dict[str, int] = {}
    if ban_cols and not team_rows.empty:
        total_games_for_bans = team_rows["gameid"].nunique() or 1
        for col in ban_cols:
            for champ, cnt in team_rows[col].value_counts().items():
                if isinstance(champ, str) and champ:
                    priority_bans[champ] = priority_bans.get(champ, 0) + int(cnt)
    else:
        total_games_for_bans = total_games

    pick_denom = max(total_games * 2, 1)
    ban_denom  = max(total_games_for_bans * 2 * len(ban_cols), 1) if ban_cols else 1
    for (champ, role) in priority_picks:
        p = priority_picks.get((champ, role), 0)
        b = priority_bans.get(champ, 0)
        priority_scores[(champ, role)] = round(
            (p / pick_denom + b / ban_denom) * 100, 1
        )
    return priority_scores


def _compute_meta_winrates(
    player_df: pd.DataFrame,
    game_df: pd.DataFrame,
    model,
) -> tuple[dict, dict, dict]:
    """
    Recompute BT-adjusted champion win rates from a filtered player_df.
    Same logic as BaseStrengthModel.fit() but on an arbitrary subset.
    Returns (role_winrates, role_counts, role_qualifies).
    """
    alpha    = model.config.TEAM_STRENGTH_ALPHA
    SHRINK_K = float(model.config.MIN_CHAMP_GAMES)

    team_side_map = {}
    if not game_df.empty and "teamname" in game_df.columns:
        gd_blue = game_df[game_df["side"] == "Blue"].set_index("gameid")[["teamname"]]
        gd_red  = game_df[game_df["side"] == "Red"].set_index("gameid")[["teamname"]]
        for gid in gd_blue.index.intersection(gd_red.index):
            b_team = str(gd_blue.at[gid, "teamname"])
            r_team = str(gd_red.at[gid,  "teamname"])
            team_side_map[(gid, "Blue")] = (b_team, r_team)
            team_side_map[(gid, "Red")]  = (r_team, b_team)

    role_win_w:   dict = {}
    role_total_w: dict = {}
    role_raw_n:   dict = {}

    for _, row in player_df.iterrows():
        champ  = row["champion"]
        role   = row["position"]
        result = int(row["result"])
        gameid = row["gameid"]
        side   = row["side"]
        pw     = float(row.get("patch_weight", 1.0))

        team_info = team_side_map.get((gameid, side))
        if team_info:
            my_team, opp_team = team_info
            winner, loser = (my_team, opp_team) if result == 1 else (opp_team, my_team)
            p_exp    = model.bt_model.expected_win_prob(winner, loser)
            surprise = 1.0 - p_exp
            team_adj = 1.0 - alpha + alpha * (surprise if result == 1 else (1.0 - surprise))
            upset_w  = model.bt_model.upset_weight(winner, loser) if result == 1 else 1.0
        else:
            team_adj = 1.0
            upset_w  = 1.0

        total_weight = pw * team_adj * upset_w
        k = (champ, role)
        role_win_w[k]   = role_win_w.get(k, 0.0)   + result * total_weight
        role_total_w[k] = role_total_w.get(k, 0.0) + total_weight
        role_raw_n[k]   = role_raw_n.get(k, 0)     + 1

    role_avg_win_w:   dict = {}
    role_avg_total_w: dict = {}
    for k, tw in role_total_w.items():
        _, role = k
        role_avg_win_w[role]   = role_avg_win_w.get(role, 0.0)   + role_win_w.get(k, 0.0)
        role_avg_total_w[role] = role_avg_total_w.get(role, 0.0) + tw
    global_total = sum(role_avg_total_w.values())
    global_wins  = sum(role_avg_win_w.values())
    global_avg   = (global_wins + 0.5) / (global_total + 1.0) if global_total > 0 else 0.5
    role_avg = {
        r: (role_avg_win_w[r] + 0.5) / (role_avg_total_w[r] + 1.0)
        for r in role_avg_win_w if role_avg_total_w.get(r, 0) > 0
    }

    role_winrates:  dict = {}
    role_counts:    dict = {}
    role_qualifies: dict = {}
    for k, raw_n in role_raw_n.items():
        champ, role = k
        tw = role_total_w.get(k, 0.0)
        ww = role_win_w.get(k, 0.0)
        if tw <= 0:
            continue
        observed_wr = (ww + 0.5) / (tw + 1.0)
        avg  = role_avg.get(role, global_avg)
        conf = raw_n / (raw_n + SHRINK_K)
        role_winrates[k]  = conf * observed_wr + (1.0 - conf) * avg
        role_counts[k]    = raw_n
        role_qualifies[k] = model.base_model._role_qualifies(champ, role)

    return role_winrates, role_counts, role_qualifies


def _colour_prob(p: float) -> str:
    if p >= 0.60: return "🟢"
    if p >= 0.52: return "🟡"
    if p >= 0.48: return "⚪"
    if p >= 0.40: return "🟠"
    return "🔴"

def _delta_str(d: float) -> str:
    return f"+{d:.1f}%" if d >= 0 else f"{d:.1f}%"

# Champions to exclude from specific roles regardless of data
_ROLE_EXCLUSIONS: dict[str, set[str]] = {
    "top": {"Nidalee"},
}

def _champs_for_role(m: DraftModel, role: str) -> list[str]:
    excluded = _ROLE_EXCLUSIONS.get(role, set())
    return sorted([
        c for c in m.base_model.known_champions()
        if c not in excluded
        and m.base_model._role_qualifies(c, role)
        and m.base_model.role_counts_.get((c, role), 0) >= m.config.MIN_CHAMP_GAMES
    ])

def _clear_draft():
    st.session_state._clear_draft_flag = True

def _apply_clear_draft():
    """Called at the TOP of the Draft tab, before any widget is rendered."""
    if st.session_state.get("_clear_draft_flag"):
        for i in range(5):
            st.session_state.pop(f"blue_{i}", None)
            st.session_state.pop(f"red_{i}",  None)
        st.session_state.pop("bans_input", None)
        st.session_state.blue_picks = [""] * 5
        st.session_state.red_picks  = [""] * 5
        st.session_state.bans       = []
        st.session_state._clear_draft_flag = False


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_draft, tab_meta, tab_tier, tab_accuracy, tab_opinion, tab_t1games, tab_scout, tab_settings = st.tabs(
    ["⚔️ Draft", "📊 Meta", "🏆 Tier List", "📈 Accuracy & Games", "🧪 Opinion Log", "🎮 Recent Games", "🔍 Team Scout", "⚙️ Settings"]
)


# ═══════════════════════════════════════════════════════════════════════════
# DRAFT TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_draft:
    _apply_clear_draft()
    _data_banner()
    m = get_model()
    all_champs = sorted(m.base_model.known_champions())

    st.markdown("### Draft Board")
    st.caption("Enter picks as you draft. Recommendations update after each pick.")

    col_blue, col_mid, col_red = st.columns([5, 3, 5])

    with col_blue:
        st.markdown("**🔵 Blue Side**")
        for i, role in enumerate(ROLES):
            val = st.session_state.blue_picks[i]
            idx = (all_champs.index(val) + 1) if val in all_champs else 0
            st.session_state.blue_picks[i] = st.selectbox(
                f"Blue {ROLE_LABELS[role]}",
                options=[""] + all_champs,
                index=idx,
                key=f"blue_{i}",
            )

    with col_red:
        st.markdown("**🔴 Red Side**")
        for i, role in enumerate(ROLES):
            val = st.session_state.red_picks[i]
            idx = (all_champs.index(val) + 1) if val in all_champs else 0
            st.session_state.red_picks[i] = st.selectbox(
                f"Red {ROLE_LABELS[role]}",
                options=[""] + all_champs,
                index=idx,
                key=f"red_{i}",
            )

    with col_mid:
        st.markdown("**Bans**")
        bans_raw = st.text_area(
            "Bans (one per line)",
            value="\n".join(st.session_state.bans),
            height=140,
            key="bans_input",
        )
        st.session_state.bans = [b.strip() for b in bans_raw.split("\n") if b.strip()]

        if st.button("🔄 Clear Draft", use_container_width=True):
            _clear_draft()
            st.rerun()

    blue = st.session_state.blue_picks
    red  = st.session_state.red_picks
    bans = st.session_state.bans

    blue_filled = [c for c in blue if c]
    red_filled  = [c for c in red  if c]

    # ── Draft pick sequence recommendations ─────────────────────────────────
    st.divider()

    # ── Live model weight controls ───────────────────────────────────────────
    with st.expander("⚙️ Model weights & patch decay", expanded=False):
        st.caption(
            "Adjust weights live — changes apply instantly to recommendations and predictions. "
            "Patch decay toggles rebuild the model (one-time cost)."
        )

        _syn_decay_on   = st.session_state.synergy_patch_decay
        _ctr_decay_on   = st.session_state.counter_patch_decay
        _all_leagues_on = st.session_state.use_all_leagues
        _league_label   = "all leagues (LCK/LPL/LEC/LCS/LCP + regional)" if _all_leagues_on else "major leagues only (LCK, LPL, LEC, LCS, LCP + internationals)"
        _ctr_league_label = _league_label if _all_leagues_on else "major leagues + supplemented with all-league matchup data"

        _base_tip = (
            "**Base strength** — role-aware champion win rate, corrected for opponent team strength "
            "(Bradley-Terry adjusted) and Bayesian-shrunk toward the role average for thin samples.\n\n"
            f"**Data:** 2025 + 2026 · {_league_label} · patch-decay weighted (recent patches count more)."
        )
        _syn_tip = (
            "**Synergy** — how much better (or worse) two champions win together vs their individual rates. "
            "For Bot+Support and Mid+Jungle pairs, also blends combined gold/CS/XP diff at 15 min.\n\n"
            f"**Data:** 2025 + 2026 · {_league_label} · "
            f"patch decay {'ON (recent patches weighted more)' if _syn_decay_on else 'OFF (all patches equal — more pair data)'}."
        )
        _ctr_tip = (
            "**Counter** — per-role win rate delta when champion A faces champion B in lane, "
            "blended with gold/CS/XP diff at 15 min for the same matchups.\n\n"
            f"**Data:** 2025 + 2026 · {_ctr_league_label} · "
            f"patch decay {'ON (recent patches weighted more)' if _ctr_decay_on else 'OFF (all patches equal — more matchup data)'}."
        )
        _fight_tip = (
            "**Teamfight@22** — each champion's average gold at ~22 min, normalised by the role average. "
            "A score > 1.0 means this champion tends to be ahead of curve at the mid-game fight window.\n\n"
            f"**Data:** 2025 + 2026 · {_league_label} · patch-decay weighted."
        )

        dw1, dw2, dw3, dw4 = st.columns(4)
        with dw1:
            st.markdown("⚖️ **Base**", help=_base_tip)
            new_wb = st.slider("Base weight", 0.0, 1.0, float(st.session_state.w_base), 0.05,
                               key="draft_wb", label_visibility="collapsed")
            if new_wb != st.session_state.w_base:
                st.session_state.w_base = new_wb
        with dw2:
            st.markdown("🔗 **Synergy**", help=_syn_tip)
            new_ws = st.slider("Synergy weight", 0.0, 1.0, float(st.session_state.w_synergy), 0.05,
                               key="draft_ws", label_visibility="collapsed")
            if new_ws != st.session_state.w_synergy:
                st.session_state.w_synergy = new_ws
        with dw3:
            st.markdown("⚔️ **Counter**", help=_ctr_tip)
            new_wc = st.slider("Counter weight", 0.0, 1.0, float(st.session_state.w_counter), 0.05,
                               key="draft_wc", label_visibility="collapsed")
            if new_wc != st.session_state.w_counter:
                st.session_state.w_counter = new_wc
        with dw4:
            st.markdown("💥 **Fight@22**", help=_fight_tip)
            new_wf = st.slider("Fight weight", 0.0, 1.0, float(st.session_state.w_fight), 0.05,
                               key="draft_wf", label_visibility="collapsed")
            if new_wf != st.session_state.w_fight:
                st.session_state.w_fight = new_wf

        # Auto-normalise so weights always sum to 1
        _draft_wsum = st.session_state.w_base + st.session_state.w_synergy + st.session_state.w_counter + st.session_state.w_fight
        if _draft_wsum > 0 and abs(_draft_wsum - 1.0) > 0.01:
            _scale = 1.0 / _draft_wsum
            st.session_state.w_base    = round(st.session_state.w_base    * _scale, 4)
            st.session_state.w_synergy = round(st.session_state.w_synergy * _scale, 4)
            st.session_state.w_counter = round(st.session_state.w_counter * _scale, 4)
            st.session_state.w_fight   = round(st.session_state.w_fight   * _scale, 4)
        st.caption(
            f"Base={st.session_state.w_base:.2f}  "
            f"Syn={st.session_state.w_synergy:.2f}  "
            f"Ctr={st.session_state.w_counter:.2f}  "
            f"Fight={st.session_state.w_fight:.2f}"
        )
        # Push updated weights into the live model config immediately
        m.config.WEIGHT_BASE    = st.session_state.w_base
        m.config.WEIGHT_SYNERGY = st.session_state.w_synergy
        m.config.WEIGHT_COUNTER = st.session_state.w_counter
        m.config.WEIGHT_FIGHT   = st.session_state.w_fight

        st.markdown("**Data scope & patch decay** *(rebuilds model — one-time cost)*")

        def _rebuild_on_toggle():
            """Called by any toggle on_change — clears cache and marks model dirty."""
            _load_model.clear()
            st.session_state.model       = None
            st.session_state.config_hash = None

        dc0, dc1, dc2 = st.columns(3)
        with dc0:
            st.toggle(
                "All leagues (base + synergy + counter)",
                key="use_all_leagues",
                on_change=_rebuild_on_toggle,
                help=(
                    "OFF (default): train on major leagues only (LCK, LPL, LEC, LCS, LCP + MSI/Worlds/EWC). "
                    "Higher data quality, less noise.\n\n"
                    "ON: train on ALL leagues including regional (CBLOL, VCS, LJL, etc). "
                    "More matchup coverage, but lower-quality games included."
                )
            )
        with dc1:
            st.toggle(
                "Synergy patch decay",
                key="synergy_patch_decay",
                on_change=_rebuild_on_toggle,
                help="ON = recent patches weighted more for synergy pairs.\nOFF = all patches equal weight (pools more pair data, ignores meta drift)."
            )
        with dc2:
            st.toggle(
                "Counter patch decay",
                key="counter_patch_decay",
                on_change=_rebuild_on_toggle,
                help="ON = recent patches weighted more for counter matchups.\nOFF = all patches equal weight (pools more matchup data across 2025+2026, recommended for rare pairs)."
            )

    next_blue_role = next((ROLES[i] for i, c in enumerate(blue) if not c), None)
    next_red_role  = next((ROLES[i] for i, c in enumerate(red)  if not c), None)

    if next_blue_role or next_red_role:
        st.markdown("#### Next pick recommendation")
        rec_col1, rec_col2 = st.columns(2)

        # Priority scores from full training data (same source as Meta tab)
        _draft_raw = m._raw_df if m._raw_df is not None else pd.DataFrame()
        _draft_raw_hash = str(id(m))  # changes when model rebuilds
        _draft_priority = _compute_priority_scores(_draft_raw_hash, _draft_raw, tuple(ROLES)) if not _draft_raw.empty else {}

        for side, next_role, team_so_far, opp_so_far, col in [
            ("🔵 Blue", next_blue_role, blue, red,  rec_col1),
            ("🔴 Red",  next_red_role,  red,  blue, rec_col2),
        ]:
            if next_role is None:
                continue
            with col:
                st.markdown(f"**{side} — {ROLE_LABELS[next_role]}**")
                _role_excl = list(_ROLE_EXCLUSIONS.get(next_role, set()))
                recs = m.recommend_pick(team_so_far, opp_so_far, next_role,
                                        banned=bans + _role_excl, top_n=10)
                if recs.empty:
                    st.info("No eligible recommendations.")
                else:
                    # Attach priority score for this role
                    recs["priority_%"] = recs["champion"].apply(
                        lambda c: _draft_priority.get((c, next_role), 0.0)
                    )
                    # Re-rank: blend win_prob (80%) + normalised priority (20%) so that
                    # a champion with near-zero real-world priority (e.g. Nidalee in a
                    # non-Nidalee meta) cannot float to #1 purely on model score.
                    _wp_min, _wp_max = recs["win_prob_%"].min(), recs["win_prob_%"].max()
                    _pr_min, _pr_max = recs["priority_%"].min(), recs["priority_%"].max()
                    recs["_wp_norm"] = (recs["win_prob_%"] - _wp_min) / max(_wp_max - _wp_min, 1e-9)
                    recs["_pr_norm"] = (recs["priority_%"] - _pr_min) / max(_pr_max - _pr_min, 1e-9)
                    recs["_rec_score"] = 0.80 * recs["_wp_norm"] + 0.20 * recs["_pr_norm"]
                    recs = recs.sort_values("_rec_score", ascending=False).head(10).reset_index(drop=True)
                    recs = recs.drop(columns=["_wp_norm", "_pr_norm", "_rec_score"])
                    recs.insert(0, "#", range(1, len(recs) + 1))
                    recs["win_prob_%"]    = recs["win_prob_%"].apply(lambda p: f"{_colour_prob(p/100)} {p:.1f}%")
                    recs["counter_adv_%"] = recs["counter_adv_%"].apply(_delta_str)
                    # Flag low matchup game counts for the opponent pairing
                    _opp_filled = [c for c in opp_so_far if c]
                    def _games_flag(row):
                        champ = row["champion"] if "champion" in row.index else ""
                        games = int(row["games"]) if "games" in row.index else 0
                        # Also check counter data vs opponent
                        ctr_games = 0
                        if _opp_filled:
                            for opp in _opp_filled:
                                k = (champ, next_role, opp)
                                ctr_games += m.counter_model.direct_counts_.get(k, 0)
                        flag = ""
                        if ctr_games > 0 and ctr_games < 10:
                            flag = f" ⚠️{ctr_games}g"
                        elif ctr_games == 0 and _opp_filled:
                            flag = " ⚠️0g"
                        return flag
                    recs["_ctr_flag"] = recs.apply(_games_flag, axis=1)
                    recs["counter_adv_%"] = recs["counter_adv_%"].astype(str) + recs["_ctr_flag"]
                    recs = recs.drop(columns=["_ctr_flag"])
                    recs.columns = ["#", "Champion", "Win%", "Base WR%", "Counter Adv", "Fight Adv", "Games", "Priority%"]
                    st.dataframe(
                        recs, use_container_width=True, hide_index=True,
                        column_config={
                            "Counter Adv": st.column_config.TextColumn(
                                "Counter Adv",
                                help="Counter advantage vs current opponent picks. "
                                     "⚠️Ng = only N games of matchup data — treat with caution. "
                                     "⚠️0g = no direct matchup data found."
                            ),
                            "Base WR%": st.column_config.NumberColumn(
                                "Base WR%",
                                help="Role-adjusted win rate (2025+2026, major leagues, BT-corrected)."
                            ),
                            "Fight Adv": st.column_config.NumberColumn(
                                "Fight Adv",
                                help="Gold@22 fight strength delta vs opponent. Positive = ahead at mid-game fight window."
                            ),
                            "Priority%": st.column_config.NumberColumn(
                                "Priority%",
                                format="%.1f",
                                help="Pick + Ban presence % from Meta tab — how contested this champion is. High = frequently picked or banned.",
                            ),
                        }
                    )

    # ── Full prediction ──────────────────────────────────────────────────────
    if blue_filled and red_filled:
        st.divider()
        st.markdown("### Prediction")

        b_pad = (blue + [""] * 5)[:5]
        r_pad = (red  + [""] * 5)[:5]
        res   = m.predict(b_pad, r_pad, team1_is_blue=True)
        wp    = res["win_probability"]

        prob_col, _ = st.columns([3, 2])
        with prob_col:
            st.markdown(f"**Blue win probability: {_colour_prob(wp)} {wp*100:.1f}%**")
            st.progress(wp)

        # ── Score breakdown — per-team component scores ──────────────────────
        st.markdown("#### Score breakdown")
        bd = res["breakdown"]

        breakdown_data = {
            "Component": ["Base strength", "Synergy", "Counter", "Teamfight@22"],
            "Blue raw":  [
                f"{res['team1_base']*100:.2f}%",
                f"{res['team1_synergy']*100:.2f}%",
                f"{res['counter_net']*100/2:+.2f}%",
                f"{res['fight_detail']['team1_fight_strength']:.4f}",
            ],
            "Red raw": [
                f"{res['team2_base']*100:.2f}%",
                f"{res['team2_synergy']*100:.2f}%",
                f"{-res['counter_net']*100/2:+.2f}%",
                f"{res['fight_detail']['team2_fight_strength']:.4f}",
            ],
            "Net (Blue − Red)": [
                f"{(res['team1_base'] - res['team2_base'])*100:+.2f}%",
                f"{(res['team1_synergy'] - res['team2_synergy'])*100:+.2f}%",
                f"{res['counter_net']*100:+.2f}%",
                f"{res['fight_net']:+.4f}",
            ],
            "Contribution to logit": [
                f"{bd['base']*100:+.2f}%",
                f"{bd['synergy']*100:+.2f}%",
                f"{bd['counter']*100:+.2f}%",
                f"{bd['fight']*100:+.2f}%",
            ],
        }
        st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

        # ── Per-role matchup detail with advantage indicator ─────────────────
        st.markdown("#### Per-role matchup")
        rows = []
        fight_t1 = {d["role"]: d for d in res["fight_detail"].get("team1_detail", [])}
        fight_t2 = {d["role"]: d for d in res["fight_detail"].get("team2_detail", [])}

        for ctr in res["counter_breakdown"]:
            role  = ctr["role"]
            bc    = ctr["team1_champ"] or "—"
            rc    = ctr["team2_champ"] or "—"
            b_wr  = m.base_model.champion_winrate(ctr["team1_champ"], role) if ctr["team1_champ"] else 0.5
            r_wr  = m.base_model.champion_winrate(ctr["team2_champ"], role) if ctr["team2_champ"] else 0.5
            b_fight = fight_t1.get(role, {}).get("fight_strength", 1.0)
            r_fight = fight_t2.get(role, {}).get("fight_strength", 1.0)

            # Advantage indicator: combine counter net + fight + base WR
            net_counter = ctr["direct_net"] / 100
            net_fight   = b_fight - r_fight
            net_wr      = b_wr - r_wr
            adv_score   = net_counter + net_fight * 0.5 + net_wr * 0.3

            if adv_score > 0.02:
                adv = f"🔵 {bc}"
            elif adv_score < -0.02:
                adv = f"🔴 {rc}"
            else:
                adv = "⚪ Even"

            # Games count for the forward matchup
            fwd_key = (ctr["team1_champ"], role, ctr["team2_champ"]) if ctr["team1_champ"] and ctr["team2_champ"] else None
            matchup_games = m.counter_model.direct_counts_.get(fwd_key, 0) if fwd_key else 0

            rows.append({
                "Role":        ROLE_LABELS.get(role, role),
                "Blue":        bc,
                "Red":         rc,
                "Advantage":   adv,
                "Blue WR%":    f"{b_wr*100:.1f}%",
                "Red WR%":     f"{r_wr*100:.1f}%",
                "Blue Δvs":    _delta_str(ctr["t1_wr_delta"]),
                "Red Δvs":     _delta_str(ctr["t2_wr_delta"]),
                "Counter Δ":   _delta_str(ctr["direct_net"]),
                "Matchup N":   matchup_games,
                "Lane @15":    f"{ctr['t1_lane_15']:.2f}" if ctr["t1_lane_15"] is not None else "—",
                "Blue fight":  f"{b_fight:.3f}",
                "Red fight":   f"{r_fight:.3f}",
                "Override":    "✅" if ctr.get("t1_overridden") else "",
            })

        st.dataframe(
            pd.DataFrame(rows), use_container_width=True, hide_index=True,
            column_config={
                "Blue Δvs":  st.column_config.TextColumn("🔵 WR Δ vs opp",
                    help="Blue champ's WR delta vs this specific opponent vs their baseline WR. e.g. +10% = wins 10pp more than usual against this matchup."),
                "Red Δvs":   st.column_config.TextColumn("🔴 WR Δ vs opp",
                    help="Red champ's WR delta vs this specific opponent vs their baseline WR."),
                "Counter Δ": st.column_config.TextColumn("Counter Δ (net)",
                    help="Net counter advantage for blue = (Blue WR Δ vs opponent) − (Red WR Δ vs opponent). Large values on thin sample sizes (low Matchup N) are unreliable."),
                "Matchup N": st.column_config.NumberColumn("Matchup N",
                    help="Number of games this exact lane matchup has been seen in training data. Low = less reliable."),
            }
        )

        # ── Synergy detail ───────────────────────────────────────────────────
        with st.expander("Synergy detail"):
            syn_c1, syn_c2 = st.columns(2)
            with syn_c1:
                st.markdown("**Blue synergies**")
                sd1 = res.get("synergy_detail_t1", [])
                if sd1:
                    st.dataframe(pd.DataFrame(sd1), use_container_width=True, hide_index=True)
            with syn_c2:
                st.markdown("**Red synergies**")
                sd2 = res.get("synergy_detail_t2", [])
                if sd2:
                    st.dataframe(pd.DataFrame(sd2), use_container_width=True, hide_index=True)

        # ── Quick opinion logger ─────────────────────────────────────────────
        st.divider()
        st.markdown("#### Log your opinion on this draft")
        op_col1, op_col2, op_col3 = st.columns([2, 2, 3])
        with op_col1:
            user_pick = st.selectbox("Your predicted winner", ["Blue", "Red"], key="op_pick")
        with op_col2:
            user_conf = st.selectbox("Confidence", ["Low", "Medium", "High"], key="op_conf")
        with op_col3:
            user_notes = st.text_input("Notes (optional)", key="op_notes")

        if st.button("📝 Log opinion", type="primary", use_container_width=True):
            _log_opinion(
                blue=blue,
                red=red,
                model_prob=wp,
                user_pick=user_pick,
                user_conf=user_conf,
                notes=user_notes,
            )
            st.success("✅ Opinion logged!")

# ═══════════════════════════════════════════════════════════════════════════
# META TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_meta:
    _data_banner()
    m = get_model()
    st.markdown("### Champion Meta")

    # ── Meta filter ─────────────────────────────────────────────────────────
    raw_df_meta = m._raw_df.copy() if m._raw_df is not None else pd.DataFrame()
    available_years   = sorted(raw_df_meta["year"].dropna().unique().astype(int).tolist()) if "year" in raw_df_meta.columns else []
    available_patches = sorted(raw_df_meta["patch"].dropna().unique().tolist())            if "patch" in raw_df_meta.columns else []

    mf1, mf2 = st.columns([2, 3])
    with mf1:
        filter_mode = st.selectbox(
            "Show win rates from",
            options=["All training data"] + [str(y) for y in available_years] + ["Custom patches"],
            key="meta_filter_mode",
            help="Filters which games compute champion WR in this tab. The fitted model (Draft/predictions) is unaffected.",
        )
    with mf2:
        selected_patches = []
        if filter_mode == "Custom patches" and available_patches:
            selected_patches = st.multiselect(
                "Select patches", options=available_patches,
                default=available_patches[-3:] if len(available_patches) >= 3 else available_patches,
                key="meta_patch_select",
            )

    # Apply filter
    use_filtered = filter_mode != "All training data" and not raw_df_meta.empty
    if use_filtered:
        if filter_mode == "Custom patches":
            mask = raw_df_meta["patch"].isin(selected_patches) if selected_patches else pd.Series(True, index=raw_df_meta.index)
        else:
            mask = raw_df_meta["year"].astype(int) == int(filter_mode)
        filtered_raw = raw_df_meta[mask]

        keep_p = ["gameid", "side", "position", "champion", "result", "patch", "patch_weight", "league", "date"]
        keep_p = [c for c in keep_p if c in filtered_raw.columns]
        filt_player_df = (filtered_raw[filtered_raw["position"] != "team"]
                          .copy()[keep_p]
                          .pipe(lambda d: d[d["champion"].notna() & (d["champion"] != "")])
                          .dropna(subset=["champion", "result", "position"]))
        filt_player_df["result"] = filt_player_df["result"].astype(int)
        filt_player_df = filt_player_df[filt_player_df["position"].isin(set(m.config.ROLES))]

        ext_cols = ["gameid", "side", "position", "champion", "result", "patch", "patch_weight", "league", "teamname"]
        ext_cols = [c for c in ext_cols if c in filtered_raw.columns]
        pext = filtered_raw[filtered_raw["position"] != "team"][ext_cols].dropna(subset=["champion"]).copy()

        def _pivot_mini(grp):
            role_map = grp.set_index("position")["champion"].to_dict()
            return pd.Series({
                "champs":       [role_map.get(r, "") for r in m.config.ROLES],
                "result":       int(grp["result"].iloc[0]),
                "patch":        grp["patch"].iloc[0],
                "patch_weight": grp["patch_weight"].iloc[0],
                "league":       grp["league"].iloc[0],
                "teamname":     str(grp["teamname"].iloc[0]) if "teamname" in grp.columns and pd.notna(grp["teamname"].iloc[0]) else "",
            })

        filt_game_df = (pext.groupby(["gameid", "side"], group_keys=False)
                        .apply(_pivot_mini).reset_index())

        role_winrates, role_counts, role_qualifies = _compute_meta_winrates(filt_player_df, filt_game_df, m)
        n_games_shown = filt_player_df["gameid"].nunique()
        filter_label  = filter_mode if filter_mode != "Custom patches" else f"patches {', '.join(str(p) for p in selected_patches)}"
        st.caption(
            f"📊 Showing **{filter_label}** · {n_games_shown:,} games · "
            f"Min {m.config.MIN_CHAMP_GAMES} games · BT-adjusted · "
            "Grade: S ≥ 55% · A ≥ 52% · B ≥ 50% · C ≥ 48% · D < 48%"
        )
        wr_col = "Adj WR%"
    else:
        role_winrates  = m.base_model.role_winrates_
        role_counts    = m.base_model.role_counts_
        role_qualifies = {k: m.base_model._role_qualifies(k[0], k[1]) for k in m.base_model.role_winrates_}
        st.caption(
            f"Ranked by adjusted win rate (team-strength-weighted, Bayesian-shrunk). "
            f"Min {m.config.MIN_CHAMP_GAMES} games. "
            "Grade: S ≥ 55% · A ≥ 52% · B ≥ 50% · C ≥ 48% · D < 48% adjusted WR."
        )
        wr_col = "Adj WR%"

    # ── Compute P+B priority scores from the filtered raw data ─────────────
    # Priority score = (picks + bans) / total_games_in_window, per role
    # Uses the same filtered raw_df as win rates so the filter applies
    _raw_for_priority = filtered_raw if use_filtered else raw_df_meta
    _meta_raw_hash = str(id(m)) + filter_mode + ("".join(str(p) for p in selected_patches) if filter_mode == "Custom patches" else "")
    _priority_scores: dict[tuple, float] = _compute_priority_scores(
        _meta_raw_hash, _raw_for_priority, tuple(m.config.ROLES)
    ) if not _raw_for_priority.empty else {}

    # ── Best First Pick ──────────────────────────────────────────────────────
    # Normalise WR and Priority within each role's pool (so roles are comparable),
    # apply patch-decay via trend score, then merge all roles and rank globally.
    with st.expander("🥇 Best First Pick — all roles", expanded=True):
        st.caption(
            "Champions ranked by **First Pick Score** = normalised adjusted WR (40%) + "
            "normalised Priority% (40%) + patch trend (20%), "
            "computed within each role's pool so roles are fairly comparable. "
            "Patch trend rewards champions rising on recent patches and penalises those falling. "
            "⚠️Ng = low sample size."
        )
        all_fp_rows = []
        for fp_role in ROLES:
            fp_rows = []
            for (c, r), wr in role_winrates.items():
                if r != fp_role:
                    continue
                n = role_counts.get((c, r), 0)
                if n < m.config.MIN_CHAMP_GAMES:
                    continue
                if not role_qualifies.get((c, r), False):
                    continue
                pr = _priority_scores.get((c, r), 0.0)
                # trend_score is in [-1, 1]; shift to [0, 1] for combining
                trend_raw = m.patch_trend_model.trend_scores_.get((c, r), 0.0)
                trend_0_1 = (trend_raw + 1.0) / 2.0
                trend_arrow = m.patch_trend_model.trend_arrow(c, r)
                fp_rows.append({
                    "Champion": c, "Role": ROLE_LABELS[fp_role],
                    "wr": wr * 100, "priority": pr,
                    "trend_0_1": trend_0_1, "Trend": trend_arrow, "games": n,
                })

            if not fp_rows:
                continue

            # Normalise WR and Priority within this role's pool so roles are comparable
            role_fp = pd.DataFrame(fp_rows)
            wr_min, wr_max = role_fp["wr"].min(), role_fp["wr"].max()
            pr_min, pr_max = role_fp["priority"].min(), role_fp["priority"].max()
            role_fp["wr_norm"] = (role_fp["wr"] - wr_min) / (wr_max - wr_min + 1e-9)
            role_fp["pr_norm"] = (role_fp["priority"] - pr_min) / (pr_max - pr_min + 1e-9)
            # Trend is already [0,1] — no need to normalise per-role
            # Weights: 40% WR, 40% priority, 20% patch trend
            role_fp["fp_score"] = (
                0.4 * role_fp["wr_norm"] +
                0.4 * role_fp["pr_norm"] +
                0.2 * role_fp["trend_0_1"]
            ) * 100
            all_fp_rows.append(role_fp)

        if all_fp_rows:
            fp_all = (
                pd.concat(all_fp_rows, ignore_index=True)
                .sort_values("fp_score", ascending=False)
                .reset_index(drop=True)
            )
            fp_all.index += 1

            # Build display columns
            fp_all["Adj WR%"]    = fp_all["wr"].round(1)
            fp_all["Priority%"]  = fp_all["priority"].round(1)
            fp_all["FP Score"]   = fp_all["fp_score"].round(1)
            fp_all["Sample"]     = fp_all["games"].apply(
                lambda n: f"⚠️{n}g" if n < m.config.MIN_CHAMP_GAMES * 2 else f"{n}g"
            )
            fp_display = fp_all[["Champion", "Role", "Adj WR%", "Priority%", "Trend", "FP Score", "Sample"]]
            st.dataframe(
                fp_display, use_container_width=True,
                column_config={
                    "Champion":  st.column_config.TextColumn("Champion",   width="medium"),
                    "Role":      st.column_config.TextColumn("Role",       width="small"),
                    "Adj WR%":   st.column_config.ProgressColumn("Adj WR%", min_value=40, max_value=65, format="%.1f%%",
                        help="BT-adjusted win rate for this champion in this role."),
                    "Priority%": st.column_config.ProgressColumn("Priority%", min_value=0, max_value=100, format="%.1f%%",
                        help="Pick + Ban presence % across all games in the selected window."),
                    "Trend":     st.column_config.TextColumn("Trend",      width="small",
                        help="Patch trend arrow — rising champions score higher in FP Score."),
                    "FP Score":  st.column_config.ProgressColumn("FP Score", min_value=0, max_value=100, format="%.1f",
                        help="First Pick Score = 40% normalised WR + 40% normalised Priority + 20% patch trend (0–100)."),
                    "Sample":    st.column_config.TextColumn("Sample",     width="small",
                        help="⚠️Ng = fewer than 2× min games — win rate is heavily shrunk, treat with caution."),
                },
            )
        else:
            st.info("No data available with current filter.")

    st.divider()

    # ── Role tabs ────────────────────────────────────────────────────────────
    role_tabs = st.tabs([ROLE_LABELS[r] for r in ROLES])
    for role, rtab in zip(ROLES, role_tabs):
        with rtab:
            # Build rows common to both sub-tabs
            rows_wr = []
            rows_pr = []
            for (c, r), wr in role_winrates.items():
                if r != role:
                    continue
                n = role_counts.get((c, r), 0)
                if n < m.config.MIN_CHAMP_GAMES:
                    continue
                if not role_qualifies.get((c, r), False):
                    continue
                wr_pct = wr * 100
                if wr_pct >= 55:   grade = "🟢 S"
                elif wr_pct >= 52: grade = "🟡 A"
                elif wr_pct >= 50: grade = "⚪ B"
                elif wr_pct >= 48: grade = "🟠 C"
                else:              grade = "🔴 D"
                trend = m.patch_trend_model.trend_arrow(c, r)
                pr_score = _priority_scores.get((c, r), 0.0)
                rows_wr.append({"Champion": c, "Grade": grade, wr_col: round(wr_pct, 1),
                                 "Priority%": pr_score, "Trend": trend, "Games": n})
                rows_pr.append({"Champion": c, "Priority%": pr_score, "Grade": grade,
                                 wr_col: round(wr_pct, 1), "Trend": trend, "Games": n})

            if not rows_wr:
                st.info("No data for this role with current filter.")
                continue

            # ── Low sample size threshold for this role ──────────────────────
            _LOW_SAMPLE_THRESHOLD = max(m.config.MIN_CHAMP_GAMES * 3, 30)

            def _sample_flag(n: int) -> str:
                """Return a warning flag for low sample sizes, similar to counter adv ⚠️."""
                if n < m.config.MIN_CHAMP_GAMES * 2:
                    return f" ⚠️{n}g"
                return ""

            # Attach sample flag to champion name
            for row in rows_wr:
                row["Champion"] = row["Champion"] + _sample_flag(row["Games"])
            for row in rows_pr:
                row["Champion"] = row["Champion"] + _sample_flag(row["Games"])

            meta_sub1, meta_sub2 = st.tabs(["📈 Adjusted Win Rate", "🎯 Priority Score (P+B)"])

            with meta_sub1:
                st.caption(
                    "Sorted by **adjusted win rate** — BT-corrected for opponent strength, "
                    "Bayesian-shrunk toward role average. Priority% shown for reference. "
                    f"⚠️Ng = fewer than {m.config.MIN_CHAMP_GAMES * 2} games — treat with caution."
                )
                df_wr = (pd.DataFrame(rows_wr)
                           .sort_values(wr_col, ascending=False)
                           .reset_index(drop=True))
                df_wr.index += 1
                st.dataframe(
                    df_wr, use_container_width=True,
                    column_config={
                        "Champion":  st.column_config.TextColumn("Champion",   width="medium",
                            help="⚠️Ng = low sample size (< 2× min games). Win rate is heavily shrunk toward role average — interpret cautiously."),
                        "Grade":     st.column_config.TextColumn("Grade",      width="small",
                            help="🟢S ≥55% · 🟡A ≥52% · ⚪B ≥50% · 🟠C ≥48% · 🔴D <48%"),
                        wr_col:      st.column_config.ProgressColumn(wr_col,   min_value=40, max_value=65, format="%.1f%%"),
                        "Priority%": st.column_config.ProgressColumn("Priority%", min_value=0, max_value=100, format="%.1f%%",
                            help="Pick + Ban presence % across all games in the selected window."),
                        "Trend":     st.column_config.TextColumn("Trend",      width="small"),
                        "Games":     st.column_config.NumberColumn("Games",    width="small"),
                    },
                )

            with meta_sub2:
                st.caption(
                    "Sorted by **Priority Score** = pick-rate in role + ban-rate across all games "
                    "in the selected window. High priority = contested every game. "
                    "A champion can have high priority but low win rate (overrated) or vice versa. "
                    f"⚠️Ng = fewer than {m.config.MIN_CHAMP_GAMES * 2} games — win rate less reliable."
                )
                df_pr = (pd.DataFrame(rows_pr)
                           .sort_values("Priority%", ascending=False)
                           .reset_index(drop=True))
                df_pr.index += 1
                # Priority grade based on P+B %
                def _pr_grade(p):
                    if p >= 80: return "🟢 S"
                    if p >= 60: return "🟡 A"
                    if p >= 40: return "⚪ B"
                    if p >= 20: return "🟠 C"
                    return "🔴 D"
                df_pr["P Grade"] = df_pr["Priority%"].apply(_pr_grade)
                df_pr = df_pr[["Champion", "P Grade", "Priority%", "Grade", wr_col, "Trend", "Games"]]
                st.dataframe(
                    df_pr, use_container_width=True,
                    column_config={
                        "Champion":  st.column_config.TextColumn("Champion",   width="medium",
                            help="⚠️Ng = low sample size (< 2× min games). Win rate is heavily shrunk toward role average — interpret cautiously."),
                        "P Grade":   st.column_config.TextColumn("P Grade",    width="small",
                            help="🟢S ≥80% · 🟡A ≥60% · ⚪B ≥40% · 🟠C ≥20% · 🔴D <20%"),
                        "Priority%": st.column_config.ProgressColumn("Priority%", min_value=0, max_value=100, format="%.1f%%",
                            help="(Picks in this role + bans) as % of available game slots. Higher = more contested."),
                        "Grade":     st.column_config.TextColumn("WR Grade",   width="small",
                            help="🟢S ≥55% · 🟡A ≥52% · ⚪B ≥50% · 🟠C ≥48% · 🔴D <48%"),
                        wr_col:      st.column_config.ProgressColumn(wr_col,   min_value=40, max_value=65, format="%.1f%%"),
                        "Trend":     st.column_config.TextColumn("Trend",      width="small"),
                        "Games":     st.column_config.NumberColumn("Games",    width="small"),
                    },
                )


# ═══════════════════════════════════════════════════════════════════════════
# TIER LIST TAB — by region
# ═══════════════════════════════════════════════════════════════════════════
with tab_tier:
    _data_banner()
    m = get_model()
    st.markdown("### Team Tier List")

    tier_df = m.bt_model.tier_table()
    tier_df = tier_df[tier_df["Games"] >= m.config.MIN_TEAM_GAMES].copy()

    tier_colours = {"S": "🟡", "A": "🟢", "B": "🔵", "C": "⚪", "D": "🔴", "?": "⬜"}

    region_order = sorted(tier_df["Region"].unique(),
                          key=lambda r: LEAGUE_REGION_TIER.get(r, 99))
    region_filter = st.multiselect("Filter by region", options=region_order,
                                   default=list(region_order), key="tier_region")
    if region_filter:
        tier_df = tier_df[tier_df["Region"].isin(region_filter)]

    # ── Year filter (mirrors Meta tab) ──────────────────────────────────────
    raw_df_tier = m._raw_df if m._raw_df is not None else pd.DataFrame()
    available_years_tier = sorted(
        raw_df_tier["year"].dropna().unique().astype(int).tolist()
    ) if "year" in raw_df_tier.columns else []

    tf1, tf2 = st.columns([2, 3])
    with tf1:
        tier_year_mode = st.selectbox(
            "Show stats from",
            options=["All training data"] + [str(y) for y in available_years_tier],
            key="tier_year_mode",
            help="Filters which games are used for BT scores and draft quality. "
                 "The fitted model is unaffected.",
        )
    with tf2:
        st.caption(
            f"Filtered to: **{tier_year_mode}** · "
            f"Click any column header to sort."
        )

    # Filter game_df and team_level for the selected year
    if tier_year_mode != "All training data" and not raw_df_tier.empty:
        yr_mask       = raw_df_tier["year"].astype(int) == int(tier_year_mode)
        raw_tier_filt = raw_df_tier[yr_mask]
        from model import DataLoader as _DL
        _keep_t = ["gameid","side","teamname","result","patch","patch_weight","league"]
        _keep_t = [c for c in _keep_t if c in raw_tier_filt.columns]
        team_level_filt = raw_tier_filt[raw_tier_filt["position"]=="team"][_keep_t].copy()
        # Rebuild a lightweight filtered game_df for draft quality
        _ext = ["gameid","side","position","champion","result","patch","patch_weight","league","teamname"]
        _ext = [c for c in _ext if c in raw_tier_filt.columns]
        _pext = raw_tier_filt[raw_tier_filt["position"]!="team"][_ext].dropna(subset=["champion"]).copy()
        def _piv(grp):
            rm = grp.set_index("position")["champion"].to_dict()
            return pd.Series({
                "champs":       [rm.get(r,"") for r in m.config.ROLES],
                "result":       int(grp["result"].iloc[0]),
                "patch":        grp["patch"].iloc[0],
                "patch_weight": grp["patch_weight"].iloc[0],
                "league":       grp["league"].iloc[0],
                "teamname":     str(grp["teamname"].iloc[0]) if "teamname" in grp.columns and pd.notna(grp["teamname"].iloc[0]) else "",
            })
        game_df_filt = (_pext.groupby(["gameid","side"], group_keys=False)
                        .apply(_piv).reset_index())
        game_df_filt = game_df_filt[
            game_df_filt["champs"].apply(lambda c: isinstance(c,list) and len(c)==5)
        ]
    else:
        team_level_filt = None
        game_df_filt    = m._game_df

    # Recompute BT on filtered data for the BT tab
    from model import TeamStrengthModel as _BT
    if team_level_filt is not None and not team_level_filt.empty:
        bt_filt = _BT(m.config).fit(team_level_filt)
    else:
        bt_filt = m.bt_model

    tier_df_filt = bt_filt.tier_table()
    tier_df_filt = tier_df_filt[tier_df_filt["Games"] >= m.config.MIN_TEAM_GAMES].copy()
    if region_filter:
        tier_df_filt = tier_df_filt[tier_df_filt["Region"].isin(region_filter)]

    tier_tab1, tier_tab2 = st.tabs(["🏆 Win Rate (Bradley-Terry)", "🎯 Draft Quality"])

    with tier_tab1:
        st.caption(
            "Regional-tier weighted + strength-of-schedule corrected Bradley-Terry ratings. "
            "Mean score = 1.0. T1 (LCK/LPL) full weight, T2 (LEC/LCS) 0.80, "
            "T3 (CBLOL/VCS/LJL) 0.50, T4 0.25. Click column headers to sort."
        )
        bt_display = tier_df_filt[["Team","BT Score","Tier","W","L","Games","Region"]].copy()
        bt_display["Tier"] = bt_display["Tier"].map(lambda t: f"{tier_colours.get(t,'')} {t}")
        st.dataframe(
            bt_display.reset_index(drop=True),
            use_container_width=True, hide_index=True,
            column_config={
                "Team":     st.column_config.TextColumn("Team",     width="medium"),
                "BT Score": st.column_config.ProgressColumn("BT Score", min_value=0, max_value=2, format="%.3f",
                    help="Bradley-Terry strength. Mean = 1.0. Higher = stronger team."),
                "Tier":     st.column_config.TextColumn("Tier",     width="small"),
                "W":        st.column_config.NumberColumn("W",       width="small"),
                "L":        st.column_config.NumberColumn("L",       width="small"),
                "Games":    st.column_config.NumberColumn("Games",   width="small"),
                "Region":   st.column_config.TextColumn("Region",   width="small"),
            },
        )

    with tier_tab2:
        st.caption(
            "Draft quality: for each game the model assigns a win-probability to the draft. "
            "That WP is bucketed into a letter grade — **S** ≥ 65% · **A** ≥ 57% · **B** ≥ 50% · **C** ≥ 43% · **D** < 43%. "
            "The table shows how often each team lands in each bucket. "
            "Sort by S% or A%+ to find teams that consistently draft well."
        )

        # Grade every draft by WP threshold (same as draft_grade_for_picks)
        def _wp_to_grade(wp: float) -> str:
            if wp >= 0.65: return "S"
            if wp >= 0.57: return "A"
            if wp >= 0.50: return "B"
            if wp >= 0.43: return "C"
            return "D"

        gdf_dq = game_df_filt if game_df_filt is not None else m._game_df
        if gdf_dq is None or "teamname" not in gdf_dq.columns:
            st.info("No draft quality data — ensure teamname column is present in the CSV.")
        else:
            blue_dq = gdf_dq[gdf_dq["side"] == "Blue"].set_index("gameid")[["champs", "teamname", "result", "patch_weight"]]
            red_dq  = gdf_dq[gdf_dq["side"] == "Red"].set_index("gameid")[["champs", "teamname", "result", "patch_weight"]]
            common_dq = blue_dq.index.intersection(red_dq.index)

            # Per-team grade counts, and global grade→win tracking
            team_grade_counts: dict[str, dict] = {}
            team_region: dict[str, str] = {}
            grade_wins  = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0}
            grade_total = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0}

            for gid in common_dq:
                b_champs = blue_dq.at[gid, "champs"]
                r_champs = red_dq.at[gid,  "champs"]
                b_team   = str(blue_dq.at[gid, "teamname"])
                r_team   = str(red_dq.at[gid,  "teamname"])
                b_result = int(blue_dq.at[gid, "result"])
                r_result = int(red_dq.at[gid,  "result"])

                if not (isinstance(b_champs, list) and len(b_champs) == 5 and all(b_champs)):
                    continue
                if not (isinstance(r_champs, list) and len(r_champs) == 5 and all(r_champs)):
                    continue

                try:
                    res  = m.predict(b_champs, r_champs, team1_is_blue=True)
                    b_wp = float(res["win_probability"])
                except Exception:
                    b_wp = 0.5
                r_wp = 1.0 - b_wp

                for team, wp, result in [(b_team, b_wp, b_result), (r_team, r_wp, r_result)]:
                    g = _wp_to_grade(wp)
                    if team not in team_grade_counts:
                        team_grade_counts[team] = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0, "games": 0}
                    team_grade_counts[team][g] += 1
                    team_grade_counts[team]["games"] += 1
                    # Both sides counted: grade X wins if that team actually won
                    grade_wins[g]  += result
                    grade_total[g] += 1

            # Collect region from BT model
            for team in team_grade_counts:
                rg = m.bt_model.league_games_.get(team, {})
                team_region[team] = max(rg, key=rg.get) if rg else "?"

            dq_rows = []
            for team, counts in team_grade_counts.items():
                n = counts["games"]
                if n < m.config.MIN_TEAM_GAMES:
                    continue
                region = team_region.get(team, "?")
                if region_filter and region not in region_filter:
                    continue
                s_pct = counts["S"] / n * 100
                a_pct = counts["A"] / n * 100
                b_pct = counts["B"] / n * 100
                c_pct = counts["C"] / n * 100
                d_pct = counts["D"] / n * 100
                dq_rows.append({
                    "Team":    team,
                    "Region":  region,
                    "Games":   n,
                    "S%":      round(s_pct, 1),
                    "A%":      round(a_pct, 1),
                    "B%":      round(b_pct, 1),
                    "C%":      round(c_pct, 1),
                    "D%":      round(d_pct, 1),
                    "A%+":     round(s_pct + a_pct, 1),   # "Elite draft" rate
                    "S":       counts["S"],
                    "A":       counts["A"],
                    "B":       counts["B"],
                    "C":       counts["C"],
                    "D":       counts["D"],
                })

            if not dq_rows:
                st.info("No draft quality data — ensure teamname column is present in the CSV.")
            else:
                dq_display = (
                    pd.DataFrame(dq_rows)
                    .sort_values("A%+", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(
                    dq_display[["Team", "Region", "Games", "S%", "A%", "A%+", "B%", "C%", "D%", "S", "A", "B", "C", "D"]],
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Team":   st.column_config.TextColumn("Team",   width="medium"),
                        "Region": st.column_config.TextColumn("Region", width="small"),
                        "Games":  st.column_config.NumberColumn("Games", width="small"),
                        "S%":     st.column_config.ProgressColumn("🟡 S%",  min_value=0, max_value=100, format="%.1f%%",
                            help="% of drafts where model gave ≥65% win probability"),
                        "A%":     st.column_config.ProgressColumn("🟢 A%",  min_value=0, max_value=100, format="%.1f%%",
                            help="% of drafts where model gave 57–65% win probability"),
                        "A%+":    st.column_config.ProgressColumn("S+A%",   min_value=0, max_value=100, format="%.1f%%",
                            help="S% + A%: share of drafts graded elite (≥57% WP)"),
                        "B%":     st.column_config.NumberColumn("🔵 B%",  width="small", format="%.1f%%",
                            help="% of drafts where model gave 50–57% win probability"),
                        "C%":     st.column_config.NumberColumn("⚪ C%",  width="small", format="%.1f%%",
                            help="% of drafts where model gave 43–50% win probability"),
                        "D%":     st.column_config.NumberColumn("🔴 D%",  width="small", format="%.1f%%",
                            help="% of drafts where model gave <43% win probability"),
                        "S":      st.column_config.NumberColumn("S",     width="small"),
                        "A":      st.column_config.NumberColumn("A",     width="small"),
                        "B":      st.column_config.NumberColumn("B",     width="small"),
                        "C":      st.column_config.NumberColumn("C",     width="small"),
                        "D":      st.column_config.NumberColumn("D",     width="small"),
                    },
                )

                # ── Win rate by draft grade ──────────────────────────────────
                st.divider()
                st.markdown("#### Win rate by draft grade")
                st.caption(
                    "Across all teams and games in the current filter: "
                    "how often does a team actually win when the model grades their draft S, A, B, C, or D?"
                )
                grade_order  = ["S", "A", "B", "C", "D"]
                grade_labels = {
                    "S": "🟡 S (≥65% WP)",
                    "A": "🟢 A (57–65%)",
                    "B": "🔵 B (50–57%)",
                    "C": "⚪ C (43–50%)",
                    "D": "🔴 D (<43%)",
                }
                wr_rows = []
                for g in grade_order:
                    total = grade_total[g]
                    wins  = grade_wins[g]
                    wr    = wins / total * 100 if total > 0 else None
                    wr_rows.append({
                        "Draft Grade": grade_labels[g],
                        "Games":       total,
                        "Wins":        wins,
                        "Win Rate":    round(wr, 1) if wr is not None else None,
                    })
                wr_df = pd.DataFrame(wr_rows)
                st.dataframe(
                    wr_df, use_container_width=True, hide_index=True,
                    column_config={
                        "Draft Grade": st.column_config.TextColumn("Draft Grade", width="medium"),
                        "Games":       st.column_config.NumberColumn("Games",     width="small"),
                        "Wins":        st.column_config.NumberColumn("Wins",      width="small"),
                        "Win Rate":    st.column_config.ProgressColumn("Win Rate", min_value=0, max_value=100, format="%.1f%%",
                            help="How often teams with this draft grade actually won the game."),
                    },
                )


# ═══════════════════════════════════════════════════════════════════════════
# ACCURACY + RECENT GAMES TAB  (merged)
# ═══════════════════════════════════════════════════════════════════════════
with tab_accuracy:
    st.markdown("### Model Accuracy & Recent Games")
    st.caption(
        "Train the model on one season, test it on another. "
        "See overall accuracy, draft grade win rates, and a per-game breakdown with champs and grades — "
        "all evaluated out-of-sample."
    )

    acc_train_col, acc_test_col = st.columns(2)

    with acc_train_col:
        st.markdown("#### 📂 Training data")
        st.caption("Upload one or more Oracle's Elixir CSVs. The model will be fitted on the combined data.")
        train_files = st.file_uploader(
            "Training CSVs",
            type="csv",
            accept_multiple_files=True,
            key="acc_train_files",
        )
        if train_files:
            st.success(f"{len(train_files)} file(s) uploaded: {', '.join(f.name for f in train_files)}")

    with acc_test_col:
        st.markdown("#### 🎯 Test data")
        st.caption("Upload one or more CSVs. The fitted model will be evaluated against ALL games in this data.")
        test_files = st.file_uploader(
            "Test CSVs",
            type="csv",
            accept_multiple_files=True,
            key="acc_test_files",
        )
        if test_files:
            st.success(f"{len(test_files)} file(s) uploaded: {', '.join(f.name for f in test_files)}")

    # League filter for accuracy evaluation
    acc_league_opts = ALL_LEAGUES
    acc_leagues = st.multiselect(
        "Filter test data to leagues (leave empty = all)",
        options=acc_league_opts,
        default=[],
        key="acc_leagues",
        help="Optionally restrict which leagues are evaluated. Leave blank to test on everything."
    )

    st.divider()

    if st.button("🚀 Run evaluation", use_container_width=True, type="primary",
                 disabled=not (train_files and test_files)):

        if not train_files:
            st.error("Please upload at least one training CSV.")
        elif not test_files:
            st.error("Please upload at least one test CSV.")
        else:
            try:
                import tempfile, os
                from model import DataLoader

                with st.spinner("Loading and combining CSVs…"):
                    train_df    = pd.concat([pd.read_csv(f, low_memory=False) for f in train_files], ignore_index=True)
                    test_df_raw = pd.concat([pd.read_csv(f, low_memory=False) for f in test_files],  ignore_index=True)

                # Apply league filter to test data
                if acc_leagues:
                    test_df_raw = test_df_raw[test_df_raw["league"].isin(acc_leagues)]

                st.info(
                    f"Training rows: **{len(train_df):,}** across {len(train_files)} file(s) | "
                    f"Test rows: **{len(test_df_raw):,}** across {len(test_files)} file(s)"
                    + (f" · filtered to: {', '.join(acc_leagues)}" if acc_leagues else "")
                )

                with st.spinner("Fitting model on training data (~30s)…"):
                    cfg_acc = _make_config()
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                        train_df.to_csv(tmp.name, index=False)
                        tmp_path = tmp.name
                    try:
                        acc_model = build_model(tmp_path, config=cfg_acc)
                    finally:
                        os.unlink(tmp_path)

                with st.spinner("Parsing test data…"):
                    dl_acc = DataLoader(cfg_acc)
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp2:
                        test_df_raw.to_csv(tmp2.name, index=False)
                        tmp2_path = tmp2.name
                    try:
                        _, test_game_df, _, test_raw_df = dl_acc.load(tmp2_path)
                    finally:
                        os.unlink(tmp2_path)

                with st.spinner("Evaluating on ALL test games…"):
                    # Evaluate on entire test set, not just last patch
                    blue_ev = test_game_df[test_game_df["side"] == "Blue"].set_index("gameid")
                    red_ev  = test_game_df[test_game_df["side"] == "Red"].set_index("gameid")
                    common_ev = blue_ev.index.intersection(red_ev.index)

                    def _wp_grade_acc(wp):
                        if wp >= 0.65: return "S"
                        if wp >= 0.57: return "A"
                        if wp >= 0.50: return "B"
                        if wp >= 0.43: return "C"
                        return "D"

                    grade_emoji_acc = {"S": "🟡 S", "A": "🟢 A", "B": "🔵 B", "C": "⚪ C", "D": "🔴 D"}

                    y_true, y_pred = [], []
                    game_patches   = []
                    per_game_rows  = []
                    grade_wins_acc  = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0}
                    grade_total_acc = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0}

                    # Build champ lookup for test raw data
                    player_ev = test_raw_df[test_raw_df["position"] != "team"].copy()
                    champ_lkp = (
                        player_ev[player_ev["champion"].notna() & (player_ev["champion"] != "")]
                        .set_index(["gameid", "side", "position"])["champion"]
                        .to_dict()
                    ) if "position" in player_ev.columns else {}

                    def _get_ev_champs(gid, side):
                        return [champ_lkp.get((gid, side, r), "") for r in ROLES]

                    for gid in common_ev:
                        bc      = list(blue_ev.at[gid, "champs"])
                        rc      = list(red_ev.at[gid,  "champs"])
                        actual  = int(blue_ev.at[gid, "result"])
                        patch   = str(blue_ev.at[gid, "patch"]) if "patch" in blue_ev.columns else "?"
                        league  = str(blue_ev.at[gid, "league"]) if "league" in blue_ev.columns else "?"
                        b_team  = str(blue_ev.at[gid, "teamname"]) if "teamname" in blue_ev.columns else "Blue"
                        r_team  = str(red_ev.at[gid,  "teamname"]) if "teamname" in red_ev.columns else "Red"

                        try:
                            res    = acc_model.predict(bc, rc, team1_is_blue=True)
                            p_blue = float(res["win_probability"])
                        except Exception:
                            p_blue = 0.5

                        p_red   = 1.0 - p_blue
                        b_grade = _wp_grade_acc(p_blue)
                        r_grade = _wp_grade_acc(p_red)
                        correct = (p_blue >= 0.5) == (actual == 1)
                        winner  = b_team if actual == 1 else r_team

                        y_true.append(actual)
                        y_pred.append(p_blue)
                        # Store patch per game for accurate per-patch breakdown
                        game_patches.append(patch)

                        # Grade win rate: count BOTH sides independently.
                        # "Did a team with grade X win their game?"
                        # This gives meaningful C/D rows — those are teams the model
                        # predicted to lose (WP < 50%), and we check if they actually won.
                        grade_total_acc[b_grade] += 1
                        grade_wins_acc[b_grade]  += actual          # blue won?
                        grade_total_acc[r_grade] += 1
                        grade_wins_acc[r_grade]  += (1 - actual)    # red won?

                        # Champ display strings
                        b_ch = _get_ev_champs(gid, "Blue")
                        r_ch = _get_ev_champs(gid, "Red")

                        per_game_rows.append({
                            "Patch":       patch,
                            "League":      league,
                            "Blue":        b_team,
                            "🔵 Grade":    grade_emoji_acc[b_grade],
                            "Prediction":  f"🔵 {p_blue*100:.0f}% / 🔴 {p_red*100:.0f}%",
                            "🔴 Grade":    grade_emoji_acc[r_grade],
                            "Red":         r_team,
                            "Winner 🏆":   winner,
                            "Model 🎯":    "✅" if correct else "❌",
                            "🔵 Top":      b_ch[0] or "?",
                            "🔵 Jng":      b_ch[1] or "?",
                            "🔵 Mid":      b_ch[2] or "?",
                            "🔵 Bot":      b_ch[3] or "?",
                            "🔵 Sup":      b_ch[4] or "?",
                            "🔴 Top":      r_ch[0] or "?",
                            "🔴 Jng":      r_ch[1] or "?",
                            "🔴 Mid":      r_ch[2] or "?",
                            "🔴 Bot":      r_ch[3] or "?",
                            "🔴 Sup":      r_ch[4] or "?",
                        })

                    y_true = np.array(y_true, dtype=float)
                    y_pred = np.array(y_pred, dtype=float)
                    n      = len(y_true)

                    accuracy = float(np.mean((y_pred >= 0.5) == y_true))
                    baseline = float(np.mean(y_true))
                    eps      = 1e-7
                    p_clip   = np.clip(y_pred, eps, 1 - eps)
                    log_loss = float(-np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip)))
                    brier    = float(np.mean((y_pred - y_true) ** 2))

                # ── Overall metrics ──────────────────────────────────────────
                st.markdown("#### Overall results")
                st.caption(f"Evaluated on **all {n:,} games** in the test data (no patch restriction).")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Accuracy",    f"{accuracy*100:.1f}%",
                           delta=f"{(accuracy - baseline)*100:+.1f}pp vs baseline")
                mc2.metric("Baseline",    f"{baseline*100:.1f}%",
                           help="Baseline = always predict the majority outcome (blue/red side WR)")
                mc3.metric("Log Loss",    f"{log_loss:.4f}")
                mc4.metric("Brier Score", f"{brier:.4f}")

                # ── Draft grade win rates ────────────────────────────────────
                st.divider()
                st.markdown("#### Win rate by draft grade (out-of-sample)")
                st.caption("How often does the model's grade predict the actual winner on unseen data?")
                grade_order_acc = ["S", "A", "B", "C", "D"]
                grade_labels_acc = {
                    "S": "🟡 S (≥65% WP)", "A": "🟢 A (57–65%)",
                    "B": "🔵 B (50–57%)",  "C": "⚪ C (43–50%)", "D": "🔴 D (<43%)",
                }
                wr_acc_rows = []
                for g in grade_order_acc:
                    tot = grade_total_acc[g]
                    w   = grade_wins_acc[g]
                    wr_acc_rows.append({
                        "Draft Grade": grade_labels_acc[g],
                        "Games":       tot,
                        "Wins":        w,
                        "Win Rate":    round(w / tot * 100, 1) if tot > 0 else None,
                    })
                st.dataframe(
                    pd.DataFrame(wr_acc_rows), use_container_width=True, hide_index=True,
                    column_config={
                        "Draft Grade": st.column_config.TextColumn("Draft Grade", width="medium"),
                        "Games":       st.column_config.NumberColumn("Games",     width="small"),
                        "Wins":        st.column_config.NumberColumn("Wins",      width="small"),
                        "Win Rate":    st.column_config.ProgressColumn("Win Rate", min_value=0, max_value=100, format="%.1f%%",
                            help="How often teams with this draft grade actually won the game."),
                    },
                )

                # ── Accuracy by patch ────────────────────────────────────────
                st.divider()
                st.markdown("#### Accuracy by patch")
                game_patches_arr = np.array(game_patches)
                patch_rows = []
                for patch in sorted(set(game_patches)):
                    mask_p = game_patches_arr == str(patch)
                    if mask_p.sum() == 0:
                        continue
                    acc_p  = float(np.mean((y_pred[mask_p] >= 0.5) == y_true[mask_p]))
                    base_p = float(np.mean(y_true[mask_p]))
                    patch_rows.append({"Patch": str(patch), "Games": int(mask_p.sum()),
                                       "Accuracy": f"{acc_p*100:.1f}%", "Baseline": f"{base_p*100:.1f}%"})
                if patch_rows:
                    st.dataframe(pd.DataFrame(patch_rows), use_container_width=True, hide_index=True)

                # ── Calibration ──────────────────────────────────────────────
                st.divider()
                st.markdown("#### Calibration")
                bins    = np.linspace(0, 1, 11)
                bin_idx = np.clip(np.digitize(y_pred, bins) - 1, 0, 9)
                cal_rows = []
                for b in range(10):
                    mask = bin_idx == b
                    if mask.sum() == 0: continue
                    cal_rows.append({
                        "Predicted Range": f"{bins[b]:.0%}–{bins[b+1]:.0%}",
                        "Predicted Mean":  round(float(y_pred[mask].mean()), 3),
                        "Actual Win Rate": round(float(y_true[mask].mean()), 3),
                        "Games":           int(mask.sum()),
                    })
                st.dataframe(pd.DataFrame(cal_rows), use_container_width=True, hide_index=True)

                # ── Per-game table with champs + grades ──────────────────────
                st.divider()
                st.markdown("#### Per-game results")
                st.caption("Every test game — draft grades, prediction, winner, champions, and whether the model got it right.")
                pg_df = pd.DataFrame(per_game_rows)
                st.dataframe(
                    pg_df, use_container_width=True, hide_index=True,
                    column_config={
                        "Patch":      st.column_config.TextColumn("Patch",      width="small"),
                        "League":     st.column_config.TextColumn("League",     width="small"),
                        "Blue":       st.column_config.TextColumn("Blue",       width="medium"),
                        "🔵 Grade":   st.column_config.TextColumn("🔵 Grade",   width="small"),
                        "Prediction": st.column_config.TextColumn("Prediction", width="medium"),
                        "🔴 Grade":   st.column_config.TextColumn("🔴 Grade",   width="small"),
                        "Red":        st.column_config.TextColumn("Red",        width="medium"),
                        "Winner 🏆":  st.column_config.TextColumn("Winner",     width="medium"),
                        "Model 🎯":   st.column_config.TextColumn("Model",      width="small"),
                        "🔵 Top":     st.column_config.TextColumn("🔵 Top",     width="small"),
                        "🔵 Jng":     st.column_config.TextColumn("🔵 Jng",     width="small"),
                        "🔵 Mid":     st.column_config.TextColumn("🔵 Mid",     width="small"),
                        "🔵 Bot":     st.column_config.TextColumn("🔵 Bot",     width="small"),
                        "🔵 Sup":     st.column_config.TextColumn("🔵 Sup",     width="small"),
                        "🔴 Top":     st.column_config.TextColumn("🔴 Top",     width="small"),
                        "🔴 Jng":     st.column_config.TextColumn("🔴 Jng",     width="small"),
                        "🔴 Mid":     st.column_config.TextColumn("🔴 Mid",     width="small"),
                        "🔴 Bot":     st.column_config.TextColumn("🔴 Bot",     width="small"),
                        "🔴 Sup":     st.column_config.TextColumn("🔴 Sup",     width="small"),
                    },
                )

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.exception(e)


# ═══════════════════════════════════════════════════════════════════════════
# OPINION LOG TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_opinion:
    st.markdown("### Opinion Log")
    st.caption(
        "Every draft you log appears here alongside the model's prediction. "
        "Disagreements (your pick ≠ model's pick) are the most interesting rows — "
        "over time they reveal where your intuition and the model diverge."
    )

    df_op = _load_opinions()

    if df_op.empty:
        st.info("No opinions logged yet. Go to the Draft tab, enter a draft, "
                "and use 'Log your opinion' at the bottom.")
    else:
        # Summary stats
        total     = len(df_op)
        agree_n   = df_op["agree"].sum()
        disagree_n = total - agree_n

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total logged",     total)
        s2.metric("Agreed with model", agree_n)
        s3.metric("Disagreed",         disagree_n)
        s4.metric("Agreement rate",    f"{100*agree_n/max(total,1):.0f}%")

        st.markdown("#### All logged drafts")
        display_df = df_op.copy()
        display_df["agree"] = display_df["agree"].map({1: "✅ Yes", 0: "❌ No"})
        display_df["model_prob"] = (display_df["model_prob"] * 100).round(1).astype(str) + "%"
        display_df.columns = ["ID", "Time", "Blue picks", "Red picks",
                               "Model P(Blue)", "Your pick", "Confidence", "Agreed?", "Notes"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("#### Disagreements only")
        dis_df = df_op[df_op["agree"] == 0].copy()
        if dis_df.empty:
            st.info("No disagreements yet.")
        else:
            dis_df["agree"] = dis_df["agree"].map({1: "✅ Yes", 0: "❌ No"})
            dis_df["model_prob"] = (dis_df["model_prob"] * 100).round(1).astype(str) + "%"
            dis_df.columns = ["ID", "Time", "Blue picks", "Red picks",
                               "Model P(Blue)", "Your pick", "Confidence", "Agreed?", "Notes"]
            st.dataframe(dis_df, use_container_width=True, hide_index=True)

        col_dl, col_del = st.columns([3, 1])
        with col_dl:
            csv_bytes = df_op.to_csv(index=False).encode()
            st.download_button("⬇️ Export to CSV", data=csv_bytes,
                               file_name="opinions.csv", mime="text/csv")
        with col_del:
            if st.button("🗑️ Clear all logs", type="secondary"):
                con = sqlite3.connect(DB_PATH)
                con.execute("DELETE FROM opinions")
                con.commit()
                con.close()
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# RECENT T1 GAMES TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_t1games:
    _data_banner()
    m = get_model()
    st.markdown("### Recent Major League Games")
    st.caption(
        "Most recent games from major regions. Shows draft grades per side, model prediction, "
        "actual winner, and all champions by role."
    )

    raw_t1 = m._raw_df if m._raw_df is not None else pd.DataFrame()

    if raw_t1.empty:
        st.info("No data loaded.")
    else:
        _MAJOR_LEAGUES = ["LCK", "LPL", "LEC", "LCS", "LCP"]
        t1c1, t1c2 = st.columns([3, 2])
        with t1c1:
            t1_region_filter = st.multiselect(
                "Leagues", options=_MAJOR_LEAGUES, default=["LCK", "LPL"], key="t1_region_filter"
            )
        with t1c2:
            n_games_show = st.slider("Games to show", 10, 100, 30, 10, key="t1_n_games")

        raw_t1_filt = raw_t1[raw_t1["league"].isin(t1_region_filter or _MAJOR_LEAGUES)]

        team_rows   = raw_t1_filt[raw_t1_filt["position"] == "team"].copy()
        player_rows = raw_t1_filt[raw_t1_filt["position"] != "team"].copy()

        if team_rows.empty:
            st.info("No team-level data found for selected regions.")
        else:
            blue_teams = team_rows[team_rows["side"] == "Blue"].set_index("gameid")
            red_teams  = team_rows[team_rows["side"] == "Red"].set_index("gameid")

            champ_lookup = (
                player_rows[player_rows["champion"].notna() & (player_rows["champion"] != "")]
                .set_index(["gameid", "side", "position"])["champion"]
                .to_dict()
            )

            def _get_champs_t1(gameid, side):
                return [champ_lookup.get((gameid, side, r), "") for r in ROLES]

            def _wp_grade_t1(wp: float) -> str:
                if wp >= 0.65: return "S"
                if wp >= 0.57: return "A"
                if wp >= 0.50: return "B"
                if wp >= 0.43: return "C"
                return "D"

            grade_emoji = {"S": "🟡 S", "A": "🟢 A", "B": "🔵 B", "C": "⚪ C", "D": "🔴 D", "?": "❔ ?"}

            common_games = blue_teams.index.intersection(red_teams.index)

            # Sort by date descending
            game_dates = {}
            if "date" in blue_teams.columns:
                for gid in common_games:
                    try:
                        game_dates[gid] = pd.to_datetime(blue_teams.at[gid, "date"])
                    except Exception:
                        game_dates[gid] = pd.Timestamp.min
                common_games = sorted(common_games, key=lambda g: game_dates.get(g, pd.Timestamp.min), reverse=True)

            common_games = list(common_games)[:n_games_show]

            game_display_rows = []
            for gid in common_games:
                b_team   = str(blue_teams.at[gid, "teamname"]) if "teamname" in blue_teams.columns else "Blue"
                r_team   = str(red_teams.at[gid,  "teamname"]) if "teamname" in red_teams.columns else "Red"
                b_result = int(blue_teams.at[gid, "result"])   if "result"   in blue_teams.columns else 0
                r_result = int(red_teams.at[gid,  "result"])   if "result"   in red_teams.columns else 0
                league   = str(blue_teams.at[gid, "league"])   if "league"   in blue_teams.columns else "?"
                date_str = game_dates[gid].strftime("%Y-%m-%d") if gid in game_dates and game_dates[gid] != pd.Timestamp.min else "?"

                b_champs = _get_champs_t1(gid, "Blue")
                r_champs = _get_champs_t1(gid, "Red")

                # Model prediction — only run if both sides have full champ data
                b_valid = all(b_champs)
                r_valid = all(r_champs)
                if b_valid and r_valid:
                    try:
                        pred      = m.predict(b_champs, r_champs, team1_is_blue=True)
                        b_wp      = float(pred["win_probability"])
                        r_wp      = 1.0 - b_wp
                        b_grade   = grade_emoji[_wp_grade_t1(b_wp)]
                        r_grade   = grade_emoji[_wp_grade_t1(r_wp)]
                        pred_str  = f"🔵 {b_wp*100:.0f}% / 🔴 {r_wp*100:.0f}%"
                        model_fav = b_team if b_wp >= 0.5 else r_team
                        correct   = "✅" if (b_wp >= 0.5 and b_result == 1) or (b_wp < 0.5 and r_result == 1) else "❌"
                    except Exception:
                        b_grade = r_grade = grade_emoji["?"]
                        pred_str  = "—"
                        model_fav = "—"
                        correct   = "—"
                else:
                    b_grade = r_grade = grade_emoji["?"]
                    pred_str  = "—"
                    model_fav = "—"
                    correct   = "—"

                winner = b_team if b_result == 1 else r_team

                game_display_rows.append({
                    "Date":        date_str,
                    "League":      league,
                    "Blue Team":   b_team,
                    "🔵 Draft":    b_grade,
                    "Prediction":  pred_str,
                    "🔴 Draft":    r_grade,
                    "Red Team":    r_team,
                    "Winner 🏆":   winner,
                    "Model 🎯":    correct,
                    "🔵 Top":      b_champs[0] or "?",
                    "🔵 Jng":      b_champs[1] or "?",
                    "🔵 Mid":      b_champs[2] or "?",
                    "🔵 Bot":      b_champs[3] or "?",
                    "🔵 Sup":      b_champs[4] or "?",
                    "🔴 Top":      r_champs[0] or "?",
                    "🔴 Jng":      r_champs[1] or "?",
                    "🔴 Mid":      r_champs[2] or "?",
                    "🔴 Bot":      r_champs[3] or "?",
                    "🔴 Sup":      r_champs[4] or "?",
                })

            if not game_display_rows:
                st.info("No games found for selected regions.")
            else:
                games_df = pd.DataFrame(game_display_rows)
                st.markdown(f"Showing **{len(games_df)}** most recent games from **{', '.join(t1_region_filter or _MAJOR_LEAGUES)}**")
                st.dataframe(
                    games_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Date":       st.column_config.TextColumn("Date",       width="small"),
                        "League":     st.column_config.TextColumn("League",     width="small"),
                        "Blue Team":  st.column_config.TextColumn("Blue",       width="medium"),
                        "🔵 Draft":   st.column_config.TextColumn("🔵 Grade",   width="small",
                            help="Model draft grade for blue side (S/A/B/C/D by predicted WP)"),
                        "Prediction": st.column_config.TextColumn("Prediction", width="medium",
                            help="Model's predicted win % for blue / red"),
                        "🔴 Draft":   st.column_config.TextColumn("🔴 Grade",   width="small",
                            help="Model draft grade for red side"),
                        "Red Team":   st.column_config.TextColumn("Red",        width="medium"),
                        "Winner 🏆":  st.column_config.TextColumn("Winner",     width="medium"),
                        "Model 🎯":   st.column_config.TextColumn("Model",      width="small",
                            help="✅ = model picked the correct winner, ❌ = upset"),
                        "🔵 Top":     st.column_config.TextColumn("🔵 Top",     width="small"),
                        "🔵 Jng":     st.column_config.TextColumn("🔵 Jng",     width="small"),
                        "🔵 Mid":     st.column_config.TextColumn("🔵 Mid",     width="small"),
                        "🔵 Bot":     st.column_config.TextColumn("🔵 Bot",     width="small"),
                        "🔵 Sup":     st.column_config.TextColumn("🔵 Sup",     width="small"),
                        "🔴 Top":     st.column_config.TextColumn("🔴 Top",     width="small"),
                        "🔴 Jng":     st.column_config.TextColumn("🔴 Jng",     width="small"),
                        "🔴 Mid":     st.column_config.TextColumn("🔴 Mid",     width="small"),
                        "🔴 Bot":     st.column_config.TextColumn("🔴 Bot",     width="small"),
                        "🔴 Sup":     st.column_config.TextColumn("🔴 Sup",     width="small"),
                    },
                )


# ═══════════════════════════════════════════════════════════════════════════
# TEAM SCOUT TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_scout:
    _data_banner()
    m = get_model()
    st.markdown("### Team Scout")
    st.caption(
        "Per-team draft tendencies: champion pool by role, most common picks, "
        "bans, first picks, and side win rates."
    )

    raw_scout = m._raw_df if m._raw_df is not None else pd.DataFrame()

    if raw_scout.empty:
        st.info("No data loaded.")
    else:
        # ── Team selector ─────────────────────────────────────────────────────
        # Build team list from player rows (teamname column)
        player_scout = raw_scout[raw_scout["position"] != "team"].copy()
        team_scout_rows = raw_scout[raw_scout["position"] == "team"].copy()

        all_teams_scout = sorted(
            t for t in team_scout_rows["teamname"].dropna().unique()
            if isinstance(t, str) and t.strip()
        ) if "teamname" in team_scout_rows.columns else []

        sc1, sc2 = st.columns([3, 2])
        with sc1:
            selected_team = st.selectbox(
                "Select team", options=all_teams_scout,
                key="scout_team",
            )
        with sc2:
            # Year / patch filter
            scout_years = sorted(
                raw_scout["year"].dropna().unique().astype(int).tolist()
            ) if "year" in raw_scout.columns else []
            scout_year_opts = ["All data"] + [str(y) for y in scout_years]
            scout_year = st.selectbox("Year filter", scout_year_opts, key="scout_year")

        if not selected_team:
            st.info("Select a team above.")
        else:
            # Apply year filter
            if scout_year != "All data" and "year" in raw_scout.columns:
                raw_scout_f = raw_scout[raw_scout["year"].astype(int) == int(scout_year)]
            else:
                raw_scout_f = raw_scout

            team_games_rows = raw_scout_f[
                (raw_scout_f["position"] == "team") &
                (raw_scout_f["teamname"] == selected_team)
            ].copy()

            team_player_rows = raw_scout_f[
                (raw_scout_f["position"] != "team") &
                (raw_scout_f["teamname"] == selected_team)
            ].copy()

            if team_games_rows.empty:
                st.warning(f"No games found for **{selected_team}** with current filter.")
            else:
                # De-duplicate: Oracle's Elixir can have duplicate team rows when
                # multiple CSVs are concatenated. One row per (gameid, side) is correct.
                team_games_dedup = team_games_rows.drop_duplicates(subset=["gameid", "side"])
                blue_games   = team_games_dedup[team_games_dedup["side"] == "Blue"]
                red_games    = team_games_dedup[team_games_dedup["side"] == "Red"]
                blue_wins    = int(blue_games["result"].sum()) if "result" in blue_games.columns else 0
                red_wins     = int(red_games["result"].sum())  if "result" in red_games.columns else 0
                blue_total   = len(blue_games)
                red_total    = len(red_games)
                total_games  = blue_total + red_total
                total_wins   = blue_wins + red_wins

                # ── Top-level metrics ─────────────────────────────────────────
                st.markdown(f"#### {selected_team} — {scout_year}")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total games", total_games)
                m2.metric("Overall W/L", f"{total_wins}–{total_games - total_wins}")
                m3.metric("Win rate", f"{100*total_wins/max(total_games,1):.1f}%")
                m4.metric(
                    "Blue side WR",
                    f"{100*blue_wins/max(blue_total,1):.1f}% ({blue_wins}/{blue_total})"
                )
                m5.metric(
                    "Red side WR",
                    f"{100*red_wins/max(red_total,1):.1f}% ({red_wins}/{red_total})"
                )

                st.divider()

                scout_sub1, scout_sub2, scout_sub3 = st.tabs(
                    ["🗡️ Champion Pool by Role", "🚫 Bans", "🥇 First Picks"]
                )

                # ── Champion pool by role ─────────────────────────────────────
                with scout_sub1:
                    st.caption(
                        "Champions played by this team in each role, sorted by games played. "
                        "Win rate shown is raw (not BT-adjusted) for this team specifically."
                    )
                    if team_player_rows.empty or "position" not in team_player_rows.columns:
                        st.info("No player-level data found.")
                    else:
                        pool_roles = st.tabs([ROLE_LABELS[r] for r in ROLES])
                        for role, rtab in zip(ROLES, pool_roles):
                            with rtab:
                                role_rows = team_player_rows[
                                    team_player_rows["position"] == role
                                ].copy()
                                if role_rows.empty:
                                    st.info(f"No {ROLE_LABELS[role]} data.")
                                    continue

                                pool_stats = []
                                for champ, grp in role_rows.groupby("champion"):
                                    if not isinstance(champ, str) or not champ:
                                        continue
                                    n   = len(grp)
                                    wins = int(grp["result"].sum()) if "result" in grp.columns else 0
                                    wr  = wins / n if n > 0 else 0.0
                                    pool_stats.append({
                                        "Champion": champ,
                                        "Games":    n,
                                        "Wins":     wins,
                                        "Win%":     round(wr * 100, 1),
                                        "Pick%":    round(100 * n / max(total_games, 1), 1),
                                    })

                                if not pool_stats:
                                    st.info("No data.")
                                    continue

                                pool_df = (
                                    pd.DataFrame(pool_stats)
                                    .sort_values("Games", ascending=False)
                                    .reset_index(drop=True)
                                )
                                def _wr_grade(w):
                                    if w >= 70: return "🟢 S"
                                    if w >= 55: return "🟡 A"
                                    if w >= 45: return "⚪ B"
                                    if w >= 30: return "🟠 C"
                                    return "🔴 D"
                                pool_df["Grade"] = pool_df["Win%"].apply(_wr_grade)
                                pool_df.index += 1
                                st.dataframe(
                                    pool_df[["Champion", "Grade", "Win%", "Games", "Wins", "Pick%"]],
                                    use_container_width=True,
                                    column_config={
                                        "Champion": st.column_config.TextColumn("Champion", width="medium"),
                                        "Grade":    st.column_config.TextColumn("Grade",    width="small",
                                            help="🟢S ≥70% · 🟡A ≥55% · ⚪B ≥45% · 🟠C ≥30% · 🔴D <30%"),
                                        "Win%":     st.column_config.ProgressColumn("Win%",
                                            min_value=0, max_value=100, format="%.1f%%"),
                                        "Games":    st.column_config.NumberColumn("Games",  width="small"),
                                        "Wins":     st.column_config.NumberColumn("Wins",   width="small"),
                                        "Pick%":    st.column_config.ProgressColumn("Pick% of games",
                                            min_value=0, max_value=100, format="%.1f%%",
                                            help="How often this champion appeared in this role as a fraction of team's total games."),
                                    },
                                )

                # ── Bans ─────────────────────────────────────────────────────
                with scout_sub2:
                    st.caption(
                        "Champions banned by this team, sorted by frequency. "
                        "Shows how often each champion is prioritised away."
                    )
                    ban_cols_scout = [
                        c for c in raw_scout_f.columns if c.startswith("ban") and c[3:].isdigit()
                    ]
                    if not ban_cols_scout:
                        st.info("No ban columns found in data.")
                    else:
                        # team-level rows for this team
                        team_ban_rows = raw_scout_f[
                            (raw_scout_f["position"] == "team") &
                            (raw_scout_f["teamname"] == selected_team)
                        ]
                        ban_counts: dict[str, int] = {}
                        for col in ban_cols_scout:
                            for champ in team_ban_rows[col].dropna():
                                if isinstance(champ, str) and champ.strip():
                                    ban_counts[champ] = ban_counts.get(champ, 0) + 1

                        if not ban_counts:
                            st.info("No ban data found.")
                        else:
                            ban_df = (
                                pd.DataFrame(
                                    [{"Champion": c, "Bans": n,
                                      "Ban%": round(100 * n / max(total_games, 1), 1)}
                                     for c, n in ban_counts.items()]
                                )
                                .sort_values("Bans", ascending=False)
                                .reset_index(drop=True)
                            )
                            ban_df.index += 1

                            bc1, bc2 = st.columns([2, 3])
                            with bc1:
                                top_n_bans = st.slider("Show top N bans", 5, min(50, len(ban_df)), 15, 5, key="scout_ban_n")
                            st.dataframe(
                                ban_df.head(top_n_bans),
                                use_container_width=True,
                                column_config={
                                    "Champion": st.column_config.TextColumn("Champion", width="medium"),
                                    "Bans":     st.column_config.NumberColumn("Times Banned", width="small"),
                                    "Ban%":     st.column_config.ProgressColumn("Ban% of games", min_value=0, max_value=100, format="%.1f%%",
                                        help="Percentage of this team's games where they banned this champion."),
                                },
                            )

                # ── First picks ───────────────────────────────────────────────
                with scout_sub3:
                    st.caption(
                        "Champions this team picked with their first pick slot (pick1 on blue, "
                        "or first red-side pick). Also shows first-pick win rate."
                    )
                    # Use pick1 column from team-level rows for blue side first picks.
                    # For red side, pick1 is the first red pick after blue's first two.
                    # We use the raw pick1 column which Oracle's Elixir populates per team.
                    team_tl_scout = raw_scout_f[
                        (raw_scout_f["position"] == "team") &
                        (raw_scout_f["teamname"] == selected_team)
                    ].copy()

                    fp_counts: dict[str, dict] = {}  # champ -> {games, wins, blue, red}

                    # Oracle's Elixir: pick1 = the team's first selection in draft order
                    if "pick1" in team_tl_scout.columns:
                        for _, row in team_tl_scout.iterrows():
                            champ = row.get("pick1")
                            if not isinstance(champ, str) or not champ.strip():
                                continue
                            result = int(row["result"]) if "result" in row and pd.notna(row["result"]) else 0
                            side   = str(row.get("side", ""))
                            if champ not in fp_counts:
                                fp_counts[champ] = {"games": 0, "wins": 0, "blue": 0, "red": 0}
                            fp_counts[champ]["games"] += 1
                            fp_counts[champ]["wins"]  += result
                            if side == "Blue":
                                fp_counts[champ]["blue"] += 1
                            else:
                                fp_counts[champ]["red"]  += 1

                    if not fp_counts:
                        st.info("No first-pick data found (pick1 column missing or empty).")
                    else:
                        def _fp_wr_grade(w):
                            if w >= 70: return "🟢 S"
                            if w >= 55: return "🟡 A"
                            if w >= 45: return "⚪ B"
                            if w >= 30: return "🟠 C"
                            return "🔴 D"
                        fp_df = pd.DataFrame([
                            {
                                "Champion": champ,
                                "First Picks": d["games"],
                                "Wins":        d["wins"],
                                "Win%":        round(100 * d["wins"] / max(d["games"], 1), 1),
                                "FP%":         round(100 * d["games"] / max(total_games, 1), 1),
                                "Blue":        d["blue"],
                                "Red":         d["red"],
                            }
                            for champ, d in fp_counts.items()
                        ]).sort_values("First Picks", ascending=False).reset_index(drop=True)
                        fp_df["Grade"] = fp_df["Win%"].apply(_fp_wr_grade)
                        fp_df.index += 1

                        st.dataframe(
                            fp_df[["Champion", "Grade", "Win%", "First Picks", "FP%", "Wins", "Blue", "Red"]],
                            use_container_width=True,
                            column_config={
                                "Champion":    st.column_config.TextColumn("Champion",     width="medium"),
                                "Grade":       st.column_config.TextColumn("Grade",        width="small",
                                    help="🟢S ≥70% · 🟡A ≥55% · ⚪B ≥45% · 🟠C ≥30% · 🔴D <30%"),
                                "Win%":        st.column_config.ProgressColumn("Win%", min_value=0, max_value=100, format="%.1f%%"),
                                "First Picks": st.column_config.NumberColumn("1st Picks",  width="small"),
                                "FP%":         st.column_config.ProgressColumn("FP% of games", min_value=0, max_value=100, format="%.1f%%",
                                    help="How often this team first-picked this champion as a % of total games."),
                                "Wins":        st.column_config.NumberColumn("Wins",        width="small"),
                                "Blue":        st.column_config.NumberColumn("Blue side",  width="small"),
                                "Red":         st.column_config.NumberColumn("Red side",   width="small"),
                            },
                        )


# ═══════════════════════════════════════════════════════════════════════════
# SETTINGS TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_settings:
    st.markdown("### Settings")
    st.caption("Weight changes apply immediately. League/min-games changes trigger a rebuild (~30s).")

    st.divider()

    # ── Prediction weights (manual override) ────────────────────────────────
    st.markdown("#### Prediction weights")
    st.caption(
        "Defaults set from rolling cross-validation: synergy is the strongest signal, "
        "base and counter roughly equal, fight small but positive. Adjust based on your own read."
    )
    st.info(
        "**Formula:** `logit = w_base·ΔBase + w_syn·ΔSynergy + w_ctr·Counter + w_fgt·ΔFight`  \n"
        "`P(win) = sigmoid(8 × logit)`  \n"
        "Where each Δ = team1 score − team2 score for that component.",
        icon="📐"
    )

    # ── Per-component enable toggles + sliders ───────────────────────────────
    components = [
        ("w_base",    "sl_base",  "en_base",    "⚖️ Base strength",  0.25,
         "Weight on ΔBase = mean(adj_WR_team1_by_role) − mean(adj_WR_team2_by_role). "
         "Adj WR is Bayesian-shrunk toward role avg and corrected for opponent team strength."),
        ("w_synergy", "sl_syn",   "en_synergy", "🔗 Synergy",        0.40,
         "Weight on ΔSynergy = mean_pair_WR_delta(team1) − mean_pair_WR_delta(team2). "
         "Pair WR delta = how much better (or worse) two champs win together vs. their individual rates. "
         "Bot+Sup and Mid+Jng pairs also incorporate @15 gold diff as a lane signal."),
        ("w_counter", "sl_ctr",   "en_counter", "⚔️ Counter",        0.25,
         "Weight on Counter = mean across roles of: direct_WR(team1_champ vs team2_champ) − direct_WR(team2_champ vs team1_champ). "
         "Blends same-role WR delta and @15 gold diff signal."),
        ("w_fight",   "sl_fgt",   "en_fight",   "💥 Teamfight@22",   0.10,
         "Weight on ΔFight = mean(gold22_norm_team1) − mean(gold22_norm_team2). "
         "gold22_norm = champion's avg gold at ~22 min ÷ role avg gold at 22 min."),
    ]

    # Initialize enable states
    for _, _, en_key, _, default_w, _ in components:
        if en_key not in st.session_state:
            st.session_state[en_key] = True

    wc1, wc2, wc3, wc4 = st.columns(4)
    for col, (w_key, sl_key, en_key, label, default_w, help_txt) in zip(
        [wc1, wc2, wc3, wc4], components
    ):
        with col:
            enabled = st.toggle(label, value=st.session_state[en_key], key=en_key)
            if enabled:
                st.session_state[w_key] = st.slider(
                    "Weight", 0.05, 1.0,
                    float(st.session_state.get(w_key, default_w)),
                    0.05, key=sl_key, help=help_txt,
                    label_visibility="collapsed",
                )
            else:
                st.session_state[w_key] = 0.0
                st.caption("Disabled (weight = 0)")

    # Auto-normalize enabled weights so they sum to 1.0
    active_keys  = [w for w, _, en, _, _, _ in components if st.session_state.get(en, True)]
    raw_total    = sum(st.session_state[w] for w in active_keys)
    if raw_total > 0 and abs(raw_total - 1.0) > 0.01:
        scale = 1.0 / raw_total
        for w in active_keys:
            st.session_state[w] = round(st.session_state[w] * scale, 4)
        st.caption(
            f"⚖️ Weights auto-normalized to sum to 1.0 "
            f"(×{scale:.3f}): "
            + "  ".join(f"{w.replace('w_','').title()}={st.session_state[w]:.2f}" for w in active_keys)
        )
    elif raw_total == 0:
        st.warning("All components disabled — prediction will always return 50%.")
    else:
        total = sum(st.session_state[w_key] for w_key, *_ in components)
        if abs(total - 1.0) > 0.01:
            st.warning(f"Weights sum to {total:.2f} — should be 1.00.")

    st.markdown("#### Counter sub-weights")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.session_state.w_ctr_wr = st.slider(
            "Counter WR vs @15", 0.0, 1.0, st.session_state.w_ctr_wr, 0.05, key="sl_cwr",
            help="Blends two counter signals: WR delta when champ_A faces champ_B in lane, "
                 "and gold/CS/XP diff at 15 min in those same matchups. "
                 "@15 captures early dominance that doesn't always show up in final WR."
        )
        st.caption(f"WR={st.session_state.w_ctr_wr:.2f}  @15={round(1-st.session_state.w_ctr_wr,2):.2f}")
    with sc2:
        st.session_state.w_syn_wr = st.slider(
            "Synergy WR vs @15", 0.0, 1.0, st.session_state.w_syn_wr, 0.05, key="sl_swr",
            help="For Bot+Sup and Mid+Jng lane pairs only: blends pair WR delta "
                 "with combined @15 gold+CS+XP diff when those two play together. "
                 "@15 signal only available when both champs appear in the same game with @15 data."
        )
        st.caption(f"WR={st.session_state.w_syn_wr:.2f}  @15={round(1-st.session_state.w_syn_wr,2):.2f}")

    # ── Matchup overrides ────────────────────────────────────────────────────
    st.markdown("#### Manual matchup overrides")
    all_champs_list = sorted(get_model().base_model.known_champions())

    with st.form("add_override", clear_on_submit=True):
        oc1, oc2, oc3, oc4 = st.columns([3, 3, 2, 1])
        with oc1: ov_a = st.selectbox("Champion A (favoured)", all_champs_list, key="ov_a")
        with oc2: ov_b = st.selectbox("Champion B (disfavoured)", all_champs_list, key="ov_b")
        with oc3: ov_d = st.number_input("Delta", min_value=-1.0, max_value=1.0, value=0.05, step=0.01, key="ov_d")
        with oc4:
            st.markdown("<br>", unsafe_allow_html=True)
            add_ov = st.form_submit_button("Add")
        if add_ov and ov_a != ov_b:
            st.session_state.overrides.append({"champ_a": ov_a, "champ_b": ov_b, "delta": ov_d})

    if st.session_state.overrides:
        st.markdown("**Active overrides:**")
        for idx, ov in enumerate(st.session_state.overrides):
            r1, r2 = st.columns([5, 1])
            with r1: st.write(f"{ov['champ_a']} vs {ov['champ_b']} → Δ={ov['delta']:+.2f}")
            with r2:
                if st.button("Remove", key=f"rm_ov_{idx}"):
                    st.session_state.overrides.pop(idx)
                    st.rerun()

    # ── Data filters ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Data filters")
    st.caption("⚠️ Changes here rebuild the model.")

    # ── Patch decay toggles ──────────────────────────────────────────────────
    st.markdown("**Data scope & patch decay**")
    st.caption(
        "Base strength always decays. Synergy and counter matchups change slowly — "
        "disabling decay pools all patches equally, giving more pair data at the cost of ignoring meta drift."
    )
    # Toggles are rendered in the Draft tab (to avoid duplicate widget key errors).
    # Settings shows the current state read-only.
    dc0, dc1, dc2 = st.columns(3)
    with dc0:
        val = st.session_state.get("use_all_leagues", False)
        st.markdown(f"**All leagues:** {'✅ ON' if val else '⬜ OFF'}")
        st.caption("Toggle in ⚔️ Draft tab → Model weights expander")
    with dc1:
        val = st.session_state.get("synergy_patch_decay", False)
        st.markdown(f"**Synergy patch decay:** {'✅ ON' if val else '⬜ OFF'}")
        st.caption("Toggle in ⚔️ Draft tab → Model weights expander")
    with dc2:
        val = st.session_state.get("counter_patch_decay", False)
        st.markdown(f"**Counter patch decay:** {'✅ ON' if val else '⬜ OFF'}")
        st.caption("Toggle in ⚔️ Draft tab → Model weights expander")

    st.divider()

    # ── Training data source ─────────────────────────────────────────────────
    st.markdown("**Training data source**")
    current_label = st.session_state.get("main_data_label", Path(DATA_PATH).name)
    st.info(f"Currently loaded: **{current_label}**", icon="📂")
    uploaded_mains = st.file_uploader(
        "Upload training CSV(s) — upload multiple to combine (e.g. 2025 + 2026)",
        type="csv",
        accept_multiple_files=True,
        key="main_csv_upload",
        help="Upload one or more Oracle's Elixir CSVs. Multiple files are concatenated before fitting.",
    )
    if uploaded_mains:
        import tempfile, os
        dfs = [pd.read_csv(f, low_memory=False, encoding='latin-1') for f in uploaded_mains]
        combined = pd.concat(dfs, ignore_index=True)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, dir=tempfile.gettempdir()) as tmp:
            combined.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
        new_label = " + ".join(f.name for f in uploaded_mains)
        if tmp_path != st.session_state.get("main_data_path") or new_label != st.session_state.get("main_data_label"):
            st.session_state.main_data_path  = tmp_path
            st.session_state.main_data_label = new_label
            st.session_state.model           = None
            st.session_state.config_hash     = None
            st.success(f"✅ Training data updated to **{new_label}** ({len(combined):,} rows). Model will rebuild on next interaction.")

    new_min = st.slider("Min games per champion", 5, 50, st.session_state.min_games, 5, key="sl_ming")
    if new_min != st.session_state.min_games:
        st.session_state.min_games   = new_min
        st.session_state.model       = None
        st.session_state.config_hash = None

    st.markdown("**League selection**")
    league_mode = st.radio(
        "Preset",
        ["Major + International (default)", "All regions", "Custom"],
        key="league_mode",
        horizontal=True,
    )

    if league_mode == "Major + International (default)":
        new_leagues = MAJOR_DEFAULT
    elif league_mode == "All regions":
        new_leagues = ALL_LEAGUES
    else:
        new_leagues = st.multiselect(
            "Select leagues",
            options=ALL_LEAGUES,
            default=st.session_state.selected_leagues,
            key="ms_leagues",
        )

    if set(new_leagues) != set(st.session_state.selected_leagues):
        st.session_state.selected_leagues = new_leagues
        st.session_state.model            = None
        st.session_state.config_hash      = None

    if st.session_state.model is None:
        st.info("Model will rebuild on next interaction.")

    # ═══════════════════════════════════════════════════════════════════════
    # MODEL MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("#### Model management")

    # ── Current model status ─────────────────────────────────────────────
    model_loaded = st.session_state.model is not None
    data_label   = st.session_state.get("main_data_label", Path(DATA_PATH).name)

    if model_loaded:
        m_status = st.session_state.model
        n_champs  = len(m_status.base_model.known_champions())
        n_teams   = len(m_status.bt_model.scores_)
        n_pairs   = len(m_status.synergy_model.pair_wr_)
        n_matchups= len(m_status.counter_model.direct_wr_)
        st.success(
            f"✅ Model loaded — **{data_label}** · "
            f"{n_champs} champions · {n_teams} teams · "
            f"{n_pairs} synergy pairs · {n_matchups} counter matchups",
            icon="🤖",
        )
    else:
        st.warning("⚠️ No model loaded — will build on next interaction.", icon="⚙️")

    # ── Force rebuild ────────────────────────────────────────────────────
    st.markdown("**Force rebuild**")
    st.caption("Clears the cached model and rebuilds from scratch using current settings and data.")
    if st.button("🔄 Force rebuild now", type="primary", use_container_width=True):
        _load_model.clear()
        st.session_state.model       = None
        st.session_state.config_hash = None
        st.info("Cache cleared — rebuilding model now…")
        get_model()
        st.success("✅ Model rebuilt successfully.")
        st.rerun()

    st.divider()

    # ── Save model ───────────────────────────────────────────────────────
    st.markdown("**Save model to disk**")
    st.caption("Serialises the fitted model to a `.pkl` file so you can reload it instantly without rebuilding.")

    save_col1, save_col2 = st.columns([3, 1])
    with save_col1:
        save_name = st.text_input(
            "Filename",
            value="model_cache.pkl",
            key="save_filename",
            label_visibility="collapsed",
            placeholder="model_cache.pkl",
        )
    with save_col2:
        if st.button("💾 Save", use_container_width=True, disabled=not model_loaded):
            try:
                save_path = str(Path(__file__).parent / save_name)
                st.session_state.model.save(save_path)
                st.success(f"✅ Saved to `{save_name}`")
            except Exception as e:
                st.error(f"Save failed: {e}")

    st.divider()

    # ── Load model ───────────────────────────────────────────────────────
    st.markdown("**Load model from disk**")
    st.caption("Load a previously saved `.pkl` file — skips the 30s rebuild entirely.")

    load_col1, load_col2 = st.columns([3, 1])
    with load_col1:
        load_name = st.text_input(
            "Filename to load",
            value="model_cache.pkl",
            key="load_filename",
            label_visibility="collapsed",
            placeholder="model_cache.pkl",
        )
    with load_col2:
        if st.button("📂 Load", use_container_width=True):
            try:
                load_path = str(Path(__file__).parent / load_name)
                if not Path(load_path).exists():
                    st.error(f"File not found: `{load_name}`")
                else:
                    loaded = DraftModel.load(load_path)
                    st.session_state.model       = loaded
                    st.session_state.config_hash = "loaded_from_disk"
                    st.success(f"✅ Loaded `{load_name}`")
                    st.rerun()
            except Exception as e:
                st.error(f"Load failed: {e}")

    # Show existing .pkl files in the app directory for convenience
    pkl_files = sorted(Path(__file__).parent.glob("*.pkl"))
    if pkl_files:
        st.caption("Saved models found: " + "  ·  ".join(f.name for f in pkl_files))
