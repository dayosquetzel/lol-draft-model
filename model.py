from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAJOR_LEAGUES   = {"LCK", "LPL", "LEC", "LCS", "LCP"}
INTERNATIONAL   = {"MSI", "Worlds", "EWC"}
DEFAULT_LEAGUES = MAJOR_LEAGUES | INTERNATIONAL   # FST excluded by default

LEAGUE_REGION_TIER: dict[str, int] = {
    # T1 — top two regions globally
    "LCK": 1, "LPL": 1,
    # T2 — strong western regions
    "LEC": 2, "LCS": 2, "LCP": 2,
    # T3 — strong regional leagues
    "CBLOL": 3, "VCS": 3, "LJL": 3, "TCL": 3,
    # T4 — all others
    "LCKC": 4, "LFL": 4, "PRM": 4, "NLC": 4,
    "AL": 4, "EBL": 4, "LPLOL": 4, "LRN": 4, "LRS": 4,
    "RL": 4, "ROL": 4, "HLL": 4, "HW": 4, "LIT": 4, "LES": 4,
    "LAS": 4, "NACL": 4, "CCWS": 4, "CD": 4, "EM": 4, "FST": 4,
}

TIER_BASE_WEIGHT: dict[int, float] = {1: 1.00, 2: 0.80, 3: 0.50, 4: 0.25}



BT_TIER_THRESHOLDS = {"S": 1.40, "A": 1.15, "B": 0.85, "C": 0.60, "D": 0.0}




# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ModelConfig:
    MIN_CHAMP_GAMES:   int   = 15
    MIN_SYNERGY_GAMES: int   = 5
    MIN_COUNTER_GAMES: int   = 5

    LEAGUES: Optional[list[str]] = None
    PATCHES: Optional[list[float]] = None

    # Prediction weights — set from rolling cross-validation stability analysis.
    # Synergy is the strongest signal (~50%), base and counter roughly equal (~25% each),
    # fight contributes a small but directionally correct amount.
    WEIGHT_BASE:      float = 0.25
    WEIGHT_SYNERGY:   float = 0.40
    WEIGHT_COUNTER:   float = 0.25
    WEIGHT_FIGHT:     float = 0.10
    WEIGHT_INTERCEPT: float = 0.0

    WEIGHT_COUNTER_WR:  float = 0.70
    WEIGHT_COUNTER_15:  float = 0.30

    WEIGHT_SYNERGY_WR: float = 0.60
    WEIGHT_SYNERGY_15: float = 0.40

    MATCHUP_OVERRIDES: dict = {}

    PATCH_DECAY:          float = 0.60
    TEAM_STRENGTH_ALPHA:  float = 0.70
    UPSET_MULTIPLIER:     float = 1.5

    MIN_ROLE_GAMES: int   = 5
    MIN_ROLE_SHARE: float = 0.20

    MIN_TEAM_GAMES: int = 5

    ROLES:      list[str] = ["top", "jng", "mid", "bot", "sup"]
    CACHE_PATH: str       = "model_cache.pkl"


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------

class DataLoader:
    def __init__(self, config: ModelConfig):
        self.config = config

    @staticmethod
    def _norm(v) -> str:
        return v.strip() if isinstance(v, str) else ""

    def load(self, path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print(f"[DataLoader] Loading {path} ...")
        raw = pd.read_csv(path, low_memory=False)
        print(f"[DataLoader] Raw shape: {raw.shape}")

        raw["champion"] = raw["champion"].apply(self._norm)

        if self.config.LEAGUES:
            raw = raw[raw["league"].isin(self.config.LEAGUES)]
            print(f"[DataLoader] League filter → {raw[raw['position'] != 'team']['gameid'].nunique()} games")

        if self.config.PATCHES:
            raw = raw[raw["patch"].isin(self.config.PATCHES)]
            print(f"[DataLoader] Patch filter → {raw[raw['position'] != 'team']['gameid'].nunique()} games")

        patches_sorted = sorted(raw["patch"].dropna().unique())
        max_rank       = len(patches_sorted)
        patch_rank     = {p: i + 1 for i, p in enumerate(patches_sorted)}
        decay          = self.config.PATCH_DECAY
        patch_weight   = {
            p: float(np.exp(-decay * (max_rank - patch_rank[p])))
            for p in patches_sorted
        }
        raw["patch_weight"] = raw["patch"].map(patch_weight).fillna(1.0)
        print(f"[DataLoader] Patch weights: { {p: round(w,3) for p,w in patch_weight.items()} }")

        keep = ["gameid", "side", "position", "champion", "result",
                "patch", "patch_weight", "league", "date"]
        keep = [c for c in keep if c in raw.columns]
        player_df = raw[raw["position"] != "team"].copy()[keep]
        player_df = player_df[player_df["champion"] != ""]
        player_df = player_df.dropna(subset=["champion", "result", "position"])
        player_df["result"] = player_df["result"].astype(int)
        player_df = player_df[player_df["position"].isin(set(self.config.ROLES))]

        ext_cols = ["gameid", "side", "position", "champion", "result",
                    "patch", "patch_weight", "league", "teamname"]
        ext_cols = [c for c in ext_cols if c in raw.columns]
        pext = raw[raw["position"] != "team"][ext_cols].dropna(subset=["champion"]).copy()
        pext["champion"] = pext["champion"].apply(self._norm)

        team_cols = ["gameid", "side", "teamname", "firstPick",
                     "pick1", "pick2", "pick3", "pick4", "pick5",
                     "ban1", "ban2", "ban3", "ban4", "ban5",
                     "result", "patch", "patch_weight", "league"]
        team_cols  = [c for c in team_cols if c in raw.columns]
        team_level = raw[raw["position"] == "team"][team_cols].copy()

        def _pivot(grp):
            role_map = grp.set_index("position")["champion"].to_dict()
            champs = [role_map.get(r, "") for r in self.config.ROLES]
            return pd.Series({
                "champs":       champs,
                "result":       int(grp["result"].iloc[0]),
                "patch":        grp["patch"].iloc[0],
                "patch_weight": grp["patch_weight"].iloc[0],
                "league":       grp["league"].iloc[0],
                "teamname":     str(grp["teamname"].iloc[0]) if "teamname" in grp.columns and pd.notna(grp["teamname"].iloc[0]) else "",
            })

        game_df = (
            pext.groupby(["gameid", "side"], group_keys=False)
            .apply(_pivot)
            .reset_index()
        )
        game_df = game_df[
            game_df["champs"].apply(lambda c: all(isinstance(x, str) and x for x in c))
        ]

        pick_ban_cols = (
            ["gameid", "side", "pick1", "pick2", "pick3", "pick4", "pick5", "firstPick"]
            + [c for c in ["ban1","ban2","ban3","ban4","ban5"] if c in team_level.columns]
        )
        pick_info = (
            team_level[[c for c in pick_ban_cols if c in team_level.columns]]
            .drop_duplicates(subset=["gameid", "side"])
        )
        game_df = game_df.merge(pick_info, on=["gameid", "side"], how="left")

        print(f"[DataLoader] player_df: {player_df.shape} | "
              f"game_df: {game_df.shape} | "
              f"games: {game_df['gameid'].nunique()} | "
              f"champions: {player_df['champion'].nunique()}")
        return player_df, game_df, team_level, raw


# ---------------------------------------------------------------------------
# Team Strength Model  (Regional-tier-weighted Bradley-Terry)
# ---------------------------------------------------------------------------

class TeamStrengthModel:
    """
    Iterative Bradley-Terry MLE for team strength estimation.

    Game weight = (tier_winner + tier_loser) / 2 using TIER_BASE_WEIGHT.
    This means LCK vs LCK carries full weight (1.0), LCK vs LEC carries 0.90,
    LEC vs LEC carries 0.80, and so on. International results between strong
    teams carry the most discriminating power.

    Within each game the BT update is further scaled by a strength-of-schedule
    factor (log(1 + s_w + s_l) normalised) so games between top teams carry
    more discriminating power than wins over weak opposition.
    """

    BT_FLOOR   = 0.05
    BT_ITERS   = 200
    BT_EPSILON = 1e-7

    def __init__(self, config: ModelConfig):
        self.config         = config
        self.scores_:       dict[str, float] = {}
        self.tiers_:        dict[str, str]   = {}
        self.games_:        dict[str, int]   = {}
        self.wl_:           dict[str, tuple[int, int]] = {}
        self.league_games_: dict[str, dict[str, int]] = {}

    def _game_bt_weight(self, league: str, winner: str, loser: str) -> float:
        t_w = LEAGUE_REGION_TIER.get(str(league), 4)
        # For international, use the higher of the two teams' home region tiers
        # (approximated by looking at the league; for MSI/Worlds/EWC use a fixed T1 weight)
        if str(league) in INTERNATIONAL:
            return 1.0
        return (TIER_BASE_WEIGHT.get(t_w, 0.25))

    def fit(self, team_level: pd.DataFrame) -> "TeamStrengthModel":
        blue = team_level[team_level["side"] == "Blue"].set_index("gameid")[
            ["teamname", "result", "league"]
        ]
        red = team_level[team_level["side"] == "Red"].set_index("gameid")[
            ["teamname", "result", "league"]
        ]
        common = blue.index.intersection(red.index)

        matchups = pd.DataFrame({
            "winner": np.where(
                blue.loc[common, "result"] == 1,
                blue.loc[common, "teamname"],
                red.loc[common, "teamname"],
            ),
            "loser": np.where(
                blue.loc[common, "result"] == 0,
                blue.loc[common, "teamname"],
                red.loc[common, "teamname"],
            ),
            "league": blue.loc[common, "league"].values,
        })

        for _, row in matchups.iterrows():
            for team in [row["winner"], row["loser"]]:
                if team not in self.league_games_:
                    self.league_games_[team] = {}
                lg = row["league"]
                self.league_games_[team][lg] = self.league_games_[team].get(lg, 0) + 1

        matchups["bt_weight"] = matchups.apply(
            lambda r: self._game_bt_weight(r["league"], r["winner"], r["loser"]), axis=1
        )

        team_counts = pd.concat([matchups["winner"], matchups["loser"]]).value_counts()
        valid_teams = set(team_counts[team_counts >= self.config.MIN_TEAM_GAMES].index)
        matchups = matchups[
            matchups["winner"].isin(valid_teams) & matchups["loser"].isin(valid_teams)
        ].reset_index(drop=True)

        teams = sorted(set(matchups["winner"]) | set(matchups["loser"]))
        idx   = {t: i for i, t in enumerate(teams)}
        n     = len(teams)
        s     = np.ones(n, dtype=float)

        w_wins = np.zeros(n)
        for _, row in matchups.iterrows():
            w_wins[idx[row["winner"]]] += row["bt_weight"]

        prev = s.copy()
        for iteration in range(self.BT_ITERS):
            sos_raw = np.array([
                np.log1p(s[idx[row["winner"]]] + s[idx[row["loser"]]])
                for _, row in matchups.iterrows()
            ])
            sos_mean = sos_raw.mean()
            sos_mul  = sos_raw / sos_mean if sos_mean > 0 else np.ones(len(sos_raw))

            new_denom = np.zeros(n)
            eff_wins  = np.zeros(n)
            for i, (_, row) in enumerate(matchups.iterrows()):
                w_idx = idx[row["winner"]]
                l_idx = idx[row["loser"]]
                bw    = row["bt_weight"] * sos_mul[i]
                denom = s[w_idx] + s[l_idx]
                if denom > 0:
                    new_denom[w_idx] += bw * s[w_idx] / denom
                    new_denom[l_idx] += bw * s[l_idx] / denom
                eff_wins[w_idx] += bw

            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.where(new_denom > 0, eff_wins / new_denom, self.BT_FLOOR)

            s = np.maximum(s, self.BT_FLOOR)
            s = s / s.mean()

            if np.max(np.abs(s - prev)) < self.BT_EPSILON:
                print(f"[BradleyTerry] Converged in {iteration+1} iterations.")
                break
            prev = s.copy()

        self.scores_ = {t: float(s[idx[t]]) for t in teams}
        self.games_  = {t: int(team_counts.get(t, 0)) for t in teams}

        win_counts  = matchups["winner"].value_counts().to_dict()
        loss_counts = matchups["loser"].value_counts().to_dict()
        self.wl_ = {t: (win_counts.get(t, 0), loss_counts.get(t, 0)) for t in teams}

        for team, score in self.scores_.items():
            for tier, threshold in BT_TIER_THRESHOLDS.items():
                if score >= threshold:
                    self.tiers_[team] = tier
                    break

        top5 = {t: round(v, 3) for t, v in sorted(
            self.scores_.items(), key=lambda x: -x[1])[:5]}
        print(f"[BradleyTerry] {len(self.scores_)} teams. Top 5: {top5}")
        return self

    def strength(self, team: str) -> float:
        return self.scores_.get(team, 1.0)

    def expected_win_prob(self, team_a: str, team_b: str) -> float:
        sa = self.strength(team_a)
        sb = self.strength(team_b)
        return sa / (sa + sb)

    def tier(self, team: str) -> str:
        return self.tiers_.get(team, "?")

    def tier_table(self) -> pd.DataFrame:
        rows = [
            {
                "Team":           t,
                "BT Score":       round(s, 3),
                "Tier":           self.tiers_.get(t, "?"),
                "W":              self.wl_.get(t, (0, 0))[0],
                "L":              self.wl_.get(t, (0, 0))[1],
                "Games":          self.games_.get(t, 0),
                "Region":         max(
                    self.league_games_.get(t, {"?": 0}).items(),
                    key=lambda x: x[1], default=("?", 0)
                )[0],
            }
            for t, s in sorted(self.scores_.items(), key=lambda x: -x[1])
        ]
        return pd.DataFrame(rows)

    def upset_weight(self, winner: str, loser: str) -> float:
        p_win = self.expected_win_prob(winner, loser)
        if p_win < 0.5:
            upset_magnitude = (0.5 - p_win) * 2.0
            return 1.0 + upset_magnitude * (self.config.UPSET_MULTIPLIER - 1.0)
        return 1.0


# ---------------------------------------------------------------------------
# Base Strength Model
# ---------------------------------------------------------------------------

class BaseStrengthModel:
    """
    Role-aware champion win rates adjusted for team strength and upset weight.
    Bayesian shrinkage toward role average for thin samples.
    """

    def __init__(self, config: ModelConfig, bt_model: TeamStrengthModel):
        self.config    = config
        self.bt        = bt_model
        self.role_winrates_:   dict[tuple[str, str], float] = {}
        self.role_counts_:     dict[tuple[str, str], int]   = {}
        self.global_winrates_: dict[str, float] = {}
        self.global_counts_:   dict[str, int]   = {}
        self._role_avg:        dict[str, float] = {}
        self._global_avg:      float            = 0.5

    def fit(self, player_df: pd.DataFrame, game_df: pd.DataFrame) -> "BaseStrengthModel":
        alpha = self.config.TEAM_STRENGTH_ALPHA

        team_side_map: dict[tuple[str, str], tuple[str, str]] = {}
        gd_blue = game_df[game_df["side"] == "Blue"].set_index("gameid")[["teamname"]]
        gd_red  = game_df[game_df["side"] == "Red"].set_index("gameid")[["teamname"]]
        for gid in gd_blue.index.intersection(gd_red.index):
            b_team = str(gd_blue.at[gid, "teamname"])
            r_team = str(gd_red.at[gid, "teamname"])
            team_side_map[(gid, "Blue")] = (b_team, r_team)
            team_side_map[(gid, "Red")]  = (r_team, b_team)

        role_win_w:   dict[tuple, float] = {}
        role_total_w: dict[tuple, float] = {}
        role_raw_n:   dict[tuple, int]   = {}
        glob_win_w:   dict[str, float]   = {}
        glob_total_w: dict[str, float]   = {}

        for _, row in player_df.iterrows():
            champ  = row["champion"]
            role   = row["position"]
            result = int(row["result"])
            gameid = row["gameid"]
            side   = row["side"]

            team_info = team_side_map.get((gameid, side))
            if team_info:
                my_team, opp_team = team_info
                winner, loser = (my_team, opp_team) if result == 1 else (opp_team, my_team)
                p_expected = self.bt.expected_win_prob(winner, loser)
                surprise   = 1.0 - p_expected
                team_adj   = 1.0 - alpha + alpha * (surprise if result == 1 else (1.0 - surprise))
                upset_w    = self.bt.upset_weight(winner, loser) if result == 1 else 1.0
            else:
                team_adj = 1.0
                upset_w  = 1.0

            total_weight = team_adj * upset_w
            k = (champ, role)
            role_win_w[k]   = role_win_w.get(k, 0.0)   + result * total_weight
            role_total_w[k] = role_total_w.get(k, 0.0) + total_weight
            role_raw_n[k]   = role_raw_n.get(k, 0)     + 1
            glob_win_w[champ]   = glob_win_w.get(champ, 0.0)   + result * total_weight
            glob_total_w[champ] = glob_total_w.get(champ, 0.0) + total_weight

        SHRINK_K = float(self.config.MIN_CHAMP_GAMES)

        role_avg_win_w:   dict[str, float] = {}
        role_avg_total_w: dict[str, float] = {}
        for k, tw in role_total_w.items():
            _, role = k
            role_avg_win_w[role]   = role_avg_win_w.get(role, 0.0)   + role_win_w.get(k, 0.0)
            role_avg_total_w[role] = role_avg_total_w.get(role, 0.0) + tw

        role_avg: dict[str, float] = {
            r: (role_avg_win_w[r] + 0.5) / (role_avg_total_w[r] + 1.0)
            for r in role_avg_win_w if role_avg_total_w.get(r, 0) > 0
        }
        global_avg = (
            (sum(role_avg_win_w.values()) + 0.5) /
            (sum(role_avg_total_w.values()) + 1.0)
        )

        for k, raw_n in role_raw_n.items():
            champ, role = k
            tw = role_total_w.get(k, 0.0)
            ww = role_win_w.get(k, 0.0)
            if tw <= 0:
                continue
            observed_wr = (ww + 0.5) / (tw + 1.0)
            avg  = role_avg.get(role, global_avg)
            conf = raw_n / (raw_n + SHRINK_K)
            self.role_winrates_[k] = conf * observed_wr + (1.0 - conf) * avg
            self.role_counts_[k]   = raw_n

        for champ in glob_win_w:
            raw_n_glob = sum(role_raw_n.get((champ, r), 0) for r in self.config.ROLES)
            tw = glob_total_w.get(champ, 0.0)
            ww = glob_win_w.get(champ, 0.0)
            if tw <= 0:
                continue
            observed_wr = (ww + 0.5) / (tw + 1.0)
            conf = raw_n_glob / (raw_n_glob + SHRINK_K)
            self.global_winrates_[champ] = conf * observed_wr + (1.0 - conf) * global_avg
            self.global_counts_[champ]   = raw_n_glob

        self._role_avg   = role_avg
        self._global_avg = global_avg

        thin = sum(1 for n in role_raw_n.values() if n < self.config.MIN_CHAMP_GAMES)
        print(f"[BaseStrength] {len(self.role_winrates_)} role-pairs ({thin} thin, shrunk toward role avg)")
        return self

    def champion_winrate(self, champ: str, role: Optional[str] = None) -> float:
        if role:
            v = self.role_winrates_.get((champ, role))
            if v is not None:
                return v
        v = self.global_winrates_.get(champ)
        if v is not None:
            return v
        if role and hasattr(self, "_role_avg"):
            return self._role_avg.get(role, 0.5)
        return 0.5

    def team_score(self, champs: list[str], roles: Optional[list[str]] = None) -> float:
        if roles is None:
            roles = self.config.ROLES
        return float(np.mean([self.champion_winrate(c, r) for c, r in zip(champs, roles)]))

    def known_champions(self) -> list[str]:
        return sorted(
            set(c for c, _ in self.role_winrates_) |
            set(self.global_winrates_.keys())
        )

    def summary(self, top_n: int = 20) -> pd.DataFrame:
        rows = [
            {"champion": champ, "adj_wr": round(wr * 100, 1), "games": self.global_counts_.get(champ, 0)}
            for champ, wr in sorted(self.global_winrates_.items(), key=lambda x: -x[1])[:top_n]
        ]
        return pd.DataFrame(rows)

    def _role_qualifies(self, champ: str, role: str) -> bool:
        """
        Returns True if this champion has enough games in this role to be treated
        as a legitimate role pick (not a one-off flex or small-sample outlier).

        Requires:
          - MIN_ROLE_GAMES raw games in this role
          - At least MIN_ROLE_SHARE fraction of their total games in this role
            (prevents a top-laner with 1 mid game from appearing as a mid laner)
        """
        role_n = self.role_counts_.get((champ, role), 0)
        total  = self.global_counts_.get(champ, 0)
        if role_n < self.config.MIN_ROLE_GAMES:
            return False
        if total > 0 and role_n / total < self.config.MIN_ROLE_SHARE:
            return False
        return True





# ---------------------------------------------------------------------------
# Synergy Model
# ---------------------------------------------------------------------------

class SynergyModel:
    # Canonical lane pairs — stored as sorted tuples so (bot,sup) and (sup,bot) are the same bucket.
    # mid+jng = rotational synergy; bot+sup = lane synergy. Both get @15 data if available.
    LANE_PAIRS: set = {("bot", "sup"), ("mid", "jng")}

    def __init__(self, config: ModelConfig, base: BaseStrengthModel, bt_model: TeamStrengthModel):
        self.config    = config
        self.base      = base
        self.bt        = bt_model
        self.pair_wr_:     dict[tuple, float] = {}
        self.pair_15_:     dict[tuple, float] = {}
        self.pair_counts_: dict[tuple, int]   = {}

    @staticmethod
    def _pair_key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def fit(self, game_df: pd.DataFrame, raw_df: pd.DataFrame) -> "SynergyModel":
        roles = self.config.ROLES

        bt_map: dict[tuple, tuple[str, str]] = {}
        blue = game_df[game_df["side"] == "Blue"].set_index("gameid")[["teamname", "result"]]
        red  = game_df[game_df["side"] == "Red"].set_index("gameid")[["teamname", "result"]]
        for gid in blue.index.intersection(red.index):
            b_won  = int(blue.at[gid, "result"]) == 1
            winner = str(blue.at[gid, "teamname"]) if b_won else str(red.at[gid, "teamname"])
            loser  = str(red.at[gid, "teamname"])  if b_won else str(blue.at[gid, "teamname"])
            bt_map[(gid, "Blue")] = (winner, loser)
            bt_map[(gid, "Red")]  = (winner, loser)

        at15_cols = ["golddiffat15", "csdiffat15", "xpdiffat15"]
        has_15 = all(c in raw_df.columns for c in at15_cols)
        at15_lookup: dict[tuple, tuple[float, float, float]] = {}
        if has_15:
            p15 = raw_df[raw_df["position"] != "team"][
                ["gameid", "side", "position"] + at15_cols
            ].copy()
            for col in at15_cols:
                p15[col] = pd.to_numeric(p15[col], errors="coerce")
            p15 = p15.dropna(subset=at15_cols)
            n_with_15 = p15["gameid"].nunique()
            n_total   = raw_df[raw_df["position"] != "team"]["gameid"].nunique()
            print(f"[Synergy] @15 data available for {n_with_15}/{n_total} games "
                  f"({100*n_with_15/max(n_total,1):.0f}%)")
            for _, r in p15.iterrows():
                at15_lookup[(r["gameid"], r["side"], r["position"])] = (
                    float(r["golddiffat15"]),
                    float(r["csdiffat15"]),
                    float(r["xpdiffat15"]),
                )

        pw_sums:  dict[tuple, float] = {}
        pw_wins:  dict[tuple, float] = {}
        raw_n:    dict[tuple, int]   = {}
        at15_acc: dict[tuple, list]  = {}

        for _, row in game_df.iterrows():
            champs = row["champs"]
            if not isinstance(champs, list) or len(champs) != 5:
                continue
            if not all(isinstance(c, str) and c for c in champs):
                continue

            result = int(row["result"])
            pw     = float(row.get("patch_weight", 1.0))
            gid    = row["gameid"]
            side   = row["side"]

            bt_pair = bt_map.get((gid, side))
            upset_w = (self.bt.upset_weight(*bt_pair) if bt_pair and result == 1 else 1.0)
            eff_pw  = pw * upset_w

            for i in range(5):
                for j in range(i + 1, 5):
                    ca, cb = champs[i], champs[j]
                    k = self._pair_key(ca, cb)
                    pw_sums[k] = pw_sums.get(k, 0.0) + eff_pw
                    pw_wins[k] = pw_wins.get(k, 0.0) + result * eff_pw
                    raw_n[k]   = raw_n.get(k, 0) + 1

                    role_i, role_j = roles[i], roles[j]
                    if tuple(sorted([role_i, role_j])) in self.LANE_PAIRS and has_15:
                        s15_i = at15_lookup.get((gid, side, role_i))
                        s15_j = at15_lookup.get((gid, side, role_j))
                        if s15_i and s15_j:
                            at15_acc.setdefault(k, []).append((
                                s15_i[0] + s15_j[0],
                                s15_i[1] + s15_j[1],
                                s15_i[2] + s15_j[2],
                                eff_pw,
                            ))

        a = 1.0
        for k, n in raw_n.items():
            if n < self.config.MIN_SYNERGY_GAMES:
                continue
            tw  = pw_sums[k]
            ww  = pw_wins[k]
            swr = (ww + a * 0.5) / (tw + a)
            ca, cb = k
            exp = (self.base.champion_winrate(ca) + self.base.champion_winrate(cb)) / 2.0
            self.pair_wr_[k]     = swr - exp
            self.pair_counts_[k] = n

        raw_15: dict[tuple, float] = {}
        for k, vals in at15_acc.items():
            if k not in self.pair_wr_:
                continue
            ws    = [v[3] for v in vals]
            gd15  = float(np.average([v[0] for v in vals], weights=ws))
            csd15 = float(np.average([v[1] for v in vals], weights=ws))
            xpd15 = float(np.average([v[2] for v in vals], weights=ws))
            raw_15[k] = gd15 / 500.0 + csd15 / 10.0 + xpd15 / 300.0

        if raw_15:
            vals_arr = np.array(list(raw_15.values()))
            mu, sd   = vals_arr.mean(), vals_arr.std()
            for k, v in raw_15.items():
                self.pair_15_[k] = float((v - mu) / sd) if sd > 0 else 0.0

        n_lane = sum(1 for k in self.pair_wr_ if k in self.pair_15_)
        print(f"[Synergy] {len(self.pair_wr_)} pairs ({n_lane} with @15 lane signal)")
        return self

    def pair_synergy(self, champ_a: str, champ_b: str, role_a: str = "", role_b: str = "") -> float:
        k = self._pair_key(champ_a, champ_b)
        wr_delta = self.pair_wr_.get(k, 0.0)
        lane_key = tuple(sorted([role_a, role_b]))
        if lane_key in self.LANE_PAIRS and k in self.pair_15_:
            w_wr = self.config.WEIGHT_SYNERGY_WR
            w_15 = self.config.WEIGHT_SYNERGY_15
            return w_wr * wr_delta + w_15 * self.pair_15_[k] * 0.02
        return wr_delta

    def synergy_score(self, champs: list[str]) -> float:
        roles = self.config.ROLES
        if len(champs) < 2:
            return 0.0
        vals = []
        for i in range(len(champs)):
            for j in range(i + 1, len(champs)):
                if not (isinstance(champs[i], str) and champs[i]):
                    continue
                if not (isinstance(champs[j], str) and champs[j]):
                    continue
                ra = roles[i] if i < len(roles) else ""
                rb = roles[j] if j < len(roles) else ""
                vals.append(self.pair_synergy(champs[i], champs[j], ra, rb))
        return float(np.mean(vals)) if vals else 0.0

    def synergy_detail(self, champs: list[str]) -> list[dict]:
        roles = self.config.ROLES
        out = []
        for i in range(len(champs)):
            for j in range(i + 1, len(champs)):
                ca, cb = champs[i], champs[j]
                if not (isinstance(ca, str) and ca and isinstance(cb, str) and cb):
                    continue
                ra = roles[i] if i < len(roles) else ""
                rb = roles[j] if j < len(roles) else ""
                k  = self._pair_key(ca, cb)
                out.append({
                    "pair":     f"{ca} + {cb}",
                    "roles":    f"{ra}/{rb}",
                    "wr_delta": round(self.pair_wr_.get(k, 0.0) * 100, 2),
                    "lane_15":  round(self.pair_15_.get(k, 0.0), 3) if k in self.pair_15_ else None,
                    "combined": round(self.pair_synergy(ca, cb, ra, rb) * 100, 2),
                    "games":    self.pair_counts_.get(k, 0),
                })
        return sorted(out, key=lambda x: -abs(x["combined"]))

    def top_pairs(self, top_n: int = 10) -> pd.DataFrame:
        rows = [
            (a, b, round(d * 100, 2), self.pair_counts_.get((a, b), 0))
            for (a, b), d in self.pair_wr_.items()
        ]
        df = pd.DataFrame(rows, columns=["champ_a", "champ_b", "wr_delta_%", "games"])
        return df.sort_values("wr_delta_%", ascending=False).head(top_n)


# ---------------------------------------------------------------------------
# Counter Model
# ---------------------------------------------------------------------------

class CounterModel:
    def __init__(self, config: ModelConfig, base: BaseStrengthModel):
        self.config = config
        self.base   = base
        self.direct_wr_:     dict[tuple, float] = {}
        self.direct_15_:     dict[tuple, float] = {}
        self.direct_counts_: dict[tuple, int]   = {}

    def fit(self, player_df: pd.DataFrame, raw_df: pd.DataFrame) -> "CounterModel":
        roles = self.config.ROLES
        min_g = self.config.MIN_COUNTER_GAMES

        at15_cols = ["golddiffat15", "csdiffat15", "xpdiffat15"]
        has_15    = all(c in raw_df.columns for c in at15_cols)
        at15_lookup: dict[tuple, tuple] = {}
        if has_15:
            p15 = raw_df[raw_df["position"] != "team"][
                ["gameid", "side", "position"] + at15_cols
            ].copy()
            for col in at15_cols:
                p15[col] = pd.to_numeric(p15[col], errors="coerce")
            p15 = p15.dropna(subset=at15_cols)
            for _, r in p15.iterrows():
                at15_lookup[(r["gameid"], r["side"], r["position"])] = (
                    float(r["golddiffat15"]),
                    float(r["csdiffat15"]),
                    float(r["xpdiffat15"]),
                )

        lookup: dict[tuple, tuple] = {}
        for _, row in player_df.iterrows():
            lookup[(row["gameid"], row["side"], row["position"])] = (
                row["champion"], int(row["result"]), float(row.get("patch_weight", 1.0))
            )

        game_ids = player_df["gameid"].unique()

        dir_win_w:   dict[tuple, float] = {}
        dir_total_w: dict[tuple, float] = {}
        dir_raw_n:   dict[tuple, int]   = {}
        dir_15_acc:  dict[tuple, list]  = {}

        for gid in game_ids:
            blue_champs: dict[str, tuple] = {}
            red_champs:  dict[str, tuple] = {}
            for role in roles:
                bk = lookup.get((gid, "Blue", role))
                rk = lookup.get((gid, "Red",  role))
                if bk: blue_champs[role] = bk
                if rk: red_champs[role]  = rk

            if not blue_champs or not red_champs:
                continue

            for role_a, (champ_a, result_a, pw_a) in blue_champs.items():
                for role_b, (champ_b, result_b, pw_b) in red_champs.items():
                    if not (champ_a and champ_b):
                        continue
                    if role_a != role_b:
                        continue
                    pw = (pw_a + pw_b) / 2.0
                    k_a = (champ_a, role_a, champ_b)
                    k_b = (champ_b, role_b, champ_a)
                    dir_win_w[k_a]   = dir_win_w.get(k_a, 0.0)   + result_a * pw
                    dir_total_w[k_a] = dir_total_w.get(k_a, 0.0) + pw
                    dir_raw_n[k_a]   = dir_raw_n.get(k_a, 0)     + 1
                    dir_win_w[k_b]   = dir_win_w.get(k_b, 0.0)   + result_b * pw
                    dir_total_w[k_b] = dir_total_w.get(k_b, 0.0) + pw
                    dir_raw_n[k_b]   = dir_raw_n.get(k_b, 0)     + 1

                    if has_15:
                        s15 = at15_lookup.get((gid, "Blue", role_a))
                        if s15:
                            dir_15_acc.setdefault(k_a, []).append((*s15, pw))
                            dir_15_acc.setdefault(k_b, []).append(
                                (-s15[0], -s15[1], -s15[2], pw)
                            )

        a = 1.0
        for k, n in dir_raw_n.items():
            if n < min_g:
                continue
            champ_a, role, champ_b = k
            tw  = dir_total_w[k]
            ww  = dir_win_w[k]
            swr = (ww + a * 0.5) / (tw + a)
            baseline = self.base.champion_winrate(champ_a, role)
            self.direct_wr_[k]     = swr - baseline
            self.direct_counts_[k] = n

        raw_15: dict[tuple, float] = {}
        for k, vals in dir_15_acc.items():
            if k not in self.direct_wr_:
                continue
            ws    = [v[3] for v in vals]
            gd15  = float(np.average([v[0] for v in vals], weights=ws))
            csd15 = float(np.average([v[1] for v in vals], weights=ws))
            xpd15 = float(np.average([v[2] for v in vals], weights=ws))
            raw_15[k] = gd15 / 500.0 + csd15 / 10.0 + xpd15 / 300.0

        if raw_15:
            arr     = np.array(list(raw_15.values()))
            mu, sd  = arr.mean(), arr.std()
            for k, v in raw_15.items():
                self.direct_15_[k] = float((v - mu) / sd) if sd > 0 else 0.0

        print(f"[CounterModel] {len(self.direct_wr_)} direct matchups ({len(self.direct_15_)} with @15)")
        return self

    def direct_score(self, champ_a: str, role: str, champ_b: str) -> float:
        k        = (champ_a, role, champ_b)
        override = self.config.MATCHUP_OVERRIDES.get((champ_a, champ_b))
        wr_delta = float(override) if override is not None else self.direct_wr_.get(k, 0.0)
        if k in self.direct_15_:
            return (self.config.WEIGHT_COUNTER_WR * wr_delta +
                    self.config.WEIGHT_COUNTER_15  * self.direct_15_[k] * 0.02)
        return wr_delta

    def direct_detail(self, champ_a: str, role: str, champ_b: str) -> dict:
        k        = (champ_a, role, champ_b)
        override = self.config.MATCHUP_OVERRIDES.get((champ_a, champ_b))
        wr_delta = float(override) if override is not None else self.direct_wr_.get(k, 0.0)
        return {
            "wr_delta":      round(wr_delta * 100, 2),
            "lane_15_score": round(self.direct_15_.get(k, 0.0), 3),
            "combined":      round(self.direct_score(champ_a, role, champ_b) * 100, 2),
            "games":         self.direct_counts_.get(k, 0),
            "overridden":    override is not None,
        }

    def counter_score(self, team1: list[str], team2: list[str]) -> tuple[float, list[dict]]:
        roles = self.config.ROLES
        direct_vals: list[float] = []
        breakdown:   list[dict]  = []

        for i, (c1, r1) in enumerate(zip(team1, roles)):
            if not (isinstance(c1, str) and c1):
                continue
            c2 = team2[i] if i < len(team2) else ""
            r2 = roles[i]

            d_fwd = self.direct_score(c1, r1, c2) if c2 else 0.0
            d_rev = self.direct_score(c2, r2, c1) if c2 else 0.0
            net_direct = d_fwd - d_rev
            direct_vals.append(net_direct)

            detail_fwd = self.direct_detail(c1, r1, c2) if c2 else {}
            detail_rev = self.direct_detail(c2, r2, c1) if c2 else {}

            breakdown.append({
                "role":          r1,
                "team1_champ":   c1,
                "team2_champ":   c2,
                "direct_net":    round(net_direct * 100, 2),
                "t1_wr_delta":   detail_fwd.get("wr_delta", 0.0),
                "t1_lane_15":    detail_fwd.get("lane_15_score"),
                "t2_wr_delta":   detail_rev.get("wr_delta", 0.0),
                "t2_lane_15":    detail_rev.get("lane_15_score"),
                "t1_overridden": detail_fwd.get("overridden", False),
            })

        total = float(np.mean(direct_vals)) if direct_vals else 0.0
        return total, breakdown

    def best_counters(self, champ: str, top_n: int = 10) -> pd.DataFrame:
        rows = []
        for (ca, role, cb), wr_d in self.direct_wr_.items():
            if cb != champ:
                continue
            rows.append({
                "counter_champ": ca,
                "role":          role,
                "wr_delta_%":    round(wr_d * 100, 2),
                "lane_15_score": round(self.direct_15_.get((ca, role, cb), 0.0), 3),
                "combined_%":    round(self.direct_score(ca, role, cb) * 100, 2),
                "games":         self.direct_counts_.get((ca, role, cb), 0),
            })
        return (
            pd.DataFrame(rows)
            .sort_values("combined_%", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Teamfight Model  (champion-level early vs late power differential)
# ---------------------------------------------------------------------------

class TeamfightModel:
    """
    Predicts the expected fight outcome at ~20-23 minutes based on each team's
    gold position at that timestamp.

    PIPELINE:
    1. Gold at 22 min — each champion has a baseline expected gold at ~22 min
       from historical data (patch-weighted average across all games they appear
       in that role). This captures item spike timing naturally: a champion that
       historically has 8,500g at 22 min is on a different item spike than one
       at 7,200g.

    2. Fight strength score — gold_at_22 / role_avg_gold_at_22.
       Normalising by role average removes the structural difference between
       roles (supports always have less gold than carries). A score > 1.0 means
       this champion tends to be ahead of where their role typically is at 22
       min. A score < 1.0 means behind the curve.

    3. Team fight strength = mean of 5 role-normalised scores.
       Net = team1 - team2. Positive = team1 is expected to be ahead in gold
       and items at the fight window.

    Fallbacks: champions with no gold data fall back to 1.0 (role average).
    """

    MIN_GAMES: int = 5

    def __init__(self, config: ModelConfig):
        self.config = config
        # (champ, role) → patch-weighted avg gold at 22 min
        self.avg_gold22_:      dict[tuple, float] = {}
        # role → avg gold at 22 min (for normalisation)
        self._role_avg_gold22: dict[str, float]   = {}

    def fit(self, player_df: pd.DataFrame, raw_df: pd.DataFrame) -> "TeamfightModel":
        has_g20 = "goldat20" in raw_df.columns
        has_g25 = "goldat25" in raw_df.columns

        if not has_g20 and not has_g25:
            print("[TeamfightModel] WARNING: no gold columns — fight scores will be neutral.")
            return self

        needed = ["gameid", "side", "position"]
        if has_g20: needed.append("goldat20")
        if has_g25: needed.append("goldat25")

        gold_raw = raw_df[raw_df["position"] != "team"][needed].copy()
        for col in needed[3:]:
            gold_raw[col] = pd.to_numeric(gold_raw[col], errors="coerce")

        # Interpolate to 22 min (between 20 and 25 if both available)
        if has_g20 and has_g25:
            gold_raw["gold22"] = (gold_raw["goldat20"] +
                                  0.4 * (gold_raw["goldat25"] - gold_raw["goldat20"]))
        elif has_g20:
            gold_raw["gold22"] = gold_raw["goldat20"]
        else:
            gold_raw["gold22"] = gold_raw["goldat25"]

        gold_raw = gold_raw.dropna(subset=["gold22"])

        merged = player_df[
            ["gameid", "side", "position", "champion", "patch_weight"]
        ].merge(gold_raw[["gameid", "side", "position", "gold22"]],
                on=["gameid", "side", "position"], how="inner")

        # ── Per (champ, role): patch-weighted avg gold at 22 ────────────────
        g_acc: dict[tuple, list] = {}
        for _, row in merged.iterrows():
            k  = (row["champion"], row["position"])
            pw = float(row["patch_weight"])
            g22 = float(row["gold22"])
            if np.isnan(g22):
                continue
            g_acc.setdefault(k, []).append((g22, pw))

        for k, vals in g_acc.items():
            if len(vals) < self.MIN_GAMES:
                continue
            ws = [w for _, w in vals]
            self.avg_gold22_[k] = float(np.average([v for v, _ in vals], weights=ws))

        # ── Role avg gold at 22 for normalisation ───────────────────────────
        role_g:  dict[str, list] = {}
        for (champ, role), g in self.avg_gold22_.items():
            role_g.setdefault(role, []).append(g)
        for role, vals in role_g.items():
            self._role_avg_gold22[role] = float(np.mean(vals))

        n_profiles = len(self.avg_gold22_)
        if n_profiles:
            print(f"[TeamfightModel] {n_profiles} champ-role gold@22 profiles")
            print(f"  Role baselines: { {r: round(g,0) for r,g in self._role_avg_gold22.items()} }")
        else:
            print("[TeamfightModel] WARNING: no gold profiles built.")
        return self

    def _expected_gold22(self, champ: str, role: str) -> float:
        """Patch-weighted average gold at ~22 min for this champion in this role."""
        return self.avg_gold22_.get(
            (champ, role),
            self._role_avg_gold22.get(role, 7500.0)
        )

    def fight_strength(self, champ: str, role: str) -> float:
        """
        Gold@22 normalised by role average.
        1.0 = exactly on the role's typical gold curve.
        >1.0 = ahead of where this role usually is → stronger at 22 min.
        <1.0 = behind the curve → weaker at 22 min.
        """
        expected = self._expected_gold22(champ, role)
        baseline = self._role_avg_gold22.get(role, expected)
        if baseline <= 0:
            return 1.0
        return float(np.clip(expected / baseline, 0.7, 1.4))

    def team_fight_strength(
        self,
        team:  list[str],
        roles: list[str],
    ) -> tuple[float, list[dict]]:
        contribs = []
        for champ, role in zip(team, roles):
            if not (isinstance(champ, str) and champ):
                contribs.append({
                    "champ": champ, "role": role,
                    "gold22": self._role_avg_gold22.get(role, 7500.0),
                    "fight_strength": 1.0,
                    "has_data": False,
                })
                continue
            fs  = self.fight_strength(champ, role)
            g22 = self._expected_gold22(champ, role)
            contribs.append({
                "champ":          champ,
                "role":           role,
                "gold22":         round(g22, 0),
                "fight_strength": round(fs, 4),
                "has_data":       (champ, role) in self.avg_gold22_,
            })
        score = float(np.mean([c["fight_strength"] for c in contribs]))
        return score, contribs

    def teamfight_score(
        self,
        team1: list[str],
        team2: list[str],
        roles: Optional[list[str]] = None,
    ) -> tuple[float, dict]:
        if roles is None:
            roles = self.config.ROLES
        t1_score, t1_detail = self.team_fight_strength(team1, roles)
        t2_score, t2_detail = self.team_fight_strength(team2, roles)
        net = t1_score - t2_score
        return float(net), {
            "team1_fight_strength": round(t1_score, 4),
            "team2_fight_strength": round(t2_score, 4),
            "net_advantage":        round(net, 4),
            "team1_detail":         t1_detail,
            "team2_detail":         t2_detail,
        }

    def gold_profile(self, champ: str, role: str) -> dict:
        return {
            "avg_gold_at_22":      round(self.avg_gold22_.get((champ, role), 0.0), 0),
            "role_avg_gold_at_22": round(self._role_avg_gold22.get(role, 0.0), 0),
            "fight_strength":      round(self.fight_strength(champ, role), 4),
        }


# ---------------------------------------------------------------------------
# Patch Trend Model
# ---------------------------------------------------------------------------

class PatchTrendModel:
    ARROW_THRESHOLDS = [(+0.40, "↑↑"), (+0.15, "↑"), (-0.15, "→"), (-0.40, "↓"), (-1.00, "↓↓")]
    WR_WEIGHT:   float = 0.15
    PB_WEIGHT:   float = 0.45
    TEAM_WEIGHT: float = 0.40

    def __init__(self, config: ModelConfig, bt_model: TeamStrengthModel):
        self.config  = config
        self.bt      = bt_model
        self.trend_scores_:       dict[tuple[str, str], float] = {}
        self.predicted_delta_:    dict[tuple[str, str], float] = {}
        self.wr_slope_:           dict[tuple[str, str], float] = {}
        self.pb_slope_:           dict[tuple[str, str], float] = {}
        self.team_quality_delta_: dict[tuple[str, str], float] = {}
        self._patches:            list = []

    def fit(self, player_df: pd.DataFrame, game_df: pd.DataFrame) -> "PatchTrendModel":
        patches = sorted(player_df["patch"].dropna().unique())
        self._patches = patches

        if len(patches) < 2:
            print("[PatchTrend] Fewer than 2 patches — no trend data.")
            return self

        recency_weight: dict = {}
        has_date = "date" in player_df.columns

        for patch in patches:
            patch_rows = player_df[player_df["patch"] == patch].drop_duplicates("gameid")
            if has_date:
                patch_rows = patch_rows.copy()
                patch_rows["_date_parsed"] = pd.to_datetime(patch_rows["date"], errors="coerce")
                patch_rows = patch_rows.sort_values("_date_parsed")
            else:
                patch_rows = patch_rows.sort_values("gameid")
            sorted_gids = patch_rows["gameid"].tolist()
            n = len(sorted_gids)
            for rank, gid in enumerate(sorted_gids):
                recency_weight[gid] = (2.0 - 1.5 * (rank / (n - 1))) if n > 1 else 1.0

        wr_by_patch:    dict[tuple, list]  = {}
        pb_by_patch:    dict[tuple, float] = {}
        sides_by_patch: dict[object, float] = {}

        for _, row in player_df.iterrows():
            champ = row["champion"]
            role  = row["position"]
            patch = row["patch"]
            res   = int(row["result"])
            wr_by_patch.setdefault((champ, role, patch), []).append(res)
            pb_by_patch[(champ, role, patch)] = pb_by_patch.get((champ, role, patch), 0.0) + 1.0
            sides_by_patch[patch] = sides_by_patch.get(patch, 0.0) + 1.0

        team_bt_by_patch: dict[tuple, list] = {}
        gid_team: dict[tuple, str] = {}
        if "teamname" in game_df.columns:
            for _, row in game_df.iterrows():
                gid_team[(row["gameid"], row["side"])] = str(row.get("teamname", ""))

        for _, row in player_df.iterrows():
            champ = row["champion"]
            role  = row["position"]
            patch = row["patch"]
            gid   = row["gameid"]
            side  = row["side"]
            tname = gid_team.get((gid, side), "")
            bt    = self.bt.strength(tname) if tname else 1.0
            rw    = recency_weight.get(gid, 1.0)
            team_bt_by_patch.setdefault((champ, role, patch), []).append((bt, rw))

        patch_idx = {p: i for i, p in enumerate(patches)}

        def _slope(xs: list[float], ys: list[float]) -> float:
            if len(xs) < 2:
                return 0.0
            xm = np.mean(xs)
            ym = np.mean(ys)
            num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
            den = sum((x - xm) ** 2 for x in xs)
            return num / den if den > 0 else 0.0

        all_keys       = set((c, r) for (c, r, _) in wr_by_patch)
        recent_patches = set(patches[-2:])
        min_total      = self.config.MIN_CHAMP_GAMES

        for (champ, role) in all_keys:
            total_games = sum(len(wr_by_patch.get((champ, role, p), [])) for p in patches)
            if total_games < min_total:
                continue
            recent_picks = sum(pb_by_patch.get((champ, role, p), 0) for p in recent_patches)
            if recent_picks < 2:
                continue

            wr_xs, wr_ys, pb_xs, pb_ys, bt_xs, bt_ys = [], [], [], [], [], []

            for patch in patches:
                xi = float(patch_idx[patch])
                results = wr_by_patch.get((champ, role, patch), [])
                if len(results) >= 5:
                    wr_xs.append(xi)
                    wr_ys.append(float(np.mean(results)))
                picks = pb_by_patch.get((champ, role, patch), 0.0)
                total = sides_by_patch.get(patch, 1.0)
                if total > 0:
                    pb_xs.append(xi)
                    pb_ys.append(picks / total)
                bt_pairs = team_bt_by_patch.get((champ, role, patch), [])
                if bt_pairs:
                    tot_w = sum(rw for _, rw in bt_pairs)
                    wt_bt = sum(bt * rw for bt, rw in bt_pairs) / tot_w if tot_w > 0 else 1.0
                    bt_xs.append(xi)
                    bt_ys.append(wt_bt)

            if len(wr_xs) < 2 and len(pb_xs) < 2:
                continue

            ws = _slope(wr_xs, wr_ys)
            ps = _slope(pb_xs, pb_ys)

            recent_patch_indices = {float(patch_idx[p]) for p in recent_patches}
            if len(bt_ys) >= 2 and bt_xs[-1] in recent_patch_indices:
                td = bt_ys[-1] - bt_ys[0]
                td += max(0.0, bt_ys[-1] - 1.0) * 0.5
            elif len(bt_ys) >= 2:
                td = 0.0
            elif len(bt_ys) == 1 and bt_xs[0] in recent_patch_indices:
                td = max(0.0, bt_ys[0] - 1.0) * 0.3
            else:
                td = 0.0

            self.wr_slope_[(champ, role)]           = round(ws, 5)
            self.pb_slope_[(champ, role)]           = round(ps, 5)
            self.team_quality_delta_[(champ, role)] = round(td, 4)

            trend = (self.WR_WEIGHT   * np.clip(ws * 10,  -1, 1) +
                     self.PB_WEIGHT   * np.clip(ps * 20,  -1, 1) +
                     self.TEAM_WEIGHT * np.clip(td * 1.5, -1, 1))
            self.trend_scores_[(champ, role)]    = round(float(np.clip(trend, -1, 1)), 3)
            self.predicted_delta_[(champ, role)] = round(float(trend) * 10, 2)

        print(f"[PatchTrend] {len(self.trend_scores_)} (champ,role) trends across {len(patches)} patches")
        return self

    def trend_arrow(self, champ: str, role: str) -> str:
        score = self.trend_scores_.get((champ, role), 0.0)
        for threshold, arrow in self.ARROW_THRESHOLDS:
            if score >= threshold:
                return arrow
        return "→"

    def trend_detail(self, champ: str, role: str) -> dict:
        return {
            "trend_score":        self.trend_scores_.get((champ, role), 0.0),
            "predicted_delta":    self.predicted_delta_.get((champ, role), 0.0),
            "wr_slope":           self.wr_slope_.get((champ, role), 0.0),
            "pb_slope":           self.pb_slope_.get((champ, role), 0.0),
            "team_quality_delta": self.team_quality_delta_.get((champ, role), 0.0),
            "arrow":              self.trend_arrow(champ, role),
        }

    def rising_champions(self, top_n: int = 10) -> pd.DataFrame:
        rows = [
            {
                "Champion":             c,
                "Role":                 r,
                "Trend":                self.trend_arrow(c, r),
                "Trend Score":          s,
                "Predicted Δ Priority": self.predicted_delta_.get((c, r), 0.0),
                "WR Slope/patch":       round(self.wr_slope_.get((c, r), 0.0) * 100, 2),
                "P+B Slope/patch":      round(self.pb_slope_.get((c, r), 0.0) * 100, 2),
                "Team Quality Δ":       round(self.team_quality_delta_.get((c, r), 0.0), 3),
            }
            for (c, r), s in self.trend_scores_.items()
        ]
        return (pd.DataFrame(rows).sort_values("Trend Score", ascending=False)
                .head(top_n).reset_index(drop=True))

    def falling_champions(self, top_n: int = 10) -> pd.DataFrame:
        rows = [
            {
                "Champion":             c,
                "Role":                 r,
                "Trend":                self.trend_arrow(c, r),
                "Trend Score":          s,
                "Predicted Δ Priority": self.predicted_delta_.get((c, r), 0.0),
                "WR Slope/patch":       round(self.wr_slope_.get((c, r), 0.0) * 100, 2),
                "P+B Slope/patch":      round(self.pb_slope_.get((c, r), 0.0) * 100, 2),
                "Team Quality Δ":       round(self.team_quality_delta_.get((c, r), 0.0), 3),
            }
            for (c, r), s in self.trend_scores_.items()
        ]
        return (pd.DataFrame(rows).sort_values("Trend Score", ascending=True)
                .head(top_n).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Model Evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    def evaluate(self, model: "DraftModel", game_df: pd.DataFrame) -> dict:
        patches = sorted(game_df["patch"].dropna().unique())

        if len(patches) >= 2:
            test_patch  = patches[-1]
            test_df     = game_df[game_df["patch"] == test_patch]
            split_label = f"patch {test_patch} (out-of-sample)"
        else:
            test_df     = game_df
            split_label = "all patches (only 1 patch available — in-sample)"
            print("[Evaluator] WARNING: only one patch — evaluation is in-sample.")

        blue_rows = test_df[test_df["side"] == "Blue"].set_index("gameid")
        red_rows  = test_df[test_df["side"] == "Red"].set_index("gameid")
        common    = blue_rows.index.intersection(red_rows.index)
        blue_rows = blue_rows.loc[common]
        red_rows  = red_rows.loc[common]

        y_true, y_pred = [], []
        for gid in common:
            bc     = blue_rows.at[gid, "champs"]
            rc     = red_rows.at[gid,  "champs"]
            actual = int(blue_rows.at[gid, "result"])
            try:
                res    = model.predict(bc, rc, team1_is_blue=True)
                p_blue = res["win_probability"]
            except Exception:
                p_blue = 0.5
            y_true.append(actual)
            y_pred.append(p_blue)

        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        n      = len(y_true)

        accuracy = float(np.mean((y_pred >= 0.5).astype(float) == y_true))
        baseline = float(np.mean(y_true))

        eps      = 1e-7
        p        = np.clip(y_pred, eps, 1 - eps)
        log_loss = float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
        brier    = float(np.mean((y_pred - y_true) ** 2))

        bins    = np.linspace(0, 1, 11)
        bin_idx = np.clip(np.digitize(y_pred, bins) - 1, 0, 9)
        cal_rows = []
        for b in range(10):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            cal_rows.append({
                "Predicted Range": f"{bins[b]:.0%}–{bins[b+1]:.0%}",
                "Predicted Mean":  round(float(y_pred[mask].mean()), 3),
                "Actual Win Rate": round(float(y_true[mask].mean()), 3),
                "Games":           int(mask.sum()),
            })

        detail_rows = [
            {
                "Game ID":       gid,
                "P(Blue Win)":   round(float(y_pred[i]), 3),
                "Actual Winner": "Blue" if y_true[i] == 1 else "Red",
                "Correct":       "✅" if (y_pred[i] >= 0.5) == (y_true[i] == 1) else "❌",
            }
            for i, gid in enumerate(common)
        ]

        print(f"[Evaluator] {split_label} | Accuracy={accuracy:.3f} | "
              f"Baseline={baseline:.3f} | LogLoss={log_loss:.4f} | Brier={brier:.4f} | N={n}")

        return {
            "accuracy":          round(accuracy, 4),
            "baseline_accuracy": round(baseline, 4),
            "log_loss":          round(log_loss, 4),
            "brier_score":       round(brier, 4),
            "n_games":           n,
            "test_patch":        str(patches[-1]) if len(patches) >= 2 else (str(patches[0]) if len(patches) == 1 else "all"),
            "calibration_df":    pd.DataFrame(cal_rows),
            "detail_df":         pd.DataFrame(detail_rows),
            "y_true":            y_true,
            "y_pred":            y_pred,
        }


# ---------------------------------------------------------------------------
# Draft Model  (Facade)
# ---------------------------------------------------------------------------

class DraftModel:
    """
    Facade combining all sub-models into a single predict/recommend interface.

    Sub-models (fitted in order):
      TeamStrengthModel   — Bradley-Terry team ratings (used to weight training data)
      BaseStrengthModel   — Role-aware champion win rates, strength-of-schedule corrected
      SynergyModel        — Champion pair synergy (win rate + @15 gold for lane pairs)
      CounterModel        — Direct lane matchup win rate deltas + @15 signal
      TeamfightModel      — Gold@22 normalised by role baseline → fight power score
      PatchTrendModel     — Win rate / pick-ban slope across patches

    Prediction formula (logit space, then sigmoid):
      logit = w_base*ΔBase + w_synergy*ΔSynergy + w_counter*Counter + w_fight*ΔFight
      P(team1 wins) = sigmoid(LOGIT_SCALE * logit)

    LOGIT_SCALE amplifies the raw logit so that meaningful draft advantages
    produce meaningfully different probabilities (e.g. 55% vs 45% not 51% vs 49%).
    """
    LOGIT_SCALE: float = 8.0   # tune here to widen/narrow probability spread

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config             = config or ModelConfig()
        self.bt_model:          Optional[TeamStrengthModel]  = None
        self.base_model:        Optional[BaseStrengthModel]  = None
        self.synergy_model:     Optional[SynergyModel]       = None
        self.counter_model:     Optional[CounterModel]       = None
        self.teamfight_model:   Optional[TeamfightModel]     = None
        self.patch_trend_model: Optional[PatchTrendModel]    = None
        self._fitted            = False
        self._game_df:          Optional[pd.DataFrame]       = None
        self._player_df:        Optional[pd.DataFrame]       = None
        self._raw_df:           Optional[pd.DataFrame]       = None

    def fit(
        self,
        player_df:  pd.DataFrame,
        game_df:    pd.DataFrame,
        team_level: pd.DataFrame,
        raw_df:     pd.DataFrame,
    ) -> "DraftModel":
        self.bt_model          = TeamStrengthModel(self.config).fit(team_level)
        self.base_model        = BaseStrengthModel(self.config, self.bt_model).fit(player_df, game_df)
        self.synergy_model     = SynergyModel(self.config, self.base_model, self.bt_model).fit(game_df, raw_df)
        self.counter_model     = CounterModel(self.config, self.base_model).fit(player_df, raw_df)
        self.teamfight_model   = TeamfightModel(self.config).fit(player_df, raw_df)
        self.patch_trend_model = PatchTrendModel(self.config, self.bt_model).fit(player_df, game_df)

        self._game_df   = game_df.copy()
        self._player_df = player_df.copy()
        self._raw_df    = raw_df
        self._fitted  = True
        return self

    def predict(
        self,
        team1: list[str],
        team2: list[str],
        team1_is_blue: bool = False,
        verbose: bool = False,
    ) -> dict:
        """
        Predict win probability for team1 vs team2.

        Formula:
          ΔBase    = mean(WR_team1_by_role) − mean(WR_team2_by_role)
          ΔSynergy = mean_pair_synergy(team1) − mean_pair_synergy(team2)
          Counter  = mean_role_direct_score(team1 vs team2)  [net, already a delta]
          ΔFight   = mean_gold22_norm(team1) − mean_gold22_norm(team2)

          logit    = w_base·ΔBase + w_syn·ΔSynergy + w_ctr·Counter + w_fgt·ΔFight
          P(win)   = sigmoid(LOGIT_SCALE · logit)

        LOGIT_SCALE (default 8.0) widens the probability distribution so that
        real draft advantages produce meaningfully different win probabilities
        rather than clustering around 50%.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")

        roles = self.config.ROLES
        w_b   = self.config.WEIGHT_BASE
        w_s   = self.config.WEIGHT_SYNERGY
        w_c   = self.config.WEIGHT_COUNTER
        w_f   = self.config.WEIGHT_FIGHT
        w_0   = self.config.WEIGHT_INTERCEPT

        t1_base = self.base_model.team_score(team1, roles)
        t2_base = self.base_model.team_score(team2, roles)
        bd      = t1_base - t2_base

        t1_syn = self.synergy_model.synergy_score(team1)
        t2_syn = self.synergy_model.synergy_score(team2)
        sd     = t1_syn - t2_syn

        counter_net, counter_breakdown = self.counter_model.counter_score(team1, team2)
        fight_net, fight_detail        = self.teamfight_model.teamfight_score(team1, team2, roles)

        logit    = w_0 + w_b * bd + w_s * sd + w_c * counter_net + w_f * fight_net
        win_prob = float(1.0 / (1.0 + np.exp(-self.LOGIT_SCALE * logit)))

        result = {
            "win_probability":   round(win_prob, 4),
            "logit_score":       round(logit, 4),
            "team1_base":        round(t1_base, 4),
            "team2_base":        round(t2_base, 4),
            "team1_synergy":     round(t1_syn, 4),
            "team2_synergy":     round(t2_syn, 4),
            "counter_net":       round(counter_net, 4),
            "fight_net":         round(fight_net, 4),
            "counter_breakdown": counter_breakdown,
            "synergy_detail_t1": self.synergy_model.synergy_detail(team1),
            "synergy_detail_t2": self.synergy_model.synergy_detail(team2),
            "fight_detail":      fight_detail,
            "breakdown": {
                "base":    round(w_b * bd,          4),
                "synergy": round(w_s * sd,          4),
                "counter": round(w_c * counter_net, 4),
                "fight":   round(w_f * fight_net,   4),
            },
        }

        if verbose:
            s1 = "Blue" if team1_is_blue else "Red"
            s2 = "Red"  if team1_is_blue else "Blue"
            print(f"\n=== DraftModel Prediction ===")
            print(f"Team 1 ({s1}): {team1}  Team 2 ({s2}): {team2}")
            print(f"Base    T1={t1_base:.4f}  T2={t2_base:.4f}  Δ={bd:+.4f}")
            print(f"Synergy T1={t1_syn:.4f}  T2={t2_syn:.4f}  Δ={sd:+.4f}")
            print(f"Counter net={counter_net:+.4f}  Fight net={fight_net:+.4f}")
            print(f"Logit={logit:.4f} × scale={self.LOGIT_SCALE} → P(T1 wins)={win_prob:.4f} ({win_prob*100:.1f}%)")

        return result

    def evaluate(self) -> dict:
        if self._game_df is None:
            raise RuntimeError("No game data stored.")
        return ModelEvaluator().evaluate(self, self._game_df)

    def champion_list(self) -> list[str]:
        return self.base_model.known_champions()

    def save(self, path: Optional[str] = None) -> None:
        out = path or self.config.CACHE_PATH
        with open(out, "wb") as f:
            pickle.dump(self, f)
        print(f"[DraftModel] Saved to {out}")

    @classmethod
    def load(cls, path: str) -> "DraftModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"[DraftModel] Loaded from {path}")
        return model

    def draft_win_expectation(
        self,
        team_champs: list[str],
        opp_champs:  list[str],
        actual_result: int,
        team_is_blue: bool = True,
    ) -> dict:
        """
        For a single completed game, return the model's predicted win probability
        and the residual (actual - predicted). Positive residual = outperformed draft.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .draft_win_expectation()")
        try:
            res = self.predict(team_champs, opp_champs, team1_is_blue=team_is_blue)
            wp  = float(res["win_probability"])
        except Exception:
            wp  = 0.5
        return {
            "predicted_wp": round(wp, 4),
            "actual":       actual_result,
            "residual":     round(actual_result - wp, 4),
        }

    def team_draft_grades(
        self,
        game_df: Optional[pd.DataFrame] = None,
        min_games: int = 5,
    ) -> pd.DataFrame:
        """
        For every team in game_df compute:
          - mean residual  = mean(actual_result - predicted_wp)  [patch-weighted]
          - draft_wr       = actual wins / games
          - model_wr       = mean predicted_wp
          - outperformance = draft_wr - model_wr  (same as mean residual)

        Teams are then ranked into percentile buckets S–D:
          S = top 10%   A = 70–90th   B = 40–70th   C = 20–40th   D = bottom 20%

        Returns a DataFrame sorted by outperformance descending.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .team_draft_grades()")

        gdf = game_df if game_df is not None else self._game_df
        if gdf is None or "teamname" not in gdf.columns:
            return pd.DataFrame()

        # Build opponent lookup: for each (gameid, side) find the other side's champs
        blue = gdf[gdf["side"] == "Blue"].set_index("gameid")[["champs", "teamname", "result", "patch_weight"]]
        red  = gdf[gdf["side"] == "Red"].set_index("gameid")[["champs", "teamname", "result", "patch_weight"]]
        common = blue.index.intersection(red.index)

        team_stats: dict[str, dict] = {}

        for gid in common:
            b_champs = blue.at[gid, "champs"]
            r_champs = red.at[gid,  "champs"]
            b_result = int(blue.at[gid, "result"])
            r_result = int(red.at[gid,  "result"])
            b_team   = str(blue.at[gid, "teamname"])
            r_team   = str(red.at[gid,  "teamname"])
            pw       = float(blue.at[gid, "patch_weight"])

            if not (isinstance(b_champs, list) and len(b_champs) == 5 and all(b_champs)):
                continue
            if not (isinstance(r_champs, list) and len(r_champs) == 5 and all(r_champs)):
                continue

            try:
                res = self.predict(b_champs, r_champs, team1_is_blue=True)
                b_wp = float(res["win_probability"])
            except Exception:
                b_wp = 0.5
            r_wp = 1.0 - b_wp

            for team, champs, result, wp in [
                (b_team, b_champs, b_result, b_wp),
                (r_team, r_champs, r_result, r_wp),
            ]:
                if team not in team_stats:
                    team_stats[team] = {
                        "weighted_residual": 0.0,
                        "weighted_actual":   0.0,
                        "weighted_model_wp": 0.0,
                        "total_weight":      0.0,
                        "games":             0,
                    }
                s = team_stats[team]
                s["weighted_residual"] += pw * (result - wp)
                s["weighted_actual"]   += pw * result
                s["weighted_model_wp"] += pw * wp
                s["total_weight"]      += pw
                s["games"]             += 1

        rows = []
        for team, s in team_stats.items():
            if s["games"] < min_games or s["total_weight"] <= 0:
                continue
            tw = s["total_weight"]
            outperf   = s["weighted_residual"] / tw
            draft_wr  = s["weighted_actual"]   / tw
            model_wr  = s["weighted_model_wp"] / tw
            region_games = self.bt_model.league_games_.get(team, {})
            region = max(region_games, key=region_games.get) if region_games else "?"
            rows.append({
                "team":           team,
                "outperformance": round(outperf,  4),
                "draft_wr":       round(draft_wr, 4),
                "model_wr":       round(model_wr, 4),
                "games":          s["games"],
                "bt_score":       round(self.bt_model.strength(team), 3),
                "region":         region,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("outperformance", ascending=False).reset_index(drop=True)

        # Percentile-based grades
        pct = df["outperformance"].rank(pct=True)
        def _grade(p: float) -> str:
            if p >= 0.90: return "S"
            if p >= 0.70: return "A"
            if p >= 0.40: return "B"
            if p >= 0.20: return "C"
            return "D"
        df["grade"] = pct.apply(_grade)

        return df

    def draft_grade_for_picks(
        self,
        team_champs: list[str],
        opp_champs:  list[str],
        team_is_blue: bool = True,
    ) -> dict:
        """
        Return a letter grade for a live draft based on predicted win probability
        relative to the distribution of win probabilities seen in training data.
        Used for the Draft tab indicator — no actual result needed.

        Grade based on predicted WP percentile vs all games in _game_df:
          S ≥ 90th pct   A ≥ 70th   B ≥ 40th   C ≥ 20th   D < 20th
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .draft_grade_for_picks()")

        try:
            res = self.predict(team_champs, opp_champs, team1_is_blue=team_is_blue)
            wp  = float(res["win_probability"])
            bd  = res["breakdown"]
        except Exception:
            return {"grade": "?", "win_probability": 0.5, "breakdown": {}}

        # Determine grade from WP alone (relative signal vs 50% baseline)
        # WP already encodes all four components so percentile grade is meaningful
        if wp >= 0.65:   grade = "S"
        elif wp >= 0.57: grade = "A"
        elif wp >= 0.50: grade = "B"
        elif wp >= 0.43: grade = "C"
        else:            grade = "D"

        return {
            "grade":           grade,
            "win_probability": round(wp, 4),
            "breakdown":       bd,
        }

    def recommend_pick(
        self,
        team1_so_far: list[str],
        team2_so_far: list[str],
        role: str,
        banned: Optional[list[str]] = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Recommend the best champion for team1's next pick in the given role.

        For each candidate champion that qualifies for this role:
          - Inserts it into team1's roster at the role slot
          - Runs predict(team1, team2) to get win probability
          - Returns top_n sorted by win probability

        Counter advantage shown is the full-team counter_net (not just the lane
        matchup) since the opponent may not have their role-slot filled yet.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .recommend_pick()")

        roles   = self.config.ROLES
        banned  = set(banned or [])
        already = set(c for c in team1_so_far + team2_so_far if isinstance(c, str) and c)

        candidates = [
            c for c in self.base_model.known_champions()
            if c not in already
            and c not in banned
            and self.base_model._role_qualifies(c, role)
            and self.base_model.role_counts_.get((c, role), 0) >= self.config.MIN_CHAMP_GAMES
        ]

        role_idx = roles.index(role) if role in roles else -1
        if role_idx < 0:
            return pd.DataFrame()

        def _fill(team, champ):
            t = list(team)
            if role_idx < len(t):
                t[role_idx] = champ
            return t

        rows = []
        for champ in candidates:
            t1 = (_fill(team1_so_far, champ) + [""] * 5)[:5]
            t2 = (list(team2_so_far)          + [""] * 5)[:5]
            res = self.predict(t1, t2)
            wp  = res["win_probability"]
            rows.append({
                "champion":      champ,
                "win_prob_%":    round(wp * 100, 1),
                "base_wr_%":     round(self.base_model.champion_winrate(champ, role) * 100, 1),
                "counter_adv_%": round(res["counter_net"] * 100, 2),
                "fight_adv":     round(res["fight_net"], 4),
                "games":         self.base_model.role_counts_.get((champ, role), 0),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("win_prob_%", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )


ROLE_LABELS_SHORT = {"top": "Top", "jng": "Jng", "mid": "Mid", "bot": "Bot", "sup": "Sup"}


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def build_model(
    csv_path: str,
    config: Optional[ModelConfig] = None,
    save_cache: bool = False,
) -> DraftModel:
    cfg = config or ModelConfig()
    ld  = DataLoader(cfg)
    player_df, game_df, team_level, raw_df = ld.load(csv_path)
    m   = DraftModel(cfg).fit(player_df, game_df, team_level, raw_df)
    if save_cache:
        m.save()
    return m


def load_or_build(
    csv_path: str,
    config: Optional[ModelConfig] = None,
    cache_path: str = "model_cache.pkl",
) -> DraftModel:
    if Path(cache_path).exists():
        print(f"[Pipeline] Loading cache from {cache_path}")
        return DraftModel.load(cache_path)
    print(f"[Pipeline] Building from {csv_path}")
    return build_model(csv_path, config, save_cache=True)
