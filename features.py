"""Construction des variables du modèle, sans fuite d'information :
chaque ligne (joueur, match) n'utilise que des données antérieures au match.

Le même code sert à l'entraînement (matchs passés) et à la prédiction
(matchs du soir ajoutés comme lignes « fantômes » sans statistiques).
"""
import numpy as np
import pandas as pd

POS_GROUP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C", "FC": "C", "GF": "F"}


def ewm_prior(s: pd.Series, groups, halflife):
    """Moyenne exponentielle des valeurs *précédentes* (shift 1) par groupe."""
    return s.groupby(groups).transform(lambda x: x.shift(1).ewm(halflife=halflife, min_periods=1).mean())


def roll_prior(s, groups, n):
    return s.groupby(groups).transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())


def load(root):
    box = pd.read_parquet(root / "data" / "box.parquet")
    games = pd.read_parquet(root / "data" / "games.parquet")
    odds = pd.read_parquet(root / "data" / "odds.parquet")
    games = games.merge(odds, on="game_id", how="left")
    return box, games


def team_games(box, games):
    """Une ligne par (équipe, match) : adversaire, domicile, rythme, repos, cotes."""
    g = games.copy()
    home = g.assign(team_id=g.home_id, opp_id=g.away_id, home=1,
                    spread=g.spread_home, pts_for=g.home_score, pts_against=g.away_score)
    away = g.assign(team_id=g.away_id, opp_id=g.home_id, home=0,
                    spread=-g.spread_home, pts_for=g.away_score, pts_against=g.home_score)
    tg = pd.concat([home, away])[["game_id", "date", "season", "team_id", "opp_id", "home",
                                  "spread", "total", "pts_for", "pts_against"]]
    tg["date"] = pd.to_datetime(tg.date)

    # Possessions estimées par équipe et par match
    played = box[~box.dnp]
    tot = played.groupby(["game_id", "team_id"])[["fga", "oreb", "tov", "fta"]].sum().reset_index()
    tot["poss"] = tot.fga - tot.oreb + tot.tov + 0.44 * tot.fta
    tg = tg.merge(tot[["game_id", "team_id", "poss"]], on=["game_id", "team_id"], how="left")

    tg = tg.sort_values(["team_id", "date"])
    tg["rest"] = tg.groupby("team_id").date.diff().dt.days.clip(upper=5).fillna(5)
    tg["b2b"] = (tg.rest == 1).astype(int)
    # 3 matchs en 4 jours
    tg["games_last4"] = tg.groupby("team_id").date.transform(
        lambda d: [(d[(d < x) & (d >= x - pd.Timedelta(days=3))]).size for x in d])
    tg["pace"] = ewm_prior(tg.poss, tg.team_id, 10)
    # Implied team total (marché) ; repli sur la moyenne de points marqués
    tg["pts_for_avg"] = ewm_prior(tg.pts_for, tg.team_id, 10)
    tg["implied"] = tg.total / 2 - tg.spread / 2
    return tg


def opponent_defense(box, tg):
    """TTFL concédé par l'adversaire à chaque groupe de poste, relatif à la moyenne ligue."""
    played = box[~box.dnp].copy()
    played["pg"] = played.pos.map(POS_GROUP).fillna("F")
    agg = played.groupby(["game_id", "team_id", "pg"]).agg(ttfl=("ttfl", "sum"), min=("min", "sum")).reset_index()
    agg["ttfl_pm"] = agg.ttfl / agg["min"].clip(lower=1)
    # côté défenseur : l'équipe adverse
    agg = agg.merge(tg[["game_id", "team_id", "opp_id", "date"]], on=["game_id", "team_id"])
    agg = agg.rename(columns={"opp_id": "def_id"}).sort_values("date")
    league = agg.groupby(["date", "pg"]).ttfl_pm.transform("mean")
    agg["rel"] = agg.ttfl_pm / league
    agg = agg.sort_values(["def_id", "pg", "date"])
    agg["def_rel"] = agg.groupby(["def_id", "pg"]).rel.transform(
        lambda x: x.shift(1).ewm(halflife=15, min_periods=1).mean())
    return agg


def build(box, games, future=None):
    """future : DataFrame (game_id, player_id, team_id, name, pos) de lignes à prédire."""
    box = box.copy()
    if future is not None and len(future):
        box = pd.concat([box, future.assign(dnp=False, _future=True)], ignore_index=True)
    box["_future"] = box.get("_future", False)
    box["_future"] = box["_future"].fillna(False).astype(bool)

    tg = team_games(box[~box._future], games)
    df = box.merge(tg, on=["game_id", "team_id"], how="inner")
    df = df.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
    df["pg"] = df.pos.map(POS_GROUP).fillna("F")

    # --- Historique du joueur (uniquement matchs joués) ---
    p = df[~df.dnp].copy()
    p["ttfl_pm"] = np.where(p["min"] > 0, p.ttfl / p["min"].clip(lower=1), np.nan)
    p["usg_pm"] = (p.fga + 0.44 * p.fta + p.tov) / p["min"].clip(lower=1)
    hist_cols = {"ttfl": "ttfl", "min": "min", "ttfl_pm": "ttfl_pm", "usg_pm": "usg_pm", "starter": "starter"}
    for c in hist_cols:
        p[c] = p[c].astype(float)
        p.loc[p._future, c] = np.nan  # les lignes futures n'ont pas de stats
    grp = p.player_id
    feats = pd.DataFrame(index=p.index)
    for c in ["ttfl", "min", "ttfl_pm"]:
        feats[f"{c}_l3"] = roll_prior(p[c], grp, 3)
        feats[f"{c}_l10"] = roll_prior(p[c], grp, 10)
        feats[f"{c}_ewm"] = ewm_prior(p[c], grp, 7)
    feats["usg_ewm"] = ewm_prior(p.usg_pm, grp, 7)
    feats["starter_ewm"] = ewm_prior(p.starter, grp, 5)
    feats["ttfl_std10"] = p.ttfl.groupby(grp).transform(lambda x: x.shift(1).rolling(10, min_periods=3).std())
    feats["gp"] = p.groupby("player_id").cumcount()
    # saison en cours vs saison précédente
    feats["gp_season"] = p.groupby(["player_id", "season"]).cumcount()
    prev = p[~p._future].groupby(["player_id", "season"]).agg(prev_ttfl=("ttfl", "mean"), prev_min=("min", "mean")).reset_index()
    prev["season"] += 1
    p = p.join(feats)
    p = p.merge(prev, on=["player_id", "season"], how="left")
    # jours depuis le dernier match joué (retour de blessure)
    p["days_since"] = p.groupby("player_id").date.diff().dt.days.fillna(30).clip(upper=30)

    # --- Coéquipiers absents ---
    # minutes récentes attendues de chaque joueur (ewm) pour chaque match de son équipe
    p = absent_teammates(df, p)

    # --- Défense adverse ---
    od = opponent_defense(box[~box._future], tg)
    last_def = od[["def_id", "pg", "date", "def_rel"]]
    p = p.sort_values("date")
    p = pd.merge_asof(p, last_def.sort_values("date").rename(columns={"def_id": "opp_id", "def_rel": "opp_def"}),
                      on="date", by=["opp_id", "pg"], allow_exact_matches=False, direction="backward")
    # rythme de l'adversaire
    opp_pace = tg[["game_id", "team_id", "pace", "b2b", "rest"]].rename(
        columns={"team_id": "opp_id", "pace": "opp_pace", "b2b": "opp_b2b", "rest": "opp_rest"})
    p = p.merge(opp_pace, on=["game_id", "opp_id"], how="left")
    p["exp_pace"] = (p.pace + p.opp_pace) / 2
    p["abs_spread"] = p.spread.abs()
    p = add_base(p)
    return p


def add_base(p):
    """Niveau de base du joueur : moyenne de saison rétrécie vers la saison précédente,
    repli sur la moyenne exponentielle (halflife 10)."""
    p = p.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
    t = p.ttfl.where(~p._future & ~p.dnp)
    p["ewm10"] = F_ewm(t, p.player_id, 10)
    p["season_avg"] = t.groupby([p.player_id, p.season]).transform(lambda x: x.shift(1).expanding().mean())
    n = p.gp_season.fillna(0)
    prior = p.prev_ttfl.fillna(p.ewm10)
    p["base"] = ((p.season_avg.fillna(0) * n + prior * 8) / (n + 8)).fillna(p.ewm10)
    p["base"] = p.base.fillna(p.prev_ttfl).fillna(10)
    return p


def F_ewm(s, g, hl):
    return s.groupby(g).transform(lambda x: x.shift(1).ewm(halflife=hl, min_periods=1).mean())


def absent_teammates(df, p):
    """Pour chaque (équipe, match), somme des minutes/usage habituels des joueurs
    de la rotation qui ne jouent pas ce soir-là."""
    # minutes attendues = ewm des minutes jouées précédentes (dnp => pas de mise à jour)
    exp = p[["player_id", "team_id", "game_id", "date", "min_ewm", "usg_ewm", "ttfl_ewm"]].copy()
    # rotation récente de chaque équipe : joueurs ayant joué pour l'équipe dans les 10 derniers jours
    team_dates = df[["team_id", "game_id", "date"]].drop_duplicates()
    played = p[~p._future][["player_id", "team_id", "date", "min", "ttfl"]]
    played = played.assign(usg=p.loc[~p._future, "usg_pm"] * p.loc[~p._future, "min"])
    present_map = df[~df.dnp].groupby(["game_id", "team_id"]).player_id.apply(set).to_dict()
    out = []
    for tid, tdates in team_dates.groupby("team_id"):
        pl = played[played.team_id == tid]
        for _, row in tdates.iterrows():
            d = row.date
            recent = pl[(pl.date < d) & (pl.date >= d - pd.Timedelta(days=12))]
            if recent.empty:
                out.append((row.game_id, tid, 0.0, 0.0, 0.0, 0))
                continue
            avg = recent.groupby("player_id").agg(m=("min", "mean"), u=("usg", "mean"), t=("ttfl", "mean"), n=("min", "size"))
            rot = avg[(avg.m >= 12) & (avg.n >= 2)]
            present = present_map.get((row.game_id, tid), set())
            miss = rot[~rot.index.isin(present)]
            out.append((row.game_id, tid, miss.m.sum(), miss.u.sum(), miss.t.sum(), len(miss)))
    miss = pd.DataFrame(out, columns=["game_id", "team_id", "miss_min", "miss_usg", "miss_ttfl", "miss_n"])
    return p.merge(miss, on=["game_id", "team_id"], how="left")


# Variables de contexte utilisées pour l'ajustement autour de la base
CONTEXT = ["home", "rest", "b2b", "games_last4", "opp_b2b", "opp_rest", "opp_def", "pace", "opp_pace",
           "exp_pace", "spread", "abs_spread", "total", "implied", "miss_min", "miss_usg", "miss_ttfl",
           "miss_n", "min_l3", "min_ewm", "starter_ewm", "usg_ewm", "days_since", "gp_season", "base"]

FEATURES = [
    "ttfl_l3", "ttfl_l10", "ttfl_ewm", "min_l3", "min_l10", "min_ewm",
    "ttfl_pm_l3", "ttfl_pm_l10", "ttfl_pm_ewm", "usg_ewm", "starter_ewm", "ttfl_std10",
    "gp", "gp_season", "prev_ttfl", "prev_min", "days_since",
    "home", "rest", "b2b", "games_last4", "opp_b2b", "opp_rest",
    "opp_def", "pace", "opp_pace", "exp_pace",
    "spread", "abs_spread", "total", "implied", "pts_for_avg",
    "miss_min", "miss_usg", "miss_ttfl", "miss_n",
]
