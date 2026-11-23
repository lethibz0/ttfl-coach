"""Probabilité qu'un joueur de la rotation joue un match donné.

Couche 1 (apprise sur l'historique) : présence récente, back-to-back, enchaînement,
retour d'absence, rôle. Entraînée sans le rapport de blessures (les rapports historiques
d'ESPN sont figés après coup, donc inutilisables).
Couche 2 (le jour J) : statut et commentaire du rapport de blessures.
"""
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATS = ["avail10", "avail30", "missed_last", "streak_missed", "b2b", "rest", "games_last4",
         "min_ewm", "starter_ewm", "home", "gp_team"]


def panel(box, games, future=None):
    """Une ligne par (joueur de la rotation, match de son équipe)."""
    g = games[["game_id", "date", "home_id", "away_id"]].copy()
    g["date"] = pd.to_datetime(g.date)
    tg = pd.concat([g.assign(team_id=g.home_id, home=1), g.assign(team_id=g.away_id, home=0)])
    tg = tg.sort_values(["team_id", "date"]).reset_index(drop=True)
    tg["rest"] = tg.groupby("team_id").date.diff().dt.days.clip(upper=5).fillna(5)
    tg["b2b"] = (tg.rest == 1).astype(int)
    tg["games_last4"] = tg.groupby("team_id").date.transform(
        lambda d: [(d[(d < x) & (d >= x - pd.Timedelta(days=3))]).size for x in d])
    tg["tg_idx"] = tg.groupby("team_id").cumcount()

    b = box[(~box.dnp) & (box["min"] > 0)][["game_id", "team_id", "player_id", "min", "starter"]]
    b = b.merge(tg[["game_id", "team_id", "tg_idx", "date"]], on=["game_id", "team_id"])
    # chaque joueur est suivi sur les matchs de son équipe à partir de son premier match avec elle,
    # jusqu'à 25 matchs d'équipe après sa dernière apparition
    rows = []
    for (pid, tid), h in b.groupby(["player_id", "team_id"]):
        idxs = h.tg_idx.values
        lo, hi = idxs.min(), idxs.max() + 25
        span = tg[(tg.team_id == tid) & (tg.tg_idx >= lo) & (tg.tg_idx <= hi)]
        played = dict(zip(h.tg_idx, h["min"]))
        starter = dict(zip(h.tg_idx, h.starter.astype(float)))
        rows.append(pd.DataFrame({
            "player_id": pid, "team_id": tid, "game_id": span.game_id.values, "tg_idx": span.tg_idx.values,
            "played": [i in played for i in span.tg_idx], "min": [played.get(i, np.nan) for i in span.tg_idx],
            "starter": [starter.get(i, np.nan) for i in span.tg_idx]}))
    p = pd.concat(rows, ignore_index=True)
    p = p.merge(tg[["game_id", "team_id", "date", "home", "rest", "b2b", "games_last4"]], on=["game_id", "team_id"])
    p = p.sort_values(["player_id", "team_id", "tg_idx"]).reset_index(drop=True)
    grp = p.groupby(["player_id", "team_id"])
    pl = p.played.astype(float)
    p["avail10"] = grp.played.transform(lambda x: x.astype(float).shift(1).rolling(10, min_periods=1).mean())
    p["avail30"] = grp.played.transform(lambda x: x.astype(float).shift(1).rolling(30, min_periods=1).mean())
    p["missed_last"] = 1 - grp.played.transform(lambda x: x.astype(float).shift(1)).fillna(1)
    # nombre de matchs consécutifs manqués juste avant
    def streak(x):
        out, s = [], 0
        for v in x.shift(1).fillna(True):
            s = 0 if v else s + 1
            out.append(s)
        return out
    p["streak_missed"] = grp.played.transform(streak)
    p["min_ewm"] = grp["min"].transform(lambda x: x.shift(1).ewm(halflife=7, ignore_na=True).mean())
    p["starter_ewm"] = grp.starter.transform(lambda x: x.shift(1).ewm(halflife=5, ignore_na=True).mean())
    p["gp_team"] = grp.cumcount()
    # rotation uniquement, et exclusion des absences longues (> 10 matchs : blessures connues)
    p = p[(p.min_ewm >= 15) & (p.gp_team >= 3) & (p.streak_missed <= 10)]
    return p


def fit(p):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                       min_samples_leaf=200, random_state=0)
    m.fit(p[FEATS], p.played.astype(int))
    return m


# --- Couche 2 : rapport de blessures du jour ---
KEYWORDS = [  # (motif, probabilité de jouer) : le premier qui correspond l'emporte
    (r"won't play|will not play|ruled out|out for|will miss|won't suit up|is out|out indefinitely|season-ending|surgery", 0.03),
    (r"doubtful", 0.15),
    (r"questionable|game-time decision|uncertain", 0.55),
    (r"probable|expected to play|likely to play|on track to play", 0.90),
    (r"will play|available|cleared|will return|set to return|upgraded", 0.95),
]
STATUS_DEFAULT = {"Out": 0.03, "Doubtful": 0.15, "Questionable": 0.55, "Day-To-Day": 0.55, "Probable": 0.90}


WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def status_prob(status, comment, game_date):
    """Probabilité de jouer d'après le rapport, ou (None, False) si le joueur n'y figure pas.
    Si le commentaire parle d'un autre jour que celui du match (ex. « won't play Sunday »
    alors qu'on joue lundi), il est périmé : on garde l'incertitude du statut seul."""
    if not status:
        return None, False
    c = (comment or "").lower()
    p = STATUS_DEFAULT.get(status, 0.55)
    tonight = WEEKDAYS[pd.Timestamp(game_date).weekday()]
    days = [d for d in WEEKDAYS if re.search(rf"\b{d}", c)]
    stale = bool(days) and tonight not in days
    if not stale:
        for pat, v in KEYWORDS:
            if re.search(pat, c):
                p = v
                break
    return p, stale


def combine(p_model, p_status):
    """Sans statut : l'absence surprise (non annoncée) est environ moitié moins fréquente
    que le taux historique brut, une partie des absences étant annoncée dans le rapport."""
    if p_status is None:
        return 1 - 0.5 * (1 - p_model)
    return p_status  # le rapport du jour prime sur l'historique
