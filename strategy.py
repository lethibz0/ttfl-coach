"""Stratégie sous contrainte « 30 jours » : affectation optimale joueurs -> soirs.

Chaque soir on résout, sur l'horizon des H prochains soirs, le problème :
    maximiser  somme valeur(joueur, soir)
    s.c.       un joueur par soir, chaque joueur au plus une fois,
               joueurs bloqués indisponibles avant la fin de leur blocage.
On ne joue que la décision du soir, puis on recommence le lendemain.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

COOLDOWN = 30


def player_state(hist, d):
    """État de chaque joueur à la date d : équipe actuelle, base, taux de présence."""
    h = hist[hist.date < d]
    last = h.sort_values("date").groupby("player_id").tail(1)[["player_id", "name", "team_id", "base", "ttfl", "date"]]
    # base mise à jour avec le dernier match (approx : moyenne pondérée)
    recent = h[h.date >= d - pd.Timedelta(days=40)]
    team_games = recent.groupby("team_id").game_id.nunique()
    played = recent[~recent.dnp].groupby("player_id").game_id.nunique()
    last = last.set_index("player_id")
    last["p_play"] = (played.reindex(last.index).fillna(0) / team_games.reindex(last.team_id).values).clip(0, 1)
    last = last[last.date >= d - pd.Timedelta(days=20)]  # joueurs actifs récemment
    return last


def plan(tonight, state, schedule, d, last_used, horizon=30, decay=1.0, n_players=80):
    """tonight : DataFrame (player_id, name, proj) joueurs disponibles ce soir.
    schedule : DataFrame (date, team_id) des matchs à venir.
    Retourne (pick_player_id, tableau de valeurs, affectation)."""
    nights = [d + pd.Timedelta(days=k) for k in range(horizon)]
    sched = schedule[(schedule.date > d) & (schedule.date < nights[-1] + pd.Timedelta(days=1))]
    teams_by_night = sched.groupby("date").team_id.apply(set).to_dict()
    nights = [d] + [n for n in nights[1:] if n in teams_by_night]

    cand = set(tonight.nlargest(40, "proj").player_id) | set(state.nlargest(n_players, "base").index)
    cand = [p for p in cand if p in state.index or p in set(tonight.player_id)]
    tproj = tonight.set_index("player_id").proj
    V = np.zeros((len(cand), len(nights)))
    for i, pid in enumerate(cand):
        unlock = last_used.get(pid)
        unlock = unlock + pd.Timedelta(days=COOLDOWN + 1) if unlock is not None else d
        if pid in tproj.index and d >= unlock:
            V[i, 0] = tproj[pid]
        if pid in state.index:
            st = state.loc[pid]
            for j, n in enumerate(nights[1:], start=1):
                if n >= unlock and st.team_id in teams_by_night.get(n, ()):
                    V[i, j] = st.base * st.p_play * decay ** j
    rows, cols = linear_sum_assignment(V, maximize=True)
    assign = {nights[c]: cand[r] for r, c in zip(rows, cols) if V[r, c] > 0}
    return assign.get(d), V, cand, nights, assign


def backtest(pred, games, start, end, horizon=30, decay=1.0):
    return backtest_full(pred, pred, games, start, end, horizon, decay)


def backtest_full(hist, pred, games, start, end, horizon=30, decay=1.0):
    """pred : lignes joueur-match avec 'proj' (prédictions walk-forward) + 'ttfl' réel."""
    sched = pd.concat([games[["date", "home_id"]].rename(columns={"home_id": "team_id"}),
                       games[["date", "away_id"]].rename(columns={"away_id": "team_id"})])
    sched["date"] = pd.to_datetime(sched.date)
    last_used, total_opt, total_greedy, lu_g = {}, 0.0, 0.0, {}
    log = []
    for d in sorted(pred[(pred.date >= start) & (pred.date <= end)].date.unique()):
        d = pd.Timestamp(d)
        tonight = pred[pred.date == d]
        state = player_state(hist, d)
        pick, *_ = plan(tonight, state, sched, d, last_used, horizon, decay)
        if pick is None:  # personne d'intéressant : on prend le meilleur dispo
            avail = tonight[[last_used.get(p) is None or (d - last_used[p]).days > COOLDOWN for p in tonight.player_id]]
            pick = avail.nlargest(1, "proj").player_id.iloc[0]
        r = tonight.set_index("player_id").loc[pick]
        last_used[pick] = d
        total_opt += r.ttfl
        # glouton de référence
        avail = tonight[[lu_g.get(p) is None or (d - lu_g[p]).days > COOLDOWN for p in tonight.player_id]]
        g = avail.nlargest(1, "proj").iloc[0]
        lu_g[g.player_id] = d
        total_greedy += g.ttfl
        log.append((d.date(), r["name"], round(r.proj, 1), r.ttfl, g["name"], round(g.proj, 1), g.ttfl))
    return total_opt, total_greedy, pd.DataFrame(log, columns=["date", "pick", "proj", "réel", "glouton", "proj_g", "réel_g"])
